import os
import sys
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = "C:/Experiments/BEBLaDII"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.beb_la_dii.model.dus import DUSModel
from src.beb_la_dii.utils.loss import safe_normalize


def normalize_key(k: str) -> str:
    """Удаляет FSDP, DDP, Dynamo, EMA и DUS префиксы для гарантированного маппинга."""
    return (
        k.replace("._orig_module.", ".")
        .replace("_orig_module.", "")
        .replace(".module.", ".")
        .replace("module.", "")
        .replace("student.model.", "")
        .replace("student.", "")
        .replace("model.", "")
    )


class LatentDecompressorDecoder(nn.Module):
    """
    Декомпрессирующий декодер для Plan C:
    Принимает сжатый холст длины N (где 1 токен = K=4 исходных слова/BPE).
    1. Контекстуализация концептов: слои ModernBERT на длине N.
    2. Геометрическая распаковка 1 -> 4: обучаемый Linear проектор (1024 -> 4 * 1024).
    3. Орфографическое и синтаксическое сглаживание: слои ModernBERT на полной длине 4N.
    4. Финальная нормализация ModernBERT final_norm.
    5. Выходная проекция 1024 -> 1536 для LM-головы Qwen.
    """
    def __init__(
        self,
        latent_dim: int = 1024,
        qwen_dim: int = 1536,
        num_context_layers: int = 2,
        num_smoothing_layers: int = 2,
        chunk_size: int = 4,
        local_window_size: int = 128,
        dus_weights_path: str = None,
        phase2_weights_path: str = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.qwen_dim = qwen_dim
        self.chunk_size = chunk_size
        self.local_window_size = local_window_size

        # Загружаем базовую модель ModernBERT для извлечения слоев
        if dus_weights_path and os.path.exists(dus_weights_path):
            print(f"Initializing Decompressor from DUS backbone: {dus_weights_path}")
            dus = DUSModel.from_scratch(weights_path=None)
            
            # Загружаем веса из чекпоинта Фазы 4 с нормализацией ключей
            try:
                p4 = torch.load(dus_weights_path, map_location="cpu", weights_only=False)
                dus_st = p4.get("dus_ema", p4.get("dus", p4.get("model_state_dict", p4)))
                clean_dus = {normalize_key(k): v for k, v in dus_st.items()}
                res = dus.model.load_state_dict(clean_dus, strict=False)
                print(f"   -> DUS weights loaded: missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)}")
            except Exception as e:
                print(f"   -> [WARN] Error loading DUS weights: {e}")

            all_layers = dus.model.layers
            # 1. Первые num_context_layers для концептуального контекста
            self.context_layers = nn.ModuleList([copy.deepcopy(all_layers[i]) for i in range(num_context_layers)])
            # 2. Финальные num_smoothing_layers из верхних слоев DUS для грамматики
            self.smoothing_layers = nn.ModuleList([copy.deepcopy(all_layers[-num_smoothing_layers + i]) for i in range(num_smoothing_layers)])
            # 3. Финальная нормализация
            self.final_norm = copy.deepcopy(dus.model.final_norm)
            del dus
        else:
            print("Warning: No DUS weights found, initializing ModernBERT layers from HuggingFace.")
            from transformers import AutoModel
            hf_bert = AutoModel.from_pretrained("answerdotai/ModernBERT-large")
            self.context_layers = nn.ModuleList([copy.deepcopy(hf_bert.layers[i]) for i in range(num_context_layers)])
            self.smoothing_layers = nn.ModuleList([copy.deepcopy(hf_bert.layers[-num_smoothing_layers + i]) for i in range(num_smoothing_layers)])
            self.final_norm = copy.deepcopy(hf_bert.final_norm)
            del hf_bert

        # Гарантируем, что у первого контекстного слоя attn_norm корректно работает (не Identity)
        if hasattr(self.context_layers[0], "attn_norm") and isinstance(self.context_layers[0].attn_norm, nn.Identity):
            self.context_layers[0].attn_norm = nn.LayerNorm(latent_dim, eps=1e-5)

        # Блок распаковки 1 -> K (Up-Projection)
        self.unpool_proj = nn.Linear(latent_dim, chunk_size * latent_dim, bias=False)
        self.unpool_norm = nn.RMSNorm(latent_dim)
        nn.init.xavier_uniform_(self.unpool_proj.weight)

        # Выходной проектор в размерность Qwen
        self.output_proj = nn.Linear(latent_dim, qwen_dim)
        nn.init.xavier_uniform_(self.output_proj.weight)

        # Если передан чекпоинт старого Phase 2 decoder, подтягиваем веса output_proj и final_norm
        if phase2_weights_path and os.path.exists(phase2_weights_path):
            try:
                ckpt = torch.load(phase2_weights_path, map_location="cpu", weights_only=False)
                st = ckpt.get("decoder", ckpt)
                clean_st = {normalize_key(k).replace("decoder.", ""): v for k, v in st.items()}
                
                proj_w = clean_st.get("output_proj.weight", None)
                if proj_w is not None and proj_w.shape == self.output_proj.weight.shape:
                    self.output_proj.weight.data.copy_(proj_w)
                    if "output_proj.bias" in clean_st and self.output_proj.bias is not None:
                        self.output_proj.bias.data.copy_(clean_st["output_proj.bias"])
                    print("   -> Loaded output_proj weights from Phase 2 checkpoint.")

                if "backbone.final_norm.weight" in clean_st:
                    self.final_norm.weight.data.copy_(clean_st["backbone.final_norm.weight"])
                    if "backbone.final_norm.bias" in clean_st and self.final_norm.bias is not None:
                        self.final_norm.bias.data.copy_(clean_st["backbone.final_norm.bias"])
                    print("   -> Loaded final_norm weights from Phase 2 checkpoint.")
            except Exception as e:
                print(f"   -> Could not load weights from Phase 2 ckpt: {e}")

    def _prepare_masks(self, attention_mask_2d: torch.Tensor, seq_len: int, device: torch.device, dtype: torch.dtype):
        """
        Формирует безопасные 4D маски внимания (global и sliding_window) по ADR 054 и ADR 104:
        0.0 для разрешенных позиций, -10000.0 для замаскированных (PAD / вне локального окна).
        """
        B = attention_mask_2d.shape[0] if attention_mask_2d is not None else 1
        if attention_mask_2d is None:
            mask_2d = torch.ones((B, seq_len), device=device, dtype=torch.bool)
        else:
            mask_2d = attention_mask_2d.bool()

        # [B, 1, 1, seq_len]
        mask_4d = mask_2d.unsqueeze(1).unsqueeze(2)
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        neg_inf = torch.tensor(-10000.0, device=device, dtype=dtype)
        global_mask = torch.where(mask_4d, zero, neg_inf)  # [B, 1, 1, seq_len]

        # Скользящее окно для локальных слоев ModernBERT
        rows = torch.arange(seq_len, device=device).unsqueeze(0)
        dist = torch.abs(rows - rows.T)  # [seq_len, seq_len]
        half_w = self.local_window_size // 2
        window_allowed = (dist <= half_w).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        sliding_mask = torch.where(window_allowed & mask_4d, zero, neg_inf)

        return global_mask, sliding_mask

    def forward(
        self,
        z_compressed: torch.Tensor,
        attention_mask_full: torch.Tensor = None,
        attention_mask_N: torch.Tensor = None
    ) -> torch.Tensor:
        """
        z_compressed: [B, N, D=1024] — сжатый холст (концепты)
        attention_mask_full: [B, 4N] — маска внимания на полной длине (1=токен, 0=PAD)
        attention_mask_N: [B, N] — маска внимания на сжатой длине
        Возвращает: logits representation [B, 4N, 1536] для Qwen LM Head
        """
        B, N, D = z_compressed.shape
        K = self.chunk_size
        device = z_compressed.device
        dtype = z_compressed.dtype

        # 1. Контекстуализация концептов на длине N
        global_mask_N, sliding_mask_N = self._prepare_masks(attention_mask_N, N, device, dtype)
        pos_ids_N = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        
        h = z_compressed
        for layer in self.context_layers:
            layer_out = layer(
                h,
                position_ids=pos_ids_N,
                attention_mask=global_mask_N,
                sliding_window_mask=sliding_mask_N
            )
            h = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        # 2. Геометрическая распаковка 1 -> K (Up-Projection)
        h_unpool = self.unpool_proj(h)  # [B, N, K * D]
        h_expanded = h_unpool.view(B, N * K, D)  # [B, N * K, D]
        h_expanded = self.unpool_norm(h_expanded)

        # 3. Орфографическое и синтаксическое сглаживание на полной длине N * K
        full_len = N * K
        global_mask_full, sliding_mask_full = self._prepare_masks(attention_mask_full, full_len, device, dtype)
        pos_ids_full = torch.arange(full_len, device=device).unsqueeze(0).expand(B, -1)

        for layer in self.smoothing_layers:
            layer_out = layer(
                h_expanded,
                position_ids=pos_ids_full,
                attention_mask=global_mask_full,
                sliding_window_mask=sliding_mask_full
            )
            h_expanded = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        # 4. Финальная нормализация ModernBERT
        h_final = self.final_norm(h_expanded)

        # 5. Проекция в пространство эмбеддингов Qwen (1536)
        out = self.output_proj(h_final)  # [B, N * K, 1536]
        return out
