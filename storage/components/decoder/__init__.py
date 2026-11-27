"""Автономная компонента Декодера-Генератора Plan C (O(8)-канонизация, Радар Шума, распак. 1 -> 4)."""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ..plan_c_layers import (
        ModernBERTLayer,
        RMSNorm,
        RotaryEmbedding128,
        WordAwareElasticChunker,
        canonicalize_cell_basis,
        clamp_to_unit_ball,
        safe_lorentzian_eigh,
    )
    from ..quantization import (
        Int8RowwiseEmbedding,
        System1QuantPolicy,
        is_quantized_checkpoint,
        load_quantized_decoder_payload,
        quantize_decoder_in_place,
        resolve_component_weights_path,
        resolve_system1_quant_policy,
    )
except ImportError:
    from plan_c_layers import (
        ModernBERTLayer,
        RMSNorm,
        RotaryEmbedding128,
        WordAwareElasticChunker,
        canonicalize_cell_basis,
        clamp_to_unit_ball,
        safe_lorentzian_eigh,
    )
    from quantization import (
        Int8RowwiseEmbedding,
        System1QuantPolicy,
        is_quantized_checkpoint,
        load_quantized_decoder_payload,
        quantize_decoder_in_place,
        resolve_component_weights_path,
        resolve_system1_quant_policy,
    )

DECODER_DIR = Path(__file__).resolve().parent


class SetCrossAttentionUnpacker(nn.Module):
    """Архитектурно O(8)-инвариантная распаковка клетки 1 -> 4 через Cross-Attention и Gram(V) в float32."""

    def __init__(self, axis_dim: int = 128, num_slots: int = 4, hidden_size: int = 1024):
        super().__init__()
        self.axis_dim = axis_dim
        self.num_slots = num_slots
        self.hidden_size = hidden_size

        self.query_slots = nn.Parameter(torch.randn(num_slots, axis_dim) * 0.02)
        self.q_norm = RMSNorm(axis_dim)
        self.k_norm = RMSNorm(axis_dim)

        self.gram_mlp = nn.Sequential(
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            RMSNorm(128),
        )
        self.out_proj = nn.Linear(axis_dim + 128, hidden_size)
        self.out_norm = RMSNorm(hidden_size)

    def forward(self, V_cell: torch.Tensor) -> torch.Tensor:
        B, N, _, D = V_cell.shape
        Q = self.q_norm(self.query_slots).unsqueeze(0).unsqueeze(0).expand(B, N, self.num_slots, D).to(V_cell.dtype)
        K_mat = self.k_norm(V_cell)

        scores = torch.matmul(Q, K_mat.transpose(-2, -1)) / (D ** 0.5)
        attn = F.softmax(scores.float(), dim=-1).to(V_cell.dtype)
        h_attn = torch.matmul(attn, V_cell)

        # Строгий float32 с отключенным autocast для матрицы Грама и gram_mlp (ADR 126)
        device_type = V_cell.device.type
        cat_dtype = h_attn.dtype
        with torch.autocast(device_type=device_type, enabled=False):
            V_f32 = V_cell.float()
            G_flat_f32 = torch.matmul(V_f32, V_f32.transpose(-2, -1)).view(B, N, 64)

            mlp_w0 = self.gram_mlp[0].weight.float()
            mlp_b0 = self.gram_mlp[0].bias.float() if self.gram_mlp[0].bias is not None else None
            h1 = F.silu(F.linear(G_flat_f32, mlp_w0, mlp_b0))
            mlp_w2 = self.gram_mlp[2].weight.float()
            mlp_b2 = self.gram_mlp[2].bias.float() if self.gram_mlp[2].bias is not None else None
            h2 = F.linear(h1, mlp_w2, mlp_b2)
            norm_w = self.gram_mlp[3].weight.float()
            variance = h2.pow(2).mean(-1, keepdim=True)
            h_normed = h2 * torch.rsqrt(variance + 1e-6) * norm_w
            h_gram = h_normed.to(cat_dtype).unsqueeze(2).expand(B, N, self.num_slots, 128)

        h_comb = torch.cat([h_attn, h_gram], dim=-1)
        proj_w_dtype = self.out_proj.weight.dtype
        if torch.is_autocast_enabled(V_cell.device.type) or h_comb.dtype == proj_w_dtype:
            h_out = self.out_norm(self.out_proj(h_comb))
        else:
            h_out = self.out_norm(self.out_proj(h_comb.to(proj_w_dtype))).to(V_cell.dtype)
        return h_out.view(B, N * self.num_slots, self.hidden_size)


class NoiseRadarHead(nn.Module):
    """
    Быстрый радар шума и неуверенности (Выход А для цикла кристаллизации Фазы 3):
    Оценивает поклеточный уровень шума sigma_cell in [0, 1] и потокенную неуверенность sigma_tok in [0, 1]
    на основе спектральных инвариантов клетки (eigvals, энергия, чистота/ранг),
    контекстного представления Этапа 1 и близости распакованных слотов к словам словаря.
    """

    def __init__(self, hidden_size: int = 1024, num_axes: int = 8, chunk_size: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_axes = num_axes
        self.chunk_size = chunk_size
        self._tied_int8_embedding: Optional[Int8RowwiseEmbedding] = None
        self._radar_gemm_dtype: Optional[torch.dtype] = None
        self._use_cpu_onednn: bool = True

        self.cell_proj = nn.Linear(hidden_size, 32)
        self.tok_proj = nn.Linear(hidden_size, 32)
        self.slot_pos_emb = nn.Parameter(torch.randn(chunk_size, 8) * 0.02)

        # Спектр (8) + энергия (1) + чистота (1) + inv_rk_eff (1) + cell_proj (32) + agg_dict_stats (6) = 49
        self.cell_mlp = nn.Sequential(
            nn.Linear(num_axes + 3 + 32 + 6, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        # tok_proj (32) + dict_stats (4) + slot_pos_emb (8) + cell_ctx (16) = 60
        self.cell_to_tok = nn.Linear(num_axes + 3 + 32, 16)
        self.tok_mlp = nn.Sequential(
            nn.Linear(32 + 4 + 8 + 16, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    @torch.no_grad()
    def compute_dict_stats(
        self,
        h_unpacked: torch.Tensor,
        lm_head_weight: Optional[torch.Tensor],
        lm_head_inv_norms: Optional[torch.Tensor],
        vocab_chunk_size: int = 32768,
    ) -> torch.Tensor:
        """
        Вычисляет близость каждого слота (B, 4N, 1024) к ближайшим словам словаря без аллокации
        полного тензора логитов (chunked max/top-2 косинус под no_grad).
        Поддерживает как float lm_head_weight, так и per-row INT8 (через _tied_int8_embedding или int8-тензор).
        Возвращает: (B, 4N, 4) -> [cos_max, cos_2nd, cos_gap, log_norm_ratio].
        """
        B, L, D = h_unpacked.shape
        device = h_unpacked.device
        if lm_head_weight is None and self._tied_int8_embedding is None:
            return torch.zeros(B, L, 4, device=device, dtype=torch.float32)

        with torch.autocast(device_type=device.type, enabled=False):
            h_f32 = h_unpacked.float().reshape(B * L, D)
            h_norms = h_f32.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
            h_unit = h_f32 / h_norms
            norm_feat = torch.log1p(h_norms / math.sqrt(D))

            if self._tied_int8_embedding is not None and (
                lm_head_weight is None or lm_head_weight.data_ptr() == self._tied_int8_embedding.q_weight.data_ptr()
            ):
                eff_dtype = self._radar_gemm_dtype or (
                    torch.float32 if device.type == "cpu" else torch.bfloat16
                )
                cos_max, cos_2nd = self._tied_int8_embedding.compute_radar_top2_cos(
                    h_unit_f32=h_unit,
                    chunk_size=vocab_chunk_size,
                    gemm_dtype=eff_dtype,
                    use_cpu_onednn=self._use_cpu_onednn,
                )
                cos_gap = (cos_max - cos_2nd).clamp(min=0.0)
                return torch.cat([cos_max, cos_2nd, cos_gap, norm_feat], dim=-1).view(B, L, 4)

            if lm_head_weight is None or lm_head_inv_norms is None:
                return torch.zeros(B, L, 4, device=device, dtype=torch.float32)

            V_size = lm_head_weight.size(0)
            top2_vals = torch.full((B * L, 2), -10.0, device=device, dtype=torch.float32)

            for start in range(0, V_size, vocab_chunk_size):
                end = min(start + vocab_chunk_size, V_size)
                w_chunk = lm_head_weight[start:end].to(device=device, dtype=torch.float32)
                inv_n = lm_head_inv_norms[start:end].to(device=device, dtype=torch.float32)
                cos_chunk = F.linear(h_unit, w_chunk) * inv_n.unsqueeze(0)
                k_take = min(2, end - start)
                chunk_top = torch.topk(cos_chunk, k=k_take, dim=-1).values
                merged = torch.cat([top2_vals, chunk_top], dim=-1)
                top2_vals = torch.topk(merged, k=2, dim=-1).values

            cos_max = top2_vals[:, 0:1]
            cos_2nd = top2_vals[:, 1:2]
            cos_gap = (cos_max - cos_2nd).clamp(min=0.0)
            return torch.cat([cos_max, cos_2nd, cos_gap, norm_feat], dim=-1).view(B, L, 4)

    def forward(
        self,
        eigvals: torch.Tensor,
        h_cell: torch.Tensor,
        h_unpacked: torch.Tensor,
        lm_head_weight: Optional[torch.Tensor] = None,
        lm_head_inv_norms: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, N, K = eigvals.shape
        device_type = h_unpacked.device.type
        dict_stats = self.compute_dict_stats(h_unpacked, lm_head_weight, lm_head_inv_norms)

        with torch.autocast(device_type=device_type, enabled=False):
            eig_f32 = eigvals.float()
            energy = eig_f32.sum(dim=-1, keepdim=True)
            purity = (eig_f32 ** 2).sum(dim=-1, keepdim=True)
            inv_rk = purity / (energy ** 2 + 1e-6)

            cell_hidden_feat = F.silu(self.cell_proj(h_cell.float()))
            cell_base = torch.cat([eig_f32, energy, purity, inv_rk, cell_hidden_feat], dim=-1)

            dict_cells = dict_stats.view(B, N, self.chunk_size, 4)
            dict_mean = dict_cells.mean(dim=2)[..., :3]
            dict_min = dict_cells.min(dim=2).values[..., :3]
            cell_in = torch.cat([cell_base, dict_mean, dict_min], dim=-1)

            sigma_cell_logits = self.cell_mlp(cell_in).squeeze(-1)
            sigma_cell_hat = torch.sigmoid(sigma_cell_logits)

            tok_hidden_feat = F.silu(self.tok_proj(h_unpacked.float()))
            cell_ctx = F.silu(self.cell_to_tok(cell_base)).unsqueeze(2).expand(B, N, self.chunk_size, 16)
            cell_ctx = cell_ctx.reshape(B, N * self.chunk_size, 16)
            slot_pos = self.slot_pos_emb.float().unsqueeze(0).unsqueeze(0).expand(B, N, self.chunk_size, 8)
            slot_pos = slot_pos.reshape(B, N * self.chunk_size, 8)

            tok_in = torch.cat([tok_hidden_feat, dict_stats, slot_pos, cell_ctx], dim=-1)
            sigma_tok_logits = self.tok_mlp(tok_in).squeeze(-1)
            sigma_tok_hat = torch.sigmoid(sigma_tok_logits)

        return {
            "sigma_cell_hat": sigma_cell_hat,
            "sigma_cell_logits": sigma_cell_logits,
            "sigma_tok_hat": sigma_tok_hat,
            "sigma_tok_logits": sigma_tok_logits,
            "dict_stats": dict_stats,
        }


class GrassmannianDecoder(nn.Module):
    """
    Иерархический O(8)-инвариантный Декодер-Генератор Plan C:
    - Входная детерминированная O(8)-канонизация базиса V_cell;
    - Этап 1: контекстный слой по холсту N + SetCrossAttentionUnpacker (1 -> 4);
    - Выход А (NoiseRadarHead): быстрая оценка sigma_cell и sigma_tok для кристаллизации;
    - Выход Б (Генератор): инъекция sigma_tok и сглаживающие/генерирующие слои (по 4N) -> logits.
    """

    def __init__(
        self,
        vocab_size: int = 131072,
        hidden_size: int = 1024,
        intermediate_size: int = 2048,
        num_axes: int = 8,
        axis_dim: int = 128,
        chunk_size: int = 4,
        max_seq_len: int = 4096,
        cell_jitter_std: float = 0.0,
        num_smooth_layers: int = 4,
        canonicalize_basis: bool = True,
        enable_noise_radar: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_axes = num_axes
        self.axis_dim = axis_dim
        self.chunk_size = chunk_size
        self.cell_jitter_std = cell_jitter_std
        self.num_smooth_layers = num_smooth_layers
        self.canonicalize_basis = canonicalize_basis
        self.enable_noise_radar = enable_noise_radar

        self.rope = RotaryEmbedding128(dim=axis_dim, max_seq_len=max_seq_len)
        self.cell_in_norm = RMSNorm(hidden_size)
        self.dec_context_layer = ModernBERTLayer(hidden_size, intermediate_size)
        self.unpacker = SetCrossAttentionUnpacker(axis_dim=axis_dim, num_slots=chunk_size, hidden_size=hidden_size)
        self.dec_smooth_layers = nn.ModuleList(
            [ModernBERTLayer(hidden_size, intermediate_size) for _ in range(num_smooth_layers)]
        )
        self.dec_final_norm = RMSNorm(hidden_size)

        if enable_noise_radar:
            self.noise_radar = NoiseRadarHead(
                hidden_size=hidden_size, num_axes=num_axes, chunk_size=chunk_size
            )
            self.noise_cond_proj = nn.Sequential(
                nn.Linear(1, 64),
                nn.SiLU(),
                nn.Linear(64, hidden_size),
            )
            nn.init.zeros_(self.noise_cond_proj[2].weight)
            nn.init.zeros_(self.noise_cond_proj[2].bias)
        else:
            self.noise_radar = None
            self.noise_cond_proj = None

        # Разделяемая матрица эмбеддингов для tied LM head (ADR 125)
        self.register_buffer("lm_head_weight", None, persistent=False)
        self.register_buffer("lm_head_inv_norms", None, persistent=False)
        self.register_buffer("lm_head_row_scale", None, persistent=False)
        self.register_buffer("lm_head_unit_scale", None, persistent=False)
        self._tied_int8_embedding: Optional[Int8RowwiseEmbedding] = None

    def tie_embeddings(self, embeddings_weight: Union[torch.Tensor, nn.Module]) -> None:
        """
        Привязывает матрицу эмбеддингов (131072 x 1024) как выходную голову и нормирует для радара.
        Принимает:
          - Int8RowwiseEmbedding (0 лишних копий памяти: одна int8-матрица на всё),
          - nn.Embedding,
          - обычный float torch.Tensor.
        """
        if isinstance(embeddings_weight, Int8RowwiseEmbedding):
            self._tied_int8_embedding = embeddings_weight
            self.lm_head_weight = embeddings_weight.q_weight
            self.lm_head_row_scale = embeddings_weight.row_scale
            # Для int8-словаря F.linear(h_unit, q_weight) * unit_scale == точный косинус
            self.lm_head_inv_norms = embeddings_weight.unit_scale
            self.lm_head_unit_scale = embeddings_weight.unit_scale
            if self.noise_radar is not None:
                self.noise_radar._tied_int8_embedding = embeddings_weight
            return

        if isinstance(embeddings_weight, nn.Embedding):
            w_tensor = embeddings_weight.weight
        else:
            w_tensor = embeddings_weight

        self._tied_int8_embedding = None
        self.lm_head_weight = w_tensor
        self.lm_head_row_scale = None
        self.lm_head_unit_scale = None
        with torch.no_grad():
            inv_norms = 1.0 / w_tensor.float().norm(p=2, dim=-1).clamp(min=1e-6)
        self.lm_head_inv_norms = inv_norms
        if self.noise_radar is not None:
            self.noise_radar._tied_int8_embedding = None

    def _compute_eigvals_desc(self, V_cell: torch.Tensor) -> torch.Tensor:
        """Вычисляет отсортированный по убыванию спектр Gram(V_cell) в float32."""
        with torch.autocast(device_type=V_cell.device.type, enabled=False):
            V_f32 = torch.nan_to_num(V_cell.float(), nan=0.0, posinf=65000.0, neginf=-65000.0)
            G = torch.matmul(V_f32, V_f32.transpose(-2, -1))
            G = 0.5 * (G + G.transpose(-2, -1))
            eigvals, _ = safe_lorentzian_eigh(G, eps=1e-4)
            return F.relu(torch.flip(eigvals, dims=[-1])).to(V_cell.dtype)

    def decode_stage1(self, V_cell: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Этап 1: канонизация базиса, клеточный контекст (N) и распаковка 1 -> 4."""
        B, N, K, D = V_cell.shape
        if self.training and self.cell_jitter_std > 0.0:
            noise_scale = torch.rand(B, N, 1, 1, device=V_cell.device, dtype=V_cell.dtype) * self.cell_jitter_std
            V_cell_in = clamp_to_unit_ball(V_cell + torch.randn_like(V_cell) * noise_scale)
        else:
            V_cell_in = V_cell

        if self.canonicalize_basis:
            V_work, eigvals = canonicalize_cell_basis(V_cell_in)
        else:
            V_work = V_cell_in
            eigvals = self._compute_eigvals_desc(V_cell_in) if self.enable_noise_radar else None

        cell_vec = self.cell_in_norm(V_work.reshape(B, N, K * D))
        cos_N, sin_N = self.rope(cell_vec, N)
        h_cell = self.dec_context_layer(cell_vec, cos_N, sin_N)
        V_cell_mod = V_work + 0.1 * h_cell.reshape(B, N, K, D)
        h_unpacked = self.unpacker(V_cell_mod)

        return {
            "V_work": V_work,
            "eigvals": eigvals,
            "h_cell": h_cell,
            "h_unpacked": h_unpacked,
        }

    def estimate_noise(
        self, V_cell: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Выход А (Быстрый радар для цикла диффузии Фазы 3):
        Возвращает (sigma_cell_hat (B, N), sigma_tok_hat (B, 4N), radar_dict).
        """
        if self.noise_radar is None:
            raise RuntimeError("NoiseRadarHead отключен (enable_noise_radar=False).")
        stage1 = self.decode_stage1(V_cell)
        radar = self.noise_radar(
            eigvals=stage1["eigvals"],
            h_cell=stage1["h_cell"],
            h_unpacked=stage1["h_unpacked"],
            lm_head_weight=self.lm_head_weight,
            lm_head_inv_norms=self.lm_head_inv_norms,
        )
        radar["eigvals"] = stage1["eigvals"]
        return radar["sigma_cell_hat"], radar["sigma_tok_hat"], radar

    def decode_hidden(
        self, V_cell: torch.Tensor, return_radar: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Декомпрессия клеток V_cell (B, N, 8, 128) в скрытые состояния (B, 4N, 1024)."""
        B, N, _, _ = V_cell.shape
        stage1 = self.decode_stage1(V_cell)
        h_unpacked = stage1["h_unpacked"]

        radar_out: Optional[Dict[str, torch.Tensor]] = None
        if self.enable_noise_radar and self.noise_radar is not None:
            radar_out = self.noise_radar(
                eigvals=stage1["eigvals"],
                h_cell=stage1["h_cell"],
                h_unpacked=h_unpacked,
                lm_head_weight=self.lm_head_weight,
                lm_head_inv_norms=self.lm_head_inv_norms,
            )
            radar_out["eigvals"] = stage1["eigvals"]
            if self.noise_cond_proj is not None:
                cond_w_dtype = self.noise_cond_proj[0].weight.dtype
                if cond_w_dtype == torch.float32:
                    with torch.autocast(device_type=V_cell.device.type, enabled=False):
                        cond = self.noise_cond_proj(radar_out["sigma_tok_hat"].float().unsqueeze(-1)).to(
                            h_unpacked.dtype
                        )
                else:
                    cond = self.noise_cond_proj(radar_out["sigma_tok_hat"].unsqueeze(-1).to(cond_w_dtype)).to(
                        h_unpacked.dtype
                    )
                h_unpacked = h_unpacked + cond

        full_len = N * self.chunk_size
        cos_full, sin_full = self.rope(h_unpacked, full_len)
        x = h_unpacked
        for layer in self.dec_smooth_layers:
            x = layer(x, cos_full, sin_full)
        x_final = self.dec_final_norm(x)
        if return_radar:
            return x_final, (radar_out or {})
        return x_final

    def get_tied_head_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Вычисление логитов строго в float32 с отключенным autocast (ADR 125)."""
        if self._tied_int8_embedding is not None:
            return self._tied_int8_embedding.linear_chunked_fp32(hidden_states)
        if self.lm_head_weight is None:
            raise RuntimeError(
                "Не задана матрица lm_head_weight. Передайте embeddings_weight в get_decoder() или вызовите tie_embeddings()."
            )
        device_type = hidden_states.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            h_f32 = hidden_states.float()
            if self.lm_head_row_scale is not None:
                v_size = int(self.lm_head_weight.shape[0])
                chunks = []
                for start in range(0, v_size, 32768):
                    end = min(start + 32768, v_size)
                    w_c = self.lm_head_weight[start:end].to(device=h_f32.device, dtype=torch.float32)
                    s_c = self.lm_head_row_scale[start:end].to(device=h_f32.device, dtype=torch.float32)
                    chunks.append(F.linear(h_f32, w_c * s_c.unsqueeze(-1)))
                return torch.cat(chunks, dim=-1)
            return F.linear(h_f32, self.lm_head_weight.float())

    def forward(
        self, V_cell: torch.Tensor, return_radar: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if return_radar:
            x, radar = self.decode_hidden(V_cell, return_radar=True)
            return self.get_tied_head_logits(x), radar
        x = self.decode_hidden(V_cell, return_radar=False)
        return self.get_tied_head_logits(x)

    @torch.no_grad()
    def decode_text(
        self,
        V_cell: torch.Tensor,
        tokenizer: Any,
        pad_token_id: int = 10,
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Декодирует батч клеток (B, N, 8, 128) в строки с удалением SLOT_PAD (id=10)."""
        logits = self.forward(V_cell)
        pred_ids = torch.argmax(logits, dim=-1)
        return [
            WordAwareElasticChunker.detokenize_non_pad(
                row, tokenizer=tokenizer, pad_token_id=pad_token_id, skip_special_tokens=skip_special_tokens
            )
            for row in pred_ids
        ]


def get_decoder(
    weights_path: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
    embeddings_weight: Optional[Union[torch.Tensor, nn.Module]] = None,
    encoder: Optional[nn.Module] = None,
    device: Union[str, torch.device] = "cpu",
    dtype: Optional[torch.dtype] = None,
    eval_mode: bool = True,
    canonicalize_basis: Optional[bool] = None,
    enable_noise_radar: Optional[bool] = None,
    num_smooth_layers: Optional[int] = None,
    use_ema: bool = True,
    quant_mode: str = "fp32",
    mode: str = "inference",
    amp_dtype: Optional[torch.dtype] = None,
) -> GrassmannianDecoder:
    """
    Загружает GrassmannianDecoder и привязывает матрицу эмбеддингов (от encoder или из ../encoder/).

    Параметры квантизации и переключения:
      - quant_mode: 'fp32' (по умолчанию, полная обратная совместимость), 'auto' (автовыбор по mode+device),
                    'half' ('fp16'/'bf16'), 'emb_int8', 'int8'.
      - mode: 'train' (слои в FP16/BF16/FP32 без int8 на слоях — для быстрого backward в L_ce)
              или 'inference' (полный per-row INT8 на CPU/GPU).
      - amp_dtype: желаемый compute dtype (например env.amp_dtype).
    Если готовый квантизованный файл (weights_fp16.pt / weights_int8.pt) отсутствует рядом с weights_path,
    конвертация выполняется автоматически на лету из weights.pt.
    """
    cfg_file = Path(config_path) if config_path else DECODER_DIR / "config.json"
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    policy: System1QuantPolicy = resolve_system1_quant_policy(
        mode=mode,
        device=device,
        quant_mode=quant_mode,
        amp_dtype=(amp_dtype if amp_dtype is not None else dtype),
    )

    use_canon = cfg.get("canonicalize_basis", True) if canonicalize_basis is None else canonicalize_basis
    use_radar = cfg.get("enable_noise_radar", True) if enable_noise_radar is None else enable_noise_radar
    n_smooth = cfg.get("num_smooth_layers", 4) if num_smooth_layers is None else num_smooth_layers

    model = GrassmannianDecoder(
        vocab_size=cfg.get("vocab_size", 131072),
        hidden_size=cfg.get("hidden_size", 1024),
        intermediate_size=cfg.get("intermediate_size", 2048),
        num_axes=cfg.get("num_axes", 8),
        axis_dim=cfg.get("axis_dim", 128),
        chunk_size=cfg.get("chunk_size", 4),
        max_seq_len=cfg.get("max_seq_len", 4096),
        cell_jitter_std=cfg.get("cell_jitter_std", 0.0),
        num_smooth_layers=n_smooth,
        canonicalize_basis=use_canon,
        enable_noise_radar=use_radar,
    )

    fallback_name = cfg.get("weights_file", "weights.pt")
    w_file, _ = resolve_component_weights_path(
        base_dir=DECODER_DIR,
        preset_filename=policy.decoder_weights_filename,
        explicit_weights_path=weights_path,
        fallback_filename=fallback_name,
    )
    raw_ckpt = torch.load(w_file, map_location="cpu", weights_only=False)

    if is_quantized_checkpoint(raw_ckpt):
        loaded_preset = load_quantized_decoder_payload(
            model,
            payload=raw_ckpt,
            layer_dtype=policy.layer_dtype,
            use_cpu_onednn=policy.use_cpu_onednn_int8,
        )
        if policy.decoder_preset == "int8" and loaded_preset != "int8":
            quantize_decoder_in_place(
                model,
                preset="int8",
                layer_dtype=policy.layer_dtype,
                use_cpu_onednn=policy.use_cpu_onednn_int8,
            )
    else:
        if isinstance(raw_ckpt, dict) and ("decoder_state_dict" in raw_ckpt or "ema_state_dict" in raw_ckpt):
            # Тренировочный чекпоинт Phase 2 (с EMA-тенями и/или сырыми весами)
            ema_shadow = (raw_ckpt.get("ema_state_dict") or {}).get("shadow") if use_ema else None
            if isinstance(ema_shadow, dict) and len(ema_shadow) > 0:
                sd = {}
                for k, v in ema_shadow.items():
                    clean_k = k.replace("module.", "")
                    if clean_k.endswith(".q_weight"):
                        v = F.normalize(v.float(), p=2, dim=-1, eps=1e-8)
                    sd[clean_k] = v
            else:
                sd = {k.replace("module.", ""): v for k, v in raw_ckpt["decoder_state_dict"].items()}
        elif isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt:
            sd = {k.replace("module.", ""): v for k, v in raw_ckpt["model_state_dict"].items()}
        else:
            sd = {k.replace("module.", ""): v for k, v in raw_ckpt.items()}

        # Извлекаем веса декодера, если передан общий чекпоинт Phase 1
        if any(k.startswith("dec_") or k.startswith("unpacker.") or k == "cell_in_norm.weight" for k in sd):
            dec_prefixes = ("cell_in_norm.", "dec_context_layer.", "unpacker.", "dec_smooth_layers.", "dec_final_norm.", "noise_radar.", "noise_cond_proj.")
            sd = {k: v for k, v in sd.items() if k.startswith(dec_prefixes)}

        is_upgraded = any(k.startswith("noise_radar.") for k in model.state_dict()) and not any(
            k.startswith("noise_radar.") for k in sd
        )
        extra_smooth = n_smooth > 2 and not any(k.startswith("dec_smooth_layers.2.") for k in sd)

        model.load_state_dict(sd, strict=not (is_upgraded or extra_smooth))

        if extra_smooth:
            with torch.no_grad():
                for idx in range(2, n_smooth):
                    layer = model.dec_smooth_layers[idx]
                    layer.o_proj.row_scale.fill_(1e-3)
                    layer.down_proj.row_scale.fill_(1e-3)

        if policy.decoder_preset != "fp32":
            quantize_decoder_in_place(
                model,
                preset=policy.decoder_preset,
                layer_dtype=policy.layer_dtype,
                use_cpu_onednn=policy.use_cpu_onednn_int8,
            )
        elif dtype is not None:
            model = model.to(dtype=dtype)

    if embeddings_weight is not None:
        model.tie_embeddings(embeddings_weight)
    elif encoder is not None and hasattr(encoder, "embeddings"):
        model.tie_embeddings(encoder.embeddings)
    else:
        enc_dir = DECODER_DIR.parent / "encoder"
        enc_file, _ = resolve_component_weights_path(
            base_dir=enc_dir,
            preset_filename=policy.encoder_weights_filename,
            explicit_weights_path=None,
            fallback_filename="weights.pt",
        )
        if enc_file.exists():
            enc_raw = torch.load(enc_file, map_location="cpu", weights_only=False)
            if is_quantized_checkpoint(enc_raw):
                enc_sd = enc_raw["state_dict"]
                if "embeddings.q_weight" in enc_sd:
                    q_w = enc_sd["embeddings.q_weight"]
                    emb_mod = Int8RowwiseEmbedding(
                        num_embeddings=int(q_w.shape[0]),
                        embedding_dim=int(q_w.shape[1]),
                        out_dtype=policy.layer_dtype,
                        use_cpu_onednn=policy.use_cpu_onednn_int8,
                    )
                    emb_mod.q_weight.copy_(q_w)
                    emb_mod.row_scale.copy_(enc_sd["embeddings.row_scale"])
                    emb_mod.inv_norms.copy_(enc_sd["embeddings.inv_norms"])
                    emb_mod.unit_scale.copy_(enc_sd["embeddings.unit_scale"])
                    model.tie_embeddings(emb_mod)
                elif "embeddings.weight" in enc_sd:
                    model.tie_embeddings(enc_sd["embeddings.weight"])
            elif isinstance(enc_raw, dict):
                enc_sd = enc_raw.get("model_state_dict", enc_raw)
                if "embeddings.weight" in enc_sd:
                    w_e = enc_sd["embeddings.weight"]
                    if policy.embedding_int8:
                        emb_mod = Int8RowwiseEmbedding.from_weight(
                            w_e, out_dtype=policy.layer_dtype, use_cpu_onednn=policy.use_cpu_onednn_int8
                        )
                        model.tie_embeddings(emb_mod)
                    else:
                        model.tie_embeddings(w_e.to(dtype=policy.layer_dtype) if policy.decoder_preset != "fp32" else w_e)

    if model.noise_radar is not None:
        model.noise_radar._radar_gemm_dtype = policy.radar_dtype
        model.noise_radar._use_cpu_onednn = policy.use_cpu_onednn_int8

    model.quant_policy = policy
    model = model.to(device)
    if model._tied_int8_embedding is not None and model._tied_int8_embedding.device != torch.device(device):
        model._tied_int8_embedding = model._tied_int8_embedding.to(device)
        model.tie_embeddings(model._tied_int8_embedding)
    if eval_mode:
        model.eval()
    return model


__all__ = [
    "DECODER_DIR",
    "SetCrossAttentionUnpacker",
    "NoiseRadarHead",
    "GrassmannianDecoder",
    "get_decoder",
]
