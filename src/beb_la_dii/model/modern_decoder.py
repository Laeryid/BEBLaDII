import torch
import torch.nn as nn
from transformers import AutoModel

class ModernLatentDecoder(nn.Module):
    """
    Decoder для Phase 2: берет последние num_layers слоев ModernBERT-large
    с предобученными весами и проецирует результат в пространство Qwen.

    Входной тензор z: (B, T, 1024) — сферический латент из VAE Phase 1.
    Выходной тензор:  (B, T, qwen_dim) — для умножения на Qwen LM Head.

    Layer forward signature (ModernBertEncoderLayer):
        hidden_states, attention_mask=None, sliding_window_mask=None,
        position_ids=None, cu_seqlens=None, max_seqlen=None,
        output_attentions=False -> Tensor
    """
    def __init__(self, num_layers=3, latent_dim=1024, qwen_dim=1536,
                 model_name="answerdotai/ModernBERT-large"):
        super().__init__()

        print(f"Loading ModernBERT-large backbone from '{model_name}'...")
        full_model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)

        # AutoModel возвращает ModernBertModel напрямую (без вложенного .model)
        # Берем последние num_layers слоев + финальную нормализацию
        self.layers = nn.ModuleList(full_model.layers[-num_layers:])
        self.final_norm = full_model.final_norm

        del full_model
        print(f"Backbone loaded: last {num_layers} layers of ModernBERT-large.")

        # Проекция из latent_dim (1024) в qwen_dim (1536)
        self.output_proj = nn.Linear(latent_dim, qwen_dim)

    def forward(self, z, attention_mask=None):
        """
        z: (B, T, 1024) — сферические латенты, работаем как hidden_states.
        attention_mask: (B, T) — стандартная маска (1=реальный токен, 0=pad).
        """
        hidden = z.to(torch.bfloat16)

        # ModernBERT ожидает additive bias в формате (B, 1, T, T):
        # 0.0 для реальных пар токенов, -inf для pad-токенов.
        if attention_mask is not None:
            # (B, T) -> (B, 1, 1, T) -> broadcast в (B, 1, T, T)
            pad_mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2)
            attn_bias = pad_mask * torch.finfo(torch.bfloat16).min
            attn_bias = attn_bias.to(dtype=torch.bfloat16)

            # sliding_window_mask: ModernBERT-large не использует sliding window
            # (sliding_window=-1 в конфиге), передаем ту же маску
            sliding_window_mask = attn_bias
        else:
            attn_bias = None
            sliding_window_mask = None

        for layer in self.layers:
            hidden = layer(
                hidden_states=hidden,
                attention_mask=attn_bias,
                sliding_window_mask=sliding_window_mask,
            )

        hidden = self.final_norm(hidden)

        # Проекция в пространство Qwen (bfloat16 -> float32 для стабильности лосса)
        out = self.output_proj(hidden.float())  # (B, T, 1536)
        return out
