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

        # Оставляем только последние num_layers слоев в модели
        full_model.layers = nn.ModuleList(full_model.layers[-num_layers:])
        full_model.config.num_hidden_layers = num_layers
        self.modernbert = full_model

        print(f"Backbone loaded: last {num_layers} layers of ModernBERT-large.")

        # Проекция из latent_dim (1024) в qwen_dim (1536)
        self.output_proj = nn.Linear(latent_dim, qwen_dim)

    def forward(self, z, attention_mask=None):
        """
        z: (B, T, 1024) — сферические латенты, работаем как hidden_states.
        attention_mask: (B, T) — стандартная маска (1=реальный токен, 0=pad).
        """
        hidden = z.to(torch.bfloat16)

        # Делегируем всю логику (RoPE, attention_mask, sliding_window) 
        # встроенному forward() ModernBERT через inputs_embeds
        outputs = self.modernbert(
            inputs_embeds=hidden,
            attention_mask=attention_mask
        )
        hidden = outputs.last_hidden_state

        # Проекция в пространство Qwen (bfloat16 -> float32 для стабильности лосса)
        out = self.output_proj(hidden.float())  # (B, T, 1536)
        return out
