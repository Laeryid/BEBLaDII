import torch
import torch.nn as nn
from .dus import DUSModel

class ModernLatentDecoder(nn.Module):
    def __init__(self, latent_dim=1024, qwen_dim=1536, num_layers=3, dus_weights_path=None):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Загружаем DUS Model (latentBERT)
        if dus_weights_path:
            print(f"Loading first {num_layers} layers from DUS latentBERT: {dus_weights_path}")
            dus = DUSModel.from_scratch(weights_path=dus_weights_path)
            self.backbone = dus.model
            
            # Берем ПОСЛЕДНИЕ слои (они лучше всего знают финальную грамматику и работают с глубокой семантикой)
            self.backbone.layers = nn.ModuleList(self.backbone.layers[-num_layers:])
            
            # Hotfix for HuggingFace ModernBERT + PyTorch DataParallel bug:
            # Replicas sometimes fail to resolve `param.device` inside `_maybe_set_compile`.
            self.backbone._maybe_set_compile = lambda *args, **kwargs: None
            
            self.use_modern_bert = True
        else:
            print("Warning: No DUS weights provided. Using random TransformerEncoder for PoC.")
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim, 
                nhead=16, 
                dim_feedforward=latent_dim * 4, 
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
            self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.use_modern_bert = False

        # Проекция из размерности латентов (1024) обратно в размерность Qwen (1536) для LM Head
        self.output_proj = nn.Linear(latent_dim, qwen_dim)

    def forward(self, z, attention_mask=None):
        # ВАЖНО ДЛЯ XLA: Мы не передаем attention_mask в ModernBERT!
        # ModernBERT при получении маски может использовать unpadding (удаление pad-токенов),
        # что делает длину тензора (Total_Active_Tokens) динамической.
        # Динамические формы заставляют XLA перекомпилировать граф на КАЖДОМ шаге (отсюда 48 секунд/шаг).
        # Так как мы умножаем финальный Loss на attention_mask, нам не страшно, что слои увидят PAD-токены.
        
        if self.use_modern_bert:
            # ModernBERT сам применит RoPE к inputs_embeds
            outputs = self.backbone(inputs_embeds=z) # Убрали attention_mask!
            hidden = outputs.last_hidden_state
        else:
            pytorch_mask = (attention_mask == 0) if attention_mask is not None else None
            hidden = self.backbone(z, src_key_padding_mask=pytorch_mask)
            
        # Проекция в пространство Qwen
        out = self.output_proj(hidden) # (B, T, 1536)
        return out
