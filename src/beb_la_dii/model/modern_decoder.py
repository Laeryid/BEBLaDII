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
            class SafeModernBertModel(type(self.backbone)):
                @property
                def device(self):
                    # Безопасный fallback, если DataParallel replica вернула пустой список
                    params = list(self.parameters())
                    return params[0].device if params else torch.device("cuda")
                
                @property
                def dtype(self):
                    params = list(self.parameters())
                    return params[0].dtype if params else torch.bfloat16
            
            self.backbone.__class__ = SafeModernBertModel
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
        # Ранее мы убирали attention_mask, боясь рекомпиляций XLA из-за unpadding.
        # Но без маски RoPE поворачивает PAD-токены и модель сжигает ёмкость на их подавление.
        # Возвращаем attention_mask. В режиме SDPA unpadding не используется, 
        # поэтому XLA не должен падать на динамических формах.
        
        if self.use_modern_bert:
            # ModernBERT сам применит RoPE к inputs_embeds
            outputs = self.backbone(inputs_embeds=z, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state
        else:
            pytorch_mask = (attention_mask == 0) if attention_mask is not None else None
            hidden = self.backbone(z, src_key_padding_mask=pytorch_mask)
            
        # Проекция в пространство Qwen
        out = self.output_proj(hidden) # (B, T, 1536)
        return out
