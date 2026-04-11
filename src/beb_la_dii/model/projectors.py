import torch
import torch.nn as nn
from .base import BEComponent

class InputProjector(BEComponent):
    """
    MLP Projector for Qwen embeddings to ModernBERT latent space.
    4096 -> 2048 -> 1024.
    """
    def __init__(self, component_id="qwen_to_bert_input", version="v1.0", config=None):
        input_dim = config.get("input_dim", 4096) if config else 4096
        output_dim = config.get("output_dim", 1024) if config else 1024
        hidden_dim = config.get("hidden_dim", 2048) if config else 2048
        super().__init__(component_id, version, {
            "input_dim": input_dim, 
            "hidden_dim": hidden_dim, 
            "output_dim": output_dim
        })
        
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    @classmethod
    def from_scratch(cls, component_id="qwen_to_bert_input", version="v1.0", **kwargs):
        config = kwargs.get("config", {"input_dim": 4096, "hidden_dim": 2048, "output_dim": 1024})
        return cls(component_id=component_id, version=version, config=config)
        
    def forward(self, x):
        return self.proj(x)

class FeatureProjector(BEComponent):
    """
    Feature Projector for ModernBERT hidden states to Qwen latent space.
    1024 -> 4096. With residual connection.
    """
    def __init__(self, component_id="bert_to_qwen_feature", version="v1.0", config=None):
        input_dim = config.get("input_dim", 1024) if config else 1024
        output_dim = config.get("output_dim", 4096) if config else 4096
        super().__init__(component_id, version, {"input_dim": input_dim, "output_dim": output_dim})
        
        # Linear approximation for residual connection
        self.residual_proj = nn.Linear(input_dim, output_dim)
        
        self.proj = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    @classmethod
    def from_scratch(cls, component_id="bert_to_qwen_feature", version="v1.0", **kwargs):
        config = kwargs.get("config", {"input_dim": 1024, "output_dim": 4096})
        return cls(component_id=component_id, version=version, config=config)
        
    def forward(self, x):
        res = self.residual_proj(x)
        out = self.proj(x)
        return out + res
