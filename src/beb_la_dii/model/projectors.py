import torch
import torch.nn as nn
from .base import BEComponent

class InputProjector(BEComponent):
    """
    MLP Projector for Qwen embeddings to ModernBERT latent space.
    3584 (Qwen2.5-7B hidden_size) -> 2048 -> 1024.
    """
    def __init__(self, component_id="qwen_to_bert_input", version="v1.0", config=None):
        # DeepSeek-R1-Distill-Qwen-7B is based on Qwen2.5, hidden_size=3584
        input_dim = config.get("input_dim", 3584) if config else 3584
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
    def from_scratch(cls, component_id="qwen_to_bert_input", version="v1.0",
                     weights_path=None, **kwargs):
        """
        Создаёт InputProjector с нуля.
        weights_path: путь к weights.pt; если None — случайная инициализация.
        """
        config = kwargs.get("config", {"input_dim": 3584, "hidden_dim": 2048, "output_dim": 1024})
        instance = cls(component_id=component_id, version=version, config=config)
        instance.load_weights(weights_path)
        return instance
        
    def forward(self, x):
        return self.proj(x)


class FeatureProjector(BEComponent):
    """
    Feature Projector for ModernBERT hidden states to Qwen latent space.
    1024 -> 3584 (Qwen2.5-7B hidden_size). With residual connection.
    """
    def __init__(self, component_id="bert_to_qwen_feature", version="v1.0", config=None):
        input_dim = config.get("input_dim", 1024) if config else 1024
        # DeepSeek-R1-Distill-Qwen-7B is based on Qwen2.5, hidden_size=3584
        output_dim = config.get("output_dim", 3584) if config else 3584
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
    def from_scratch(cls, component_id="bert_to_qwen_feature", version="v1.0",
                     weights_path=None, **kwargs):
        """
        Создаёт FeatureProjector с нуля.
        weights_path: путь к weights.pt; если None — случайная инициализация.
        """
        config = kwargs.get("config", {"input_dim": 1024, "output_dim": 3584})
        instance = cls(component_id=component_id, version=version, config=config)
        instance.load_weights(weights_path)
        return instance
        
    def forward(self, x):
        res = self.residual_proj(x)
        out = self.proj(x)
        return out + res
