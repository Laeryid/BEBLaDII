import torch
import torch.nn as nn
from .base import BEComponent

class InputProjector(BEComponent):
    """
    Проектор входных эмбеддингов Qwen в латентное пространство ModernBERT.
    4096 -> 768.
    """
    def __init__(self, component_id="qwen_to_bert_input", version="v1.0", config=None):
        input_dim = config.get("input_dim", 4096) if config else 4096
        output_dim = config.get("output_dim", 768) if config else 768
        super().__init__(component_id, version, {"input_dim": input_dim, "output_dim": output_dim})
        
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x):
        return self.proj(x)

class FeatureProjector(BEComponent):
    """
    Проектор скрытых состояний ModernBERT в латентное пространство Qwen.
    768 -> 4096. С residual-связью.
    """
    def __init__(self, component_id="bert_to_qwen_feature", version="v1.0", config=None):
        input_dim = config.get("input_dim", 768) if config else 768
        output_dim = config.get("output_dim", 4096) if config else 4096
        super().__init__(component_id, version, {"input_dim": input_dim, "output_dim": output_dim})
        
        # Для residual связи при разной размерности используем линейный аппроксиматор
        self.residual_proj = nn.Linear(input_dim, output_dim)
        
        self.proj = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x):
        res = self.residual_proj(x)
        out = self.proj(x)
        return out + res

if __name__ == "__main__":
    # Тест
    in_proj = InputProjector()
    feat_proj = FeatureProjector()
    
    x_qwen = torch.randn(1, 10, 4096)
    x_bert = in_proj(x_qwen)
    print(f"InputProjection (Qwen->BERT): {x_qwen.shape} -> {x_bert.shape}")
    
    x_back = feat_proj(x_bert)
    print(f"FeatureProjection (BERT->Qwen): {x_bert.shape} -> {x_back.shape}")
