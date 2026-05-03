import torch
from transformers import AutoConfig, AutoModel
import copy
from .base import BEComponent

class DUSModel(BEComponent):
    """
    Wrapper for Depth Up-Scaling (DUS) model.

    Скелет (40-слойная архитектура) всегда создаётся заново через DUS из ModernBERT-large.
    Предобученные веса опционально загружаются поверх скелета из файла.
    """
    def __init__(self, component_id="latentBERT", version="v1.0", config=None):
        config = config or {}
        base_model_id = config.get("base_model_id", "answerdotai/ModernBERT-large")
        target_layers = config.get("target_layers", 40)
        
        super().__init__(component_id, version, {"base_model_id": base_model_id, "target_layers": target_layers})
        
        # Create model skeleton via DUS
        self.model = create_latentbert(base_model_id, target_layers)
        
    @classmethod
    def from_scratch(cls, component_id="latentBERT", version="v1.0",
                     weights_path=None, **kwargs):
        """
        Создаёт DUSModel с нуля: скелет строится через DUS из ModernBERT-large.
        weights_path: путь к weights.pt; если None — используются веса из ModernBERT (DUS).
        """
        config = kwargs.get("config", {
            "base_model_id": "answerdotai/ModernBERT-large",
            "target_layers": 40
        })
        instance = cls(component_id=component_id, version=version, config=config)
        # Для DUSModel weights_path загружает поверх DUS-инициализированных весов
        instance.load_weights(weights_path)
        return instance
        
    def load_weights(self, weights_path):
        """
        Загружает веса в self.model (внутренний HF модуль, не в self).
        Переопределяет базовый метод, т.к. веса хранятся в self.model.
        """
        import torch
        import os
        if weights_path and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(state)
            print(f"  Weights loaded: {weights_path}")
        elif weights_path:
            print(f"  WARN: weights_path not found, using DUS init: {weights_path}")
        return self

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse)


def create_latentbert(model_id="answerdotai/ModernBERT-large", target_layers=40):
    """
    Creates latentBERT via Depth Up-Scaling (DUS).
    Always builds the skeleton from scratch from ModernBERT-large.
    """
    import torch
    print(f"Loading model architecture from {model_id}...")
    
    # Пытаемся загрузить конфиг. Если это папка, он загрузится локально.
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    
    # If model already has target_layers (e.g. saved skeleton), skip DUS
    current_layers = len(base_model.layers)
    if current_layers == target_layers:
        print(f"Model already has {target_layers} layers. Skipping DUS logic.")
        base_model.gradient_checkpointing_enable()
        return base_model

    layers = base_model.layers
    num_base_layers = len(layers)
    
    # Block 1: layers 0-19 (first 20)
    new_layers = torch.nn.ModuleList([copy.deepcopy(layers[i]) for i in range(20)])
    
    # Block 2: layers 8-27 (last 20, overlapping middle)
    new_layers.extend([copy.deepcopy(layers[i]) for i in range(8, 28)])
    
    print(f"Created {len(new_layers)} layers from {num_base_layers} base layers (DUS applied).")
    
    # Inject layers
    base_model.layers = new_layers
    base_model.config.num_hidden_layers = target_layers
    
    # Enable Gradient Checkpointing for memory efficiency
    base_model.gradient_checkpointing_enable()
    
    return base_model
