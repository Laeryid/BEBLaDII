import torch
from transformers import AutoConfig, AutoModelForMaskedLM
import copy
from .base import BEComponent

class DUSModel(BEComponent):
    """
    Wrapper for Depth Up-Scaling (DUS) model.
    """
    def __init__(self, component_id="modernbert_dus_40", version="v1.0", config=None):
        base_model_id = config.get("base_model_id", "answerdotai/ModernBERT-large") if config else "answerdotai/ModernBERT-large"
        target_layers = config.get("target_layers", 40) if config else 40
        super().__init__(component_id, version, {"base_model_id": base_model_id, "target_layers": target_layers})
        
        # Create model
        self.model = create_latentbert(base_model_id, target_layers)
        
    @classmethod
    def from_scratch(cls, component_id="latentBERT", version="v1.0", **kwargs):
        """Создает новую 40-слойную модель через DUS."""
        config = kwargs.get("config", {
            "base_model_id": "answerdotai/ModernBERT-large",
            "target_layers": 40
        })
        return cls(component_id=component_id, version=version, config=config)
        
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict)

def create_latentbert(model_id="answerdotai/ModernBERT-large", target_layers=40):
    """
    Creates latentBERT via Depth Up-Scaling (DUS).
    
    40-layer scheme from 28 layers:
    Block 1: Layers 0-19 (20 layers)
    Block 2: Layers 8-27 (20 layers)
    """
    print(f"Loading model architecture from {model_id}...")
    
    # 1. Загрузка конфигурации с защитой от "битых" локальных конфигов
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        # Проверяем, что конфиг корректный (в нем должен быть model_type)
        if not hasattr(config, "model_type"):
            raise ValueError("model_type not found in config")
    except Exception as e:
        print(f"Warning: Failed to load valid config from {model_id} ({e}).")
        print("Downloading original config from internet (answerdotai/ModernBERT-large)...")
        config = AutoConfig.from_pretrained("answerdotai/ModernBERT-large", trust_remote_code=True)
        # Так как это наш 40-слойный скелет, обновляем количество слоев:
        config.num_hidden_layers = target_layers

    # 2. Загрузка весов, передаем наш 'починенный' конфигурационный объект
    base_model = AutoModelForMaskedLM.from_pretrained(model_id, config=config, trust_remote_code=True)
    
    # Check if the model is already scaled
    current_layers = len(base_model.model.layers)
    if current_layers == target_layers:
        print(f"Model already has {target_layers} layers. Skipping DUS logic.")
        base_model.gradient_checkpointing_enable()
        return base_model

    layers = base_model.model.layers
    num_base_layers = len(layers)
    
    # 1. First block (0-19)
    new_layers = torch.nn.ModuleList([copy.deepcopy(layers[i]) for i in range(20)])
    
    # 2. Second block (8-27)
    new_layers.extend([copy.deepcopy(layers[i]) for i in range(8, 28)])
    
    print(f"Created {len(new_layers)} layers from {num_base_layers} base layers (DUS applied).")
    
    # Inject layers
    base_model.model.layers = new_layers
    base_model.config.num_hidden_layers = target_layers
    
    # Enable Gradient Checkpointing
    base_model.gradient_checkpointing_enable()
    
    return base_model
