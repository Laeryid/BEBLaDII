import torch
from transformers import AutoConfig, AutoModelForMaskedLM
import copy
from .base import BEComponent

class DUSModel(BEComponent):
    """
    Компонент-обертка для модели с расширенной глубиной (DUS).
    """
    def __init__(self, component_id="modernbert_dus_40", version="v1.0", config=None):
        base_model_id = config.get("base_model_id", "answerdotai/ModernBERT-large") if config else "answerdotai/ModernBERT-large"
        target_layers = config.get("target_layers", 40) if config else 40
        super().__init__(component_id, version, {"base_model_id": base_model_id, "target_layers": target_layers})
        
        # Создаем модель
        self.model = create_latentbert(base_model_id, target_layers)
        
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
    Создает latentBERT через Depth Up-Scaling (DUS) базовой модели ModernBERT.
    
    Схема для 40 слоев из 28:
    Блок 1: Слои 0-19 (20 слоев)
    Блок 2: Слои 8-27 (20 слоев)
    """
    print(f"Загрузка базовой модели {model_id}...")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    base_model = AutoModelForMaskedLM.from_pretrained(model_id, trust_remote_code=True)
    
    # Изменяем конфигурацию
    new_config = copy.deepcopy(config)
    new_config.num_hidden_layers = target_layers
    
    # Создаем новую модель с новой конфигурацией
    # Важно: ModernBERT использует специфичные слои, поэтому мы создаем пустую модель и копируем веса.
    # Для целей Фазы 1 мы можем просто вручную собрать список слоев.
    
    layers = base_model.model.layers
    num_base_layers = len(layers)
    print(f"Базовая модель имеет {num_base_layers} слоев.")
    
    # 1. Первый блок (0-19)
    new_layers = torch.nn.ModuleList([copy.deepcopy(layers[i]) for i in range(20)])
    
    # 2. Второй блок (8-27)
    new_layers.extend([copy.deepcopy(layers[i]) for i in range(8, 28)])
    
    print(f"Создано {len(new_layers)} слоев.")
    
    # Подменяем слои в базовой модели (или создаем новую структуру)
    base_model.model.layers = new_layers
    base_model.config.num_hidden_layers = target_layers
    
    # Включаем Gradient Checkpointing
    base_model.gradient_checkpointing_enable()
    
    return base_model

if __name__ == "__main__":
    # Тест без загрузки весов (только конфиг если возможно, либо маленькая модель)
    # В реальном сценарии на Kaggle будет загружаться полная модель.
    try:
        model = create_latentbert()
        print(f"LatentBERT успешно создан. Количество слоев: {len(model.model.layers)}")
    except Exception as e:
        print(f"Ошибка при тестировании: {e}")
