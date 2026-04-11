import torch.nn as nn
import json
import os

class BEComponent(nn.Module):
    """
    Базовый класс для всех компонентов архитектуры BEBLaDII.
    Обеспечивает поддержку версионирования и хранения конфигурации.
    """
    def __init__(self, component_id: str, version: str = "v1.0", config: dict = None):
        super().__init__()
        self.component_id = component_id
        self.version = version
        self.config = config or {}
        
    def get_metadata(self):
        """Возвращает словарь с метаданными компонента."""
        return {
            "component_id": self.component_id,
            "version": self.version,
            "config": self.config,
            "class_name": self.__class__.__name__
        }
    
    def save_metadata(self, path: str):
        """Сохраняет метаданные в JSON файл."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.get_metadata(), f, indent=4, ensure_ascii=False)

    @classmethod
    def from_scratch(cls, component_id: str, version: str = "v1.0", **kwargs):
        """Метод для инициализации нового компонента с нуля."""
        raise NotImplementedError("Этот метод должен быть реализован в подклассе.")
