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

    def load_weights(self, weights_path: str):
        """
        Загружает веса из файла weights.pt в текущий компонент.

        Основной подход для всех компонентов:
          - Скелет (архитектура) всегда создаётся заново из кода через from_scratch().
          - Веса опционально подгружаются из файла поверх скелета.
          - Если weights_path не передан или файл не найден — используется случайная инициализация.

        Args:
            weights_path: Путь к файлу weights.pt, или None для случайной инициализации.
        """
        import torch
        if weights_path and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location="cpu")
            self.load_state_dict(state)
            print(f"  Weights loaded: {weights_path}")
        elif weights_path:
            print(f"  WARN: weights_path not found, using random init: {weights_path}")
        return self

    @classmethod
    def from_scratch(cls, component_id: str, version: str = "v1.0", **kwargs):
        """
        Создаёт компонент с нуля (случайная инициализация).
        Принимает опциональный `weights_path` — если передан, загружает веса из файла.
        """
        raise NotImplementedError("Этот метод должен быть реализован в подклассе.")
