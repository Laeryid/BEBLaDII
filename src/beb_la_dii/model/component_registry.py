import os
import torch
import json
from typing import Dict, Type
from .base import BEComponent

class ComponentRegistry:
    """
    Реестр компонентов для BEBLaDII.
    Управляет версиями весов и конфигураций в хранилище.
    """
    def __init__(self, storage_root: str = "storage/components"):
        self.storage_root = storage_root
        os.makedirs(self.storage_root, exist_ok=True)

    def _get_path(self, component_type: str, component_id: str, version: str):
        return os.path.join(self.storage_root, component_type, component_id, version)

    def save_component(self, component: BEComponent, component_type: str):
        """
        Сохраняет веса и метаданные компонента в реестр.
        """
        path = self._get_path(component_type, component.component_id, component.version)
        os.makedirs(path, exist_ok=True)
        
        # Сохранение весов
        torch.save(component.state_dict(), os.path.join(path, "weights.pt"))
        
        # Сохранение метаданных
        component.save_metadata(os.path.join(path, "config.json"))
        print(f"Компонент {component.component_id} ({component.version}) сохранен в {path}")

    def load_component(self, 
                       component_class: Type[BEComponent], 
                       component_type: str, 
                       component_id: str, 
                       version: str):
        """
        Загружает компонент указанной версии из реестра.
        """
        path = self._get_path(component_type, component_id, version)
        config_path = os.path.join(path, "config.json")
        weights_path = os.path.join(path, "weights.pt")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Метаданные не найдены по адресу {config_path}")
            
        with open(config_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        # Создаем экземпляр
        instance = component_class(
            component_id=meta["component_id"],
            version=meta["version"],
            config=meta["config"]
        )
        
        if os.path.exists(weights_path):
            instance.load_state_dict(torch.load(weights_path, map_location="cpu"))
            print(f"Веса для {component_id} ({version}) загружены.")
        else:
            print(f"ВНИМАНИЕ: Веса для {component_id} {version} не найдены. Использована инициализация по умолчанию.")
            
        return instance

    def list_versions(self, component_type: str, component_id: str):
        """Возвращает список доступных версий для компонента."""
        id_path = os.path.join(self.storage_root, component_type, component_id)
        if not os.path.exists(id_path):
            return []
        return [d for d in os.listdir(id_path) if os.path.isdir(os.path.join(id_path, d))]
