import os
import torch
import json
from typing import Type
from .base import BEComponent


class ComponentRegistry:
    """
    Реестр компонентов BEBLaDII.
    Отвечает только за сохранение компонентов после обучения.

    Загрузка весов теперь производится через weights_map в ModelAssembler:
    скелет каждого компонента строится из кода, веса загружаются напрямую
    из файла через BEComponent.load_weights(path).
    """
    def __init__(self, storage_root: str = "storage/components"):
        self.storage_root = storage_root
        os.makedirs(self.storage_root, exist_ok=True)

    def component_path(self, component_type: str, component_id: str, version: str) -> str:
        """Возвращает путь к директории компонента в storage."""
        return os.path.join(self.storage_root, component_type, component_id, version)

    def weights_path(self, component_type: str, component_id: str, version: str) -> str:
        """Возвращает путь к weights.pt компонента."""
        return os.path.join(self.component_path(component_type, component_id, version), "weights.pt")

    def save_component(self, component: BEComponent, component_type: str):
        """
        Сохраняет веса и метаданные компонента в storage.
        Структура: storage/components/{type}/{id}/{version}/weights.pt
        """
        path = self.component_path(component_type, component.component_id, component.version)
        os.makedirs(path, exist_ok=True)

        weights_file = os.path.join(path, "weights.pt")
        torch.save(component.state_dict(), weights_file)

        component.save_metadata(os.path.join(path, "config.json"))
        print(f"Saved {component.component_id} ({component.version}) -> {path}")
        return weights_file

    def list_versions(self, component_type: str, component_id: str):
        """Возвращает список доступных версий для компонента."""
        id_path = os.path.join(self.storage_root, component_type, component_id)
        if not os.path.exists(id_path):
            return []
        return [d for d in os.listdir(id_path) if os.path.isdir(os.path.join(id_path, d))]
