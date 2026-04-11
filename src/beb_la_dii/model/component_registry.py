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
    def __init__(self, storage_root: str = "storage/components", alt_roots: list = None):
        self.storage_root = storage_root
        self.alt_roots = alt_roots or []
        os.makedirs(self.storage_root, exist_ok=True)

    def _get_path(self, component_type: str, component_id: str, version: str, root: str = None):
        target_root = root or self.storage_root
        return os.path.join(target_root, component_type, component_id, version)

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
        Проверяет основное хранилище и список alt_roots.
        """
        # Сначала ищем в основном хранилище, затем в альтернативных
        search_roots = [self.storage_root] + self.alt_roots
        
        final_config_path = None
        final_weights_path = None
        
        for root in search_roots:
            # Пытаемся найти по стандартной структуре реестра
            path = self._get_path(component_type, component_id, version, root)
            config_path = os.path.join(path, "config.json")
            weights_path = os.path.join(path, "weights.pt")
            
            if os.path.exists(config_path):
                final_config_path = config_path
                final_weights_path = weights_path
                break
                
            # Специальная обработка для плоской структуры (например, корень датасета Kaggle)
            # Если это latentBERT, ищем weights.pt в корне root
            if component_id == "latentBERT" and os.path.exists(os.path.join(root, "weights.pt")):
                # Создаем временный конфиг в памяти или используем дефолтный, если нет config.json
                # Но по правилам DUS нам нужен config.json. 
                # Если его нет даже в корне, мы упадем позже.
                if os.path.exists(os.path.join(root, "config.json")):
                    final_config_path = os.path.join(root, "config.json")
                    final_weights_path = os.path.join(root, "weights.pt")
                    break

        if not final_config_path or not os.path.exists(final_config_path):
            raise FileNotFoundError(f"Метаданные для {component_id} ({version}) не найдены в поисковых путях.")
            
        with open(final_config_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        # Создаем экземпляр
        instance = component_class(
            component_id=meta["component_id"],
            version=meta["version"],
            config=meta["config"]
        )
        
        if final_weights_path and os.path.exists(final_weights_path):
            instance.load_state_dict(torch.load(final_weights_path, map_location="cpu"))
            print(f"Веса для {component_id} ({version}) загружены из {final_weights_path}.")
        else:
            print(f"ВНИМАНИЕ: Веса для {component_id} {version} не найдены. Использована инициализация по умолчанию.")
            
        return instance

    def list_versions(self, component_type: str, component_id: str):
        """Возвращает список доступных версий для компонента."""
        id_path = os.path.join(self.storage_root, component_type, component_id)
        if not os.path.exists(id_path):
            return []
        return [d for d in os.listdir(id_path) if os.path.isdir(os.path.join(id_path, d))]
