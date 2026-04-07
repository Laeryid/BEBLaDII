import os
import json
import torch
import datetime
from typing import Dict, Any
from ..model.component_registry import ComponentRegistry

class ExperimentManager:
    """
    Класс для управления экспериментами и снимками состояний (Snapshots).
    """
    def __init__(self, 
                 experiment_root: str = "storage/experiments", 
                 registry: ComponentRegistry = None):
        self.experiment_root = experiment_root
        self.registry = registry or ComponentRegistry()
        os.makedirs(self.experiment_root, exist_ok=True)

    def create_experiment(self, name: str, config: Dict[str, Any]):
        """
        Создает новую папку эксперимента и сохраняет начальный конфиг.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"{name}_{timestamp}"
        exp_path = os.path.join(self.experiment_root, exp_id)
        os.makedirs(exp_path, exist_ok=True)
        
        config["experiment_id"] = exp_id
        config["start_time"] = timestamp
        
        with open(os.path.join(exp_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
            
        print(f"Эксперимент {exp_id} инициализирован.")
        return exp_id, exp_path

    def save_snapshot(self, exp_path: str, components: Dict[str, Any], results: Dict[str, Any]):
        """
        Сохраняет финальное состояние эксперимента: ссылки на версии компонентов и метрики.
        """
        snapshot = {
            "timestamp": datetime.datetime.now().isoformat(),
            "results": results,
            "composition": {}
        }
        
        # Сохраняем информацию о каждом компоненте
        for name, comp in components.items():
            if hasattr(comp, "get_metadata"):
                snapshot["composition"][name] = {
                    "id": comp.component_id,
                    "version": comp.version,
                    "is_trainable": any(p.requires_grad for p in comp.parameters())
                }
            else:
                snapshot["composition"][name] = "External (frozen/native)"
                
        snapshot_path = os.path.join(exp_path, "snapshot.json")
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=4, ensure_ascii=False)
            
        print(f"Снимок эксперимента сохранен: {snapshot_path}")
        return snapshot_path
