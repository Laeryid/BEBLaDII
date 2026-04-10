import os
import json
import datetime
import subprocess
import platform
import sys
import torch
from typing import Any, Dict, Optional, List

class ExperimentTracker:
    """
    Класс для отслеживания экспериментов, сохранения метаданных и истории обучения.
    Ориентирован на работу в Kaggle и легкий перенос данных в систему.
    """
    def __init__(self, 
                 project_root: str, 
                 stage: str, 
                 experiment_name: Optional[str] = None,
                 storage_dir: str = "storage/experiments"):
        self.project_root = os.path.abspath(project_root)
        self.stage = stage
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Генерация run_id
        name_prefix = f"{experiment_name}_" if experiment_name else ""
        self.run_id = f"{self.timestamp}_{name_prefix}{stage}"
        
        # Директории
        self.exp_dir = os.path.join(self.project_root, storage_dir, self.run_id)
        self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoints")
        self.logs_dir = os.path.join(self.exp_dir, "logs")
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.history_file = os.path.join(self.logs_dir, "history.jsonl")
        self.metadata_file = os.path.join(self.exp_dir, "experiment.json")

    def _get_git_commit(self) -> str:
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], 
                                         cwd=self.project_root, 
                                         stderr=subprocess.DEVNULL).decode("ascii").strip()
        except Exception:
            return "unknown"

    def _get_env_info(self) -> Dict[str, str]:
        info = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
        }
        try:
            import transformers
            info["transformers"] = transformers.__version__
        except ImportError:
            pass
        return info

    def log_metadata(self, 
                     hyperparams: Dict[str, Any], 
                     dataset_configs: List[Dict[str, Any]]):
        """
        Сохраняет начальные метаданные эксперимента.
        """
        metadata = {
            "run_id": self.run_id,
            "stage": self.stage,
            "timestamp": self.timestamp,
            "git_commit": self._get_git_commit(),
            "env": self._get_env_info(),
            "hyperparams": hyperparams,
            "datasets": dataset_configs,
            "status": "started"
        }
        
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        
        print(f"Experiment metadata saved to: {self.metadata_file}")

    def log_step(self, step: int, metrics: Dict[str, Any]):
        """
        Добавляет запись в историю обучения (history.jsonl).
        """
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "step": step,
            **metrics
        }
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def save_checkpoint(self, 
                        state: Dict[str, Any], 
                        optimizer_state: Optional[Dict[str, Any]] = None, 
                        name: str = "checkpoint"):
        """
        Сохраняет веса модели и, опционально, состояние оптимизатора.
        """
        save_data = state.copy()
        if optimizer_state is not None:
            save_data["optimizer_state_dict"] = optimizer_state
            
        save_path = os.path.join(self.checkpoint_dir, f"{name}.pt")
        torch.save(save_data, save_path)
        print(f"Checkpoint saved: {save_path}")
        
    def finish(self, status: str = "completed"):
        """
        Обновляет статус эксперимента в метаданных.
        """
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            metadata["status"] = status
            metadata["finished_at"] = datetime.datetime.now().isoformat()
            
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
