import torch.nn as nn
from .component_registry import ComponentRegistry
from .dus import DUSModel
from .projectors import InputProjector, FeatureProjector
from .distiller import ReasoningDistiller

class ModelAssembler:
    """
    Класс-сборщик системы BEBLaDII.
    Реализует логику: "Найти в реестре -> если нет, инициализировать с нуля".
    """
    def __init__(self, registry: ComponentRegistry = None):
        self.registry = registry or ComponentRegistry()

    def get_component(self, cls, component_type: str, component_id: str, version: str = "v1.0", **kwargs):
        """
        Пытается загрузить компонент из реестра. 
        В случае неудачи — создает новый через from_scratch.
        """
        try:
            instance = self.registry.load_component(cls, component_type, component_id, version)
            return instance
        except Exception as e:
            print(f"Инфо: Компонент {component_id} ({version}) не найден в реестре. Создание с нуля... ({str(e)})")
            return cls.from_scratch(component_id=component_id, version=version, **kwargs)

    def assemble_phase1_distiller(self, 
                                 teacher_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
                                 version="v1.0",
                                 device_map="auto"):
        """
        Собирает полную систему для Фазы 1 дистилляции.
        """
        print(f"Сборка системы дистилляции (версия {version})...")
        
        # 1. Загрузка/Инициализация latentBERT
        student = self.get_component(DUSModel, "model", "latentBERT", version)
        
        # 2. Загрузка/Инициализация входного проектора
        input_projector = self.get_component(InputProjector, "projector", "qwen_to_bert_input", version)
        
        # 3. Загрузка/Инициализация проекторов признаков
        feature_projectors = nn.ModuleDict({
            "20": self.get_component(FeatureProjector, "projector", "feat_proj_20", version),
            "30": self.get_component(FeatureProjector, "projector", "feat_proj_30", version),
            "40": self.get_component(FeatureProjector, "projector", "feat_proj_40", version)
        })
        
        # 4. Сборка дистиллятора
        distiller = ReasoningDistiller(
            teacher_id=teacher_id,
            student=student,
            input_projector=input_projector,
            feature_projectors=feature_projectors,
            device_map=device_map
        )
        
        return distiller
