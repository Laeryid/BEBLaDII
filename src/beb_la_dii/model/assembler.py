import torch.nn as nn
from .component_registry import ComponentRegistry
from .dus import DUSModel
from .projectors import InputProjector, FeatureProjector
from .distiller import ReasoningDistiller


class ModelAssembler:
    """
    Класс-сборщик системы BEBLaDII.

    Основной подход к весам: скелет каждого компонента создаётся заново из кода,
    веса опционально загружаются из файла через weights_map.

    weights_map: dict[component_id -> str (путь к weights.pt)]
    Если ключ отсутствует или путь не найден — используется случайная инициализация.

    Пример:
        weights_map = {
            "latentBERT":         "storage/components/model/latentBERT/v1.0/weights.pt",
            "qwen_to_bert_input": "storage/components/projector/qwen_to_bert_input/v1.0/weights.pt",
            "feat_proj_20":       "storage/components/projector/feat_proj_20/v1.0/weights.pt",
            "feat_proj_30":       "storage/components/projector/feat_proj_30/v1.0/weights.pt",
            "feat_proj_40":       "storage/components/projector/feat_proj_40/v1.0/weights.pt",
        }
    """
    def __init__(self, storage_root: str = "storage/components"):
        # Registry используется только для сохранения компонентов после обучения
        self.registry = ComponentRegistry(storage_root=storage_root)

    def assemble_phase1_distiller(self,
                                  teacher_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                                  student_base_id="answerdotai/ModernBERT-large",
                                  version="v1.0",
                                  weights_map=None,
                                  device_map="auto",
                                  student_device=None):
        """
        Собирает полную систему для Фазы 1 дистилляции.

        Args:
            teacher_id:      HF ID или локальный путь к модели-учителю.
            student_base_id: HF ID или локальный путь к базовой модели студента (или пребилту).
            version:         Версия компонентов (используется при сохранении).
            weights_map:     dict[component_id -> weights_path]. Если None — все компоненты
                             инициализируются случайно.
            device_map:      Стратегия размещения учителя ('auto' для bitsandbytes).
        """
        weights_map = weights_map or {}
        print(f"Assembling distillation system (version {version})...")

        # 1. latentBERT — скелет строится через DUS, веса опционально из файла
        print(f"[1/3] Building latentBERT skeleton (from {student_base_id})...")
        student = DUSModel.from_scratch(
            component_id="latentBERT",
            version=version,
            weights_path=weights_map.get("latentBERT"),
            config={"base_model_id": student_base_id, "target_layers": 40}
        )

        # 2. InputProjector
        print("[2/3] Building InputProjector...")
        input_projector = InputProjector.from_scratch(
            component_id="qwen_to_bert_input",
            version=version,
            weights_path=weights_map.get("qwen_to_bert_input"),
        )

        # 3. FeatureProjectors x3
        print("[3/3] Building FeatureProjectors...")
        feature_projectors = nn.ModuleDict({
            "20": FeatureProjector.from_scratch(
                component_id="feat_proj_20", version=version,
                weights_path=weights_map.get("feat_proj_20"),
            ),
            "30": FeatureProjector.from_scratch(
                component_id="feat_proj_30", version=version,
                weights_path=weights_map.get("feat_proj_30"),
            ),
            "40": FeatureProjector.from_scratch(
                component_id="feat_proj_40", version=version,
                weights_path=weights_map.get("feat_proj_40"),
            ),
        })

        # 4. Сборка дистиллятора
        distiller = ReasoningDistiller(
            teacher_id=teacher_id,
            student=student,
            input_projector=input_projector,
            feature_projectors=feature_projectors,
            device_map=device_map,
            student_device=student_device,
        )

        return distiller
