import sys
import os

# Добавляем путь к src, чтобы импорты работали
sys.path.append(os.path.abspath("src"))

import torch
from beb_la_dii.model.component_registry import ComponentRegistry
from beb_la_dii.model.dus import DUSModel
from beb_la_dii.model.projectors import InputProjector, FeatureProjector

def init_all():
    registry = ComponentRegistry(storage_root="storage/components")
    
    print("--- Initializing latentBERT (ModernBERT 40 layers) ---")
    # Using default config for DUS
    student = DUSModel(
        component_id="latentBERT",
        version="v1.0",
        config={"base_model_id": "answerdotai/ModernBERT-large", "target_layers": 40}
    )
    registry.save_component(student, component_type="model")
    
    print("\n--- Initializing Input Projector ---")
    input_proj = InputProjector(
        component_id="qwen_to_bert_input",
        version="v1.0",
        config={"input_dim": 4096, "output_dim": 1024}
    )
    registry.save_component(input_proj, component_type="projector")
    
    print("\n--- Initializing Feature Projectors ---")
    for layer in [20, 30, 40]:
        feat_proj = FeatureProjector(
            component_id=f"feat_proj_{layer}",
            version="v1.0",
            config={"input_dim": 1024, "output_dim": 4096}
        )
        registry.save_component(feat_proj, component_type="projector")

    print("\nAll components successfully initialized and saved in storage/components")

if __name__ == "__main__":
    init_all()
