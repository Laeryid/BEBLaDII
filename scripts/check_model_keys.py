import torch
import sys
import os

# Добавляем путь к src
sys.path.insert(0, os.getcwd())

try:
    from src.beb_la_dii.model.assembler import ModelAssembler
    
    assembler = ModelAssembler()
    # Собираем минимальный дистиллятор для теста (без учителя, если можно)
    # Но нам нужны только ключи.
    
    # Имитируем параметры из train_phase1_kaggle
    distiller = assembler.assemble_phase1_distiller(
        teacher_id="hf-internal-testing/tiny-random-BertModel", # Заглушка
        student_base_id="answerdotai/ModernBERT-large",
        version="v1.0",
        device_map="cpu"
    )
    
    keys = list(distiller.state_dict().keys())
    print(f"Total keys in distiller: {len(keys)}")
    print("Example keys:")
    for k in keys:
        if "layers.0" in k:
            print(f"  {k}")
            break
    for k in keys:
        if "input_projector" in k:
            print(f"  {k}")
            break
            
except Exception as e:
    print(f"Error: {e}")
