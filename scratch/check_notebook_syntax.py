import os
import sys
from pathlib import Path
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
import wandb
import shutil

# Симуляция импортов проекта (проверка синтаксиса использования)
# from src.beb_la_dii.model.distiller import ReasoningDistiller

BASE_MODEL_NAME = "answerdotai/ModernBERT-large"
TEACHER_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MAX_LENGTH = 512
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
EPOCHS = 1
LEARNING_RATE = 5e-5
STAGE = 'awakening'
VERSION = "v1.0"

def setup_kaggle():
    if not os.path.exists("/kaggle/input"):
        return None
    os.makedirs("data", exist_ok=True)
    input_base = "/kaggle/input"
    resource_ds = None
    for ds_name in os.listdir(input_base):
        if "bebladii" in ds_name.lower():
            resource_ds = os.path.join(input_base, ds_name)
            break
    if not resource_ds:
        return None
    ds_data_path = os.path.join(resource_ds, "data")
    if os.path.exists(ds_data_path):
        for folder in os.listdir(ds_data_path):
            src = os.path.join(ds_data_path, folder)
            dst = os.path.join("data", folder)
            if not os.path.exists(dst):
                os.symlink(src, dst)
    return resource_ds
