import torch
from transformers import AutoConfig, AutoModel

model_id = "answerdotai/ModernBERT-large"
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

# Инициализируем модель на meta-девайсе, чтобы не грузить веса
with torch.device("meta"):
    model = AutoModel.from_config(config, trust_remote_code=True)

# 28 layers base
num_layers_base = len(model.layers)
params_base = sum(p.numel() for p in model.parameters())
params_layers_only = sum(p.numel() for p in model.layers.parameters())
params_per_layer = params_layers_only / num_layers_base

# 40 layers target
target_layers = 40
total_params_40 = params_base + (target_layers - num_layers_base) * params_per_layer

print(f"Base model layers: {num_layers_base}")
print(f"Params per layer: {params_per_layer / 1e6:.2f}M")
print(f"Base model total: {params_base / 1e6:.2f}M")
print(f"Target 40-layer model (without CA/Head): {total_params_40 / 1e6:.2f}M")
