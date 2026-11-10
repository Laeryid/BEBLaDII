import torch
import os

ckpt_path = r"C:\Experiments\BEBLaDII\experiments\phase3\planB_phase3_checkpoints_phase3_step_8000.pth"
state = torch.load(ckpt_path, map_location="cpu")

dus = state.get("dus_ema", state.get("dus", state))
print(f"Loaded DUS keys: {len(dus.keys())}")

# Check variance of last layer weights
last_layer_weight = dus.get('student.model.layers.39.mlp.dense.weight', None)
if last_layer_weight is None:
    last_layer_weight = dus.get('layers.39.mlp.dense.weight', None)
if last_layer_weight is not None:
    print(f"Layer 39 MLP dense weight std: {last_layer_weight.std().item()}")
    
# Check sum of norms of all weights
total_norm = 0.0
for k, v in dus.items():
    if v.is_floating_point():
        total_norm += v.norm().item()
print(f"Total norm of all DUS weights: {total_norm}")

c_proj = state.get("confidence_proj_ema", state.get("confidence_proj", {}))
if c_proj:
    print("Confidence Proj weights:")
    for k, v in c_proj.items():
        print(f"  {k}: norm = {v.norm().item()}")
