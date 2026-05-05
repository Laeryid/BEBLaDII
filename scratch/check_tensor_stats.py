import torch

ckpt_path = r"c:\Experiments\BEBLaDII\storage\experiments\20260504 Phase + Reasoning 14500 steps\checkpoints_ckpt_14500.pt"
ckpt = torch.load(ckpt_path, map_location='cpu')
sd = ckpt.get('model_state_dict', {})

def check_layer(idx):
    keys = [k for k in sd.keys() if f"layers.{idx}." in k and "weight" in k]
    if not keys:
        print(f"Layer {idx}: No weights found")
        return
    
    print(f"Layer {idx}:")
    for k in keys[:2]: # Проверим пару тензоров
        v = sd[k]
        print(f"  {k}: shape={list(v.shape)}, mean={v.float().mean().item():.6f}, std={v.float().std().item():.6f}")

check_layer(19) # Последний слой первого блока
check_layer(39) # Последний слой модели (40-й)

print("\nChecking Projectors:")
for p_idx in ["20", "40"]:
    keys = [k for k in sd.keys() if f"feature_projectors.{p_idx}." in k and "weight" in k]
    for k in keys[:1]:
        v = sd[k]
        print(f"  Projector {p_idx} ({k}): shape={list(v.shape)}, mean={v.float().mean().item():.6f}")
