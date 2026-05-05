import torch
import os

ckpt_path = r"c:\Experiments\BEBLaDII\storage\experiments\20260504 Phase + Reasoning 14500 steps\checkpoints_ckpt_14500.pt"
if not os.path.exists(ckpt_path):
    print(f"File {ckpt_path} not found.")
else:
    print(f"Checking {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt.get('model_state_dict', {})
    
    # 1. Проверка слоев студента
    layer_keys = [k for k in sd.keys() if "student.model.layers" in k]
    layers = set()
    for k in layer_keys:
        parts = k.split('.')
        try:
            idx = parts.index("layers")
            layers.add(int(parts[idx+1]))
        except: pass
            
    if not layers:
        print("[-] No student layers found.")
    else:
        print(f"[+] Found {len(layers)} student layers (max index: {max(layers)})")

    # 2. Проверка проекторов
    proj_keys = [k for k in sd.keys() if "feature_projectors" in k]
    projs = set()
    for k in proj_keys:
        parts = k.split('.')
        # feature_projectors.20.xxx
        if "feature_projectors" in parts:
            idx = parts.index("feature_projectors")
            projs.add(parts[idx+1])
    
    print(f"[+] Found feature projectors for layers: {sorted(list(projs))}")
    
    if 'global_step' in ckpt:
        print(f"[+] Global step: {ckpt['global_step']}")
