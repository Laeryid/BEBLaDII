import torch
path = r"c:\Experiments\BEBLaDII\storage\experiments\20260414 Phase 1 Reasoning failed start\BEST_MODEL.pt"
print(f"Inspecting BEST_MODEL.pt: {path}")
try:
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict):
        print(f"Top level keys: {list(ckpt.keys())}")
        for k in list(ckpt.keys()):
            val = ckpt[k]
            if isinstance(val, dict):
                print(f"  {k}: dict with {len(val)} items. First 3 keys: {list(val.keys())[:3]}")
    else:
        print(f"Loaded object is {type(ckpt)}")
except Exception as e:
    print(f"Error: {e}")
