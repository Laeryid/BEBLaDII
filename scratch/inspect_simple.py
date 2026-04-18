import torch
path = r"c:\Experiments\BEBLaDII\storage\experiments\20260414 Phase 1 Reasoning failed start\BEST_MODEL.pt"
print(f"Loading {path}...")
try:
    ckpt = torch.load(path, map_location='cpu')
    print(f"Success! Type: {type(ckpt)}")
    if hasattr(ckpt, 'keys'):
        print(f"Keys: {list(ckpt.keys())[:20]}")
    else:
        print("Object has no keys.")
except Exception as e:
    print(f"FAIL: {e}")
