import torch

ckpt_path = r"c:\Experiments\BEBLaDII\storage\experiments\20260504 Phase + Reasoning 14500 steps\checkpoints_ckpt_14500.pt"
ckpt = torch.load(ckpt_path, map_location='cpu')
sd = ckpt.get('model_state_dict', {})

print("First 10 keys:")
for k in list(sd.keys())[:10]:
    print(k)

print("\nLayer 40 keys (first 5):")
l40_keys = [k for k in sd.keys() if "layers.39" in k][:5]
for k in l40_keys: print(k)

print("\nProjector 40 keys (first 5):")
p40_keys = [k for k in sd.keys() if "feature_projectors.40" in k][:5]
for k in p40_keys: print(k)
