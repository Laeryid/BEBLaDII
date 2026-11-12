import torch

ckpt = torch.load("experiments/phase 1/checkpoints/phase1_best_model.pt", map_location="cpu", weights_only=False)
sd = ckpt["model_state_dict"]
for name in ["unpacker.gram_mlp.0.weight", "unpacker.gram_mlp.0.bias", "unpacker.gram_mlp.2.weight", "unpacker.out_proj.weight"]:
    if name in sd:
        w = sd[name]
        print(f"{name}: shape={list(w.shape)}, min={w.min().item():.5f}, max={w.max().item():.5f}, norm={w.norm().item():.5f}")
