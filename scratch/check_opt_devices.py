import torch

ckpt = torch.load("experiments/phase 1/checkpoints/phase1_best_model.pt", map_location="cpu", weights_only=False)
if "opt_muon_state_dict" in ckpt:
    state = ckpt["opt_muon_state_dict"]["state"]
    for k, v in list(state.items())[:5]:
        for buf_name, buf_val in v.items():
            if isinstance(buf_val, torch.Tensor):
                print(f"Muon state {k} {buf_name}: device={buf_val.device}, shape={buf_val.shape}")

if "opt_adamw_state_dict" in ckpt:
    state = ckpt["opt_adamw_state_dict"]["state"]
    for k, v in list(state.items())[:5]:
        for buf_name, buf_val in v.items():
            if isinstance(buf_val, torch.Tensor):
                print(f"AdamW state {k} {buf_name}: device={buf_val.device}, shape={buf_val.shape}")
