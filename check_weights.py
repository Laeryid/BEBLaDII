import torch

ckpt_path = r"C:\Experiments\BEBLaDII\experiments\phase 6\checkpoints\planB_phase6_checkpoints_phase6_ca_layers_step_3000.pth"
ckpt = torch.load(ckpt_path, map_location='cpu')
all_nonzero = True
total_zeros = 0
total_params = 0

print("Checking checkpoint for zero weights...")
for k, v in ckpt.items():
    if isinstance(v, torch.Tensor):
        zeros = (v == 0).sum().item()
        total_zeros += zeros
        total_params += v.numel()
        if zeros > 0:
            print(f"Layer {k} has {zeros}/{v.numel()} zero weights.")
            all_nonzero = False

if all_nonzero:
    print(f"All {total_params} weights in the checkpoint are non-zero.")
else:
    print(f"Found {total_zeros} zero weights out of {total_params} total weights.")
