import torch

ckpt_path = r"C:\Experiments\BEBLaDII\experiments\phase3\planB_phase3_checkpoints_phase3_step_17995.pth"
print(f"Loading checkpoint {ckpt_path}...")
ckpt = torch.load(ckpt_path, map_location="cpu")

adaln = ckpt.get("adaLN", {})
if not adaln:
    print("No adaLN found in checkpoint!")
    exit()

print("AdaLN weights found.")
# adaLN is a ModuleList of AdaLNModulation
# Each has a .modulation (Sequential: SiLU, Linear)
# So keys look like:
# '0.modulation.1.weight', '0.modulation.1.bias'
# The bias was initialized to [zeros(D), ones(D)]

D = 1024
num_layers = 40 # Assuming ModernBERT-large or so, let's find out from keys
max_layer = -1
for k in adaln.keys():
    parts = k.split('.')
    if parts[0].isdigit():
        max_layer = max(max_layer, int(parts[0]))

print(f"Detected {max_layer + 1} layers.")

# Let's compute average shift and scale bias
mean_shift_bias = 0
mean_scale_bias = 0
for i in range(max_layer + 1):
    bias = adaln.get(f"{i}.modulation.1.bias")
    if bias is not None:
        shift_b = bias[:D]
        scale_b = bias[D:]
        
        # print specific layer biases
        if i % 10 == 0 or i == max_layer:
            print(f"Layer {i:02d} | shift_bias mean: {shift_b.mean().item():.4f}, std: {shift_b.std().item():.4f} | scale_bias mean: {scale_b.mean().item():.4f}, std: {scale_b.std().item():.4f} (init=1.0)")
            
        mean_shift_bias += shift_b.mean().item()
        mean_scale_bias += scale_b.mean().item()

print(f"\nOverall Avg Shift Bias: {mean_shift_bias / (max_layer + 1):.4f} (init=0.0)")
print(f"Overall Avg Scale Bias: {mean_scale_bias / (max_layer + 1):.4f} (init=1.0)")

# Also let's check t_emb weights magnitude to see if t_emb heavily influences this
# weight shape is [2*D, t_emb_dim]
mean_scale_weight_norm = 0
for i in range(max_layer + 1):
    w = adaln.get(f"{i}.modulation.1.weight")
    if w is not None:
        scale_w = w[D:, :]
        mean_scale_weight_norm += scale_w.norm().item()
print(f"Overall Avg Scale Weight Norm: {mean_scale_weight_norm / (max_layer + 1):.4f}")

