import torch
import torch.nn.functional as F

def safe_normalize(x, dim=-1, eps=1e-8):
    norm = torch.norm(x, dim=dim, keepdim=True)
    return x / (norm + eps)

torch.manual_seed(42)
# Simulate a random latent vector (like from LatentEncoder)
z_clean = torch.randn(1, 512, 1024, dtype=torch.float32)
z_clean = safe_normalize(z_clean, dim=-1)

# Now convert to bfloat16 as done in training
z_clean_bf16 = z_clean.to(torch.bfloat16)

# The loss function logic:
z_clean_normed = safe_normalize(z_clean_bf16, dim=-1)
z_noisy_normed = safe_normalize(z_clean_bf16, dim=-1) # for unnoised tokens, z_noisy = z_clean

c_true = torch.clamp((z_clean_normed * z_noisy_normed).sum(dim=-1), min=0.0)

print(f"Mean c_true (bfloat16): {c_true.mean().item():.5f}")
print(f"Min c_true (bfloat16): {c_true.min().item():.5f}")
print(f"Max c_true (bfloat16): {c_true.max().item():.5f}")

# And what happens with gamma=20?
gamma = 20.0
w_c = torch.pow(c_true.float(), gamma)

print(f"Mean w_c (penalty multiplier): {w_c.mean().item():.5f}")
print(f"Min w_c: {w_c.min().item():.5f}")
