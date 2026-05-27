import torch

def rkd_buggy(s_centered):
    s_normed = s_centered / s_centered.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return s_normed

def rkd_safe(s_centered):
    s_norm = torch.sqrt(torch.sum(s_centered ** 2, dim=-1, keepdim=True) + 1e-8)
    s_normed = s_centered / s_norm
    return s_normed

# Создаем тензор с очень маленькими, но ненулевыми значениями
s_centered_1 = torch.ones(1, 4, 1024) * 1e-7
s_centered_1.requires_grad_(True)

s_centered_2 = torch.ones(1, 4, 1024) * 1e-7
s_centered_2.requires_grad_(True)

# 1. Buggy version
out_buggy = rkd_buggy(s_centered_1)
loss_bug = out_buggy.sum()
loss_bug.backward()
print("Buggy Grad has NaN:", torch.isnan(s_centered_1.grad).any().item())
print("Buggy Grad Max/Min:", s_centered_1.grad.max().item(), s_centered_1.grad.min().item())

# 2. Safe version
out_safe = rkd_safe(s_centered_2)
loss_safe = out_safe.sum()
loss_safe.backward()
print("Safe Grad has NaN:", torch.isnan(s_centered_2.grad).any().item())
print("Safe Grad Max/Min:", s_centered_2.grad.max().item(), s_centered_2.grad.min().item())
