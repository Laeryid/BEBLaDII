import torch

def safe_cosine_similarity(x1, x2, dim=-1, eps=1e-6):
    norm1 = torch.sqrt(torch.sum(x1 ** 2, dim=dim, keepdim=True) + eps)
    norm2 = torch.sqrt(torch.sum(x2 ** 2, dim=dim, keepdim=True) + eps)
    return torch.sum(x1 * x2, dim=dim) / (norm1 * norm2)

ds = torch.zeros(1, 1024, requires_grad=True)
dt = torch.randn(1, 1024)

cos_sim = safe_cosine_similarity(ds, dt, dim=-1, eps=1e-6)
loss = cos_sim.sum()
loss.backward()

print("Safe F.cosine_similarity (eps=1e-6) Grad Max/Min:", ds.grad.max().item(), ds.grad.min().item())
