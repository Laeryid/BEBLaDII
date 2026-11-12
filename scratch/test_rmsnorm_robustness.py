import torch
import torch.nn as nn

class OldRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x.float() ** 2, dim=-1, keepdim=True)
        normed = x * torch.rsqrt(var + self.eps).to(x.dtype)
        return normed * self.weight

class RobustRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Полный расчет нормализации в float32 с защитой от inf
        x_f = torch.nan_to_num(x.float(), nan=0.0, posinf=65000.0, neginf=-65000.0)
        var = torch.mean(x_f ** 2, dim=-1, keepdim=True)
        normed = x_f * torch.rsqrt(var + self.eps)
        return (normed * self.weight.float()).to(x.dtype)

def test_norm():
    dim = 1024
    old_norm = OldRMSNorm(dim)
    robust_norm = RobustRMSNorm(dim)

    # Вход с переполнением FP16 через .half()
    x_fp16 = (torch.randn(2, 10, dim) * 100000.0).half()
    print("Вход x_fp16 contains inf:", torch.isinf(x_fp16).any().item())

    out_old = old_norm(x_fp16)
    print("OldRMSNorm output finite:", torch.isfinite(out_old).all().item(), "min/max:", out_old.min().item(), out_old.max().item())

    out_robust = robust_norm(x_fp16)
    print("RobustRMSNorm output finite:", torch.isfinite(out_robust).all().item(), "min/max:", out_robust.min().item(), out_robust.max().item())

if __name__ == "__main__":
    test_norm()
