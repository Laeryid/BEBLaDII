"""Общие математические примитивы и архитектурные слои Plan C (Burer-Monteiro Gr(8, 128))."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LorentzianEigh(torch.autograd.Function):
    """Спектральное разложение симметричной матрицы с лоренцевой регуляризацией градиентов."""

    @staticmethod
    def forward(ctx, G: torch.Tensor, eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = G.dtype
        G_f32 = 0.5 * (G.float() + G.float().transpose(-2, -1))
        K = G_f32.size(-1)

        diag_shift = torch.arange(1, K + 1, device=G_f32.device, dtype=torch.float32) * 1e-6
        G_reg = G_f32 + torch.diag(diag_shift)

        L_reg: Optional[torch.Tensor] = None
        V: Optional[torch.Tensor] = None
        try:
            L_reg, V = torch.linalg.eigh(G_reg)
            is_valid = torch.isfinite(L_reg).all() and torch.isfinite(V).all()
        except Exception:
            is_valid = False

        if not is_valid:
            try:
                L_64, V_64 = torch.linalg.eigh(G_reg.double())
                L_reg = L_64.float()
                V = V_64.float()
            except Exception:
                L_reg = torch.full(G_reg.shape[:-1], 1e-5, device=G_reg.device, dtype=torch.float32)
                V = torch.eye(K, device=G_reg.device, dtype=torch.float32).expand_as(G_reg).clone()
            L_reg = torch.nan_to_num(L_reg, nan=1e-5, posinf=1.0, neginf=1e-5)
            V = torch.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)

        ctx.save_for_backward(L_reg, V)
        ctx.eps = eps
        ctx.orig_dtype = orig_dtype

        L_clean = torch.clamp(L_reg - diag_shift, min=1e-8)
        return L_clean.to(orig_dtype), V.to(orig_dtype)

    @staticmethod
    def backward(ctx, grad_L: torch.Tensor, grad_V: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        L_reg, V = ctx.saved_tensors
        eps = ctx.eps

        diff = L_reg.unsqueeze(-2) - L_reg.unsqueeze(-1)
        lorentz = diff / (diff ** 2 + eps ** 2)
        lorentz.diagonal(dim1=-2, dim2=-1).zero_()

        vt_gv = torch.matmul(V.transpose(-2, -1), grad_V.float())
        vt_gv_skew = 0.5 * (vt_gv - vt_gv.transpose(-2, -1))
        mid = torch.diag_embed(grad_L.float()) + lorentz * vt_gv_skew

        grad_G = torch.matmul(torch.matmul(V, mid), V.transpose(-2, -1))
        grad_G = 0.5 * (grad_G + grad_G.transpose(-2, -1))
        grad_G = torch.nan_to_num(grad_G, nan=0.0, posinf=0.0, neginf=0.0)
        return grad_G.to(ctx.orig_dtype), None


def safe_lorentzian_eigh(G: torch.Tensor, eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    """Безопасное спектральное разложение с лоренцевым ядром в backward."""
    return LorentzianEigh.apply(G, eps)


def apply_rope_16d(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Partial RoPE для последних 16 координат (8 пар) по гармонической сетке [pi/8, pi/2]."""
    thetas = torch.linspace(math.pi / 8.0, math.pi / 2.0, steps=8, device=x.device, dtype=x.dtype)
    angles = positions.unsqueeze(-1).to(x.dtype) * thetas.unsqueeze(0)
    sin = torch.sin(angles).unsqueeze(-2)
    cos = torch.cos(angles).unsqueeze(-2)

    shape_prefix = x.shape[:-1]
    x_pairs = x.view(*shape_prefix, 8, 2)
    x0 = x_pairs[..., 0]
    x1 = x_pairs[..., 1]

    x0_rot = x0 * cos - x1 * sin
    x1_rot = x0 * sin + x1 * cos
    return torch.stack([x0_rot, x1_rot], dim=-1).view(*shape_prefix, 16)


def clamp_to_unit_ball(V: torch.Tensor) -> torch.Tensor:
    """Радиальная проекция строк в единичный шар B^128 без сжатия векторов с нормой <= 1."""
    norms = torch.norm(V, p=2, dim=-1, keepdim=True)
    return V / norms.clamp(min=1.0)


class RMSNorm(nn.Module):
    """RMSNorm с расчетом в float32 и защитой от нечисловых переполнений."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = torch.nan_to_num(x.float(), nan=0.0, posinf=65000.0, neginf=-65000.0)
        var = torch.mean(x_f ** 2, dim=-1, keepdim=True)
        normed = x_f * torch.rsqrt(var + self.eps)
        return (normed * self.weight.float()).to(x.dtype)


class SphereLinear(nn.Module):
    """Линейный слой со сферической параметризацией строк для OrthMuon (ADR 131): W = diag(row_scale) @ q_weight."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        w_init = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(w_init, a=math.sqrt(5))
        row_norms = w_init.norm(p=2, dim=-1).clamp(min=1e-7)

        self.q_weight = nn.Parameter(w_init / row_norms.unsqueeze(-1))
        self.row_scale = nn.Parameter(row_norms)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def merged_weight(self) -> torch.Tensor:
        """Собирает эффективную плотную матрицу весов W = diag(row_scale) @ q_weight."""
        return self.q_weight * self.row_scale.unsqueeze(-1).to(self.q_weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.merged_weight(), self.bias)


class RotaryEmbedding128(nn.Module):
    """RoPE для ModernBERT attention (head_dim=128)."""

    def __init__(self, dim: int = 128, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len].to(x.dtype), self.sin_cached[:seq_len].to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class ModernBERTLayer(nn.Module):
    """Слой трансформера ModernBERT с Pre-RMSNorm, per-head QK-norm, RoPE и SwiGLU на SphereLinear."""

    def __init__(self, hidden_size: int = 1024, intermediate_size: int = 2048, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.attn_norm = RMSNorm(hidden_size)
        self.q_proj = SphereLinear(hidden_size, hidden_size, bias=False)
        self.k_proj = SphereLinear(hidden_size, hidden_size, bias=False)
        self.v_proj = SphereLinear(hidden_size, hidden_size, bias=False)
        self.o_proj = SphereLinear(hidden_size, hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.mlp_norm = RMSNorm(hidden_size)
        self.gate_proj = SphereLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = SphereLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = SphereLinear(intermediate_size, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        normed_x = self.attn_norm(x)
        q = self.q_norm(self.q_proj(normed_x).view(B, T, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(normed_x).view(B, T, self.num_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(normed_x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.o_proj(attn_out)

        normed_mlp = self.mlp_norm(x)
        gate = F.silu(self.gate_proj(normed_mlp).float())
        up = self.up_proj(normed_mlp).float()
        gate_up = (gate * up).clamp(-65000.0, 65000.0).to(x.dtype)
        x = x + self.down_proj(gate_up)
        return x


__all__ = [
    "LorentzianEigh",
    "safe_lorentzian_eigh",
    "apply_rope_16d",
    "clamp_to_unit_ball",
    "RMSNorm",
    "SphereLinear",
    "RotaryEmbedding128",
    "rotate_half",
    "apply_rotary_pos_emb",
    "ModernBERTLayer",
]
