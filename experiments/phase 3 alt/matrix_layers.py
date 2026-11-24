"""
Геометрически-согласованные матричные слои для Phase 3 Alt Plan C (раздел 10 ARCHITECTURE_AND_IDEAS.md):

  1. `PHMLinear` и `PHMFullSABlock`:
     - Заменяет плотные проекции `1024 x 1024` (и опционально `SwiGLU FFN`) в `FullSABlock` на
       факторизованную сумму Кронекера `Y = sum_{m=1..r} A_m @ X @ F_m^T` без материализации
       полной `1024 x 1024` матрицы (всего 2 плотных GEMM, полная совместимость с `SphereLinear`,
       `OrthMuon` и `EMA`).
     - Нативный почти блочно-диагональный старт (`A[:, :, 0] = sqrt(1 - alpha^2) * I_8`).
  2. `GrassAlignSABlock`:
     - Заменяет тяжёлое `(4, 1024, 1024)` внимание в `GrassFull` на точный Грассманиановый скор
       `s_ij = ||Q_i K_j^T||_F^2 / 8` с компактным `N x N (128 x 128)` softmax и нормированным
       Procrustes-роутером `M_tilde_ij` (`8 x 8`), выравнивающим базис ячейки j к ячейке i.
  3. `BilinearGrassSABlock` (шаг `L * X * R` из Monarch):
     - Добавляет в `GrassmannianSABlock` (и `GrassAlignSABlock`) обучаемые матрицы левого
       кросс-осевого смешивания `axis_mix_attn (8 x 8)` и `axis_mix_mlp (8 x 8)` (`init = I_8`,
       128 параметров на слой, ~0 FLOPs).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from storage.components.plan_c_layers import (
    RMSNorm,
    SphereLinear,
    apply_rotary_pos_emb,
)
from train_phase3_alt import (
    AXIS_DIM,
    FullSABlock,
    GrassmannianAltDiffusionCore,
    GrassmannianSABlock,
    HIDDEN_SIZE,
    N_CANVAS,
    NUM_AXES,
    OrthMuon,
    PerAxisGRNv3,
    PerHeadRMSNorm,
    _apply_cell_rope_to_vectors,
    _clamp_fp16_safe,
    _clamp_swiglu,
    _compute_adaln_stats,
    _init_soft_gate_adaln,
    _repeat_adaln_stats_to_4096,
)

MATRIX_VARIANTS: Tuple[str, ...] = (
    "ref_dense",
    "opt1_phm_attn",
    "opt1_phm_full",
    "opt2_grass_align",
    "opt3_bilinear_lxr",
    "all_matrix_safe",
    "all_matrix_ultra",
)


# =============================================================================
# 1. Факторизованный PHM-слой (Parameterized Hypercomplex Multiplication)
# =============================================================================
class PHMLinear(nn.Module):
    """
    Факторизованный 2D-линейный слой над ячейкой X in R^{K_in x D_in}:
        Y = sum_{m=1..r} A_m @ X @ F_m^T  in R^{K_out x D_out}

    Вычисляется ровно за 2 плотных GEMM без материализации плотной (K_out*D_out, K_in*D_in):
      1. `f_proj`: SphereLinear(D_in, rank * D_out) -> (B, N, K_in, rank * D_out)
      2. `axis_mix` (K_out, K_in * rank) @ H (B*N, K_in * rank, D_out) -> (B, N, K_out, D_out)

    Полная совместимость с OrthMuon/EMA (`f_proj.q_weight` имеет строго единичные строки).
    """

    def __init__(
        self,
        in_axes: int = NUM_AXES,
        in_dim: int = AXIS_DIM,
        out_axes: int = NUM_AXES,
        out_dim: int = AXIS_DIM,
        rank: int = 4,
        cross_alpha: float = 0.25,
    ):
        super().__init__()
        self.in_axes = int(in_axes)
        self.in_dim = int(in_dim)
        self.out_axes = int(out_axes)
        self.out_dim = int(out_dim)
        self.rank = int(rank)
        self.in_features = self.in_axes * self.in_dim
        self.out_features = self.out_axes * self.out_dim

        self.f_proj = SphereLinear(self.in_dim, self.rank * self.out_dim, bias=False)
        # Матрица осевого смешивания A: (K_out, K_in * rank), нормируется так, чтобы
        # L2-норма каждой выходной строки была ~1.0 (сохраняя Kaiming-дисперсию SphereLinear).
        self.axis_mix = nn.Parameter(torch.empty(self.out_axes, self.in_axes * self.rank))
        self._init_axis_mix(cross_alpha=cross_alpha)

    @torch.no_grad()
    def _init_axis_mix(self, cross_alpha: float = 0.25) -> None:
        A = torch.zeros(self.out_axes, self.in_axes, self.rank, dtype=torch.float32)
        if self.rank == 1 or cross_alpha <= 0.0:
            if self.out_axes == self.in_axes:
                A[:, :, 0] = torch.eye(self.out_axes, dtype=torch.float32)
            else:
                nn.init.orthogonal_(A[:, :, 0])
        else:
            main_w = math.sqrt(max(0.0, 1.0 - cross_alpha ** 2))
            if self.out_axes == self.in_axes:
                A[:, :, 0] = main_w * torch.eye(self.out_axes, dtype=torch.float32)
            else:
                nn.init.orthogonal_(A[:, :, 0])
                A[:, :, 0].mul_(main_w)
            noise = torch.randn(self.out_axes, self.in_axes, self.rank - 1, dtype=torch.float32)
            noise_norm = noise.flatten(1).norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6).view(self.out_axes, 1, 1)
            A[:, :, 1:] = (noise / noise_norm) * cross_alpha
        self.axis_mix.copy_(A.reshape(self.out_axes, self.in_axes * self.rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Принимает x формы (B, N, K_in, D_in) или (B, N, K_in * D_in);
        возвращает тензор той же ранга (4D или 3D).
        """
        was_3d = x.ndim == 3
        if was_3d:
            B, N, _ = x.shape
            x4 = x.view(B, N, self.in_axes, self.in_dim)
        else:
            B, N, _, _ = x.shape
            x4 = x

        # 1. Правый GEMM: (B, N, K_in, D_in) -> (B, N, K_in, rank * D_out)
        h = self.f_proj(x4)
        # Преобразуем в (B * N, K_in * rank, D_out) для одного батчевого левого GEMM
        h_mat = h.reshape(B * N, self.in_axes * self.rank, self.out_dim)
        # 2. Левый GEMM: (K_out, K_in * rank) @ (B*N, K_in * rank, D_out) -> (B*N, K_out, D_out)
        a_mat = self.axis_mix.to(dtype=h_mat.dtype)
        y4 = torch.matmul(a_mat, h_mat).view(B, N, self.out_axes, self.out_dim)
        if was_3d:
            return y4.reshape(B, N, self.out_features)
        return y4


class PHMFullSABlock(nn.Module):
    """
    Drop-in замена `FullSABlock` с PHM-проекциями на Q/KV/O (и опционально на SwiGLU FFN).
    Сохраняет полную совместимость по входу/выходу (x_in, h_ca, x_prev, c, cos_full, sin_full, m_active),
    `PerAxisGRNv3` DCA и `Soft-Gate AdaLN`.
    """

    def __init__(
        self,
        hidden_size: int = HIDDEN_SIZE,
        intermediate_size: int = 2048,
        num_heads: int = 8,
        has_ca: bool = True,
        cond_dim: int = 256,
        phm_attn: bool = True,
        phm_ffn: bool = False,
        phm_rank: int = 4,
        cross_alpha: float = 0.25,
    ):
        super().__init__()
        if intermediate_size % NUM_AXES != 0:
            raise ValueError(f"intermediate_size={intermediate_size} должен делиться на NUM_AXES={NUM_AXES}")
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.phm_attn = phm_attn
        self.phm_ffn = phm_ffn
        self.phm_rank = phm_rank

        self.dca_q = PerAxisGRNv3(NUM_AXES, AXIS_DIM, has_ca=has_ca)
        self.dca_kv = PerAxisGRNv3(NUM_AXES, AXIS_DIM, has_ca=has_ca)

        self.attn_norm_q = RMSNorm(hidden_size)
        self.attn_norm_kv = RMSNorm(hidden_size)

        if phm_attn:
            self.q_proj: nn.Module = PHMLinear(NUM_AXES, AXIS_DIM, NUM_AXES, AXIS_DIM, rank=phm_rank, cross_alpha=cross_alpha)
            self.kv_proj: nn.Module = PHMLinear(NUM_AXES, AXIS_DIM, NUM_AXES, AXIS_DIM, rank=phm_rank, cross_alpha=cross_alpha)
            self.o_proj: nn.Module = PHMLinear(NUM_AXES, AXIS_DIM, NUM_AXES, AXIS_DIM, rank=phm_rank, cross_alpha=cross_alpha)
        else:
            self.q_proj = SphereLinear(hidden_size, hidden_size, bias=False)
            self.kv_proj = SphereLinear(hidden_size, hidden_size, bias=False)
            self.o_proj = SphereLinear(hidden_size, hidden_size, bias=False)

        self.q_norm = PerHeadRMSNorm(num_heads, self.head_dim)
        self.k_norm = PerHeadRMSNorm(num_heads, self.head_dim)
        self.v_gate = nn.Parameter(torch.ones(1, num_heads, 1, self.head_dim))

        self.mlp_norm = RMSNorm(hidden_size)
        ffn_axis_dim = intermediate_size // NUM_AXES
        if phm_ffn:
            self.gate_proj: nn.Module = PHMLinear(NUM_AXES, AXIS_DIM, NUM_AXES, ffn_axis_dim, rank=phm_rank, cross_alpha=cross_alpha)
            self.up_proj: nn.Module = PHMLinear(NUM_AXES, AXIS_DIM, NUM_AXES, ffn_axis_dim, rank=phm_rank, cross_alpha=cross_alpha)
            self.down_proj: nn.Module = PHMLinear(NUM_AXES, ffn_axis_dim, NUM_AXES, AXIS_DIM, rank=phm_rank, cross_alpha=cross_alpha)
        else:
            self.gate_proj = SphereLinear(hidden_size, intermediate_size, bias=False)
            self.up_proj = SphereLinear(hidden_size, intermediate_size, bias=False)
            self.down_proj = SphereLinear(intermediate_size, hidden_size, bias=False)

        self.adaln_proj = nn.Linear(cond_dim, 6 * hidden_size)
        _init_soft_gate_adaln(self.adaln_proj, hidden_size, gate_bias=0.02)

    def forward(
        self,
        x_in: torch.Tensor,
        h_ca: Optional[torch.Tensor],
        x_prev: torch.Tensor,
        c: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        m_active: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        B, N, K, D = x_prev.shape
        C = K * D

        mods = self.adaln_proj(c.to(self.adaln_proj.weight.dtype)).to(x_prev.dtype)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods.chunk(6, dim=-1)
        M = torch.cat([shift_msa, scale_msa, shift_mlp, scale_mlp], dim=-1)
        stats = _compute_adaln_stats(M, m_active)

        x_q_flat = self.dca_q(x_in, h_ca, x_prev).reshape(B, N, C)
        x_kv_flat = self.dca_kv(x_in, h_ca, x_prev).reshape(B, N, C)
        x_flat = x_prev.reshape(B, N, C)

        h_q = self.attn_norm_q(x_q_flat) * (1.0 + scale_msa) + shift_msa
        h_kv = self.attn_norm_kv(x_kv_flat) * (1.0 + scale_msa) + shift_msa

        q = self.q_proj(h_q).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        u = self.kv_proj(h_kv).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(u)
        v = u * self.v_gate.to(u.dtype)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        x_flat = x_flat + gate_msa * self.o_proj(attn_out)

        h_mlp = self.mlp_norm(x_flat) * (1.0 + scale_mlp) + shift_mlp
        gated = _clamp_swiglu(self.gate_proj(h_mlp), self.up_proj(h_mlp))
        mlp_out = _clamp_fp16_safe(self.down_proj(gated))
        x_flat = x_flat + gate_mlp * mlp_out
        return x_flat.view(B, N, K, D), stats


# =============================================================================
# 2. GrassAlignSABlock: Грассманиановый скор ||Q_i K_j^T||_F^2 + Procrustes-роутер
# =============================================================================
class GrassAlignSABlock(GrassmannianSABlock):
    """
    Drop-in замена глобального `GrassFull` (`GrassmannianSABlock(local_radius=None)`):
    - Вместо 4 подголов с матрицей внимания `(4, 1024, 1024)` проецирует Q и K в единое
      32-мерное пространство (`128 -> 32`, совместимое с `cos_grass/sin_grass` размерности 32).
    - Считает общую матрицу попарных косинусов `M_ij = (Q_i K_j^T) / sqrt(32)` размера `(B, N, N, 8, 8)`.
    - Точный инвариант на Грассманиане (без SVD): `s_ij = temp * ||M_ij||_F^2 / 8` -> `softmax`
      только по `N x N (128 x 128)` (в 256x меньше softmax-элементов!).
    - Строчно-нормированная `M_tilde_ij` выступает Procrustes-выравнивателем базиса (со знаком):
      поворачивает `V_j (8 x 128)` в систему координат ячейки `i` перед взвешиванием на `alpha_ij`.
    - Наследует от `GrassmannianSABlock` (без создания лишних весов), чтобы бесшовно работать
      в `GrassmannianAltDiffusionCore.forward_with_cached_prompt` и `idea5_kmeans_ffn`.
    """

    def __init__(
        self,
        axis_dim: int = AXIS_DIM,
        ffn_dim: int = 512,
        qk_dim: int = 32,
        has_ca: bool = True,
        cond_dim: int = 256,
        use_bilinear_lxr: bool = False,
    ):
        nn.Module.__init__(self)
        self.axis_dim = axis_dim
        self.qk_dim = qk_dim
        self.num_sub_heads = 1
        self.sub_head_dim = qk_dim
        self.local_radius = None
        self.use_bilinear_lxr = bool(use_bilinear_lxr)

        self.dca_q = PerAxisGRNv3(NUM_AXES, axis_dim, has_ca=has_ca)
        self.dca_kv = PerAxisGRNv3(NUM_AXES, axis_dim, has_ca=has_ca)

        self.attn_norm_q = RMSNorm(axis_dim)
        self.attn_norm_kv = RMSNorm(axis_dim)
        self.norm_emb_q = nn.Linear(1, axis_dim)
        self.norm_emb_kv = nn.Linear(1, axis_dim)
        nn.init.zeros_(self.norm_emb_q.weight)
        nn.init.zeros_(self.norm_emb_q.bias)
        nn.init.zeros_(self.norm_emb_kv.weight)
        nn.init.zeros_(self.norm_emb_kv.bias)

        # Компактные проекции для Q/K (128 -> 32) и полные V/O (128 -> 128)
        self.q_align_proj = SphereLinear(axis_dim, qk_dim, bias=False)
        self.k_align_proj = SphereLinear(axis_dim, qk_dim, bias=False)
        self.v_proj = SphereLinear(axis_dim, axis_dim, bias=False)
        self.o_proj = SphereLinear(axis_dim, axis_dim, bias=False)

        self.q_norm = PerHeadRMSNorm(1, qk_dim)
        self.k_norm = PerHeadRMSNorm(1, qk_dim)
        self.v_gate = nn.Parameter(torch.ones(1, 1, 1, axis_dim))
        self.score_temp = nn.Parameter(torch.tensor(1.0))

        self.mlp_norm = RMSNorm(axis_dim)
        self.gate_proj = SphereLinear(axis_dim, ffn_dim, bias=False)
        self.up_proj = SphereLinear(axis_dim, ffn_dim, bias=False)
        self.down_proj = SphereLinear(ffn_dim, axis_dim, bias=False)

        if self.use_bilinear_lxr:
            self.axis_mix_attn = nn.Parameter(torch.eye(NUM_AXES))
            self.axis_mix_mlp = nn.Parameter(torch.eye(NUM_AXES))

        self.adaln_proj = nn.Linear(cond_dim, 6 * axis_dim)
        _init_soft_gate_adaln(self.adaln_proj, axis_dim, gate_bias=0.02)
        self.register_buffer("win_mask_default", None, persistent=False)

    def forward(
        self,
        x_in: torch.Tensor,
        h_ca: Optional[torch.Tensor],
        x_prev: torch.Tensor,
        c: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        m_active: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        B, N, K, D = x_prev.shape

        mods = self.adaln_proj(c.to(self.adaln_proj.weight.dtype)).to(x_prev.dtype)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods.chunk(6, dim=-1)
        M_mod = torch.cat([shift_msa, scale_msa, shift_mlp, scale_mlp], dim=-1)
        stats = _repeat_adaln_stats_to_4096(_compute_adaln_stats(M_mod, m_active))

        s_msa = shift_msa.unsqueeze(2)
        sc_msa = scale_msa.unsqueeze(2)
        g_msa = gate_msa.unsqueeze(2)
        s_mlp = shift_mlp.unsqueeze(2)
        sc_mlp = scale_mlp.unsqueeze(2)
        g_mlp = gate_mlp.unsqueeze(2)

        x_q = self.dca_q(x_in, h_ca, x_prev)
        x_kv = self.dca_kv(x_in, h_ca, x_prev)

        ln_q = torch.log1p(x_q.float().norm(p=2, dim=-1, keepdim=True)).to(x_q.dtype)
        ln_kv = torch.log1p(x_kv.float().norm(p=2, dim=-1, keepdim=True)).to(x_kv.dtype)
        h_q = (self.attn_norm_q(x_q) + self.norm_emb_q(ln_q)) * (1.0 + sc_msa) + s_msa
        h_kv = (self.attn_norm_kv(x_kv) + self.norm_emb_kv(ln_kv)) * (1.0 + sc_msa) + s_msa

        # Q, K в R^32 с ячеечным RoPE: (B, 1, N, K, 32)
        q = self.q_align_proj(h_q).unsqueeze(1)
        k = self.k_align_proj(h_kv).unsqueeze(1)
        q = self.q_norm(q.reshape(B, 1, N * K, self.qk_dim)).view(B, 1, N, K, self.qk_dim)
        k = self.k_norm(k.reshape(B, 1, N * K, self.qk_dim)).view(B, 1, N, K, self.qk_dim)
        q = _apply_cell_rope_to_vectors(q, cos, sin).reshape(B, N * K, self.qk_dim)
        k = _apply_cell_rope_to_vectors(k, cos, sin).reshape(B, N * K, self.qk_dim)

        # V в R^128 без RoPE: (B, N*K, 128)
        v = (self.v_proj(h_kv) * self.v_gate.to(h_kv.dtype)).reshape(B, N * K, D)

        # 1. Один компактный matmul (B, N*K, 32) @ (B, 32, N*K) -> M: (B, N, K, N, K)
        #    Переупорядочиваем в (B, N, N, K, K) только для подсчёта ||M_ij||_F^2 и строчной нормы
        m_raw = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(self.qk_dim)
        m_blocks = m_raw.view(B, N, K, N, K)

        # 2. Точный Грассманиановый скор ||M_ij||_F^2 -> softmax по (B, N, N) (128 x 128!)
        temp = self.score_temp.float().clamp(min=0.05, max=20.0)
        cell_scores = temp * m_blocks.pow(2).sum(dim=(2, 4)) / float(K)  # (B, N, N)
        cell_probs = F.softmax(cell_scores, dim=-1)  # (B, N, N)

        # 3. Строчно-нормированный Procrustes-роутер M_tilde_ij (сохраняет знак и норму <= 1)
        row_norm = m_blocks.norm(p=2, dim=4, keepdim=True).clamp(min=1e-4)
        m_router = m_blocks / row_norm  # (B, N, K, N, K)

        # Комбинированный вес W[(i,a), (j,b)] = alpha_ij * M_tilde_ij[a, b]
        w_full = (m_router * cell_probs.view(B, N, 1, N, 1)).reshape(B, N * K, N * K).to(v.dtype)
        attn_out = torch.matmul(w_full, v).view(B, N, K, D)

        delta_attn = self.o_proj(attn_out)
        if self.use_bilinear_lxr:
            delta_attn = torch.matmul(self.axis_mix_attn.to(delta_attn.dtype), delta_attn)
        x = x_prev + g_msa * delta_attn

        h_mlp = self.mlp_norm(x) * (1.0 + sc_mlp) + s_mlp
        gated = _clamp_swiglu(self.gate_proj(h_mlp), self.up_proj(h_mlp))
        mlp_out = _clamp_fp16_safe(self.down_proj(gated))
        if self.use_bilinear_lxr:
            mlp_out = torch.matmul(self.axis_mix_mlp.to(mlp_out.dtype), mlp_out)
        x = x + g_mlp * mlp_out
        return x, stats


# =============================================================================
# 3. Билинейный шаг L * X * R (Monarch-миксер осей 8x8) внутри GrassmannianSABlock
# =============================================================================
class BilinearGrassSABlock(GrassmannianSABlock):
    """
    Расширение `GrassmannianSABlock` с левыми матрицами кросс-осевого смешивания `8 x 8`
    (`axis_mix_attn` и `axis_mix_mlp`, старт строго `I_8`, всего 128 параметров на слой).
    """

    def __init__(
        self,
        axis_dim: int = AXIS_DIM,
        ffn_dim: int = 512,
        num_sub_heads: int = 4,
        local_radius: Optional[int] = 6,
        has_ca: bool = True,
        cond_dim: int = 256,
    ):
        super().__init__(
            axis_dim=axis_dim,
            ffn_dim=ffn_dim,
            num_sub_heads=num_sub_heads,
            local_radius=local_radius,
            has_ca=has_ca,
            cond_dim=cond_dim,
        )
        self.axis_mix_attn = nn.Parameter(torch.eye(NUM_AXES))
        self.axis_mix_mlp = nn.Parameter(torch.eye(NUM_AXES))

    def forward(
        self,
        x_in: torch.Tensor,
        h_ca: Optional[torch.Tensor],
        x_prev: torch.Tensor,
        c: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        m_active: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        B, N, K, D = x_prev.shape
        H, dh = self.num_sub_heads, self.sub_head_dim

        mods = self.adaln_proj(c.to(self.adaln_proj.weight.dtype)).to(x_prev.dtype)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods.chunk(6, dim=-1)
        M = torch.cat([shift_msa, scale_msa, shift_mlp, scale_mlp], dim=-1)
        stats = _repeat_adaln_stats_to_4096(_compute_adaln_stats(M, m_active))

        s_msa = shift_msa.unsqueeze(2)
        sc_msa = scale_msa.unsqueeze(2)
        g_msa = gate_msa.unsqueeze(2)
        s_mlp = shift_mlp.unsqueeze(2)
        sc_mlp = scale_mlp.unsqueeze(2)
        g_mlp = gate_mlp.unsqueeze(2)

        x_q = self.dca_q(x_in, h_ca, x_prev)
        x_kv = self.dca_kv(x_in, h_ca, x_prev)

        ln_q = torch.log1p(x_q.float().norm(p=2, dim=-1, keepdim=True)).to(x_q.dtype)
        ln_kv = torch.log1p(x_kv.float().norm(p=2, dim=-1, keepdim=True)).to(x_kv.dtype)
        h_q = (self.attn_norm_q(x_q) + self.norm_emb_q(ln_q)) * (1.0 + sc_msa) + s_msa
        h_kv = (self.attn_norm_kv(x_kv) + self.norm_emb_kv(ln_kv)) * (1.0 + sc_msa) + s_msa

        q = self.q_proj(h_q).view(B, N, K, H, dh).permute(0, 3, 1, 2, 4)
        u = self.kv_proj(h_kv).view(B, N, K, H, dh).permute(0, 3, 1, 2, 4)

        q = self.q_norm(q.reshape(B, H, N * K, dh)).view(B, H, N, K, dh)
        k = self.k_norm(u.reshape(B, H, N * K, dh)).view(B, H, N, K, dh)
        v_flat = u.reshape(B, H, N * K, dh) * self.v_gate.to(u.dtype)

        q = _apply_cell_rope_to_vectors(q, cos, sin)
        k = _apply_cell_rope_to_vectors(k, cos, sin)

        if self.local_radius is not None:
            v = v_flat.view(B, H, N, K, dh)
            attn_out = self._local_window_attention(q, k, v)
        else:
            q_f = q.reshape(B, H, N * K, dh)
            k_f = k.reshape(B, H, N * K, dh)
            attn_out = F.scaled_dot_product_attention(
                q_f, k_f, v_flat, attn_mask=None, dropout_p=0.0, is_causal=False
            ).view(B, H, N, K, dh)

        attn_out = attn_out.permute(0, 2, 3, 1, 4).reshape(B, N, K, D)
        delta_attn = torch.matmul(self.axis_mix_attn.to(attn_out.dtype), self.o_proj(attn_out))
        x = x_prev + g_msa * delta_attn

        h_mlp = self.mlp_norm(x) * (1.0 + sc_mlp) + s_mlp
        gated = _clamp_swiglu(self.gate_proj(h_mlp), self.up_proj(h_mlp))
        mlp_out = _clamp_fp16_safe(self.down_proj(gated))
        mlp_out = torch.matmul(self.axis_mix_mlp.to(mlp_out.dtype), mlp_out)
        x = x + g_mlp * mlp_out
        return x, stats


# =============================================================================
# 4. Фабрика моделей под 7 вариантов матричной абляции + оптимизатор
# =============================================================================
def get_matrix_variant_flags(variant: str) -> Dict[str, Any]:
    if variant not in MATRIX_VARIANTS:
        raise ValueError(f"Неизвестный матричный вариант: {variant}. Доступны: {MATRIX_VARIANTS}")
    return {
        "phm_attn": variant in ("opt1_phm_attn", "opt1_phm_full", "all_matrix_safe", "all_matrix_ultra"),
        "phm_ffn": variant in ("opt1_phm_full", "all_matrix_ultra"),
        "grass_align": variant in ("opt2_grass_align", "all_matrix_safe", "all_matrix_ultra"),
        "bilinear_lxr": variant in ("opt3_bilinear_lxr", "all_matrix_safe", "all_matrix_ultra"),
    }


def build_matrix_alt_core(
    void_ref: torch.Tensor,
    variant: str = "ref_dense",
    preset: str = "tiny",
    phm_rank: int = 4,
    cross_alpha: float = 0.25,
    **core_kwargs: Any,
) -> GrassmannianAltDiffusionCore:
    """
    Собирает `GrassmannianAltDiffusionCore` и заменяет целевые блоки согласно матричному варианту.
    Сохраняет единую сигнатуру, DCA, сэмплеры и совместимость с `init_strategies.py`.
    """
    flags = get_matrix_variant_flags(variant)
    model = GrassmannianAltDiffusionCore(void_ref=void_ref, preset=preset, **core_kwargs)
    if variant == "ref_dense":
        return model

    for idx, (kind, layer) in enumerate(zip(model.stack_spec, model.layers)):
        if kind == "full" and (flags["phm_attn"] or flags["phm_ffn"]):
            assert isinstance(layer, FullSABlock)
            new_layer = PHMFullSABlock(
                hidden_size=layer.hidden_size,
                intermediate_size=layer.gate_proj.out_features,
                num_heads=layer.num_heads,
                has_ca=layer.dca_q.has_ca,
                cond_dim=layer.adaln_proj.in_features,
                phm_attn=flags["phm_attn"],
                phm_ffn=flags["phm_ffn"],
                phm_rank=phm_rank,
                cross_alpha=cross_alpha,
            )
            model.layers[idx] = new_layer
        elif kind == "grass_full" and flags["grass_align"]:
            assert isinstance(layer, GrassmannianSABlock)
            new_layer = GrassAlignSABlock(
                axis_dim=layer.axis_dim,
                ffn_dim=layer.gate_proj.out_features,
                qk_dim=layer.sub_head_dim,
                has_ca=layer.dca_q.has_ca,
                cond_dim=layer.adaln_proj.in_features,
                use_bilinear_lxr=flags["bilinear_lxr"],
            )
            model.layers[idx] = new_layer
        elif kind in ("grass_full", "grass_local") and flags["bilinear_lxr"]:
            assert isinstance(layer, GrassmannianSABlock)
            new_layer = BilinearGrassSABlock(
                axis_dim=layer.axis_dim,
                ffn_dim=layer.gate_proj.out_features,
                num_sub_heads=layer.num_sub_heads,
                local_radius=layer.local_radius,
                has_ca=layer.dca_q.has_ca,
                cond_dim=layer.adaln_proj.in_features,
            )
            model.layers[idx] = new_layer

    return model


def create_matrix_optimizers(
    model: nn.Module,
    muon_lr: float = 1e-3,
    adamw_lr: float = 3e-4,
    scale_1d_lr: float = 1.2e-3,
    weight_decay: float = 0.01,
) -> Tuple[Optional[OrthMuon], torch.optim.AdamW]:
    """
    Аналог `create_alt_optimizers`, дополнительно отправляющий матрицы осевого смешивания
    (`axis_mix`, `axis_mix_attn`, `axis_mix_mlp`, `score_temp`) в AdamW с `weight_decay = 0.0`,
    чтобы WD не разрушал стартовую единичную диагональ I_8.
    """
    raw_model = getattr(model, "module", model)
    sphere_params = []
    adamw_decay_params = []
    adamw_nodecay_params = []
    scale_1d_params = []

    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".q_weight"):
            sphere_params.append(param)
        elif (
            name.endswith(".row_scale")
            or name.endswith(".v_gate")
            or name.endswith(".ls_attn")
            or name.endswith(".ls_mlp")
            or name.endswith(".score_temp")
            or ("_norm" in name and "norm_emb" not in name)
            or ("q_norm." in name)
            or ("k_norm." in name)
        ):
            scale_1d_params.append(param)
        elif (
            "dca_q." in name
            or "dca_kv." in name
            or "final_dca." in name
            or "norm_emb" in name
            or "axis_mix" in name
            or name.endswith(".bias")
            or param.ndim < 2
            or "mlp.3." in name
        ):
            adamw_nodecay_params.append(param)
        else:
            adamw_decay_params.append(param)

    opt_muon = None
    if sphere_params:
        opt_muon = OrthMuon(
            [{"params": sphere_params, "lr": muon_lr, "weight_decay": 0.0, "sphere": True}],
            lr=muon_lr,
            momentum=0.95,
            ns_steps=5,
        )

    adamw_groups = []
    if adamw_decay_params:
        adamw_groups.append(
            {"params": adamw_decay_params, "lr": adamw_lr, "base_lr": adamw_lr, "weight_decay": weight_decay}
        )
    if adamw_nodecay_params:
        adamw_groups.append(
            {"params": adamw_nodecay_params, "lr": adamw_lr, "base_lr": adamw_lr, "weight_decay": 0.0}
        )
    if scale_1d_params:
        adamw_groups.append(
            {"params": scale_1d_params, "lr": scale_1d_lr, "base_lr": scale_1d_lr, "weight_decay": 0.0}
        )
    opt_adamw = torch.optim.AdamW(adamw_groups, lr=adamw_lr, betas=(0.9, 0.999), eps=1e-8)
    return opt_muon, opt_adamw
