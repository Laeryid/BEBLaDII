"""
Стратегии геометрической и data-driven инициализации весов для Phase 3 Alt (`GrassmannianAltDiffusionCore`).

Реализует 5 независимых идей из `INIT_OPTIMIZATION_IDEAS.md` и их комбинации:
  1. `idea1_ca_copy`:     «Match-and-Copy» изометрия в GrassmannianCABlock + void-выравненный k_sink.
  2. `idea2_sa_gs`:       Разнознаковые подголовы «Denoise (+R^T) + Gram-Schmidt (-R^T)» в GrassmannianSABlock.
  3. `idea3_full_blkdiag`: Почти блочно-диагональный старт (alpha=0.25) в FullSABlock (1024x1024).
  4. `idea4_wiener_head`: Поосевой Винеровский MMSE-фильтр (closed-form beta_0..7) в GrassmannianOutputHead.
  5. `idea5_kmeans_ffn`:  Сферический K-Means «посев памяти» по реальным латентам в FFN (Full + Grass SA).
  6. `all_geo`:           Комбинация 1 + 2 + 3 (чистая геометрия, без прогона данных).
  7. `all_5`:             Все 5 идей вместе.

Все генераторы случайных чисел строго локальны (`torch.Generator`), поэтому инициализация
не сдвигает глобальный RNG и гарантирует бит-в-бит идентичные батчи в A/B-экспериментах.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from storage.components.plan_c_layers import SphereLinear, canonicalize_cell_basis
from train_phase3_alt import (
    AXIS_DIM,
    FullSABlock,
    GrassmannianAltDiffusionCore,
    GrassmannianCABlock,
    GrassmannianOutputHeadV2,
    GrassmannianSABlock,
    HIDDEN_SIZE,
    N_CANVAS,
    N_PROMPT,
    NUM_AXES,
    SIGMA_DATA,
    sample_mixed_noise_schedule,
)

ABLATION_VARIANTS: Tuple[str, ...] = (
    "baseline",
    "idea1_ca_copy",
    "idea2_sa_gs",
    "idea3_full_blkdiag",
    "idea4_wiener_head",
    "idea5_kmeans_ffn",
    "all_geo",
    "all_5",
    "compat",
)


def _norm_rows(mat: torch.Tensor) -> torch.Tensor:
    return F.normalize(mat.float(), p=2, dim=-1, eps=1e-8)


def _random_orthogonal(dim: int, gen: torch.Generator) -> torch.Tensor:
    """Генерирует случайную ортогональную матрицу R in O(dim) на CPU через QR-разложение."""
    a = torch.randn(dim, dim, generator=gen, dtype=torch.float32)
    q, r = torch.linalg.qr(a)
    d = torch.sign(torch.diag(r))
    d = torch.where(d == 0, torch.ones_like(d), d)
    return q * d.unsqueeze(0)


def _set_sphere_weight(
    layer: SphereLinear,
    q_new: torch.Tensor,
    uniform_scale: Optional[float] = None,
) -> None:
    """
    Обновляет `SphereLinear.q_weight` нормированными строками и опционально выравнивает `row_scale`
    до единого скаляра (чтобы диагональный `row_scale` не искажал ортогональность матрицы R).
    """
    q_norm = _norm_rows(q_new).to(device=layer.q_weight.device, dtype=layer.q_weight.dtype)
    layer.q_weight.data.copy_(q_norm)
    if uniform_scale is not None:
        layer.row_scale.data.fill_(float(uniform_scale))


# =============================================================================
# Идея 1: «Match-and-Copy» изометрия в GrassmannianCABlock + void-выравненный k_sink
# =============================================================================
@torch.no_grad()
def apply_idea1_ca_copy(
    model: GrassmannianAltDiffusionCore,
    void_ref: torch.Tensor,
    eps: float = 0.05,
    seed: int = 101,
) -> Dict[str, Any]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    void_canon, _ = canonicalize_cell_basis(void_ref.float().cpu())
    v_void_0 = F.normalize(void_canon[0], p=2, dim=-1, eps=1e-8)  # (128,)

    n_blocks = 0
    for layer in model.layers:
        if not isinstance(layer, GrassmannianCABlock):
            continue
        d_qk = int(getattr(layer, "qk_dim", layer.axis_dim))
        d_out = int(layer.axis_dim)
        if d_qk == 112:
            # В v2 сохраняем блочную изоляцию [96 sem | 16 syn] с ортогональными блоками
            R = torch.zeros(112, 112, dtype=torch.float32)
            R[:96, :96] = _random_orthogonal(96, gen)
            R[96:112, 96:112] = _random_orthogonal(16, gen)
        else:
            R = _random_orthogonal(d_qk, gen)
        noise_q = torch.randn(d_qk, d_qk, generator=gen, dtype=torch.float32) * (eps / math.sqrt(d_qk))

        W_kv = R
        W_q = _norm_rows(R + noise_q)

        kv_scale = float(layer.kv_proj.row_scale.float().mean().item())
        q_scale = float(layer.q_proj.row_scale.float().mean().item())
        o_scale = float(layer.o_proj.row_scale.float()[:d_qk].mean().item())

        _set_sphere_weight(layer.kv_proj, W_kv, uniform_scale=kv_scale)
        _set_sphere_weight(layer.q_proj, W_q, uniform_scale=q_scale)

        if d_out == d_qk:
            _set_sphere_weight(layer.o_proj, R.T, uniform_scale=o_scale)
        else:
            # o_proj (112 -> 128): первые 112 строк = R^T, хвост 112:128 = малый cross-talk (scale * 0.10)
            W_o = torch.zeros(d_out, d_qk, dtype=torch.float32)
            W_o[:d_qk, :] = R.T
            W_o[d_qk:, :] = _norm_rows(torch.randn(d_out - d_qk, d_qk, generator=gen, dtype=torch.float32))
            _set_sphere_weight(layer.o_proj, W_o, uniform_scale=o_scale)
            layer.o_proj.row_scale.data[d_qk:].mul_(0.10)

        # k_sink выравниваем с образом главной оси void_concept в пространстве ключей R
        v_void_sub = F.normalize(v_void_0[:d_qk], p=2, dim=-1, eps=1e-8)
        k_void = (v_void_sub @ R.T).view(1, layer.num_sub_heads, 1, layer.sub_head_dim)
        k_void = F.normalize(k_void, p=2, dim=-1, eps=1e-8) * (0.02 * math.sqrt(layer.sub_head_dim))
        layer.k_sink.data.copy_(k_void.to(device=layer.k_sink.device, dtype=layer.k_sink.dtype))
        n_blocks += 1

    return {"idea1_ca_blocks": n_blocks}


# =============================================================================
# Идея 2: Разнознаковые подголовы «Denoise (+R^T) + Gram-Schmidt (-R^T)» в GrassmannianSABlock
# =============================================================================
@torch.no_grad()
def apply_idea2_sa_denoise_gs(
    model: GrassmannianAltDiffusionCore,
    eps: float = 0.05,
    seed: int = 202,
) -> Dict[str, Any]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n_blocks = 0
    for layer in model.layers:
        if not isinstance(layer, GrassmannianSABlock) or not hasattr(layer, "kv_proj"):
            continue
        d = layer.axis_dim
        H = layer.num_sub_heads
        dh = layer.sub_head_dim
        if d == 128 and H == 8 and dh == 16:
            # В v2 сохраняем блочную структуру [96 sem (6x16) | 16 syn | 16 pos]
            R = torch.zeros(128, 128, dtype=torch.float32)
            R[:96, :96] = _random_orthogonal(96, gen)
            R[96:112, 96:112] = _random_orthogonal(16, gen)
            R[112:128, 112:128] = _random_orthogonal(16, gen)
            signs = torch.ones(128, dtype=torch.float32)
            # В sem (6 голов): 3 притягивающие (+), 3 Грама-Шмидта (-); syn и pos: притягивающие (+)
            signs[3 * 16 : 6 * 16] = -1.0
        else:
            R = _random_orthogonal(d, gen)
            half_pos = (H // 2) * dh
            signs = torch.ones(d, dtype=torch.float32)
            signs[half_pos:] = -1.0
        noise_q = torch.randn(d, d, generator=gen, dtype=torch.float32) * (eps / math.sqrt(d))

        W_kv = R
        W_q = _norm_rows(R + noise_q)
        W_o = R.T * signs.unsqueeze(0)  # строки остаются строго единичной нормы!

        kv_scale = float(layer.kv_proj.row_scale.float().mean().item())
        q_scale = float(layer.q_proj.row_scale.float().mean().item())
        o_scale = float(layer.o_proj.row_scale.float().mean().item())

        _set_sphere_weight(layer.kv_proj, W_kv, uniform_scale=kv_scale)
        _set_sphere_weight(layer.q_proj, W_q, uniform_scale=q_scale)
        _set_sphere_weight(layer.o_proj, W_o, uniform_scale=o_scale)
        n_blocks += 1

    return {"idea2_sa_blocks": n_blocks}


# =============================================================================
# Идея 3: Почти блочно-диагональный старт в FullSABlock (1024x1024)
# =============================================================================
@torch.no_grad()
def apply_idea3_full_blockdiag(
    model: GrassmannianAltDiffusionCore,
    alpha: float = 0.25,
    eps: float = 0.05,
    seed: int = 303,
) -> Dict[str, Any]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n_blocks = 0
    diag_w = math.sqrt(max(0.0, 1.0 - alpha * alpha))

    def _offdiag_unit_rows(num_axes: int, axis_dim: int) -> torch.Tensor:
        dim = num_axes * axis_dim
        m = torch.randn(dim, dim, generator=gen, dtype=torch.float32)
        for k in range(num_axes):
            s = k * axis_dim
            m[s : s + axis_dim, s : s + axis_dim] = 0.0
        return _norm_rows(m)

    for layer in model.layers:
        if not isinstance(layer, FullSABlock):
            continue
        K = layer.num_heads  # 8
        D = layer.head_dim   # 128
        C = K * D            # 1024

        B_diag = torch.zeros(C, C, dtype=torch.float32)
        for k in range(K):
            Rk = _random_orthogonal(D, gen)
            s = k * D
            B_diag[s : s + D, s : s + D] = Rk

        M_kv = _offdiag_unit_rows(K, D)
        M_o = _offdiag_unit_rows(K, D)

        W_kv = _norm_rows(diag_w * B_diag + alpha * M_kv)
        noise_q = torch.randn(C, C, generator=gen, dtype=torch.float32) * (eps / math.sqrt(C))
        W_q = _norm_rows(W_kv + noise_q)
        W_o = _norm_rows(diag_w * B_diag.T + alpha * M_o)

        kv_scale = float(layer.kv_proj.row_scale.float().mean().item())
        q_scale = float(layer.q_proj.row_scale.float().mean().item())
        o_scale = float(layer.o_proj.row_scale.float().mean().item())

        _set_sphere_weight(layer.kv_proj, W_kv, uniform_scale=kv_scale)
        _set_sphere_weight(layer.q_proj, W_q, uniform_scale=q_scale)
        _set_sphere_weight(layer.o_proj, W_o, uniform_scale=o_scale)
        n_blocks += 1

    return {"idea3_full_blocks": n_blocks, "alpha": alpha}


# =============================================================================
# Идея 4: Поосевой Винеровский MMSE-фильтр в GrassmannianOutputHead
# =============================================================================
@torch.no_grad()
def compute_wiener_axis_betas(
    train_items: Sequence[Dict[str, torch.Tensor]],
    sep_concept: torch.Tensor,
    num_samples: int = 128,
    num_mc_draws: int = 4,
    seed: int = 404,
) -> Dict[str, Any]:
    """
    Вычисляет аналитический поосевой MMSE-оптимум beta_k для GrassmannianOutputHead:
        V_0_raw = V_t + SIGMA_DATA * g(t) * out_proj(RMSNorm(V_t / SIGMA_DATA))
    Минимизируя E[ || V_{0,k} - (V_{t,k} + beta_k * SIGMA_DATA * g(t) * h_k) ||^2 ]
    по реальному распределению шума `sample_mixed_noise_schedule` и реальным ячейкам датасета.
    """
    n = min(num_samples, len(train_items))
    v_ans = torch.stack([train_items[i]["V_ans"].float() for i in range(n)], dim=0)  # (n, 127, 8, 128)
    sep_canon, _ = canonicalize_cell_basis(sep_concept.float().cpu())
    sep_exp = sep_canon.view(1, 1, NUM_AXES, AXIS_DIM).expand(n, 1, NUM_AXES, AXIS_DIM)
    V_0 = torch.cat([sep_exp, v_ans], dim=1)  # (n, 128, 8, 128)
    i_last = torch.stack([train_items[i]["i_last"].long() for i in range(n)], dim=0)

    # Эмпирический RMS по каждой из 8 осей на реальных контентных ячейках
    axis_rms = v_ans.pow(2).mean(dim=(0, 1, 3)).sqrt()  # (8,)

    num = torch.zeros(NUM_AXES, dtype=torch.float64)
    den = torch.zeros(NUM_AXES, dtype=torch.float64)

    # Сохраняем/восстанавливаем глобальный RNG, т.к. sample_mixed_noise_schedule использует torch.rand
    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        for _ in range(num_mc_draws):
            t_mat = sample_mixed_noise_schedule(n, torch.device("cpu"), i_last=i_last)  # (n, 128)
            noise = torch.randn_like(V_0) * SIGMA_DATA
            t_exp = t_mat.unsqueeze(-1).unsqueeze(-1)
            V_t = (1.0 - t_exp) * V_0 + t_exp * noise
            V_t[:, 0] = sep_exp[:, 0]

            # h = RMSNorm(V_t / SIGMA_DATA) по всей 1024-мерной клетке (как в GrassmannianOutputHead)
            x_in = (V_t / SIGMA_DATA).reshape(n, N_CANVAS, HIDDEN_SIZE)
            rms = x_in.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
            h = (x_in / rms).view(n, N_CANVAS, NUM_AXES, AXIS_DIM)

            # Глобальный гейт g(t) = clamp(t_bar / 0.02, 0, 1), g[:, 0] = 0
            t_bar = t_mat[:, 1:].mean(dim=1, keepdim=True).expand(n, N_CANVAS)
            g_t = torch.clamp(t_bar / 0.02, min=0.0, max=1.0).clone()
            g_t[:, 0] = 0.0
            feat = (SIGMA_DATA * g_t.unsqueeze(-1).unsqueeze(-1) * h)[:, 1:]  # без SEP
            target = (V_0 - V_t)[:, 1:]

            num += (target.double() * feat.double()).sum(dim=(0, 1, 3))
            den += feat.double().pow(2).sum(dim=(0, 1, 3))
    finally:
        torch.random.set_rng_state(rng_state)

    betas = (num / den.clamp(min=1e-12)).float()
    # Клиппинг ради стабильности: разумный диапазон [-2.0, 0.0]
    betas = betas.clamp(min=-2.0, max=0.0)
    return {
        "betas": betas,          # (8,)
        "axis_rms": axis_rms,    # (8,)
    }


@torch.no_grad()
def apply_idea4_wiener_output_head(
    model: GrassmannianAltDiffusionCore,
    wiener_info: Dict[str, Any],
    eps: float = 0.01,
    damp: float = 1.0,
    seed: int = 404,
) -> Dict[str, Any]:
    if isinstance(model.output_head, GrassmannianOutputHeadV2):
        # В v2 EDM-голова уже встроена аналитически через c_skip_base(t_k)/c_out_base(t_k) по axis_sigma.
        # При наличии эмпирического axis_rms можно синхронизировать буфер axis_sigma.
        axis_rms = wiener_info["axis_rms"].float().clamp(min=1e-4)
        model.output_head.axis_sigma.copy_(axis_rms.to(device=model.output_head.axis_sigma.device))
        return {
            "idea4_mode": "edm_v2_axis_sigma",
            "idea4_axis_rms": [round(float(x), 5) for x in axis_rms.tolist()],
        }

    gen = torch.Generator(device="cpu").manual_seed(seed)
    betas: torch.Tensor = wiener_info["betas"].float() * float(damp)  # (8,)

    W = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, generator=gen, dtype=torch.float32) * (eps / math.sqrt(HIDDEN_SIZE))
    for k in range(NUM_AXES):
        s = k * AXIS_DIM
        idx = torch.arange(s, s + AXIS_DIM)
        W[idx, idx] = float(betas[k].item())

    out_proj = model.output_head.out_proj
    out_proj.weight.data.copy_(W.to(device=out_proj.weight.device, dtype=out_proj.weight.dtype))
    return {
        "idea4_betas": [round(float(x), 4) for x in betas.tolist()],
        "idea4_damp": float(damp),
        "idea4_axis_rms": [round(float(x), 4) for x in wiener_info["axis_rms"].tolist()],
    }


# =============================================================================
# Идея 5: Сферический K-Means «посев памяти» для FFN (FullSABlock + GrassmannianSABlock)
# =============================================================================
@torch.no_grad()
def spherical_kmeans(
    X_unit: torch.Tensor,
    num_clusters: int,
    n_iters: int = 8,
    gen: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Быстрый сферический K-Means на CPU/GPU (X_unit формы (M, D) с единичной нормой строк).
    Возвращает центроиды C формы (num_clusters, D) строго на единичной сфере.
    """
    M, D = X_unit.shape
    if gen is None:
        gen = torch.Generator(device="cpu").manual_seed(0)

    if M >= num_clusters:
        perm = torch.randperm(M, generator=gen)[:num_clusters]
        C = X_unit[perm].clone()
    else:
        # Если данных меньше K, дублируем с лёгкой пертурбацией
        idx = torch.randint(0, M, (num_clusters,), generator=gen)
        noise = torch.randn(num_clusters, D, generator=gen, dtype=torch.float32) * 0.1
        C = _norm_rows(X_unit[idx] + noise)

    # Итерации Ллойда по косинусному сходству (батчево для экономии RAM)
    chunk = 2048
    for _ in range(n_iters):
        sums = torch.zeros_like(C)
        counts = torch.zeros(num_clusters, dtype=torch.long)
        for start in range(0, M, chunk):
            xb = X_unit[start : start + chunk]
            sims = xb @ C.T  # (b, K)
            assign = sims.argmax(dim=-1)  # (b,)
            sums.index_add_(0, assign, xb)
            counts.index_add_(0, assign, torch.ones_like(assign, dtype=torch.long))

        empty = counts == 0
        if empty.any():
            n_empty = int(empty.sum().item())
            repl_idx = torch.randint(0, M, (n_empty,), generator=gen)
            noise = torch.randn(n_empty, D, generator=gen, dtype=torch.float32) * 0.05
            sums[empty] = _norm_rows(X_unit[repl_idx] + noise)

        C = _norm_rows(sums)

    return C


@torch.no_grad()
def build_kmeans_pools(
    train_items: Sequence[Dict[str, torch.Tensor]],
    max_cells: int = 8192,
    max_vecs: int = 16384,
    seed: int = 505,
) -> Dict[str, torch.Tensor]:
    """
    Извлекает и нормирует пул реальных ячеек (1024d) и базисных векторов (128d) из кэша латентов,
    плюс эталонную выборку после RMSNorm для калибровки `row_scale` в FFN.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    cells_list: List[torch.Tensor] = []
    for item in train_items:
        v_ans = item["V_ans"].float()  # (127, 8, 128)
        # Берем реальные контентные ячейки + до 8 VOID-ячеек для полноты многообразия
        i_last = int(item["i_last"].item())
        n_take = min(v_ans.shape[0], max(1, i_last) + 8)
        cells_list.append(v_ans[:n_take])

    all_cells = torch.cat(cells_list, dim=0)  # (M_total, 8, 128)
    M_total = all_cells.shape[0]

    # Пул 1024d ячеек
    n_cells = min(max_cells, M_total)
    idx_c = torch.randperm(M_total, generator=gen)[:n_cells]
    sel_cells = all_cells[idx_c]  # (n_cells, 8, 128)
    flat_cells = sel_cells.reshape(n_cells, HIDDEN_SIZE)
    cells_unit = _norm_rows(flat_cells)
    # RMSNorm(1024) вход в FullSABlock.FFN имеет норму sqrt(1024) = 32
    cells_rmsnorm = cells_unit * math.sqrt(HIDDEN_SIZE)

    # Пул 128d базисных векторов
    all_vecs = sel_cells.reshape(-1, AXIS_DIM)
    n_vecs = min(max_vecs, all_vecs.shape[0])
    idx_v = torch.randperm(all_vecs.shape[0], generator=gen)[:n_vecs]
    vecs_unit = _norm_rows(all_vecs[idx_v])
    vecs_rmsnorm = vecs_unit * math.sqrt(AXIS_DIM)

    return {
        "cells_unit": cells_unit,
        "cells_rmsnorm": cells_rmsnorm[:1024],
        "vecs_unit": vecs_unit,
        "vecs_rmsnorm": vecs_rmsnorm[:2048],
    }


@torch.no_grad()
def apply_idea5_kmeans_ffn(
    model: GrassmannianAltDiffusionCore,
    kmeans_pools: Dict[str, torch.Tensor],
    eps_up: float = 0.08,
    n_iters: int = 8,
    include_full: bool = True,
    seed: int = 505,
) -> Dict[str, Any]:
    """
    Зашивает центроиды Spherical K-Means в `gate_proj` и `up_proj` слоев `FullSABlock` и `GrassmannianSABlock`,
    калибруя `row_scale` так, чтобы стартовая RMS-амплитуда преактиваций точно совпадала со стандартным Kaiming.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    cells_unit = kmeans_pools["cells_unit"]
    cells_rms = kmeans_pools["cells_rmsnorm"]
    vecs_unit = kmeans_pools["vecs_unit"]
    vecs_rms = kmeans_pools["vecs_rmsnorm"]

    n_full = 0
    n_grass = 0

    for layer in model.layers:
        is_full_dense_ffn = (
            include_full
            and hasattr(layer, "gate_proj")
            and hasattr(layer, "up_proj")
            and isinstance(layer.gate_proj, SphereLinear)
            and isinstance(layer.up_proj, SphereLinear)
            and layer.gate_proj.in_features == HIDDEN_SIZE
        )
        if is_full_dense_ffn:
            K_ffn = layer.gate_proj.out_features
            # Подвыборка для разнообразия между слоями
            m_sub = min(cells_unit.shape[0], max(K_ffn * 3, 2048))
            sub_idx = torch.randperm(cells_unit.shape[0], generator=gen)[:m_sub]
            C = spherical_kmeans(cells_unit[sub_idx], K_ffn, n_iters=n_iters, gen=gen)  # (K_ffn, 1024)
            noise_up = torch.randn_like(C, generator=gen) * (eps_up / math.sqrt(HIDDEN_SIZE))
            C_up = _norm_rows(C + noise_up)

            # Калибровка row_scale: сохраняем точно такую же стартовую RMS преактиваций, как в baseline
            # В baseline для случайной единичной строки q и h=RMSNorm(x): E[(h·q)^2] = 1.0 -> rms = mean(row_scale)
            orig_gate_scale = float(layer.gate_proj.row_scale.float().mean().item())
            orig_up_scale = float(layer.up_proj.row_scale.float().mean().item())
            rms_gate = float((cells_rms @ C.T).pow(2).mean().sqrt().clamp(min=1e-3).item())
            rms_up = float((cells_rms @ C_up.T).pow(2).mean().sqrt().clamp(min=1e-3).item())

            _set_sphere_weight(layer.gate_proj, C, uniform_scale=orig_gate_scale / rms_gate)
            _set_sphere_weight(layer.up_proj, C_up, uniform_scale=orig_up_scale / rms_up)
            n_full += 1

        elif isinstance(layer, GrassmannianSABlock):
            K_ffn = layer.gate_proj.out_features
            m_sub = min(vecs_unit.shape[0], max(K_ffn * 6, 2048))
            sub_idx = torch.randperm(vecs_unit.shape[0], generator=gen)[:m_sub]
            C = spherical_kmeans(vecs_unit[sub_idx], K_ffn, n_iters=n_iters, gen=gen)  # (K_ffn, 128)
            noise_up = torch.randn_like(C, generator=gen) * (eps_up / math.sqrt(AXIS_DIM))
            C_up = _norm_rows(C + noise_up)

            orig_gate_scale = float(layer.gate_proj.row_scale.float().mean().item())
            orig_up_scale = float(layer.up_proj.row_scale.float().mean().item())
            rms_gate = float((vecs_rms @ C.T).pow(2).mean().sqrt().clamp(min=1e-3).item())
            rms_up = float((vecs_rms @ C_up.T).pow(2).mean().sqrt().clamp(min=1e-3).item())

            _set_sphere_weight(layer.gate_proj, C, uniform_scale=orig_gate_scale / rms_gate)
            _set_sphere_weight(layer.up_proj, C_up, uniform_scale=orig_up_scale / rms_up)
            n_grass += 1

    return {"idea5_full_ffn_blocks": n_full, "idea5_grass_ffn_blocks": n_grass}


# =============================================================================
# Единый диспетчер и проверка инвариантов
# =============================================================================
def apply_init_strategy(
    model: GrassmannianAltDiffusionCore,
    strategy_name: str,
    void_ref: torch.Tensor,
    sep_concept: torch.Tensor,
    train_items: Optional[Sequence[Dict[str, torch.Tensor]]] = None,
    precomputed_stats: Optional[Dict[str, Any]] = None,
    wiener_damp: float = 1.0,
    base_seed: int = 1000,
) -> Dict[str, Any]:
    if strategy_name not in ABLATION_VARIANTS:
        raise ValueError(f"Неизвестная стратегия: {strategy_name}. Доступны: {ABLATION_VARIANTS}")

    info: Dict[str, Any] = {"strategy": strategy_name}
    if strategy_name == "baseline":
        return info

    use_1 = strategy_name in ("idea1_ca_copy", "all_geo", "all_5", "compat")
    use_2 = strategy_name in ("idea2_sa_gs", "all_geo", "all_5", "compat")
    use_3 = strategy_name in ("idea3_full_blkdiag", "all_geo", "all_5", "compat")
    use_4 = strategy_name in ("idea4_wiener_head", "all_5", "compat")
    use_5 = strategy_name in ("idea5_kmeans_ffn", "all_5", "compat")
    include_full_k5 = strategy_name != "compat"

    if use_1:
        info.update(apply_idea1_ca_copy(model, void_ref=void_ref, seed=base_seed + 1))
    if use_2:
        info.update(apply_idea2_sa_denoise_gs(model, seed=base_seed + 2))
    if use_3:
        info.update(apply_idea3_full_blockdiag(model, seed=base_seed + 3))
    if use_4:
        wiener_info = (precomputed_stats or {}).get("wiener_info")
        if wiener_info is None:
            if train_items is None:
                raise ValueError("Для idea4_wiener_head необходим train_items или precomputed_stats['wiener_info']")
            wiener_info = compute_wiener_axis_betas(train_items, sep_concept=sep_concept, seed=base_seed + 4)
        info.update(
            apply_idea4_wiener_output_head(model, wiener_info=wiener_info, damp=wiener_damp, seed=base_seed + 4)
        )
    if use_5:
        kmeans_pools = (precomputed_stats or {}).get("kmeans_pools")
        if kmeans_pools is None:
            if train_items is None:
                raise ValueError("Для idea5_kmeans_ffn необходим train_items или precomputed_stats['kmeans_pools']")
            kmeans_pools = build_kmeans_pools(train_items, seed=base_seed + 5)
        info.update(
            apply_idea5_kmeans_ffn(model, kmeans_pools=kmeans_pools, include_full=include_full_k5, seed=base_seed + 5)
        )

    return info


@torch.no_grad()
def verify_init_invariants(
    model: GrassmannianAltDiffusionCore,
    sep_concept: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """Проверяет сферичность всех *.q_weight, отсутствие NaN/Inf, SEP t=0 и нулевую утечку CFG."""
    max_sph_err = 0.0
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            raise RuntimeError(f"NaN/Inf в параметре {name} после инициализации!")
        if name.endswith(".q_weight"):
            err = float((p.float().norm(p=2, dim=-1) - 1.0).abs().max().item())
            max_sph_err = max(max_sph_err, err)
    if max_sph_err > 1e-5:
        raise RuntimeError(f"Нарушена сферичность *.q_weight после инициализации: max_err={max_sph_err}")

    model.eval()
    gen = torch.Generator(device="cpu").manual_seed(777)
    sep_canon, _ = canonicalize_cell_basis(sep_concept.float().cpu())
    B = 1
    V_t = (torch.randn(B, N_CANVAS, NUM_AXES, AXIS_DIM, generator=gen) * SIGMA_DATA).to(device)
    V_t[:, 0] = sep_canon.to(device)
    t_mat = torch.full((B, N_CANVAS), 0.5, device=device)
    t_mat[:, 0] = 0.0
    Vp_a = (torch.randn(B, N_PROMPT, NUM_AXES, AXIS_DIM, generator=gen) * SIGMA_DATA).to(device)
    Vp_b = (torch.randn(B, N_PROMPT, NUM_AXES, AXIS_DIM, generator=gen) * SIGMA_DATA).to(device)
    p_valid = torch.ones(B, N_PROMPT, dtype=torch.bool, device=device)
    p_uncond = torch.zeros(B, N_PROMPT, dtype=torch.bool, device=device)
    m_zero = torch.zeros(B, device=device)

    out_a, _, sw = model(V_t, t_mat, Vp_a, p_valid, p_uncond, m_zero, return_sink_weights=True)
    out_b, _, _ = model(V_t, t_mat, Vp_b, p_valid, p_uncond, m_zero)

    leak_diff = float((out_a - out_b).abs().max().item())
    sep_err = float((out_a[:, 0] - sep_canon.to(device)).abs().max().item())
    sink_uncond = float(sw.mean().item()) if sw is not None else 0.0
    if leak_diff > 1e-5:
        raise RuntimeError(f"Утечка CFG после инициализации: leak_diff={leak_diff}")
    if sep_err > 1e-5:
        raise RuntimeError(f"Нарушено SEP t=0 тождество после инициализации: sep_err={sep_err}")
    if not torch.isfinite(out_a).all():
        raise RuntimeError("NaN/Inf на выходе модели после инициализации!")

    return {
        "max_sph_err": max_sph_err,
        "cfg_leak_diff": leak_diff,
        "sep_err": sep_err,
        "sink_uncond": sink_uncond,
    }
