"""
Абляционный эксперимент по 5 идеям инициализации весов на модели `tiny` (Phase 3 Alt Plan C).

Сравнивает 8 конфигураций в абсолютно равных условиях (бит-в-бит одинаковые обучающие батчи,
идентичные шумы t_mat / eps / CFG dropout на каждом шаге и идентичная валидация):
  1. baseline           — стандартная случайная инициализация
  2. idea1_ca_copy      — Match-and-Copy изометрия в GrassmannianCABlock + void k_sink
  3. idea2_sa_gs        — Разнознаковые подголовы Denoise (+R^T) / Gram-Schmidt (-R^T) в GrassmannianSABlock
  4. idea3_full_blkdiag — Почти блочно-диагональный старт (alpha=0.25) в FullSABlock
  5. idea4_wiener_head  — Поосевой Винеровский MMSE-фильтр в GrassmannianOutputHead
  6. idea5_kmeans_ffn   — Сферический K-Means посев в FFN (Full + Grass SA)
  7. all_geo            — Идеи 1 + 2 + 3 вместе (чистая геометрия, 0 оверхеда)
  8. all_5              — Все 5 идей вместе

Измеряет:
  - Мгновенный эффект ДО обучения (step 0) на фиксированной валидации (Sim_Gr, CFG, Loss, TokAcc/CE).
  - Траекторию обучения (кривые loss/total, loss/basis, loss/gr, loss/energy + AUC кривой).
  - Финальные валидационные метрики после `steps` шагов + дельты относительно `baseline`.

Примеры запуска:
  # Быстрая проверка всех 8 вариантов (smoke-test, ~1.5-2 мин на CPU):
  .venv\\Scripts\\python.exe -B "experiments/phase 3 alt/exp_init_ablation.py" --smoke-test

  # Основной эксперимент на CPU (tiny, 40 шагов, batch=4, 128 train / 32 val):
  .venv\\Scripts\\python.exe -B "experiments/phase 3 alt/exp_init_ablation.py" --steps 40 --batch-size 4

  # Глубокий прогон (100 шагов) или только выбранные варианты:
  .venv\\Scripts\\python.exe -B "experiments/phase 3 alt/exp_init_ablation.py" --steps 100 --variants baseline,idea1_ca_copy,all_geo,all_5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ALT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ALT_DIR.parents[1]
PHASE3_DIR = PROJECT_ROOT / "experiments" / "phase 3"
for _p in (str(PROJECT_ROOT), str(PHASE3_DIR), str(ALT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from storage.components.decoder import get_decoder
from storage.components.encoder import get_encoder
from storage.components.plan_c_layers import canonicalize_cell_basis
from storage.components.tokenizer import get_tokenizer

from init_strategies import (
    ABLATION_VARIANTS,
    apply_init_strategy,
    build_kmeans_pools,
    compute_wiener_axis_betas,
    verify_init_invariants,
)
from train_phase3 import sim_gr
from train_phase3_alt import (
    ARCH_PRESETS,
    AXIS_DIM,
    AltEnvironmentConfig,
    GrassmannianAltDiffusionCore,
    N_CANVAS,
    NUM_AXES,
    PAD_TOKEN_ID,
    PromptAnswerDataset,
    SIGMA_DATA,
    apply_prompt_dropout,
    build_or_load_latent_cache,
    compute_phase3_loss,
    create_alt_optimizers,
    sample_mixed_noise_schedule,
    update_optimizers_lr,
)
from vocab_optim import FastVocabOptimizer


def _collate_items(items: Sequence[Dict[str, torch.Tensor]], indices: Sequence[int]) -> Dict[str, torch.Tensor]:
    batch_items = [items[i] for i in indices]
    return {
        "V_prompt": torch.stack([x["V_prompt"].float() for x in batch_items], dim=0),
        "V_ans": torch.stack([x["V_ans"].float() for x in batch_items], dim=0),
        "ans_targets": torch.stack([x["ans_targets"].long() for x in batch_items], dim=0),
        "n_real_prompt": torch.stack([x["n_real_prompt"].long() for x in batch_items], dim=0),
        "i_last": torch.stack([x["i_last"].long() for x in batch_items], dim=0),
        "w_cell": torch.stack([x["w_cell"].float() for x in batch_items], dim=0),
    }


@torch.no_grad()
def build_fixed_train_schedule(
    train_items: Sequence[Dict[str, torch.Tensor]],
    steps: int,
    batch_size: int,
    seed: int = 2026,
) -> List[Dict[str, Any]]:
    """
    Заранее фиксирует для каждого шага 1..steps индексы примеров, расписание шума t_mat,
    гауссов шум eps и маску CFG dropout — чтобы все 8 вариантов видели БИТ-В-БИТ
    одинаковую обучающую последовательность.
    """
    n_train = len(train_items)
    rng_state = torch.random.get_rng_state()
    schedule: List[Dict[str, Any]] = []
    try:
        torch.manual_seed(seed)
        perm = torch.randperm(n_train).tolist()
        cursor = 0
        for _ in range(steps):
            indices: List[int] = []
            while len(indices) < batch_size:
                if cursor >= n_train:
                    perm = torch.randperm(n_train).tolist()
                    cursor = 0
                indices.append(perm[cursor])
                cursor += 1

            i_last = torch.stack([train_items[i]["i_last"].long() for i in indices], dim=0)
            n_real_prompt = torch.stack([train_items[i]["n_real_prompt"].long() for i in indices], dim=0)
            p_valid = n_real_prompt > 0
            p_attn_mask, m_prompt = apply_prompt_dropout(p_valid)
            t_mat = sample_mixed_noise_schedule(batch_size, torch.device("cpu"), i_last=i_last)
            # Храним шум в FP16 для экономии RAM (при шаге 100 и batch=4 это всего ~16 MB)
            noise_fp16 = (torch.randn(batch_size, N_CANVAS, NUM_AXES, AXIS_DIM) * SIGMA_DATA).to(torch.float16)

            schedule.append(
                {
                    "indices": indices,
                    "t_mat": t_mat,
                    "noise_fp16": noise_fp16,
                    "p_attn_mask": p_attn_mask,
                    "m_prompt": m_prompt,
                }
            )
    finally:
        torch.random.set_rng_state(rng_state)
    return schedule


@torch.no_grad()
def build_fixed_val_pack(
    val_items: Sequence[Dict[str, torch.Tensor]],
    seed: int = 9999,
) -> Dict[str, Any]:
    """Фиксирует валидационный пакет и детерминированный шум на уровнях t=0.25, t=0.50, t=0.75."""
    n_val = len(val_items)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise_low = torch.randn(n_val, N_CANVAS, NUM_AXES, AXIS_DIM, generator=gen) * SIGMA_DATA
    noise_mid = torch.randn(n_val, N_CANVAS, NUM_AXES, AXIS_DIM, generator=gen) * SIGMA_DATA
    noise_hi = torch.randn(n_val, N_CANVAS, NUM_AXES, AXIS_DIM, generator=gen) * SIGMA_DATA
    return {
        "items": val_items,
        "noise_low": noise_low,
        "noise_mid": noise_mid,
        "noise_hi": noise_hi,
    }


@torch.no_grad()
def evaluate_fixed_val(
    model: GrassmannianAltDiffusionCore,
    val_pack: Dict[str, Any],
    sep_canon: torch.Tensor,
    void_canon: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    batch_size: int = 8,
    decoder: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """
    Детерминированная оценка на фиксированном val-наборе на 3 уровнях шума (t=0.25, 0.50, 0.75):
    - loss/total, loss/basis, loss/gr, loss/energy на t=0.50;
    - Sim_Gr (low=0.25, mid=0.50, hi=0.75);
    - CFG contrast (mid=0.50, hi=0.75);
    - VOID F1 (mid=0.50) и эффективный ранг rk_eff;
    - при наличии `decoder`: токенная точность TokAcc и CE на low (0.25) и hi (0.75).
    """
    model.eval()
    items = val_pack["items"]
    n_val = len(items)
    sep_dev = sep_canon.to(device=device, dtype=torch.float32)
    void_dev = void_canon.to(device=device, dtype=torch.float32)

    acc: Dict[str, List[float]] = {
        "val/loss_total": [],
        "val/loss_basis": [],
        "val/loss_gr": [],
        "val/loss_energy": [],
        "val/sim_gr_low": [],
        "val/sim_gr_mid": [],
        "val/sim_gr_hi": [],
        "val/cfg_mid": [],
        "val/cfg_hi": [],
        "val/cfg_dsim_mid": [],
        "val/cfg_dsim_hi": [],
        "val/rk_eff": [],
        "val/tok_acc_low": [],
        "val/ce_low": [],
        "val/tok_acc_hi": [],
        "val/ce_hi": [],
    }
    tp_void = fp_void = fn_void = 0

    for start in range(0, n_val, batch_size):
        end = min(n_val, start + batch_size)
        idxs = list(range(start, end))
        batch = _collate_items(items, idxs)
        B = len(idxs)

        V_prompt = batch["V_prompt"].to(device)
        V_ans = batch["V_ans"].to(device)
        ans_targets = batch["ans_targets"].to(device)
        n_real_prompt = batch["n_real_prompt"].to(device)
        i_last = batch["i_last"].to(device)
        w_cell = batch["w_cell"].to(device)

        sep_exp = sep_dev.view(1, 1, NUM_AXES, AXIS_DIM).expand(B, 1, NUM_AXES, AXIS_DIM)
        V_0 = torch.cat([sep_exp, V_ans], dim=1)
        p_valid = n_real_prompt > 0
        p_uncond = torch.zeros_like(p_valid)
        m_one = torch.ones(B, device=device)
        m_zero = torch.zeros(B, device=device)
        in_dtype = amp_dtype if use_amp else torch.float32

        def _forward_at(t_val: float, noise_slice: torch.Tensor, cond: bool = True):
            t_mat = torch.full((B, N_CANVAS), t_val, device=device, dtype=torch.float32)
            t_mat[:, 0] = 0.0
            noise = noise_slice.to(device=device, dtype=torch.float32)
            V_t = (1.0 - t_mat.unsqueeze(-1).unsqueeze(-1)) * V_0 + t_mat.unsqueeze(-1).unsqueeze(-1) * noise
            V_t[:, 0] = sep_exp[:, 0]
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                vh, stats, sw = model(
                    V_t=V_t.to(in_dtype),
                    t_mat=t_mat,
                    V_prompt=V_prompt.to(in_dtype),
                    prompt_valid_mask=p_valid,
                    prompt_attn_mask=p_valid if cond else p_uncond,
                    m_prompt=m_one if cond else m_zero,
                    radar_sigma=None,
                    return_sink_weights=True,
                )
            return vh.float(), stats, sw, t_mat

        vh_low, _, _, _ = _forward_at(0.25, val_pack["noise_low"][start:end], cond=True)
        vh_mid, st_mid, sw_mid, t_mid = _forward_at(0.50, val_pack["noise_mid"][start:end], cond=True)
        vh_mid_u, _, _, _ = _forward_at(0.50, val_pack["noise_mid"][start:end], cond=False)
        vh_hi, _, _, _ = _forward_at(0.75, val_pack["noise_hi"][start:end], cond=True)
        vh_hi_u, _, _, _ = _forward_at(0.75, val_pack["noise_hi"][start:end], cond=False)

        _, m_mid = compute_phase3_loss(
            V_0_hat=vh_mid,
            V_0=V_0,
            t_mat=t_mid,
            w_cell=w_cell,
            i_last=i_last,
            adaln_stats=st_mid,
            step=1000,
            m_prompt=m_one,
            sink_weights=sw_mid,
            world_size=1,
        )
        acc["val/loss_total"].append(m_mid["loss/total"])
        acc["val/loss_basis"].append(m_mid["loss/basis"])
        acc["val/loss_gr"].append(m_mid["loss/gr"])
        acc["val/loss_energy"].append(m_mid["loss/energy"])

        # Sim_Gr на реальных контентных ячейках
        pos = torch.arange(1, N_CANVAS, device=device).unsqueeze(0)
        is_content = pos <= i_last.unsqueeze(1)  # (B, 127)
        if is_content.any():
            acc["val/sim_gr_low"].append(float(sim_gr(vh_low[:, 1:], V_0[:, 1:])[is_content].mean().item()))
            acc["val/sim_gr_mid"].append(float(sim_gr(vh_mid[:, 1:], V_0[:, 1:])[is_content].mean().item()))
            acc["val/sim_gr_hi"].append(float(sim_gr(vh_hi[:, 1:], V_0[:, 1:])[is_content].mean().item()))

            # Эффективный ранг контентных ячеек на t=0.50
            sv = torch.linalg.svdvals(vh_mid[:, 1:][is_content])
            p_sv = (sv ** 2) / (sv ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-12)
            rk = torch.exp(-(p_sv * (p_sv.clamp(min=1e-12)).log()).sum(dim=-1)).mean()
            acc["val/rk_eff"].append(float(rk.item()))

        acc["val/cfg_mid"].append(float((vh_mid[:, 1:] - vh_mid_u[:, 1:]).norm(p=2, dim=(-2, -1)).mean().item()))
        acc["val/cfg_hi"].append(float((vh_hi[:, 1:] - vh_hi_u[:, 1:]).norm(p=2, dim=(-2, -1)).mean().item()))
        # CFG-ΔSim: полезность условия = SimGr(cond, V_0) − SimGr(uncond, V_0) (>0 только если промпт помогает)
        if is_content.any():
            for tag_d, vh_c, vh_u in (("mid", vh_mid, vh_mid_u), ("hi", vh_hi, vh_hi_u)):
                d_sim = sim_gr(vh_c[:, 1:], V_0[:, 1:]) - sim_gr(vh_u[:, 1:], V_0[:, 1:])
                acc[f"val/cfg_dsim_{tag_d}"].append(float(d_sim[is_content].mean().item()))

        # VOID F1 на t=0.50
        sim_void = sim_gr(vh_mid[:, 1:], void_dev.view(1, 1, NUM_AXES, AXIS_DIM).expand_as(vh_mid[:, 1:]))
        pred_void = sim_void > 0.70
        true_void = ~is_content
        tp_void += int((pred_void & true_void).sum().item())
        fp_void += int((pred_void & ~true_void).sum().item())
        fn_void += int((~pred_void & true_void).sum().item())

        # Опциональная оценка реальным Декодером Системы 1 (4D вход (1, n_keep, 8, 128) по примеру)
        if decoder is not None and is_content.any():
            dec_dtype = next(decoder.parameters()).dtype
            for tag, vh_cur in (("low", vh_low), ("hi", vh_hi)):
                ce_b, acc_b = [], []
                for b in range(B):
                    n_keep = min(int(i_last[b].item()), 64)
                    if n_keep <= 0:
                        continue
                    c_lat = vh_cur[b : b + 1, 1 : 1 + n_keep]   # (1, n_keep, 8, 128)
                    c_tgt = ans_targets[b, : n_keep * 4]        # (n_keep * 4,) — 4 токена на ячейку
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        logits = decoder(c_lat.to(dec_dtype))   # (1, n_keep * 4, V)
                    logits_f = logits.float().reshape(-1, logits.shape[-1])
                    tgt_f = c_tgt.reshape(-1)
                    valid_tok = (tgt_f != -100) & (tgt_f != PAD_TOKEN_ID)
                    if valid_tok.any():
                        ce_b.append(float(F.cross_entropy(logits_f[valid_tok], tgt_f[valid_tok]).item()))
                        pred_tok = logits_f[valid_tok].argmax(dim=-1)
                        acc_b.append(float((pred_tok == tgt_f[valid_tok]).float().mean().item()))
                if ce_b:
                    acc[f"val/ce_{tag}"].append(float(np.mean(ce_b)))
                    acc[f"val/tok_acc_{tag}"].append(float(np.mean(acc_b)))

    res: Dict[str, float] = {}
    for k, vals in acc.items():
        if vals:
            res[k] = float(np.mean(vals))
    denom_f1 = 2 * tp_void + fp_void + fn_void
    res["val/void_f1"] = float(2 * tp_void / denom_f1) if denom_f1 > 0 else 0.0
    return res


def run_single_variant(
    variant: str,
    args: argparse.Namespace,
    env: AltEnvironmentConfig,
    train_items: Sequence[Dict[str, torch.Tensor]],
    train_schedule: Sequence[Dict[str, Any]],
    val_pack: Dict[str, Any],
    sep_canon: torch.Tensor,
    void_canon: torch.Tensor,
    precomputed_stats: Dict[str, Any],
    decoder: Optional[nn.Module],
) -> Dict[str, Any]:
    t_start = time.time()
    print(f"\n==================== [{variant}] ====================", flush=True)

    # 1. Идентичная базовая инициализация модели для всех вариантов
    torch.manual_seed(args.model_seed)
    model = GrassmannianAltDiffusionCore(
        void_ref=void_canon,
        preset=args.arch_preset,
    ).to(env.device)

    # 2. Применяем конкретную стратегию инициализации (с изолированным RNG)
    t_init0 = time.time()
    init_info = apply_init_strategy(
        model=model,
        strategy_name=variant,
        void_ref=void_canon,
        sep_concept=sep_canon,
        train_items=train_items,
        precomputed_stats=precomputed_stats,
        wiener_damp=args.wiener_damp,
        base_seed=args.init_seed,
    )
    init_sec = time.time() - t_init0

    # 3. Проверяем строгие инварианты сразу после инициализации
    inv = verify_init_invariants(model, sep_concept=sep_canon, device=env.device)
    print(
        f"  [Init OK] {init_sec:.2f}s | sph_err={inv['max_sph_err']:.2e} | "
        f"cfg_leak={inv['cfg_leak_diff']:.2e} | sink_uncond={inv['sink_uncond']:.4f} | info={init_info}",
        flush=True,
    )

    # 4. Замер ДО обучения (step 0) — показывает «врождённые рефлексы» инициализации
    eval_dec = None if args.no_decoder_eval else decoder
    val_step0 = evaluate_fixed_val(
        model=model,
        val_pack=val_pack,
        sep_canon=sep_canon,
        void_canon=void_canon,
        device=env.device,
        use_amp=env.use_amp,
        amp_dtype=env.amp_dtype,
        batch_size=min(args.batch_size * 2, 8),
        decoder=eval_dec,
    )
    print(
        f"  [Step   0 Eval] loss={val_step0['val/loss_total']:.4f} "
        f"(basis={val_step0['val/loss_basis']:.4f}, gr={val_step0['val/loss_gr']:.4f}, en={val_step0['val/loss_energy']:.4f}) | "
        f"SimGr(low/mid/hi)={val_step0['val/sim_gr_low']:.3f}/{val_step0['val/sim_gr_mid']:.3f}/{val_step0['val/sim_gr_hi']:.3f} | "
        f"CFG(mid)={val_step0['val/cfg_mid']:.4f} | "
        f"TokAcc(low/hi)={val_step0.get('val/tok_acc_low', -1.0):.3f}/{val_step0.get('val/tok_acc_hi', -1.0):.3f}",
        flush=True,
    )

    # 5. Обучение по строго фиксированному расписанию батчей и шумов
    opt_muon, opt_adamw = create_alt_optimizers(
        model,
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
        scale_1d_lr=args.scale_1d_lr,
    )
    sep_dev = sep_canon.to(device=env.device, dtype=torch.float32)
    in_dtype = env.amp_dtype if env.use_amp else torch.float32
    scaler = (
        torch.amp.GradScaler("cuda", init_scale=16384.0, growth_interval=1000)
        if env.use_grad_scaler
        else None
    )

    history: List[Dict[str, float]] = []
    val_curve: Dict[int, Dict[str, float]] = {0: val_step0}
    step1_zero_grads = -1
    step1_grad_norm = 0.0

    model.train()
    for step_idx, sched in enumerate(train_schedule, start=1):
        t0 = time.time()
        lr_mult = update_optimizers_lr(
            opt_muon, opt_adamw, step_idx, args.steps, args.lr_warmup_steps, args.muon_lr
        )
        if opt_muon is not None:
            opt_muon.zero_grad(set_to_none=True)
        opt_adamw.zero_grad(set_to_none=True)

        batch = _collate_items(train_items, sched["indices"])
        B = len(sched["indices"])
        V_prompt = batch["V_prompt"].to(env.device)
        V_ans = batch["V_ans"].to(env.device)
        n_real_prompt = batch["n_real_prompt"].to(env.device)
        i_last = batch["i_last"].to(env.device)
        w_cell = batch["w_cell"].to(env.device)
        p_valid = n_real_prompt > 0
        p_attn_mask = sched["p_attn_mask"].to(env.device)
        m_prompt = sched["m_prompt"].to(env.device)
        t_mat = sched["t_mat"].to(env.device)
        noise = sched["noise_fp16"].to(device=env.device, dtype=torch.float32)

        sep_exp = sep_dev.view(1, 1, NUM_AXES, AXIS_DIM).expand(B, 1, NUM_AXES, AXIS_DIM)
        V_0 = torch.cat([sep_exp, V_ans], dim=1)
        V_t = (1.0 - t_mat.unsqueeze(-1).unsqueeze(-1)) * V_0 + t_mat.unsqueeze(-1).unsqueeze(-1) * noise
        V_t[:, 0] = sep_exp[:, 0]

        with torch.autocast(device_type=env.device.type, dtype=env.amp_dtype, enabled=env.use_amp):
            V_0_hat, adaln_stats, sink_weights = model(
                V_t=V_t.to(in_dtype),
                t_mat=t_mat,
                V_prompt=V_prompt.to(in_dtype),
                prompt_valid_mask=p_valid,
                prompt_attn_mask=p_attn_mask,
                m_prompt=m_prompt,
                radar_sigma=None,
                return_sink_weights=True,
            )

        loss, m_dict = compute_phase3_loss(
            V_0_hat=V_0_hat,
            V_0=V_0,
            t_mat=t_mat,
            w_cell=w_cell,
            i_last=i_last,
            adaln_stats=adaln_stats,
            step=step_idx,
            m_prompt=m_prompt,
            sink_weights=sink_weights,
            world_size=1,
        )

        if scaler is not None:
            scaler.scale(loss).backward()
            if opt_muon is not None:
                scaler.unscale_(opt_muon)
            scaler.unscale_(opt_adamw)
        else:
            loss.backward()

        if step_idx == 1:
            step1_zero_grads = sum(
                1
                for p in model.parameters()
                if p.requires_grad and (p.grad is None or float(p.grad.abs().max().item()) == 0.0)
            )

        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item())
        if step_idx == 1:
            step1_grad_norm = grad_norm

        if scaler is not None:
            if opt_muon is not None:
                scaler.step(opt_muon)
            scaler.step(opt_adamw)
            scaler.update()
        else:
            if opt_muon is not None:
                opt_muon.step()
            opt_adamw.step()

        dt = time.time() - t0
        row = {
            "step": step_idx,
            "loss_total": float(m_dict["loss/total"]),
            "loss_basis": float(m_dict["loss/basis"]),
            "loss_gr": float(m_dict["loss/gr"]),
            "loss_energy": float(m_dict["loss/energy"]),
            "loss_sink": float(m_dict["loss/sink"]),
            "grad_norm": grad_norm,
            "step_sec": dt,
        }
        history.append(row)

        if step_idx == 1 or step_idx % args.log_every == 0 or step_idx == args.steps:
            print(
                f"  [{variant} | step {step_idx:3d}/{args.steps}] "
                f"tot={row['loss_total']:.4f} | basis={row['loss_basis']:.4f} | "
                f"gr={row['loss_gr']:.4f} | en={row['loss_energy']:.4f} | "
                f"gnorm={grad_norm:.3f} | lr_m={lr_mult:.3f} | {dt:.2f}s",
                flush=True,
            )

        if args.eval_every > 0 and (step_idx % args.eval_every == 0) and step_idx < args.steps:
            v_mid = evaluate_fixed_val(
                model=model,
                val_pack=val_pack,
                sep_canon=sep_canon,
                void_canon=void_canon,
                device=env.device,
                use_amp=env.use_amp,
                amp_dtype=env.amp_dtype,
                batch_size=min(args.batch_size * 2, 8),
                decoder=None,  # на промежуточных точках только быстрые латентные метрики
            )
            val_curve[step_idx] = v_mid
            model.train()

    # 6. Финальная валидация (включая реальный Декодер Системы 1)
    val_final = evaluate_fixed_val(
        model=model,
        val_pack=val_pack,
        sep_canon=sep_canon,
        void_canon=void_canon,
        device=env.device,
        use_amp=env.use_amp,
        amp_dtype=env.amp_dtype,
        batch_size=min(args.batch_size * 2, 8),
        decoder=eval_dec,
    )
    val_curve[args.steps] = val_final
    dca_final = model.get_dca_summary()

    k_win = min(5, len(history))
    train_loss_first = float(np.mean([r["loss_total"] for r in history[:k_win]]))
    train_loss_last = float(np.mean([r["loss_total"] for r in history[-k_win:]]))
    train_loss_auc = float(np.mean([r["loss_total"] for r in history]))
    avg_step_sec = float(np.mean([r["step_sec"] for r in history]))
    total_sec = time.time() - t_start

    print(
        f"  [Final @{args.steps:3d} Eval] loss={val_final['val/loss_total']:.4f} "
        f"(basis={val_final['val/loss_basis']:.4f}, gr={val_final['val/loss_gr']:.4f}, en={val_final['val/loss_energy']:.4f}) | "
        f"SimGr(low/mid/hi)={val_final['val/sim_gr_low']:.3f}/{val_final['val/sim_gr_mid']:.3f}/{val_final['val/sim_gr_hi']:.3f} | "
        f"CFG(mid)={val_final['val/cfg_mid']:.4f} | "
        f"TokAcc(low/hi)={val_final.get('val/tok_acc_low', -1.0):.3f}/{val_final.get('val/tok_acc_hi', -1.0):.3f} | "
        f"zero_grad_s1={step1_zero_grads} | total={total_sec:.1f}s",
        flush=True,
    )

    return {
        "variant": variant,
        "init_sec": init_sec,
        "init_info": init_info,
        "invariants": inv,
        "step1_zero_grads": step1_zero_grads,
        "step1_grad_norm": step1_grad_norm,
        "train_loss_first5": train_loss_first,
        "train_loss_last5": train_loss_last,
        "train_loss_auc": train_loss_auc,
        "avg_step_sec": avg_step_sec,
        "total_sec": total_sec,
        "dca_final": dca_final,
        "val_step0": val_step0,
        "val_final": val_final,
        "val_curve": {str(k): v for k, v in val_curve.items()},
        "history": history,
    }


def render_markdown_report(
    results: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
    model_params: int,
) -> str:
    base = results.get("baseline")

    def _delta(cur: float, ref: Optional[float], lower_is_better: bool = True) -> str:
        if ref is None:
            return ""
        d = cur - ref
        good = (d < 0) if lower_is_better else (d > 0)
        mark = "+" if d >= 0 else ""
        tag = " ✓" if (abs(d) > 1e-4 and good) else ""
        return f" ({mark}{d:.4f}{tag})"

    lines: List[str] = [
        f"# Абляция 5 идей инициализации — Phase 3 Alt (`{args.arch_preset}`, {model_params/1e6:.2f}M)",
        "",
        f"- **Шагов обучения:** `{args.steps}` | **Batch:** `{args.batch_size}` | **Train/Val кэш:** `{args.train_samples}/{args.val_samples}`",
        f"- **Оптимизатор:** `OrthMuon(lr={args.muon_lr}) + AdamW(lr={args.adamw_lr}, scale_1d={args.scale_1d_lr})`, warmup=`{args.lr_warmup_steps}`",
        f"- **Контроль равенства:** фиксированные батчи, фиксированные `(t_mat, eps, CFG mask)` на каждом шаге и фиксированная валидация.",
        "",
        "## 1. Мгновенный эффект на шаге 0 (ДО единого шага градиентного спуска)",
        "",
        "| Вариант | Val Loss (t=0.5) | Val Basis | Val Gr | Val Energy | SimGr low/mid/hi | CFG mid | TokAcc low/hi |",
        "|---|---:|---:|---:|---:|---|---:|---|",
    ]
    for v_name in ABLATION_VARIANTS:
        if v_name not in results:
            continue
        r = results[v_name]
        s0 = r["val_step0"]
        b0 = base["val_step0"] if base else None
        tok_s = f"{s0.get('val/tok_acc_low', 0.0):.3f} / {s0.get('val/tok_acc_hi', 0.0):.3f}" if "val/tok_acc_low" in s0 else "—"
        lines.append(
            f"| `{v_name}` | "
            f"{s0['val/loss_total']:.4f}{_delta(s0['val/loss_total'], b0['val/loss_total'] if b0 else None, True)} | "
            f"{s0['val/loss_basis']:.4f} | {s0['val/loss_gr']:.4f} | {s0['val/loss_energy']:.4f} | "
            f"{s0['val/sim_gr_low']:.3f} / {s0['val/sim_gr_mid']:.3f} / {s0['val/sim_gr_hi']:.3f} | "
            f"{s0['val/cfg_mid']:.4f} | {tok_s} |"
        )

    lines += [
        "",
        f"## 2. Итоговые метрики после {args.steps} шагов обучения",
        "",
        "| Вариант | Val Loss (t=0.5) | Train AUC | SimGr low/mid/hi | CFG mid/hi | CFG-ΔSim mid/hi | VOID F1 | TokAcc low/hi | CE low/hi | GradNorm s1 |",
        "|---|---:|---:|---|---|---|---:|---|---|---:|",
    ]
    for v_name in ABLATION_VARIANTS:
        if v_name not in results:
            continue
        r = results[v_name]
        sf = r["val_final"]
        bf = base["val_final"] if base else None
        b_auc = base["train_loss_auc"] if base else None
        tok_s = f"{sf.get('val/tok_acc_low', 0.0):.3f} / {sf.get('val/tok_acc_hi', 0.0):.3f}" if "val/tok_acc_low" in sf else "—"
        ce_s = f"{sf.get('val/ce_low', 0.0):.2f} / {sf.get('val/ce_hi', 0.0):.2f}" if "val/ce_low" in sf else "—"
        dsim_s = (
            f"{sf['val/cfg_dsim_mid']:+.4f} / {sf.get('val/cfg_dsim_hi', 0.0):+.4f}" if "val/cfg_dsim_mid" in sf else "—"
        )
        lines.append(
            f"| `{v_name}` | "
            f"{sf['val/loss_total']:.4f}{_delta(sf['val/loss_total'], bf['val/loss_total'] if bf else None, True)} | "
            f"{r['train_loss_auc']:.4f}{_delta(r['train_loss_auc'], b_auc, True)} | "
            f"{sf['val/sim_gr_low']:.3f} / {sf['val/sim_gr_mid']:.3f} / {sf['val/sim_gr_hi']:.3f} | "
            f"{sf['val/cfg_mid']:.4f} / {sf['val/cfg_hi']:.4f} | {dsim_s} | "
            f"{sf['val/void_f1']:.3f} | {tok_s} | {ce_s} | {r['step1_grad_norm']:.2f} |"
        )

    # Рейтинг по финальному Val Loss и Train AUC
    ranked = sorted(results.values(), key=lambda x: x["val_final"]["val/loss_total"])
    lines += [
        "",
        "## 3. Рейтинг по финальному `val/loss_total` (ниже — лучше)",
        "",
    ]
    for pos, r in enumerate(ranked, start=1):
        sf = r["val_final"]
        delta_s = ""
        if base is not None:
            d = sf["val/loss_total"] - base["val_final"]["val/loss_total"]
            delta_s = f" (vs baseline: {d:+.4f})"
        lines.append(
            f"{pos}. **`{r['variant']}`** — `val/loss={sf['val/loss_total']:.4f}`{delta_s}, "
            f"`Train AUC={r['train_loss_auc']:.4f}`, `SimGr_mid={sf['val/sim_gr_mid']:.4f}`, "
            f"`CFG_mid={sf['val/cfg_mid']:.4f}`, `init={r['init_sec']:.2f}s`"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Абляционный эксперимент 5 идей инициализации для Phase 3 Alt (tiny)"
    )
    parser.add_argument("--smoke-test", "--smoke_test", dest="smoke_test", action="store_true", help="Быстрая проверка всех 8 вариантов (2 шага, 8 train / 4 val)")
    parser.add_argument("--arch-preset", "--arch_preset", dest="arch_preset", type=str, default="tiny", choices=list(ARCH_PRESETS.keys()))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--cpu-bf16", "--cpu_bf16", dest="cpu_bf16", action="store_true")
    parser.add_argument("--num-threads", "--num_threads", dest="num_threads", type=int, default=0)

    parser.add_argument("--variants", type=str, default="all", help="Список через запятую или 'all'")
    parser.add_argument("--steps", type=int, default=40, help="Число шагов обучения на каждый вариант")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=4)
    parser.add_argument("--train-samples", "--train_samples", dest="train_samples", type=int, default=128)
    parser.add_argument("--val-samples", "--val_samples", dest="val_samples", type=int, default=32)
    parser.add_argument("--eval-every", "--eval_every", dest="eval_every", type=int, default=10)
    parser.add_argument("--log-every", "--log_every", dest="log_every", type=int, default=5)
    parser.add_argument("--no-decoder-eval", "--no_decoder_eval", dest="no_decoder_eval", action="store_true", help="Отключить оценку через замороженный Декодер (ускоряет val)")
    parser.add_argument("--no-fast-vocab", "--no_fast_vocab", dest="no_fast_vocab", action="store_true", help="Отключить FastVocabOptimizer на декодере")
    parser.add_argument("--wiener-damp", "--wiener_damp", dest="wiener_damp", type=float, default=1.0, help="Демпфирование коэффициентов Винера beta_k (например 0.5)")

    parser.add_argument("--muon-lr", "--muon_lr", dest="muon_lr", type=float, default=1e-3)
    parser.add_argument("--adamw-lr", "--adamw_lr", dest="adamw_lr", type=float, default=3e-4)
    parser.add_argument("--scale-1d-lr", "--scale_1d_lr", dest="scale_1d_lr", type=float, default=1.2e-3)
    parser.add_argument("--lr-warmup-steps", "--lr_warmup_steps", dest="lr_warmup_steps", type=int, default=10)

    parser.add_argument("--model-seed", "--model_seed", dest="model_seed", type=int, default=42)
    parser.add_argument("--init-seed", "--init_seed", dest="init_seed", type=int, default=1000)
    parser.add_argument("--schedule-seed", "--schedule_seed", dest="schedule_seed", type=int, default=2026)

    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default="")
    parser.add_argument("--cache-dir", "--cache_dir", dest="cache_dir", type=str, default="")
    parser.add_argument("--resume", action="store_true", help="Подгрузить уже завершенные варианты из ablation_results.json и пропустить их")
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    if args.smoke_test:
        print("=== SMOKE TEST: АБЛЯЦИЯ 5 ИДЕЙ ИНИЦИАЛИЗАЦИИ (8 ВАРИАНТОВ) ===", flush=True)
        args.steps = 2
        args.batch_size = 2
        args.train_samples = 8
        args.val_samples = 4
        args.eval_every = 2
        args.log_every = 1
        args.lr_warmup_steps = 1

    if args.variants.strip().lower() == "all":
        selected_variants = [v for v in ABLATION_VARIANTS if v != "compat"]
    else:
        selected_variants = [v.strip() for v in args.variants.split(",") if v.strip()]
        for v in selected_variants:
            if v not in ABLATION_VARIANTS:
                raise ValueError(f"Неизвестный вариант '{v}'. Доступны: {ABLATION_VARIANTS}")

    out_dir = args.output_dir or str(ALT_DIR / "output" / f"init_ablation_{args.arch_preset}")
    cache_dir = args.cache_dir or str(ALT_DIR / "output")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    env = AltEnvironmentConfig.detect(
        override_device=args.device,
        override_output_dir=out_dir,
        override_checkpoint_dir=out_dir,
        cpu_bf16=args.cpu_bf16,
    )

    print(
        f"=== Ablation Setup: preset={args.arch_preset}, device={env.device}, "
        f"steps={args.steps}, batch={args.batch_size}, train={args.train_samples}, val={args.val_samples}, "
        f"variants={selected_variants} ===",
        flush=True,
    )

    # 1. Загружаем токенизатор, замороженные Энкодер/Декодер и опорные концепты
    tokenizer = get_tokenizer()
    encoder = get_encoder(weights_path=env.encoder_weights_path, device=env.device, eval_mode=True)
    decoder = (
        None
        if args.no_decoder_eval
        else get_decoder(weights_path=env.decoder_weights_path, encoder=encoder, device=env.device, eval_mode=True)
    )
    for p in list(encoder.parameters()) + (list(decoder.parameters()) if decoder is not None else []):
        p.requires_grad_(False)

    if decoder is not None and not args.no_fast_vocab:
        fv_dtype = env.amp_dtype if env.use_amp else torch.float32
        FastVocabOptimizer.from_decoder(decoder=decoder, gemm_dtype=fv_dtype, attach=True)

    sep_concept = torch.load(env.sep_concept_path, map_location="cpu", weights_only=False)
    void_concept = torch.load(env.void_concept_path, map_location="cpu", weights_only=False)
    sep_canon, _ = canonicalize_cell_basis(sep_concept.float())
    void_canon, _ = canonicalize_cell_basis(void_concept.float())

    if os.path.exists(env.stopwords_path):
        sw_obj = torch.load(env.stopwords_path, map_location="cpu", weights_only=False)
        stopword_mask = (sw_obj["mask"] if isinstance(sw_obj, dict) else sw_obj).bool()
    else:
        stopword_mask = torch.zeros(131072, dtype=torch.bool)

    # 2. Единый кэш латентов (train + val) — энкодер вызывается ровно 1 раз
    train_cache_path = os.path.join(cache_dir, f"ablation_train_cache_{args.train_samples}.pt")
    val_cache_path = os.path.join(cache_dir, f"ablation_val_cache_{args.val_samples}.pt")

    raw_train = PromptAnswerDataset(
        parquet_path=env.train_data_path,
        tokenizer=tokenizer,
        stopword_mask=stopword_mask,
        max_rows=args.train_samples,
        is_validation=False,
        elastic_drop_prob=0.0,
    )
    train_ds = build_or_load_latent_cache(
        raw_dataset=raw_train,
        encoder=encoder,
        stopword_mask=stopword_mask,
        device=env.device,
        use_amp=env.use_amp,
        amp_dtype=env.amp_dtype,
        num_samples=args.train_samples,
        cache_path=train_cache_path,
    )

    raw_val = PromptAnswerDataset(
        parquet_path=env.val_data_path,
        tokenizer=tokenizer,
        stopword_mask=stopword_mask,
        max_rows=args.val_samples,
        is_validation=True,
        elastic_drop_prob=0.0,
    )
    val_ds = build_or_load_latent_cache(
        raw_dataset=raw_val,
        encoder=encoder,
        stopword_mask=stopword_mask,
        device=env.device,
        use_amp=env.use_amp,
        amp_dtype=env.amp_dtype,
        num_samples=args.val_samples,
        cache_path=val_cache_path,
    )

    # Освобождаем память энкодера после предкодирования
    del encoder
    if env.device.type == "cuda":
        torch.cuda.empty_cache()

    train_items = train_ds.items
    val_items = val_ds.items

    # 3. Предвычисляем data-driven статистики 1 раз (переиспользуются в idea4, idea5 и all_5)
    precomputed_stats: Dict[str, Any] = {}
    if any(v in ("idea4_wiener_head", "all_5", "compat") for v in selected_variants):
        print("[Prep] Вычисление поосевых Винеровских коэффициентов beta_0..7 по train-кэшу...", flush=True)
        precomputed_stats["wiener_info"] = compute_wiener_axis_betas(
            train_items, sep_concept=sep_canon, seed=args.init_seed + 4
        )
        print(
            f"  -> axis_rms={[round(x, 4) for x in precomputed_stats['wiener_info']['axis_rms'].tolist()]}\n"
            f"  -> wiener_betas={[round(x, 4) for x in precomputed_stats['wiener_info']['betas'].tolist()]}",
            flush=True,
        )
    if any(v in ("idea5_kmeans_ffn", "all_5", "compat") for v in selected_variants):
        print("[Prep] Подготовка нормированных пулов для Spherical K-Means FFN...", flush=True)
        precomputed_stats["kmeans_pools"] = build_kmeans_pools(train_items, seed=args.init_seed + 5)
        print(
            f"  -> cells_pool={tuple(precomputed_stats['kmeans_pools']['cells_unit'].shape)}, "
            f"vecs_pool={tuple(precomputed_stats['kmeans_pools']['vecs_unit'].shape)}",
            flush=True,
        )

    # 4. Фиксированные расписания train и val (гарантия бит-в-бит равных условий)
    train_schedule = build_fixed_train_schedule(
        train_items=train_items,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.schedule_seed,
    )
    val_pack = build_fixed_val_pack(val_items=val_items, seed=args.schedule_seed + 777)

    # Размер модели для заголовка отчета
    tmp_m = GrassmannianAltDiffusionCore(void_ref=void_canon, preset=args.arch_preset)
    model_params = sum(p.numel() for p in tmp_m.parameters() if p.requires_grad)
    del tmp_m

    results_json_path = os.path.join(out_dir, "ablation_results.json")
    report_md_path = os.path.join(out_dir, "ablation_report.md")
    results: Dict[str, Dict[str, Any]] = {}
    if args.resume and os.path.exists(results_json_path):
        with open(results_json_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"[Resume] Загружены готовые варианты: {list(results.keys())}", flush=True)

    # 5. Последовательный прогон всех выбранных вариантов
    for variant in selected_variants:
        if args.resume and variant in results:
            print(f"[Skip] Вариант '{variant}' уже есть в {results_json_path}", flush=True)
            continue
        res = run_single_variant(
            variant=variant,
            args=args,
            env=env,
            train_items=train_items,
            train_schedule=train_schedule,
            val_pack=val_pack,
            sep_canon=sep_canon,
            void_canon=void_canon,
            precomputed_stats=precomputed_stats,
            decoder=decoder,
        )
        results[variant] = res

        # Инкрементально сохраняем JSON и Markdown после каждого варианта
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        report_md = render_markdown_report(results, args=args, model_params=model_params)
        with open(report_md_path, "w", encoding="utf-8") as f:
            f.write(report_md)

    print("\n" + "=" * 72, flush=True)
    print(render_markdown_report(results, args=args, model_params=model_params), flush=True)
    print(f"Отчет сохранен: {report_md_path}", flush=True)
    print(f"JSON сохранен:  {results_json_path}", flush=True)


if __name__ == "__main__":
    main()
