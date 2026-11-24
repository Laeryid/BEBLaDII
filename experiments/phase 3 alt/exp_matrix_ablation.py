"""
Абляционный эксперимент по 3 геометрически-согласованным матричным оптимизациям
(PHM в FullSA, Грассманиановый скор ||Q K^T||_F^2 + Procrustes-роутер, билинейный шаг L*X*R)
со встроенным TPU/GPU-оптимизатором словаря `FastVocabOptimizer`.

Сравнивает 7 конфигураций на бит-в-бит идентичных батчах, шумах (t_mat, eps, CFG) и валидации:
  1. ref_dense         — текущий `tiny` (плотные FullSA + GrassFull)
  2. opt1_phm_attn     — PHM (r=4) на Q/KV/O в FullSA (FFN плотный)
  3. opt1_phm_full     — PHM (r=4) на Q/KV/O и SwiGLU FFN в FullSA (~3x меньше весов!)
  4. opt2_grass_align  — GrassAlignSABlock (128x128 softmax + Procrustes M_ij) вместо GrassFull
  5. opt3_bilinear_lxr — Билинейный шаг L*X*R (8x8) во всех GrassSA
  6. all_matrix_safe   — opt1_phm_attn + opt2_grass_align + opt3_bilinear_lxr
  7. all_matrix_ultra  — opt1_phm_full + opt2_grass_align + opt3_bilinear_lxr

База инициализации по умолчанию: `--init-base compat --wiener-damp 0.5`
(единая для всех вариантов часть `all_5`: idea1 + idea2 + idea4*0.5 + idea5_grass,
плюс родная блочно-диагональная геометрия для плотных/PHM FullSA).

Примеры запуска:
  # Быстрая проверка всех 7 вариантов (smoke-test):
  .venv\\Scripts\\python.exe -B "experiments/phase 3 alt/exp_matrix_ablation.py" --smoke-test

  # Основной эксперимент на готовом кэше 128/32 (40 шагов, batch=4):
  .venv\\Scripts\\python.exe -B "experiments/phase 3 alt/exp_matrix_ablation.py" --steps 40 --batch-size 4

  # С включённым быстрым CE-лоссом и BF16-радаром на обучении:
  .venv\\Scripts\\python.exe -B "experiments/phase 3 alt/exp_matrix_ablation.py" --steps 40 --batch-size 4 --use-radar --ce-weight 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

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

from exp_init_ablation import (
    _collate_items,
    build_fixed_train_schedule,
    build_fixed_val_pack,
    evaluate_fixed_val,
)
from init_strategies import (
    apply_init_strategy,
    build_kmeans_pools,
    compute_wiener_axis_betas,
    verify_init_invariants,
)
from matrix_layers import (
    MATRIX_VARIANTS,
    build_matrix_alt_core,
    create_matrix_optimizers,
)
from train_phase3 import compute_decoder_ce_loss, compute_radar_sigma
from train_phase3_alt import (
    ARCH_PRESETS,
    AXIS_DIM,
    AltEnvironmentConfig,
    NUM_AXES,
    PromptAnswerDataset,
    build_or_load_latent_cache,
    compute_phase3_loss,
    update_optimizers_lr,
)
from vocab_optim import FastVocabOptimizer


def run_single_matrix_variant(
    variant: str,
    args: argparse.Namespace,
    env: AltEnvironmentConfig,
    train_items: Sequence[Dict[str, torch.Tensor]],
    train_schedule: Sequence[Dict[str, Any]],
    val_pack: Dict[str, Any],
    sep_canon: torch.Tensor,
    void_canon: torch.Tensor,
    stopword_mask: torch.Tensor,
    precomputed_stats: Dict[str, Any],
    decoder: Optional[nn.Module],
    fast_vocab: Optional[FastVocabOptimizer],
) -> Dict[str, Any]:
    print(f"\n==================== [{variant}] ====================", flush=True)

    # 1. Фиксированный seed создания модели
    torch.manual_seed(args.model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.model_seed)
    model = build_matrix_alt_core(
        void_ref=void_canon,
        variant=variant,
        preset=args.arch_preset,
        phm_rank=args.phm_rank,
        cross_alpha=args.phm_alpha,
    ).to(env.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 2. Базовая инициализация (compat / baseline / all_5)
    t_init0 = time.time()
    init_info = apply_init_strategy(
        model=model,
        strategy_name=args.init_base,
        void_ref=void_canon,
        sep_concept=sep_canon,
        train_items=train_items,
        precomputed_stats=precomputed_stats,
        wiener_damp=args.wiener_damp,
        base_seed=args.init_seed,
    )
    init_sec = time.time() - t_init0

    # 3. Проверка строгих инвариантов (сферичность, нулевая утечка CFG, SEP t=0)
    inv = verify_init_invariants(model, sep_concept=sep_canon, device=env.device)
    print(
        f"  [Model] params={n_params / 1e6:.2f}M | [Init={args.init_base}, damp={args.wiener_damp}] {init_sec:.2f}s | "
        f"sph_err={inv['max_sph_err']:.2e} | cfg_leak={inv['cfg_leak_diff']:.2e} | sink_uncond={inv['sink_uncond']:.4f}",
        flush=True,
    )

    # 4. Оценка ДО обучения (step 0)
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

    # 5. Обучение по строго фиксированному расписанию
    opt_muon, opt_adamw = create_matrix_optimizers(
        model,
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
        scale_1d_lr=args.scale_1d_lr,
    )
    sep_dev = sep_canon.to(device=env.device, dtype=torch.float32)
    sw_dev = stopword_mask.to(env.device)
    in_dtype = env.amp_dtype if env.use_amp else torch.float32
    scaler = (
        torch.amp.GradScaler("cuda", init_scale=16384.0, growth_interval=1000)
        if env.use_grad_scaler
        else None
    )

    history: List[Dict[str, float]] = []
    val_curve: Dict[int, Dict[str, float]] = {0: val_step0}
    step_times: List[float] = []
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
        ans_targets = batch["ans_targets"].to(env.device)
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

        radar_sigma = None
        if args.use_radar and decoder is not None:
            with torch.no_grad():
                radar_sigma = compute_radar_sigma(
                    decoder, V_t, use_amp=env.use_amp, amp_dtype=env.amp_dtype
                )

        with torch.autocast(device_type=env.device.type, dtype=env.amp_dtype, enabled=env.use_amp):
            V_0_hat, adaln_stats, sink_weights = model(
                V_t=V_t.to(in_dtype),
                t_mat=t_mat,
                V_prompt=V_prompt.to(in_dtype),
                prompt_valid_mask=p_valid,
                prompt_attn_mask=p_attn_mask,
                m_prompt=m_prompt,
                radar_sigma=radar_sigma,
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

        ce_val = 0.0
        if decoder is not None and float(args.ce_weight) > 0.0:
            if fast_vocab is not None:
                l_ce, _ = fast_vocab.compute_decoder_ce_loss(
                    decoder=decoder,
                    V_0_hat=V_0_hat,
                    t_mat=t_mat,
                    ans_targets=ans_targets,
                    ce_max_tokens=args.ce_max_tokens,
                    ce_t_max=args.ce_t_max,
                    stopword_mask=sw_dev,
                    use_amp=env.use_amp,
                    amp_dtype=env.amp_dtype,
                    ce_batch_fraction=args.ce_batch_fraction,
                )
            else:
                l_ce, _ = compute_decoder_ce_loss(
                    decoder=decoder,
                    V_0_hat=V_0_hat,
                    t_mat=t_mat,
                    ans_targets=ans_targets,
                    ce_max_tokens=args.ce_max_tokens,
                    ce_t_max=args.ce_t_max,
                    stopword_mask=sw_dev,
                    use_amp=env.use_amp,
                    amp_dtype=env.amp_dtype,
                )
            loss = loss + float(args.ce_weight) * l_ce
            ce_val = float(l_ce.item())
            m_dict["loss/total"] = float(loss.item())

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

        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        if step_idx == 1:
            step1_grad_norm = gnorm

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
        step_times.append(dt)
        rec = {
            "step": step_idx,
            "loss_total": float(m_dict["loss/total"]),
            "loss_basis": float(m_dict["loss/basis"]),
            "loss_gr": float(m_dict["loss/gr"]),
            "loss_energy": float(m_dict["loss/energy"]),
            "loss_ce": ce_val,
            "grad_norm": gnorm,
            "lr_mult": float(lr_mult),
            "dt_sec": dt,
        }
        history.append(rec)

        if step_idx == 1 or step_idx % args.log_every == 0 or step_idx == args.steps:
            ce_str = f" | ce={ce_val:.3f}" if float(args.ce_weight) > 0.0 else ""
            print(
                f"  [{variant} | step {step_idx:3d}/{args.steps}] "
                f"tot={rec['loss_total']:.4f} | basis={rec['loss_basis']:.4f} | "
                f"gr={rec['loss_gr']:.4f} | en={rec['loss_energy']:.4f}{ce_str} | "
                f"gnorm={gnorm:.3f} | {dt:.2f}s",
                flush=True,
            )

        if step_idx < args.steps and args.eval_every > 0 and (step_idx % args.eval_every == 0):
            val_mid = evaluate_fixed_val(
                model=model,
                val_pack=val_pack,
                sep_canon=sep_canon,
                void_canon=void_canon,
                device=env.device,
                use_amp=env.use_amp,
                amp_dtype=env.amp_dtype,
                batch_size=min(args.batch_size * 2, 8),
                decoder=None,
            )
            val_curve[step_idx] = val_mid
            model.train()

    # 6. Финальная полная валидация (включая реальный Декодер)
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

    # Исключаем шаг 1 (холодный прогрев) из медианы при steps >= 3
    steady_times = step_times[1:] if len(step_times) >= 3 else step_times
    med_step_sec = float(np.median(steady_times))
    mean_step_sec = float(np.mean(steady_times))
    total_train_sec = float(sum(step_times))
    train_auc = float(np.mean([h["loss_total"] for h in history]))

    print(
        f"  [Final @ {args.steps} Eval] params={n_params / 1e6:.2f}M | med_step={med_step_sec:.2f}s | "
        f"loss={val_final['val/loss_total']:.4f} "
        f"(basis={val_final['val/loss_basis']:.4f}, gr={val_final['val/loss_gr']:.4f}, en={val_final['val/loss_energy']:.4f}) | "
        f"SimGr(low/mid/hi)={val_final['val/sim_gr_low']:.3f}/{val_final['val/sim_gr_mid']:.3f}/{val_final['val/sim_gr_hi']:.3f} | "
        f"CFG(mid)={val_final['val/cfg_mid']:.4f} | "
        f"TokAcc(low/hi)={val_final.get('val/tok_acc_low', -1.0):.3f}/{val_final.get('val/tok_acc_hi', -1.0):.3f} | "
        f"zero_grad_s1={step1_zero_grads}",
        flush=True,
    )

    return {
        "variant": variant,
        "params": int(n_params),
        "init_base": args.init_base,
        "init_info": init_info,
        "init_sec": init_sec,
        "invariants": inv,
        "step1_zero_grads": step1_zero_grads,
        "step1_grad_norm": step1_grad_norm,
        "med_step_sec": med_step_sec,
        "mean_step_sec": mean_step_sec,
        "total_train_sec": total_train_sec,
        "train_loss_auc": train_auc,
        "val_step0": val_step0,
        "val_final": val_final,
        "val_curve": {str(k): v for k, v in val_curve.items()},
        "history": history,
    }


def render_matrix_report(
    results: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    base = results.get("ref_dense")
    ordered = [results[v] for v in MATRIX_VARIANTS if v in results]

    def _delta(val: float, base_val: Optional[float], lower_is_better: bool = True) -> str:
        if base_val is None:
            return ""
        d = val - base_val
        good = (d < 0) if lower_is_better else (d > 0)
        mark = " ✓" if good and abs(d) > 1e-4 else ""
        return f" ({d:+.4f}{mark})"

    lines: List[str] = [
        f"# Абляция 3 матричных оптимизаций — Phase 3 Alt (`{args.arch_preset}`)",
        "",
        f"- **Шагов:** `{args.steps}` | **Batch:** `{args.batch_size}` | **Train/Val кэш:** `{args.train_samples}/{args.val_samples}`",
        f"- **База инициализации:** `{args.init_base}` (`wiener_damp={args.wiener_damp}`) | **PHM rank:** `{args.phm_rank}` (`alpha={args.phm_alpha}`)",
        f"- **FastVocab:** `{'OFF' if args.no_fast_vocab else 'ON'}` | **Train Radar:** `{'ON' if args.use_radar else 'OFF'}` | **Train CE weight:** `{args.ce_weight}`",
        "",
        f"## 1. Скорость, параметры и качество после {args.steps} шагов",
        "",
        "| Вариант | Params | Мед. шаг (Speedup) | Val Loss (t=0.5) | Train AUC | SimGr low/mid/hi | CFG mid/hi | VOID F1 | TokAcc low/hi | CE low |",
        "|---|---:|---|---:|---:|---|---|---:|---|---:|",
    ]
    for r in ordered:
        sf = r["val_final"]
        p_m = r["params"] / 1e6
        p_ratio = f" ({base['params'] / r['params']:.2f}×)" if base is not None else ""
        sp_str = f" ({base['med_step_sec'] / max(r['med_step_sec'], 1e-6):.2f}×)" if base is not None else ""
        b_loss = base["val_final"]["val/loss_total"] if base else None
        b_auc = base["train_loss_auc"] if base else None
        tok_s = (
            f"{sf['val/tok_acc_low']:.3f} / {sf['val/tok_acc_hi']:.3f}"
            if "val/tok_acc_low" in sf
            else "n/a"
        )
        ce_s = f"{sf['val/ce_low']:.2f}" if "val/ce_low" in sf else "n/a"
        lines.append(
            f"| `{r['variant']}` | {p_m:.2f}M{p_ratio} | {r['med_step_sec']:.2f}s{sp_str} | "
            f"{sf['val/loss_total']:.4f}{_delta(sf['val/loss_total'], b_loss, True)} | "
            f"{r['train_loss_auc']:.4f}{_delta(r['train_loss_auc'], b_auc, True)} | "
            f"{sf['val/sim_gr_low']:.3f} / {sf['val/sim_gr_mid']:.3f} / {sf['val/sim_gr_hi']:.3f} | "
            f"{sf['val/cfg_mid']:.4f} / {sf['val/cfg_hi']:.4f} | "
            f"{sf['val/void_f1']:.3f} | {tok_s} | {ce_s} |"
        )

    ranked = sorted(ordered, key=lambda r: r["val_final"]["val/loss_total"])
    lines += [
        "",
        "## 2. Рейтинг по финальному `val/loss_total` (ниже — лучше)",
        "",
    ]
    for pos, r in enumerate(ranked, start=1):
        sf = r["val_final"]
        d_s = f" (vs ref: {sf['val/loss_total'] - base['val_final']['val/loss_total']:+.4f})" if base else ""
        sp = f"{base['med_step_sec'] / max(r['med_step_sec'], 1e-6):.2f}×" if base else "1.00×"
        lines.append(
            f"{pos}. **`{r['variant']}`** (`{r['params']/1e6:.2f}M`, speedup `{sp}`) — "
            f"`val/loss={sf['val/loss_total']:.4f}`{d_s}, `SimGr_mid={sf['val/sim_gr_mid']:.4f}`, "
            f"`CFG_mid={sf['val/cfg_mid']:.4f}`, `med_step={r['med_step_sec']:.2f}s`"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Абляционный эксперимент 3 матричных оптимизаций (PHM, GrassAlign, L*X*R) + FastVocabOptimizer"
    )
    parser.add_argument("--smoke-test", "--smoke_test", dest="smoke_test", action="store_true")
    parser.add_argument("--arch-preset", "--arch_preset", dest="arch_preset", type=str, default="tiny", choices=list(ARCH_PRESETS.keys()))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--cpu-bf16", "--cpu_bf16", dest="cpu_bf16", action="store_true")
    parser.add_argument("--num-threads", "--num_threads", dest="num_threads", type=int, default=0)

    parser.add_argument("--variants", type=str, default="all", help="Список через запятую или 'all'")
    parser.add_argument("--init-base", "--init_base", dest="init_base", type=str, default="compat", choices=["compat", "baseline", "all_5"])
    parser.add_argument("--wiener-damp", "--wiener_damp", dest="wiener_damp", type=float, default=0.5)
    parser.add_argument("--phm-rank", "--phm_rank", dest="phm_rank", type=int, default=4)
    parser.add_argument("--phm-alpha", "--phm_alpha", dest="phm_alpha", type=float, default=0.25)

    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=4)
    parser.add_argument("--train-samples", "--train_samples", dest="train_samples", type=int, default=128)
    parser.add_argument("--val-samples", "--val_samples", dest="val_samples", type=int, default=32)
    parser.add_argument("--eval-every", "--eval_every", dest="eval_every", type=int, default=10)
    parser.add_argument("--log-every", "--log_every", dest="log_every", type=int, default=5)
    parser.add_argument("--no-decoder-eval", "--no_decoder_eval", dest="no_decoder_eval", action="store_true")

    # Опции быстрого словаря, радара и сквозного CE на обучении
    parser.add_argument("--no-fast-vocab", "--no_fast_vocab", dest="no_fast_vocab", action="store_true")
    parser.add_argument("--use-radar", "--use_radar", dest="use_radar", action="store_true", help="Включить NoiseRadarHead на train-шаге")
    parser.add_argument("--ce-weight", "--ce_weight", dest="ce_weight", type=float, default=0.0, help="Вес сквозного CE-лосса на train-шаге (0 = выкл)")
    parser.add_argument("--ce-max-tokens", "--ce_max_tokens", dest="ce_max_tokens", type=int, default=512)
    parser.add_argument("--ce-t-max", "--ce_t_max", dest="ce_t_max", type=float, default=0.50)
    parser.add_argument("--ce-pool-hard", "--ce_pool_hard", dest="ce_pool_hard", type=int, default=1536)
    parser.add_argument("--ce-pool-rand", "--ce_pool_rand", dest="ce_pool_rand", type=int, default=2048)
    parser.add_argument("--ce-batch-fraction", "--ce_batch_fraction", dest="ce_batch_fraction", type=float, default=0.5)
    parser.add_argument("--vocab-svd-rank", "--vocab_svd_rank", dest="vocab_svd_rank", type=int, default=64)

    parser.add_argument("--muon-lr", "--muon_lr", dest="muon_lr", type=float, default=1e-3)
    parser.add_argument("--adamw-lr", "--adamw_lr", dest="adamw_lr", type=float, default=3e-4)
    parser.add_argument("--scale-1d-lr", "--scale_1d_lr", dest="scale_1d_lr", type=float, default=1.2e-3)
    parser.add_argument("--lr-warmup-steps", "--lr_warmup_steps", dest="lr_warmup_steps", type=int, default=10)

    parser.add_argument("--model-seed", "--model_seed", dest="model_seed", type=int, default=42)
    parser.add_argument("--init-seed", "--init_seed", dest="init_seed", type=int, default=1000)
    parser.add_argument("--schedule-seed", "--schedule_seed", dest="schedule_seed", type=int, default=2026)

    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default="")
    parser.add_argument("--cache-dir", "--cache_dir", dest="cache_dir", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    if args.smoke_test:
        print("=== SMOKE TEST: АБЛЯЦИЯ 3 МАТРИЧНЫХ ОПТИМИЗАЦИЙ (7 ВАРИАНТОВ) ===", flush=True)
        args.steps = 2
        args.batch_size = 2
        args.train_samples = 8
        args.val_samples = 4
        args.eval_every = 2
        args.log_every = 1
        args.lr_warmup_steps = 1

    if args.variants.strip().lower() == "all":
        selected_variants = list(MATRIX_VARIANTS)
    else:
        selected_variants = [v.strip() for v in args.variants.split(",") if v.strip()]
        for v in selected_variants:
            if v not in MATRIX_VARIANTS:
                raise ValueError(f"Неизвестный вариант '{v}'. Доступны: {MATRIX_VARIANTS}")

    out_dir = args.output_dir or str(ALT_DIR / "output" / f"matrix_ablation_{args.arch_preset}")
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
        f"=== Matrix Ablation Setup: preset={args.arch_preset}, device={env.device}, "
        f"init_base={args.init_base}(damp={args.wiener_damp}), phm_rank={args.phm_rank}, "
        f"steps={args.steps}, batch={args.batch_size}, train={args.train_samples}, val={args.val_samples}, "
        f"fast_vocab={'OFF' if args.no_fast_vocab else 'ON'}, radar={args.use_radar}, ce_w={args.ce_weight}, "
        f"variants={selected_variants} ===",
        flush=True,
    )

    # 1. Загружаем токенизатор, замороженные Энкодер/Декодер и опорные концепты
    tokenizer = get_tokenizer()
    need_decoder = (not args.no_decoder_eval) or args.use_radar or (float(args.ce_weight) > 0.0)
    encoder = get_encoder(weights_path=env.encoder_weights_path, device=env.device, eval_mode=True)
    decoder = (
        get_decoder(weights_path=env.decoder_weights_path, encoder=encoder, device=env.device, eval_mode=True)
        if need_decoder
        else None
    )
    for p in list(encoder.parameters()) + (list(decoder.parameters()) if decoder is not None else []):
        p.requires_grad_(False)

    fast_vocab: Optional[FastVocabOptimizer] = None
    if decoder is not None and not args.no_fast_vocab:
        fv_t0 = time.time()
        fv_dtype = env.amp_dtype if env.use_amp else torch.float32
        fast_vocab = FastVocabOptimizer.from_decoder(
            decoder=decoder,
            gemm_dtype=fv_dtype,
            svd_rank=args.vocab_svd_rank,
            ce_pool_hard=args.ce_pool_hard,
            ce_pool_rand=args.ce_pool_rand,
            ce_batch_fraction=args.ce_batch_fraction,
            attach=True,
        )
        print(
            f"[FastVocab] Подключён за {time.time() - fv_t0:.2f}s "
            f"(dtype={fv_dtype}, svd_rank={fast_vocab.svd_rank}, "
            f"ce_pool={args.ce_max_tokens}+{fast_vocab.ce_pool_hard}+{fast_vocab.ce_pool_rand})",
            flush=True,
        )

    sep_concept = torch.load(env.sep_concept_path, map_location="cpu", weights_only=False)
    void_concept = torch.load(env.void_concept_path, map_location="cpu", weights_only=False)
    sep_canon, _ = canonicalize_cell_basis(sep_concept.float())
    void_canon, _ = canonicalize_cell_basis(void_concept.float())

    if os.path.exists(env.stopwords_path):
        sw_obj = torch.load(env.stopwords_path, map_location="cpu", weights_only=False)
        stopword_mask = (sw_obj["mask"] if isinstance(sw_obj, dict) else sw_obj).bool()
    else:
        stopword_mask = torch.zeros(131072, dtype=torch.bool)

    # 2. Единый кэш латентов (переиспользует готовый ablation_train_cache_128.pt / ablation_val_cache_32.pt!)
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

    del encoder
    if env.device.type == "cuda":
        torch.cuda.empty_cache()

    train_items = train_ds.items
    val_items = val_ds.items

    # 3. Предвычисляем data-driven статистики для init_base (compat / all_5)
    precomputed_stats: Dict[str, Any] = {}
    if args.init_base in ("compat", "all_5"):
        print("[Prep] Вычисление Винеровских beta_0..7 и K-Means пулов по train-кэшу...", flush=True)
        precomputed_stats["wiener_info"] = compute_wiener_axis_betas(
            train_items, sep_concept=sep_canon, seed=args.init_seed + 4
        )
        precomputed_stats["kmeans_pools"] = build_kmeans_pools(train_items, seed=args.init_seed + 5)
        eff_betas = [round(x * args.wiener_damp, 4) for x in precomputed_stats["wiener_info"]["betas"].tolist()]
        print(f"  -> wiener_betas(damp={args.wiener_damp})={eff_betas}", flush=True)

    # 4. Фиксированные расписания train и val (бит-в-бит равные условия)
    train_schedule = build_fixed_train_schedule(
        train_items=train_items,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.schedule_seed,
    )
    val_pack = build_fixed_val_pack(val_items=val_items, seed=args.schedule_seed + 777)

    results_json_path = os.path.join(out_dir, "matrix_ablation_results.json")
    report_md_path = os.path.join(out_dir, "matrix_ablation_report.md")
    results: Dict[str, Dict[str, Any]] = {}
    if args.resume and os.path.exists(results_json_path):
        with open(results_json_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"[Resume] Загружены готовые варианты: {list(results.keys())}", flush=True)

    # 5. Последовательный прогон всех выбранных матричных вариантов
    for variant in selected_variants:
        if args.resume and variant in results:
            print(f"[Skip] Вариант '{variant}' уже есть в {results_json_path}", flush=True)
            continue
        res = run_single_matrix_variant(
            variant=variant,
            args=args,
            env=env,
            train_items=train_items,
            train_schedule=train_schedule,
            val_pack=val_pack,
            sep_canon=sep_canon,
            void_canon=void_canon,
            stopword_mask=stopword_mask,
            precomputed_stats=precomputed_stats,
            decoder=decoder,
            fast_vocab=fast_vocab,
        )
        results[variant] = res

        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        report_md = render_matrix_report(results, args=args)
        with open(report_md_path, "w", encoding="utf-8") as f:
            f.write(report_md)

    print("\n" + "=" * 72, flush=True)
    print(render_matrix_report(results, args=args), flush=True)
    print(f"Отчет сохранен: {report_md_path}", flush=True)
    print(f"JSON сохранен:  {results_json_path}", flush=True)


if __name__ == "__main__":
    main()
