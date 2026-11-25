"""
Проверка точности FastVocabOptimizer на РЕАЛЬНЫХ (острых) распределениях декодера Системы 1.

Зачем: юнит-тест test_phase3_alt.py [9] и CPU-бенчмарк проверяли sampled softmax в почти равномерном
режиме (CE ≈ ln V), где любая выборка точна «бесплатно». Здесь скрытые состояния h берутся из
замороженного Декодера на реальных латентах (кэш энкодера) при t ∈ {0, 0.25, 0.5}, т.е. в режиме
острых softmax-распределений, где всё решают жёсткие негативы.

Измеряется:
  A. Sampled CE (Batch-Shared Stratified, как в обучении, чанками по M=512 токенов):
     - CE_exact vs CE_fast (смещение и разброс по R повторам случайного хвоста);
     - косинус градиента dL/dh (глобальный и медианный потокенный);
     - Recall@k: доля top-k истинных конкурентов (по точным логитам), попавших в пул (hard ∪ in-batch);
     - покрытие массы: доля нецелевой softmax-массы, попавшей в точную часть пула;
     - TokAcc по пулу vs точный TokAcc (инфляция метрики train/ce_tok_acc).
  B. NoiseRadar в BF16 vs оригинальный FP32 compute_dict_stats:
     - ошибки признаков [cos_max, cos_2nd, cos_gap] (абсолютные и относительно среднего cos_gap);
     - расхождение sigma_cell / sigma_tok радара;
     - сквозной эффект через noise_cond_proj: согласие argmax и TokAcc/CE декодера.

Запуск (CPU, ~2–5 мин на 16 примерах):
  .venv\\Scripts\\python.exe -B "experiments\\phase 3 alt\\exp_vocab_fidelity.py" --samples 16 --max-cells 32
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

ALT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ALT_DIR.parents[1]
PHASE3_DIR = PROJECT_ROOT / "experiments" / "phase 3"
for _p in (str(PROJECT_ROOT), str(PHASE3_DIR), str(ALT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from storage.components.decoder import get_decoder  # noqa: E402
from storage.components.encoder import get_encoder  # noqa: E402

from train_phase3_alt import PAD_TOKEN_ID, SIGMA_DATA, AltEnvironmentConfig  # noqa: E402
from vocab_optim import FastVocabOptimizer  # noqa: E402


def _dtype_from_str(s: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[s]


def build_batch(items: List[Dict[str, torch.Tensor]], n: int, max_cells: int):
    sel = items[:n]
    V = torch.stack([x["V_ans"].float()[:max_cells] for x in sel], dim=0)  # (B, N, 8, 128)
    tg = torch.stack([x["ans_targets"].long()[: max_cells * 4] for x in sel], dim=0)  # (B, 4N)
    i_last = torch.stack([x["i_last"].long() for x in sel], dim=0)
    n_keep = i_last.clamp(max=max_cells)
    cell_idx = torch.arange(max_cells).unsqueeze(0)
    cell_ok = cell_idx < n_keep.unsqueeze(1)  # (B, N)
    tok_ok = cell_ok.unsqueeze(-1).expand(-1, -1, 4).reshape(len(sel), max_cells * 4)
    valid = tok_ok & (tg != -100) & (tg != PAD_TOKEN_ID)
    return V, tg, valid, cell_ok


def eval_sampled_ce(
    h_all: torch.Tensor,
    tgt_all: torch.Tensor,
    W: torch.Tensor,
    fv: FastVocabOptimizer,
    chunk: int,
    repeats: int,
    topk_recall: int,
    seed: int,
) -> Dict[str, float]:
    """h_all: (M_tot, D) реальные скрытые состояния; чанки по `chunk` строк имитируют M=512 в обучении."""
    M_tot = h_all.shape[0]
    perm = torch.randperm(M_tot, generator=torch.Generator().manual_seed(seed))
    h_all = h_all[perm]
    tgt_all = tgt_all[perm]

    # --- 1. Точный CE/градиент и диагностические величины пула (детерминированная часть) ---
    g_exact = torch.zeros_like(h_all)
    ce_exact_sum, acc_exact_sum = 0.0, 0.0
    recall_hits, recall_tot = 0.0, 0.0
    mass_cov: List[float] = []
    p_tgt_list: List[float] = []
    for s in range(0, M_tot, chunk):
        e = min(M_tot, s + chunk)
        h = h_all[s:e].clone().requires_grad_(True)
        t = tgt_all[s:e]
        logits = F.linear(h, W)
        ce = F.cross_entropy(logits, t, reduction="sum") / M_tot
        ce.backward()
        g_exact[s:e] = h.grad
        with torch.no_grad():
            ce_exact_sum += float(ce.item()) * M_tot
            acc_exact_sum += float((logits.argmax(-1) == t).float().sum().item())
            p = logits.softmax(-1)
            p_t = p.gather(1, t.unsqueeze(1)).squeeze(1)
            p_tgt_list += p_t.tolist()
            # Пул точной части: hard_ids (batch-shared) ∪ in-batch цели этого чанка
            hard_ids = fv.select_batch_hard_negatives(h.detach(), torch.ones(e - s, dtype=torch.bool))
            in_pool = torch.zeros(W.shape[0], dtype=torch.bool)
            in_pool[hard_ids] = True
            in_pool[t] = True
            # Recall@k по top-k нецелевым конкурентам
            lg = logits.detach().clone()
            lg.scatter_(1, t.unsqueeze(1), float("-inf"))
            topk_ids = lg.topk(topk_recall, dim=-1).indices
            recall_hits += float(in_pool[topk_ids].float().sum().item())
            recall_tot += float(topk_ids.numel())
            # Покрытие нецелевой массы точной частью пула
            p_nt = p.clone()
            p_nt.scatter_(1, t.unsqueeze(1), 0.0)
            covered = (p_nt * in_pool.unsqueeze(0).float()).sum(-1)
            denom = (1.0 - p_t).clamp(min=1e-9)
            mass_cov += (covered / denom).tolist()

    ce_exact = ce_exact_sum / M_tot
    acc_exact = acc_exact_sum / M_tot

    # --- 2. Sampled CE: R повторов случайного хвоста ---
    ce_fast_runs, acc_pool_runs, gcos_runs, gcos_tok_med_runs = [], [], [], []
    g_fast_mean = torch.zeros_like(h_all)
    for r in range(repeats):
        torch.manual_seed(seed * 100 + r)
        g_fast = torch.zeros_like(h_all)
        ce_sum, acc_sum = 0.0, 0.0
        for s in range(0, M_tot, chunk):
            e = min(M_tot, s + chunk)
            h = h_all[s:e].clone().requires_grad_(True)
            t = tgt_all[s:e]
            per_tok, is_corr = fv.compute_sampled_ce_from_hidden(
                h_sub=h, tgt_sub=t, valid_sel_mask=torch.ones(e - s, dtype=torch.bool), lm_head_weight=W
            )
            loss = per_tok.sum() / M_tot
            loss.backward()
            g_fast[s:e] = h.grad
            ce_sum += float(per_tok.detach().sum().item())
            acc_sum += float(is_corr.float().sum().item())
        ce_fast_runs.append(ce_sum / M_tot)
        acc_pool_runs.append(acc_sum / M_tot)
        gcos_runs.append(float(F.cosine_similarity(g_fast.flatten(), g_exact.flatten(), dim=0).item()))
        tok_cos = F.cosine_similarity(g_fast, g_exact, dim=-1)
        gcos_tok_med_runs.append(float(tok_cos.median().item()))
        g_fast_mean += g_fast / repeats

    p_arr = np.array(p_tgt_list)
    return {
        "n_tokens": int(M_tot),
        "ce_exact": ce_exact,
        "ce_fast_mean": float(np.mean(ce_fast_runs)),
        "ce_fast_std": float(np.std(ce_fast_runs)),
        "ce_bias": float(np.mean(ce_fast_runs) - ce_exact),
        "ce_bias_rel": float((np.mean(ce_fast_runs) - ce_exact) / max(abs(ce_exact), 1e-9)),
        "grad_cos_global": float(np.mean(gcos_runs)),
        "grad_cos_global_min": float(np.min(gcos_runs)),
        "grad_cos_avgR": float(F.cosine_similarity(g_fast_mean.flatten(), g_exact.flatten(), dim=0).item()),
        "grad_cos_tok_median": float(np.mean(gcos_tok_med_runs)),
        "acc_exact": acc_exact,
        "acc_pool": float(np.mean(acc_pool_runs)),
        "acc_inflation": float(np.mean(acc_pool_runs) - acc_exact),
        f"recall_at_{topk_recall}": recall_hits / max(recall_tot, 1.0),
        "mass_coverage_mean": float(np.mean(mass_cov)),
        "mass_coverage_p10": float(np.percentile(mass_cov, 10)),
        "p_tgt_median": float(np.median(p_arr)),
        "frac_peaked_p_tgt_gt_0.5": float((p_arr > 0.5).mean()),
    }


@torch.no_grad()
def eval_radar(decoder, V: torch.Tensor, tg: torch.Tensor, valid: torch.Tensor, cell_ok: torch.Tensor,
               fv_bf16: FastVocabOptimizer) -> Dict[str, float]:
    radar = getattr(decoder, "noise_radar", None)
    if radar is None:
        return {"radar_present": 0.0}
    W = decoder.lm_head_weight
    stage1 = decoder.decode_stage1(V)
    h_unp = stage1["h_unpacked"]
    st_ref = radar.compute_dict_stats(h_unp, W, decoder.lm_head_inv_norms)  # оригинальный FP32
    st_bf = fv_bf16.fast_compute_dict_stats(h_unp)
    m = valid
    d = (st_bf - st_ref).abs()[m]  # (n, 4)
    gap_ref = st_ref[..., 2][m]
    out: Dict[str, float] = {
        "radar_present": 1.0,
        "cos_max_ref_mean": float(st_ref[..., 0][m].mean().item()),
        "cos_gap_ref_mean": float(gap_ref.mean().item()),
        "cos_gap_ref_median": float(gap_ref.median().item()),
        "err_cos_max_max": float(d[:, 0].max().item()),
        "err_cos_max_mean": float(d[:, 0].mean().item()),
        "err_cos_gap_max": float(d[:, 2].max().item()),
        "err_cos_gap_mean": float(d[:, 2].mean().item()),
        "err_cos_gap_rel_to_mean_gap": float(d[:, 2].mean().item() / max(float(gap_ref.mean().item()), 1e-9)),
    }

    # Сквозной эффект: sigma радара и выход декодера (noise_cond_proj) с FP32 vs BF16 признаками
    sig_c_ref, sig_t_ref, _ = decoder.estimate_noise(V)
    h_ref = decoder.decode_hidden(V)
    fv_bf16.attach_to_decoder(decoder)
    try:
        sig_c_bf, sig_t_bf, _ = decoder.estimate_noise(V)
        h_bf = decoder.decode_hidden(V)
    finally:
        fv_bf16.detach_from_decoder()

    out["sigma_cell_absdiff_mean"] = float((sig_c_bf - sig_c_ref).abs()[cell_ok].mean().item())
    out["sigma_cell_absdiff_max"] = float((sig_c_bf - sig_c_ref).abs()[cell_ok].max().item())
    out["sigma_tok_absdiff_mean"] = float((sig_t_bf - sig_t_ref).abs()[m].mean().item())
    out["sigma_tok_absdiff_max"] = float((sig_t_bf - sig_t_ref).abs()[m].max().item())

    hr = h_ref[m].float()
    hb = h_bf[m].float()
    t = tg[m]
    lr = F.linear(hr, W.float())
    lb = F.linear(hb, W.float())
    out["dec_argmax_agree"] = float((lr.argmax(-1) == lb.argmax(-1)).float().mean().item())
    out["dec_tokacc_ref"] = float((lr.argmax(-1) == t).float().mean().item())
    out["dec_tokacc_bf16radar"] = float((lb.argmax(-1) == t).float().mean().item())
    out["dec_ce_ref"] = float(F.cross_entropy(lr, t).item())
    out["dec_ce_bf16radar"] = float(F.cross_entropy(lb, t).item())
    return out


def render_report(res: Dict[str, Any], args: argparse.Namespace) -> str:
    L: List[str] = [
        "# Точность FastVocabOptimizer на реальных распределениях декодера",
        "",
        f"- **Примеров:** `{args.samples}` | **Ячеек на пример:** `≤{args.max_cells}` | **Чанк (M):** `{args.chunk}` | "
        f"**Повторов хвоста:** `{args.repeats}` | **Пул:** `M + {args.ce_pool_hard} hard + {args.ce_pool_rand} rand`, SVD rank `{args.svd_rank}`",
        "",
        "## A. Sampled CE vs точный CE",
        "",
        f"| t | sketch | токенов | p_tgt med | CE exact | CE fast (±std) | смещение | grad cos (глоб/мин/ср.R) | grad cos med/ток | Recall@{args.topk_recall} | покрытие массы (ср/p10) | TokAcc exact → pool |",
        "|---|---|---:|---:|---:|---|---:|---|---:|---:|---|---|",
    ]
    for key, r in res["ce"].items():
        t_s, sk = key.split("|")
        L.append(
            f"| {t_s} | {sk} | {r['n_tokens']} | {r['p_tgt_median']:.3f} | {r['ce_exact']:.4f} | "
            f"{r['ce_fast_mean']:.4f} ± {r['ce_fast_std']:.4f} | {r['ce_bias']:+.4f} ({r['ce_bias_rel']:+.1%}) | "
            f"{r['grad_cos_global']:.4f} / {r['grad_cos_global_min']:.4f} / {r['grad_cos_avgR']:.4f} | "
            f"{r['grad_cos_tok_median']:.4f} | {r[f'recall_at_{args.topk_recall}']:.3f} | "
            f"{r['mass_coverage_mean']:.3f} / {r['mass_coverage_p10']:.3f} | {r['acc_exact']:.3f} → {r['acc_pool']:.3f} |"
        )
    L += [
        "",
        "## B. NoiseRadar BF16 vs FP32",
        "",
        "| t | cos_gap ср/мед (FP32) | ошибка cos_max ср/макс | ошибка cos_gap ср/макс (отн. к ср. gap) | Δσ_cell ср/макс | Δσ_tok ср/макс | согласие argmax | TokAcc FP32 → BF16 | CE FP32 → BF16 |",
        "|---|---|---|---|---|---|---:|---|---|",
    ]
    for t_s, r in res["radar"].items():
        if not r.get("radar_present"):
            L.append(f"| {t_s} | радар отсутствует | | | | | | | |")
            continue
        L.append(
            f"| {t_s} | {r['cos_gap_ref_mean']:.4f} / {r['cos_gap_ref_median']:.4f} | "
            f"{r['err_cos_max_mean']:.2e} / {r['err_cos_max_max']:.2e} | "
            f"{r['err_cos_gap_mean']:.2e} / {r['err_cos_gap_max']:.2e} ({r['err_cos_gap_rel_to_mean_gap']:.1%}) | "
            f"{r['sigma_cell_absdiff_mean']:.2e} / {r['sigma_cell_absdiff_max']:.2e} | "
            f"{r['sigma_tok_absdiff_mean']:.2e} / {r['sigma_tok_absdiff_max']:.2e} | "
            f"{r['dec_argmax_agree']:.4f} | {r['dec_tokacc_ref']:.3f} → {r['dec_tokacc_bf16radar']:.3f} | "
            f"{r['dec_ce_ref']:.3f} → {r['dec_ce_bf16radar']:.3f} |"
        )
    L += [
        "",
        "## Критерии приёмки (предварительные)",
        "",
        "- Sampled CE: `grad cos (глоб) ≥ 0.98`, `|смещение| ≤ 2%`, `покрытие массы p10 ≥ 0.9` при t ≤ 0.25.",
        "  Если покрытие/Recall низкие — поднять `--vocab-svd-rank` (64 → 128/256) или `--ce-pool-hard`.",
        "- Радар BF16: `согласие argmax ≥ 0.995`, `|ΔTokAcc| ≤ 0.5 п.п.`, ошибка cos_gap `≤ 10%` от среднего gap.",
        "  Иначе на GPU/TPU оставить радар в FP32 (`FastVocabOptimizer(gemm_dtype=torch.float32)`).",
        "",
    ]
    return "\n".join(L)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Проверка точности FastVocabOptimizer на реальных латентах")
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--max-cells", dest="max_cells", type=int, default=32)
    p.add_argument("--t-levels", dest="t_levels", type=str, default="0.0,0.25,0.5")
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--topk-recall", dest="topk_recall", type=int, default=10)
    p.add_argument("--svd-rank", dest="svd_rank", type=int, default=64)
    p.add_argument("--ce-pool-hard", dest="ce_pool_hard", type=int, default=1536)
    p.add_argument("--ce-pool-rand", dest="ce_pool_rand", type=int, default=2048)
    p.add_argument("--sketch-dtypes", dest="sketch_dtypes", type=str, default="float32,bfloat16",
                   help="dtype SVD-эскиза для отбора hard negatives (в обучении на GPU/TPU = bfloat16)")
    p.add_argument("--no-radar", dest="no_radar", action="store_true")
    p.add_argument("--cache-path", dest="cache_path", type=str, default="")
    p.add_argument("--output-dir", dest="output_dir", type=str, default="")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--num-threads", dest="num_threads", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    out_dir = args.output_dir or str(ALT_DIR / "output" / "vocab_fidelity")
    os.makedirs(out_dir, exist_ok=True)
    cache_path = args.cache_path or str(ALT_DIR / "output" / "ablation_val_cache_32.pt")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Не найден кэш латентов {cache_path}. Сначала запустите exp_init_ablation.py (создаёт ablation_val_cache_32.pt) "
            f"или укажите --cache-path."
        )

    env = AltEnvironmentConfig.detect(override_device=args.device, override_output_dir=out_dir, override_checkpoint_dir=out_dir)
    device = env.device
    if device.type != "cpu":
        print("[Warn] Скрипт рассчитан на CPU-тензоры; принудительно используется CPU.", flush=True)
        device = torch.device("cpu")

    t0 = time.time()
    encoder = get_encoder(weights_path=env.encoder_weights_path, device=device, eval_mode=True)
    decoder = get_decoder(weights_path=env.decoder_weights_path, encoder=encoder, device=device, eval_mode=True)
    del encoder
    decoder = decoder.float()
    for p_ in decoder.parameters():
        p_.requires_grad_(False)
    W = decoder.lm_head_weight.detach().float()
    print(f"[Load] Декодер загружен за {time.time() - t0:.1f}s, словарь {tuple(W.shape)}", flush=True)

    items = torch.load(cache_path, map_location="cpu", weights_only=False)
    V0, tg, valid, cell_ok = build_batch(items, min(args.samples, len(items)), args.max_cells)
    print(f"[Data] {V0.shape[0]} примеров, валидных токенов: {int(valid.sum())}", flush=True)
    noise = torch.randn(tuple(V0.shape), generator=torch.Generator().manual_seed(args.seed)) * SIGMA_DATA

    fv_by_dtype: Dict[str, FastVocabOptimizer] = {}
    for ds in [s.strip() for s in args.sketch_dtypes.split(",") if s.strip()]:
        fv_by_dtype[ds] = FastVocabOptimizer(
            lm_head_weight=W, gemm_dtype=_dtype_from_str(ds), svd_rank=args.svd_rank,
            ce_pool_hard=args.ce_pool_hard, ce_pool_rand=args.ce_pool_rand,
        )
    fv_bf16 = fv_by_dtype.get("bfloat16") or FastVocabOptimizer(lm_head_weight=W, gemm_dtype=torch.bfloat16, svd_rank=args.svd_rank)

    res: Dict[str, Any] = {"ce": {}, "radar": {}}
    for t_val in [float(x) for x in args.t_levels.split(",")]:
        V_t = (1.0 - t_val) * V0 + t_val * noise
        t_s = f"{t_val:.2f}"
        with torch.no_grad():
            h = decoder.decode_hidden(V_t).float()
        h_all = h[valid].contiguous()
        tgt_all = tg[valid].contiguous()
        for ds, fv in fv_by_dtype.items():
            t1 = time.time()
            r = eval_sampled_ce(h_all, tgt_all, W, fv, args.chunk, args.repeats, args.topk_recall, args.seed)
            res["ce"][f"{t_s}|{ds}"] = r
            print(
                f"[CE t={t_s} sketch={ds}] exact={r['ce_exact']:.4f} fast={r['ce_fast_mean']:.4f}±{r['ce_fast_std']:.4f} "
                f"bias={r['ce_bias_rel']:+.2%} gcos={r['grad_cos_global']:.4f} (tok_med={r['grad_cos_tok_median']:.4f}) "
                f"R@{args.topk_recall}={r[f'recall_at_{args.topk_recall}']:.3f} cov={r['mass_coverage_mean']:.3f}/p10={r['mass_coverage_p10']:.3f} "
                f"acc {r['acc_exact']:.3f}->{r['acc_pool']:.3f} p_tgt_med={r['p_tgt_median']:.3f} | {time.time() - t1:.1f}s",
                flush=True,
            )
        if not args.no_radar:
            t1 = time.time()
            rr = eval_radar(decoder, V_t, tg, valid, cell_ok, fv_bf16)
            res["radar"][t_s] = rr
            if rr.get("radar_present"):
                print(
                    f"[Radar t={t_s}] gap_mean={rr['cos_gap_ref_mean']:.4f} err_gap={rr['err_cos_gap_mean']:.2e} "
                    f"({rr['err_cos_gap_rel_to_mean_gap']:.1%}) dσ_tok={rr['sigma_tok_absdiff_mean']:.2e} "
                    f"agree={rr['dec_argmax_agree']:.4f} acc {rr['dec_tokacc_ref']:.3f}->{rr['dec_tokacc_bf16radar']:.3f} "
                    f"| {time.time() - t1:.1f}s",
                    flush=True,
                )

    with open(os.path.join(out_dir, "vocab_fidelity.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    md = render_report(res, args)
    with open(os.path.join(out_dir, "vocab_fidelity_report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("\n" + md, flush=True)
    print(f"Отчёт: {os.path.join(out_dir, 'vocab_fidelity_report.md')} | всего {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
