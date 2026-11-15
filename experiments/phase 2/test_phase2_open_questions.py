"""
Воспроизводимые эмпирические проверки и замеры по вопросам Q1–Q7 для Фазы 2 Plan C.

Проверяет на замороженных компонентах Системы 1 (`storage/components/`) и локальных датасетах:
- Q1: Реконструкцию формата N=127 (508 токенов), сравнение с N=64/128, происхождение и влияние `sep_concept.pt`.
- Q2: Калибровку порогов анти-коллапса `tau_adj` и `tau_bary` по квантилям реальных текстов.
- Q3: Энергию и `Sim_Gr(., V_void)` граничных ячеек (n_real in {1, 2, 3}), включая худшие случаи со стоп-словами.
- Q4: Баланс слагаемых лосса (`L_basis`, `L_Gr`, `L_adj`, `L_bary`) по значениям и нормам градиентов.
- Q6: Инициализацию `Soft-Gate AdaLN` и стартовые значения `sigma_adaln` / `L_adaln`.
- Q7: Стабильность масштаба `sigma_data = RMS(V_i)` по разным доменам и для `sep_concept.pt`.

Запуск из корня репозитория:
    .venv\\Scripts\\python.exe "experiments/phase 2/test_phase2_open_questions.py"
"""

import argparse
import glob
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from storage.components.decoder import get_decoder
from storage.components.encoder import get_encoder
from storage.components.tokenizer import get_tokenizer

PAD_ID = 10
DEFAULT_SEED = 42


def encode_ids(tokenizer: Any, text: str) -> List[int]:
    """Унифицированное извлечение token_ids для Fast и raw tokenizers."""
    out = tokenizer.encode(text, add_special_tokens=False)
    return out.ids if hasattr(out, "ids") else list(out)


def sim_gr(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Нормированное O(8)-инвариантное сходство операторов плотности в [0, 1]."""
    A_f = A.float()
    B_f = B.float()
    cross = torch.matmul(A_f, B_f.transpose(-2, -1))
    Ga = torch.matmul(A_f, A_f.transpose(-2, -1))
    Gb = torch.matmul(B_f, B_f.transpose(-2, -1))
    num = (cross ** 2).sum(dim=(-2, -1))
    norm_a = torch.sqrt((Ga ** 2).sum(dim=(-2, -1)) + 1e-16)
    norm_b = torch.sqrt((Gb ** 2).sum(dim=(-2, -1)) + 1e-16)
    return num / (norm_a * norm_b + eps)


def compute_rk_eff(V: torch.Tensor) -> torch.Tensor:
    """Эффективный спектральный ранг клетки V в R^{..., 8, 128}."""
    V_f = V.float()
    G = torch.matmul(V_f, V_f.transpose(-2, -1))
    tr_g = torch.diagonal(G, dim1=-2, dim2=-1).sum(dim=-1)
    g_frob_sq = (G ** 2).sum(dim=(-2, -1))
    return (tr_g ** 2) / (g_frob_sq + 1e-8)


def make_pooling_weights(input_ids: torch.Tensor, stopword_mask: torch.Tensor) -> torch.Tensor:
    """Двухуровневые веса спектрального пулинга Фазы 1 (1.0 / 0.02 / 0.001)."""
    is_pad = input_ids == PAD_ID
    valid_idx = torch.clamp(input_ids, 0, len(stopword_mask) - 1)
    is_stop = stopword_mask[valid_idx] & (~is_pad)
    raw_w = torch.where(
        is_pad,
        torch.tensor(0.001, dtype=torch.float32),
        torch.where(
            is_stop,
            torch.tensor(0.02, dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float32),
        ),
    )
    B, T = input_ids.shape
    raw_w = raw_w.view(B, T // 4, 4)
    norm_w = raw_w / raw_w.sum(dim=-1, keepdim=True)
    return norm_w.view(B, T)


def load_domain_texts(max_per_domain: int = 40) -> Dict[str, List[str]]:
    """Загружает выборку реальных текстов из 5 локальных доменов."""
    domains: Dict[str, List[str]] = {}

    ru_path = REPO_ROOT / "data/russian-reasoning/train-00000-of-00002.parquet"
    if ru_path.exists():
        ru_df = pd.read_parquet(ru_path)
        ru_texts = []
        for conv in ru_df["conversation"].head(max_per_domain * 2):
            for msg in conv:
                if msg.get("role") == "assistant" and len(msg.get("content", "")) > 200:
                    ru_texts.append(msg["content"])
                    break
            if len(ru_texts) >= max_per_domain:
                break
        domains["russian_reasoning"] = ru_texts

    for d_name, key in [("camel_math", "message_2"), ("camel_physics", "message_2")]:
        files = sorted(glob.glob(str(REPO_ROOT / f"data/{d_name}/*.json")))[:max_per_domain]
        texts = []
        for f in files:
            with open(f, "r", encoding="utf-8") as fp:
                d = json.load(fp)
                if d.get(key):
                    texts.append(d[key])
        if texts:
            domains[d_name] = texts

    magpie_path = REPO_ROOT / "data/magpie_reasoning/data/train-00000-of-00012.parquet"
    if magpie_path.exists():
        m_df = pd.read_parquet(magpie_path)
        domains["magpie_reasoning"] = [
            t for t in m_df["response"].dropna().tolist() if len(t) > 200
        ][:max_per_domain]

    py_files = sorted(glob.glob(str(REPO_ROOT / "src/**/*.py"), recursive=True)) + sorted(
        glob.glob(str(REPO_ROOT / "storage/components/**/*.py"), recursive=True)
    )
    code_texts = []
    for pf in py_files:
        txt = Path(pf).read_text(encoding="utf-8", errors="ignore")
        for start in range(0, len(txt), 1800):
            chunk = txt[start : start + 2200]
            if len(chunk) > 300:
                code_texts.append(chunk)
            if len(code_texts) >= max_per_domain:
                break
        if len(code_texts) >= max_per_domain:
            break
    if code_texts:
        domains["python_code"] = code_texts

    return domains


def quant_stats(arr: np.ndarray) -> Dict[str, Any]:
    """Вычисляет квантильную сводку по массиву."""
    if len(arr) == 0:
        return {"count": 0}
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50_median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def check_q1(
    encoder: nn.Module,
    decoder: nn.Module,
    tokenizer: Any,
    sep_concept: torch.Tensor,
    void_concept: torch.Tensor,
    stopword_mask: torch.Tensor,
    all_texts: List[str],
) -> Dict[str, Any]:
    """Q1: проверка формата N=127 (508 токенов) и совместимости sep_concept.pt."""
    sep_ids = torch.tensor([[3648, 5972, 13951, 24661]], dtype=torch.long)
    sep_w_uniform = torch.full((1, 4), 0.25, dtype=torch.float32)
    sep_w_phase1 = make_pooling_weights(sep_ids, stopword_mask)
    with torch.no_grad():
        V_sep_unif = encoder(sep_ids, sep_w_uniform)[0, 0]
        V_sep_ph1 = encoder(sep_ids, sep_w_phase1)[0, 0]
        logits_sep_only = decoder(sep_concept.unsqueeze(0).unsqueeze(0))
        sep_decoded_ids = logits_sep_only.argmax(dim=-1)[0].tolist()

    long_batches: List[List[int]] = []
    short_batches: List[List[int]] = []
    for txt in all_texts:
        ids = encode_ids(tokenizer, txt)
        if len(ids) >= 508 and len(long_batches) < 10:
            long_batches.append(ids[:508])
        elif 100 <= len(ids) <= 420 and len(short_batches) < 10:
            short_batches.append(ids + [PAD_ID] * (508 - len(ids)))
        if len(long_batches) >= 10 and len(short_batches) >= 10:
            break

    def eval_recon(samples_list: List[List[int]], tag: str) -> Dict[str, Any]:
        inp = torch.tensor(samples_list, dtype=torch.long)
        w = make_pooling_weights(inp, stopword_mask)
        with torch.no_grad():
            V_ans = encoder(inp, w)
            logits_127 = decoder(V_ans)
            preds_127 = logits_127.argmax(dim=-1)

            sep_exp = sep_concept.unsqueeze(0).unsqueeze(0).expand(V_ans.size(0), 1, 8, 128).to(V_ans.dtype)
            V_128 = torch.cat([sep_exp, V_ans], dim=1)
            logits_128 = decoder(V_128)
            preds_128_tail = logits_128[:, 4:].argmax(dim=-1)
            preds_128_sep_pos = logits_128[:, :4].argmax(dim=-1)

        non_pad = inp != PAD_ID
        pad_mask = ~non_pad
        return {
            "tag": tag,
            "num_samples": len(samples_list),
            "V_ans_shape": list(V_ans.shape),
            "acc_127_pct": float((preds_127[non_pad] == inp[non_pad]).float().mean().item() * 100.0),
            "correct_127": int((preds_127[non_pad] == inp[non_pad]).sum().item()),
            "total_non_pad": int(non_pad.sum().item()),
            "acc_128_with_sep_prefix_pct": float(
                (preds_128_tail[non_pad] == inp[non_pad]).float().mean().item() * 100.0
            ),
            "max_logit_diff_first4": float((logits_127[:, :4] - logits_128[:, 4:8]).abs().max().item()),
            "max_logit_diff_all": float((logits_127 - logits_128[:, 4:]).abs().max().item()),
            "sep_prefix_exact_token_match_pct": float(
                (preds_128_sep_pos == sep_ids.expand(V_ans.size(0), 4)).float().mean().item() * 100.0
            ),
            "pad_argmax_acc_127_pct": (
                float((preds_127[pad_mask] == PAD_ID).float().mean().item() * 100.0) if pad_mask.any() else None
            ),
        }

    return {
        "sep_frob_sq": float((sep_concept.float() ** 2).sum().item()),
        "sep_rms": float(torch.sqrt((sep_concept.float() ** 2).mean()).item()),
        "sep_max_diff_vs_encoder_uniform": float((sep_concept.float() - V_sep_unif.float()).abs().max().item()),
        "sep_max_diff_vs_encoder_phase1_w": float((sep_concept.float() - V_sep_ph1.float()).abs().max().item()),
        "sep_only_decoded_ids": sep_decoded_ids,
        "void_frob_sq": float((void_concept.float() ** 2).sum().item()),
        "long_508_tokens": eval_recon(long_batches, "long_508_full"),
        "short_padded_to_508": eval_recon(short_batches, "short_padded_to_508"),
    }


def encode_domains(
    encoder: nn.Module,
    tokenizer: Any,
    stopword_mask: torch.Tensor,
    domains: Dict[str, List[str]],
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Кодирует тексты доменов в тензоры (B, 127, 8, 128)."""
    domain_cells = {}
    for d_name, texts in domains.items():
        batch_ids = []
        for txt in texts:
            ids = encode_ids(tokenizer, txt)[:508]
            if len(ids) < 508:
                ids = ids + [PAD_ID] * (508 - len(ids))
            batch_ids.append(ids)
        inp_t = torch.tensor(batch_ids, dtype=torch.long)
        w_t = make_pooling_weights(inp_t, stopword_mask)
        V_parts = []
        with torch.no_grad():
            for s in range(0, len(inp_t), 10):
                V_parts.append(encoder(inp_t[s : s + 10], w_t[s : s + 10]))
        V_all = torch.cat(V_parts, dim=0)
        n_real_cells = (inp_t.view(-1, 127, 4) != PAD_ID).sum(dim=-1)
        domain_cells[d_name] = (V_all, n_real_cells, inp_t)
    return domain_cells


def check_q7(
    domain_cells: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    sep_concept: torch.Tensor,
) -> Dict[str, Any]:
    """Q7: междоменная стабильность sigma_data = RMS(V_i) и энергии."""
    q7_res: Dict[str, Any] = {}
    all_content_V = []
    for d_name, (V_all, n_real_cells, _) in domain_cells.items():
        V_full = V_all[n_real_cells == 4].float()
        all_content_V.append(V_full)
        frob_sq = (V_full ** 2).sum(dim=(-2, -1)).numpy()
        rms = np.sqrt(frob_sq / 1024.0)
        rk = compute_rk_eff(V_full).numpy()
        q7_res[d_name] = {
            "num_full_cells": int(len(V_full)),
            "frob_sq_mean": float(np.mean(frob_sq)),
            "frob_sq_std": float(np.std(frob_sq)),
            "rms_mean": float(np.mean(rms)),
            "rms_std": float(np.std(rms)),
            "rms_p10": float(np.percentile(rms, 10)),
            "rms_p90": float(np.percentile(rms, 90)),
            "rk_eff_mean": float(np.mean(rk)),
        }

    V_corpus = torch.cat(all_content_V, dim=0)
    frob_sq_c = (V_corpus ** 2).sum(dim=(-2, -1)).numpy()
    rms_c = np.sqrt(frob_sq_c / 1024.0)
    q7_res["ALL_COMBINED_FULL_CELLS"] = {
        "num_full_cells": int(len(V_corpus)),
        "frob_sq_mean": float(np.mean(frob_sq_c)),
        "frob_sq_std": float(np.std(frob_sq_c)),
        "rms_mean": float(np.mean(rms_c)),
        "rms_std": float(np.std(rms_c)),
        "global_rms_sqrt_mean_sq": float(np.sqrt(np.mean(frob_sq_c) / 1024.0)),
    }
    q7_res["SEP_CONCEPT"] = {
        "frob_sq": float((sep_concept.float() ** 2).sum().item()),
        "rms": float(torch.sqrt((sep_concept.float() ** 2).mean()).item()),
        "rk_eff": float(compute_rk_eff(sep_concept).item()),
    }
    return q7_res


def check_q2(domain_cells: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Dict[str, Any]:
    """Q2: квантили Sim_Gr для соседних клеток и клеток из разных примеров."""
    adj_sims: List[float] = []
    same_sample_nonadj_sims: List[float] = []
    all_sample_cells: List[torch.Tensor] = []

    for _, (V_all, n_real_cells, _) in domain_cells.items():
        for b in range(V_all.shape[0]):
            full_idx = torch.where(n_real_cells[b] == 4)[0]
            if len(full_idx) < 2:
                continue
            for k in range(len(full_idx) - 1):
                if full_idx[k + 1] == full_idx[k] + 1:
                    adj_sims.append(float(sim_gr(V_all[b, full_idx[k]], V_all[b, full_idx[k + 1]]).item()))
            V_b = V_all[b, full_idx]
            all_sample_cells.append(V_b)
            if len(full_idx) >= 4:
                idx1 = torch.randint(0, len(full_idx), (100,))
                idx2 = torch.randint(0, len(full_idx), (100,))
                valid = torch.abs(full_idx[idx1] - full_idx[idx2]) > 1
                if valid.any():
                    same_sample_nonadj_sims.extend(sim_gr(V_b[idx1[valid]], V_b[idx2[valid]]).numpy().tolist())

    n_samples = len(all_sample_cells)
    s1 = np.random.randint(0, n_samples, size=50000)
    s2 = np.random.randint(0, n_samples, size=50000)
    mask_diff = s1 != s2
    Va_t = torch.stack([all_sample_cells[a][np.random.randint(0, len(all_sample_cells[a]))] for a in s1[mask_diff]])
    Vb_t = torch.stack([all_sample_cells[b][np.random.randint(0, len(all_sample_cells[b]))] for b in s2[mask_diff]])
    cross_arr = sim_gr(Va_t, Vb_t).numpy()

    adj_arr = np.array(adj_sims, dtype=np.float64)
    nonadj_arr = np.array(same_sample_nonadj_sims, dtype=np.float64)

    def exceed(arr: np.ndarray, thresholds: List[float]) -> Dict[str, float]:
        return {f">{t:.2f}": float(np.mean(arr > t) * 100.0) for t in thresholds}

    return {
        "adjacent_full_cells": {
            **quant_stats(adj_arr),
            "exceed_pct": exceed(adj_arr, [0.70, 0.75, 0.80, 0.85, 0.90]),
        },
        "same_sample_nonadjacent": {
            **quant_stats(nonadj_arr),
            "exceed_pct": exceed(nonadj_arr, [0.40, 0.50, 0.70, 0.85]),
        },
        "cross_sample_different_examples": {
            **quant_stats(cross_arr),
            "exceed_pct": exceed(cross_arr, [0.30, 0.40, 0.50, 0.60, 0.70]),
        },
    }


def check_q3(
    encoder: nn.Module,
    tokenizer: Any,
    void_concept: torch.Tensor,
    stopword_mask: torch.Tensor,
    flat_texts: List[str],
) -> Dict[str, Any]:
    """Q3: энергия и Sim_Gr(., void_concept) граничной ячейки n_real in {1, 2, 3}."""
    boundary_stats: Dict[int, List[Dict[str, Any]]] = {1: [], 2: [], 3: [], 0: []}
    stop_only_boundary: Dict[int, List[Dict[str, Any]]] = {1: [], 2: [], 3: []}
    test_inp_list: List[List[int]] = []
    test_meta: List[Tuple[int, int, bool]] = []

    for txt in flat_texts[:120]:
        ids = encode_ids(tokenizer, txt)
        if len(ids) < 20:
            continue
        for target_rem in [1, 2, 3]:
            L = min(len(ids), 4 * 10 + target_rem)
            if L % 4 != target_rem:
                L = (L // 4) * 4 + target_rem
                if L > len(ids):
                    L -= 4
            if L <= 0 or L > 64:
                continue
            sub = ids[:L]
            cell_idx = L // 4
            rem_toks = sub[cell_idx * 4 :]
            all_stop = all(bool(stopword_mask[t].item()) for t in rem_toks)
            padded = sub + [PAD_ID] * (64 - len(sub))
            test_inp_list.append(padded)
            test_meta.append((cell_idx, target_rem, all_stop))

    if test_inp_list:
        b_inp = torch.tensor(test_inp_list, dtype=torch.long)
        b_w = make_pooling_weights(b_inp, stopword_mask)
        with torch.no_grad():
            V_b_all = torch.cat([encoder(b_inp[s : s + 32], b_w[s : s + 32]) for s in range(0, len(b_inp), 32)], dim=0)
        for idx, (c_idx, rem, all_stop) in enumerate(test_meta):
            v_cell = V_b_all[idx, c_idx]
            en = float((v_cell.float() ** 2).sum().item())
            sv = float(sim_gr(v_cell, void_concept).item())
            rec = {"energy": en, "sim_void": sv, "is_void_triggered": (en < 0.35 or sv > 0.80)}
            boundary_stats[rem].append(rec)
            if all_stop:
                stop_only_boundary[rem].append(rec)
            if c_idx + 1 < V_b_all.shape[1]:
                v_next = V_b_all[idx, c_idx + 1]
                en_v = float((v_next.float() ** 2).sum().item())
                sv_v = float(sim_gr(v_next, void_concept).item())
                boundary_stats[0].append({"energy": en_v, "sim_void": sv_v, "is_void_triggered": (en_v < 0.35 or sv_v > 0.80)})

    worst_phrases = [".", "\n", ",", "!", "?", " и", " the", ". ", ", and"]
    worst_samples = []
    base_ids = encode_ids(tokenizer, "Математика изучает числа и структуры")[:4]
    for wp in worst_phrases:
        wp_ids = encode_ids(tokenizer, wp)
        if 1 <= len(wp_ids) <= 3:
            seq = base_ids + wp_ids + [PAD_ID] * (16 - 4 - len(wp_ids))
            worst_samples.append((seq, wp, wp_ids))

    worst_case_res = []
    if worst_samples:
        w_inp = torch.tensor([x[0] for x in worst_samples], dtype=torch.long)
        w_w = make_pooling_weights(w_inp, stopword_mask)
        with torch.no_grad():
            V_w = encoder(w_inp, w_w)
        for idx, (_, wp_txt, wp_ids) in enumerate(worst_samples):
            v_cell = V_w[idx, 1]
            en = float((v_cell.float() ** 2).sum().item())
            sv = float(sim_gr(v_cell, void_concept).item())
            worst_case_res.append(
                {
                    "phrase": repr(wp_txt),
                    "n_real": len(wp_ids),
                    "energy": en,
                    "sim_void": sv,
                    "is_void_triggered": (en < 0.35 or sv > 0.80),
                }
            )

    def summarize(lst: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not lst:
            return {"count": 0}
        ens = np.array([x["energy"] for x in lst])
        svs = np.array([x["sim_void"] for x in lst])
        return {
            "count": len(lst),
            "energy_min": float(np.min(ens)),
            "energy_median": float(np.percentile(ens, 50)),
            "energy_mean": float(np.mean(ens)),
            "sim_void_max": float(np.max(svs)),
            "is_void_triggered_pct": float(np.mean([x["is_void_triggered"] for x in lst]) * 100.0),
        }

    return {
        "natural_boundary_by_n_real": {str(k): summarize(v) for k, v in boundary_stats.items()},
        "stopword_only_boundary_by_n_real": {str(k): summarize(v) for k, v in stop_only_boundary.items()},
        "worst_case_explicit_stops": worst_case_res,
    }


def check_q4_and_q6(
    domain_cells: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    sep_concept: torch.Tensor,
    void_concept: torch.Tensor,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Q4 (баланс лоссов и градиентов) и Q6 (старт Soft-Gate AdaLN и L_adaln)."""
    sample_Vs, sample_nreal = [], []
    for d_name in ["russian_reasoning", "camel_math", "camel_physics", "magpie_reasoning"]:
        if d_name in domain_cells:
            V_all, n_rc, _ = domain_cells[d_name]
            sample_Vs.append(V_all[:2])
            sample_nreal.append(n_rc[:2])
    V_ans_b = torch.cat(sample_Vs, dim=0)
    n_rc_b = torch.cat(sample_nreal, dim=0)
    B = V_ans_b.shape[0]
    V0 = torch.cat([sep_concept.unsqueeze(0).unsqueeze(0).expand(B, 1, 8, 128), V_ans_b], dim=1).float()

    w_cell = torch.zeros(B, 128, dtype=torch.float32)
    for b in range(B):
        nz = torch.where(n_rc_b[b] > 0)[0]
        i_last = (int(nz[-1].item()) if len(nz) > 0 else 0) + 1
        for i in range(1, 128):
            if i < i_last:
                w_cell[b, i] = 1.0
            elif i in (i_last, i_last + 1):
                w_cell[b, i] = 2.0
            else:
                w_cell[b, i] = 0.25

    def eval_losses(t_val: float, pred_mode: str) -> Dict[str, Any]:
        t_mat = torch.full((B, 128), t_val, dtype=torch.float32)
        t_mat[:, 0] = 0.0
        Vt = (1.0 - t_mat.unsqueeze(-1).unsqueeze(-1)) * V0 + t_mat.unsqueeze(-1).unsqueeze(-1) * 0.05 * torch.randn_like(V0)
        if pred_mode == "init_identity":
            V_hat_init = Vt + 0.001 * torch.randn_like(V0)
        elif pred_mode == "mid_50pct":
            V_hat_init = 0.5 * V0 + 0.5 * Vt + 0.01 * torch.randn_like(V0)
        else:
            V_hat_init = V0 + 0.01 * torch.randn_like(V0)

        m_active = t_mat > 1e-3
        w_snr = torch.clamp(1.0 / torch.clamp(t_mat, min=0.1), max=5.0)
        w_tilde = m_active.float() * w_cell * w_snr

        def grad_norm(fn) -> Tuple[float, float]:
            vh = V_hat_init.clone().detach().requires_grad_(True)
            val = fn(vh)
            val.backward()
            return float(val.item()), float(vh.grad.norm().item())

        vb, gb = grad_norm(lambda vh: (w_tilde * F.smooth_l1_loss(vh, V0, beta=0.02, reduction="none").mean(dim=(-2, -1))).sum() / (w_tilde.sum() + 1e-6))

        def l_gr_fn(vh: torch.Tensor) -> torch.Tensor:
            Gh = torch.matmul(vh, vh.transpose(-2, -1))
            G0 = torch.matmul(V0, V0.transpose(-2, -1))
            cross = torch.matmul(vh, V0.transpose(-2, -1))
            l_cell = (Gh ** 2).sum(dim=(-2, -1)) + (G0 ** 2).sum(dim=(-2, -1)) - 2.0 * (cross ** 2).sum(dim=(-2, -1))
            return (w_tilde * l_cell).sum() / (w_tilde.sum() + 1e-6)

        vg, gg = grad_norm(l_gr_fn)
        vgc, ggc = grad_norm(lambda vh: (w_tilde * (1.0 - sim_gr(vh, V0))).sum() / (w_tilde.sum() + 1e-6))

        return {
            "L_basis": {"val": vb, "grad": gb},
            "0.5*L_Gr": {"val": 0.5 * vg, "grad": 0.5 * gg, "raw_val": vg, "raw_grad": gg},
            "0.02*L_Gr": {"val": 0.02 * vg, "grad": 0.02 * gg},
            "ratio_0.5Gr_over_Basis_val": (0.5 * vg) / max(vb, 1e-12),
            "ratio_0.5Gr_over_Basis_grad": (0.5 * gg) / max(gb, 1e-12),
            "1-SimGr": {"val": vgc, "grad": ggc},
        }

    q4_res = {
        mode: {f"t={tv}": eval_losses(tv, mode) for tv in [0.1, 0.5, 1.0]}
        for mode in ["init_identity", "mid_50pct", "near_converged"]
    }

    # Q6: ConditionEmbedder + 9x Soft-Gate AdaLN
    class SinusoidalTimeEmbedding(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.dim = dim

        def forward(self, t: torch.Tensor) -> torch.Tensor:
            half = self.dim // 2
            freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1))
            args = t.float().unsqueeze(-1) * freqs
            return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    class ConditionEmbedder(nn.Module):
        def __init__(self, void_ref: torch.Tensor):
            super().__init__()
            self.register_buffer("void_ref", void_ref.float())
            self.t_local_emb = SinusoidalTimeEmbedding(128)
            self.t_global_emb = SinusoidalTimeEmbedding(64)
            self.mlp = nn.Sequential(nn.Linear(200, 256), nn.SiLU(), nn.Linear(256, 256), nn.LayerNorm(256))

        def forward(self, V_t: torch.Tensor, t_mat: torch.Tensor, m_prompt: torch.Tensor) -> torch.Tensor:
            B_c, N_c, _, _ = V_t.shape
            Vf = V_t.float()
            G = torch.matmul(Vf, Vf.transpose(-2, -1))
            energy = torch.diagonal(G, dim1=-2, dim2=-1).sum(dim=-1)
            s0 = torch.clamp(energy / 2.52, max=2.0)
            s1 = (G ** 2).sum(dim=(-2, -1)) / ((energy + 1e-6) ** 2)
            s2 = 1.0 / (8.0 * torch.clamp(s1, min=1e-4))
            s3 = sim_gr(Vf, self.void_ref.view(1, 1, 8, 128).expand(B_c, N_c, 8, 128))
            s4 = (t_mat < 1e-3).float()
            s5 = m_prompt.float().view(B_c, 1).expand(B_c, N_c)
            sensors = torch.stack([s0, s1, s2, s3, s4, s5, torch.zeros_like(s0), torch.zeros_like(s0)], dim=-1)
            e_loc = self.t_local_emb(t_mat)
            e_glob = self.t_global_emb(t_mat[:, 1:].mean(dim=-1, keepdim=True).expand(B_c, N_c))
            return self.mlp(torch.cat([e_loc, e_glob, sensors], dim=-1))

    cond_emb = ConditionEmbedder(void_concept)
    adaln_projs = nn.ModuleList([nn.Linear(256, 6 * 1024) for _ in range(9)])
    for proj in adaln_projs:
        nn.init.normal_(proj.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(proj.bias)
        with torch.no_grad():
            proj.bias[2 * 1024 : 3 * 1024] = 0.02
            proj.bias[5 * 1024 : 6 * 1024] = 0.02

    t_mix = torch.zeros(B, 128, dtype=torch.float32)
    for b in range(B):
        if b < B // 2:
            t_mix[b, 1:] = torch.sigmoid(torch.randn(1) * 1.1 + 0.2).item()
        elif b < (3 * B) // 4:
            tvs = torch.empty(127).uniform_(0.25, 1.0)
            tvs[torch.rand(127) < 0.35] = 0.0
            t_mix[b, 1:] = tvs
        else:
            t_mix[b, 1:] = torch.linspace(0.1, 0.95, 127)

    Vt_mix = (1.0 - t_mix.unsqueeze(-1).unsqueeze(-1)) * V0 + t_mix.unsqueeze(-1).unsqueeze(-1) * 0.05 * torch.randn_like(V0)
    c = cond_emb(Vt_mix, t_mix, (torch.rand(B) > 0.15).float())
    m_active_mix = t_mix > 1e-3

    layer_stds, l_terms = [], []
    for proj in adaln_projs:
        out = proj(c)
        M = torch.cat([out[..., 0:2048], out[..., 3072:5120]], dim=-1)
        std_d = M[m_active_mix].std(dim=0, unbiased=False).mean()
        layer_stds.append(float(std_d.item()))
        l_terms.append(F.relu(1e-2 - std_d))

    l_adaln = torch.stack(l_terms).mean()
    (0.05 * l_adaln).backward()
    q6_res = {
        "c_rms": float(torch.sqrt((c.detach() ** 2).mean()).item()),
        "c_std_across_tokens": float(c.detach()[m_active_mix].std(dim=0).mean().item()),
        "sigma_adaln_mean_at_step0": float(np.mean(layer_stds)),
        "L_adaln_raw_at_step0": float(l_adaln.item()),
        "L_adaln_weighted_0.05_at_step0": float((0.05 * l_adaln).item()),
        "adaln_proj_weight_grad_norm_mean": float(np.mean([p.weight.grad.norm().item() for p in adaln_projs])),
    }
    return q4_res, q6_res


def run_all_checks(max_per_domain: int = 40) -> Dict[str, Any]:
    random.seed(DEFAULT_SEED)
    np.random.seed(DEFAULT_SEED)
    torch.manual_seed(DEFAULT_SEED)

    tokenizer = get_tokenizer()
    encoder = get_encoder(device="cpu", eval_mode=True)
    decoder = get_decoder(encoder=encoder, device="cpu", eval_mode=True)
    for p in list(encoder.parameters()) + list(decoder.parameters()):
        p.requires_grad_(False)

    sep_concept = torch.load(REPO_ROOT / "storage/components/sep_concept.pt", map_location="cpu", weights_only=False)
    void_concept = torch.load(REPO_ROOT / "storage/components/void_concept.pt", map_location="cpu", weights_only=False)
    stopword_mask = torch.load(REPO_ROOT / "data/plan C/tekken_stopwords_mask.pt", map_location="cpu", weights_only=False)["mask"].bool()

    domains = load_domain_texts(max_per_domain=max_per_domain)
    all_texts = [t for ts in domains.values() for t in ts]

    q1 = check_q1(encoder, decoder, tokenizer, sep_concept, void_concept, stopword_mask, all_texts)
    domain_cells = encode_domains(encoder, tokenizer, stopword_mask, domains)
    q7 = check_q7(domain_cells, sep_concept)
    q2 = check_q2(domain_cells)
    q3 = check_q3(encoder, tokenizer, void_concept, stopword_mask, all_texts)
    q4, q6 = check_q4_and_q6(domain_cells, sep_concept, void_concept)

    # Базовые инварианты регрессионного теста
    assert q1["sep_max_diff_vs_encoder_uniform"] < 1e-6, "sep_concept.pt не совпал с выходом энкодера"
    assert q1["long_508_tokens"]["acc_127_pct"] > 99.0, "Деградация реконструкции на N=127 (508 токенов)"
    assert q2["adjacent_full_cells"]["p99"] < 0.85, "Соседний p99 превысил tau_adj=0.85"
    assert q2["cross_sample_different_examples"]["p99"] < 0.50, "Межпримерный p99 превысил tau_bary=0.50"
    for rem in ["1", "2", "3"]:
        assert q3["natural_boundary_by_n_real"][rem]["is_void_triggered_pct"] == 0.0, f"Ложное срабатывание is_void при n_real={rem}"
    assert all(not w["is_void_triggered"] for w in q3["worst_case_explicit_stops"]), "Ложное срабатывание is_void на стоп-токене"
    assert 0.04 <= q7["ALL_COMBINED_FULL_CELLS"]["rms_mean"] <= 0.06, "Выход sigma_data за ожидаемый диапазон"

    return {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "Q6": q6, "Q7": q7}


def main():
    parser = argparse.ArgumentParser(description="Верификация и замеры Q1–Q7 для Фазы 2 Plan C")
    parser.add_argument("--max-per-domain", type=int, default=40, help="Макс. текстов на каждый домен")
    parser.add_argument("--save-json", type=str, default=None, help="Путь для сохранения подробного JSON-отчёта")
    args = parser.parse_args()

    results = run_all_checks(max_per_domain=args.max_per_domain)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nALL PHASE 2 PRE-FLIGHT CHECKS PASSED.")


if __name__ == "__main__":
    main()
