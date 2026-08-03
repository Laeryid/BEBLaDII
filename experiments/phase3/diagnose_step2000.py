"""
Комплексная диагностика чекпоинта phase3_step_2000.pth
======================================================
Анализирует:
  1. Метрики из истории обучения (metrics_history)
  2. Нормы весов по слоям (DUS + AdaLN); cosine(EMA, raw)
  3. AdaLN Sensitivity: насколько сильно модулируется по t
  4. Residual Stream Drift: нормы активаций по слоям для t=0.1/0.5/0.9
  5. Jacobian Lens: grad_norm/act_norm по каждому слою, маркеры DEAD/EXPLODING
  6. Topological Metrics: isotropy / rank1 / norm_cv для L4/L19/L39
  7. Identity Preservation: cos(pred, clean) для t=0.1..0.9
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

project_root = r"C:\Experiments\BEBLaDII"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.phase3.inspect_and_test_phase3 import (
    load_checkpoint_and_inspect, load_models, encode_clean_text
)
from src.beb_la_dii.utils.loss import safe_normalize

CKPT_PATH = r"C:\Experiments\BEBLaDII\experiments\phase3\local_checkpoints\phase3_step_2000.pth"


# ─────────────────────────────────────────────
# 1. Метрики из history
# ─────────────────────────────────────────────
def analyze_metrics_history(state):
    print("\n" + "=" * 70)
    print("БЛОК 1: История метрик (последние 10 шагов из metrics_history)")
    print("=" * 70)
    history = state.get("metrics_history", [])
    if not history:
        print("  [!] metrics_history отсутствует в чекпоинте.")
        return

    print(f"  Всего записей: {len(history)}")
    last = history[-10:]
    keys_of_interest = [
        "loss", "denoising_loss", "seq_rkd_loss", "prior_loss", "cov_loss",
        "var_match_loss", "div_loss", "adaln_l2_loss", "grad_norm",
        "cos_sim_t_low", "cos_sim_t_mid", "cos_sim_t_high",
        "cos_h39_t_low", "cos_h39_t_mid", "cos_h39_t_high",
        "adaln_attn_scale_dev", "adaln_attn_shift_norm", "adaln_w_norm",
        "lr_dus", "step"
    ]

    print(f"{'step':>6}", end="")
    for k in keys_of_interest:
        if k != "step":
            print(f" | {k[:14]:>14}", end="")
    print()
    print("-" * (6 + len([k for k in keys_of_interest if k != "step"]) * 17))

    for rec in last:
        step_val = int(rec.get("step", -1))
        print(f"{step_val:>6}", end="")
        for k in keys_of_interest:
            if k != "step":
                val = rec.get(k, float('nan'))
                try:
                    print(f" | {float(val):>14.5f}", end="")
                except Exception:
                    print(f" | {'?':>14}", end="")
        print()

    # Тренд
    if len(history) >= 10:
        first5 = history[:5]
        last5  = history[-5:]
        print("\n  Тренд (ср. первых 5 vs последних 5):")
        for k in ["loss", "denoising_loss", "div_loss", "adaln_l2_loss",
                  "adaln_attn_scale_dev", "grad_norm", "cos_h39_t_low"]:
            try:
                v_f = sum(float(r.get(k, 0)) for r in first5) / 5
                v_l = sum(float(r.get(k, 0)) for r in last5) / 5
                arrow = "↓" if v_l < v_f else "↑"
                print(f"    {k:<30}: {v_f:.5f} → {v_l:.5f}  {arrow}")
            except Exception:
                pass


# ─────────────────────────────────────────────
# 2. Нормы весов
# ─────────────────────────────────────────────
def analyze_weight_norms(state):
    print("\n" + "=" * 70)
    print("БЛОК 2: Нормы весов (DUS + AdaLN + EMA divergence)")
    print("=" * 70)

    dus = state.get("dus", state.get("dus_ema", {}))
    if not dus:
        print("  [!] DUS веса не найдены.")
    else:
        layer_norms = {}
        for k, v in dus.items():
            if ".layers." in k:
                try:
                    idx = int(k.split(".layers.")[1].split(".")[0])
                    if idx not in layer_norms:
                        layer_norms[idx] = []
                    layer_norms[idx].append(v.float().norm().item())
                except Exception:
                    pass

        print(f"  DUS: слоев с параметрами: {len(layer_norms)}")
        avg_norms = {i: sum(v) / len(v) for i, v in layer_norms.items()}
        print(f"  {'Layer':>6} | {'AvgWeightNorm':>14}")
        for i in sorted(avg_norms.keys()):
            if i % 5 == 0 or i == max(avg_norms.keys()):
                print(f"  {i:>6} | {avg_norms[i]:>14.4f}")

    for comp in ["adaLN_attn", "adaLN_mlp"]:
        sd = state.get(comp, state.get(f"{comp}_ema", {}))
        if not sd:
            continue
        norms_by_layer = {}
        for k, v in sd.items():
            try:
                idx = int(k.split(".")[0])
                norms_by_layer.setdefault(idx, []).append(v.float().norm().item())
            except Exception:
                pass
        avg = {i: sum(v) / len(v) for i, v in norms_by_layer.items()}
        min_n = min(avg.values()); max_n = max(avg.values())
        mean_n = sum(avg.values()) / len(avg)
        print(f"  {comp}: mean_w_norm={mean_n:.4f}, min={min_n:.4f}, max={max_n:.4f}")

    print("\n  EMA vs non-EMA (cosine sim весов DUS):")
    dus_raw = state.get("dus", {})
    dus_ema = state.get("dus_ema", {})
    if dus_raw and dus_ema:
        sims = []
        for k in list(dus_raw.keys())[:50]:
            if k in dus_ema:
                w1 = dus_raw[k].float().flatten()
                w2 = dus_ema[k].float().flatten()
                if w1.numel() > 0:
                    sims.append(F.cosine_similarity(w1.unsqueeze(0), w2.unsqueeze(0)).item())
        if sims:
            print(f"    DUS cosine(raw, ema) = {sum(sims)/len(sims):.6f}  (1.0 = идентичны)")
    else:
        print("    [!] Один из dus / dus_ema отсутствует.")


# ─────────────────────────────────────────────
# 3. AdaLN Sensitivity
# ─────────────────────────────────────────────
def analyze_adaln_sensitivity(diff_model, device):
    print("\n" + "=" * 70)
    print("БЛОК 3: AdaLN Sensitivity")
    print("=" * 70)

    t_vals = torch.tensor([0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], device=device)
    with torch.no_grad():
        t_sin = diff_model.t_sin_embed(t_vals)
        t_emb = diff_model.t_proj(t_sin)

    print(f"  Первые 5 слоев: delta(t=0.02, t=1.0)")
    print(f"  {'L':>3} | {'attn|Δshift|':>12} | {'attn|Δscale|':>12} | {'mlp|Δshift|':>11} | {'mlp|Δscale|':>11}")
    print(f"  {'-'*3}-+-{'-'*12}-+-{'-'*12}-+-{'-'*11}-+-{'-'*11}")
    with torch.no_grad():
        for i in range(min(5, len(diff_model.adaLN_attn))):
            sh_a1, sc_a1 = diff_model.adaLN_attn[i](t_emb[0:1])
            sh_a2, sc_a2 = diff_model.adaLN_attn[i](t_emb[-1:])
            sh_m1, sc_m1 = diff_model.adaLN_mlp[i](t_emb[0:1])
            sh_m2, sc_m2 = diff_model.adaLN_mlp[i](t_emb[-1:])
            print(f"  {i:>3} | {(sh_a2-sh_a1).abs().mean().item():>12.6f} | "
                  f"{(sc_a2-sc_a1).abs().mean().item():>12.6f} | "
                  f"{(sh_m2-sh_m1).abs().mean().item():>11.6f} | "
                  f"{(sc_m2-sc_m1).abs().mean().item():>11.6f}")

        # Средние по всем слоям при t=0.5
        sh_means, sc_means = [], []
        for i in range(len(diff_model.adaLN_attn)):
            sh, sc = diff_model.adaLN_attn[i](t_emb[3:4])
            sh_means.append(sh.squeeze().abs().mean().item())
            sc_means.append((sc.squeeze() - 1.0).abs().mean().item())
        print(f"\n  Все слои, t=0.5:")
        print(f"    mean |shift|   = {sum(sh_means)/len(sh_means):.6f}")
        print(f"    mean |scale-1| = {sum(sc_means)/len(sc_means):.6f}")

        # Попарное cosine sim между выходами для разных t
        outs = []
        for i in range(len(diff_model.adaLN_attn)):
            sh, sc = diff_model.adaLN_attn[i](t_emb)
            delta = torch.cat([sh.squeeze(1), sc.squeeze(1) - 1.0], dim=-1)
            outs.append(safe_normalize(delta, dim=-1))
        out_stack = torch.stack(outs).mean(dim=0)  # [7, 2048]
        sim = out_stack @ out_stack.T
        mask_off = ~torch.eye(7, dtype=torch.bool, device=device)
        batch_div = sim[mask_off].mean().item()
        print(f"    cosine_sim(AdaLN_ti, AdaLN_tj) mean = {batch_div:.6f}  (→1 плохо, →0 хорошо)")


# ─────────────────────────────────────────────
# 4. Residual Stream Drift
# ─────────────────────────────────────────────
def analyze_residual_stream(diff_model, embeddings, encoder, device, dtype, sep_embed):
    print("\n" + "=" * 70)
    print("БЛОК 4: Residual Stream Drift (норма активаций по слоям)")
    print("=" * 70)

    from src.beb_la_dii.utils.tokenizer import get_tokenizer
    tokenizer = get_tokenizer("Qwen/Qwen2.5-1.5B")
    text = "The quick brown fox jumps over the lazy dog."
    z_clean, attn_mask = encode_clean_text(text, tokenizer, embeddings, encoder, device, dtype)

    hidden_norms = {}
    hooks = []

    def get_hook(idx):
        def h(module, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            hidden_norms[idx] = x.detach().float().norm(dim=-1).mean().item()
        return h

    for i, layer in enumerate(diff_model.dus.layers):
        hooks.append(layer.register_forward_hook(get_hook(i)))

    diff_model.eval()
    for t_val in [0.1, 0.5, 0.9]:
        hidden_norms.clear()
        with torch.no_grad():
            _ = diff_model(z_clean, attn_mask,
                           torch.tensor([t_val], device=device),
                           sep_embed=sep_embed)
        norms = [hidden_norms.get(i, 0.0) for i in range(len(diff_model.dus.layers))]
        snap = {0: norms[0], 9: norms[9], 19: norms[19], 29: norms[29], 39: norms[39]}
        print(f"  t={t_val}: L0={snap[0]:.2f} L9={snap[9]:.2f} L19={snap[19]:.2f} "
              f"L29={snap[29]:.2f} L39={snap[39]:.2f} | "
              f"max={max(norms):.2f}@L{norms.index(max(norms))} ratio={max(norms)/max(norms[0],1e-6):.1f}x")

    for h in hooks:
        h.remove()


# ─────────────────────────────────────────────
# 5. Jacobian Lens
# ─────────────────────────────────────────────
def analyze_gradient_flow(diff_model, embeddings, encoder, device, dtype, sep_embed):
    print("\n" + "=" * 70)
    print("БЛОК 5: Jacobian Lens — Gradient Flow")
    print("=" * 70)

    from src.beb_la_dii.utils.tokenizer import get_tokenizer
    tokenizer = get_tokenizer("Qwen/Qwen2.5-1.5B")
    text = "The quick brown fox jumps over the lazy dog."
    z_clean, attn_mask = encode_clean_text(text, tokenizer, embeddings, encoder, device, dtype)

    for t_val in [0.9, 0.5, 0.1]:
        print(f"\n  --- t={t_val} ---")

        diff_model.train()
        for p in diff_model.parameters():
            p.requires_grad_(True)
        diff_model.zero_grad()

        mu    = math.cos(t_val * math.pi / 2)
        sigma = math.sin(t_val * math.pi / 2)
        eps   = torch.randn_like(z_clean)
        z_t   = safe_normalize(mu * z_clean + sigma * eps, dim=-1)
        z_t.requires_grad_(True)

        hidden_states = {}
        hooks = []

        def get_h(idx):
            def hook(m, inp, out):
                x = out[0] if isinstance(out, tuple) else out
                x.retain_grad()
                hidden_states[idx] = x
            return hook

        for i, layer in enumerate(diff_model.dus.layers):
            hooks.append(layer.register_forward_hook(get_h(i)))

        with torch.enable_grad():
            pred = diff_model(z_t, attn_mask,
                              torch.tensor([t_val], device=device),
                              sep_embed=sep_embed)
            loss = F.mse_loss(pred[attn_mask.bool()], z_clean[attn_mask.bool()])
            loss.backward()

        for h in hooks:
            h.remove()

        dead = exploding = 0
        print(f"  {'L':>3} | {'GradNorm':>10} | {'ActNorm':>10} | Flag")
        print(f"  {'-'*3}-+-{'-'*10}-+-{'-'*10}-+--------")

        for i in range(len(diff_model.dus.layers)):
            if i not in hidden_states or hidden_states[i].grad is None:
                if i % 10 == 0:
                    print(f"  {i:>3} | {'NO GRAD':>10} | {'':>10} |")
                continue
            gn = hidden_states[i].grad.norm().item()
            an = hidden_states[i].norm().item()
            flag = ""
            if gn < 1e-6:
                flag = "<< DEAD >>"; dead += 1
            elif gn > 10.0:
                flag = "<< EXPLODING >>"; exploding += 1
            elif gn > 2.0:
                flag = "! high"
            # Печатаем каждые 5 слоев + все с флагами
            if i % 5 == 0 or i == len(diff_model.dus.layers) - 1 or flag:
                print(f"  {i:>3} | {gn:>10.6f} | {an:>10.4f} | {flag}")

        print(f"  >> Итого: DEAD={dead}, EXPLODING={exploding}, "
              f"{'OK' if dead == 0 and exploding == 0 else 'ВНИМАНИЕ!'}")


# ─────────────────────────────────────────────
# 6. Topological Metrics
# ─────────────────────────────────────────────
def compute_topological_metrics(h):
    centered = h - h.mean(dim=0, keepdim=True)
    _, s, _  = torch.linalg.svd(centered.float(), full_matrices=False)
    iso  = (s.sum() ** 2) / (len(s) * (s ** 2).sum()) if len(s) > 0 else torch.tensor(0.0)
    r1   = (s[0] ** 2) / (s ** 2).sum() if len(s) > 0 else torch.tensor(0.0)
    norms = h.norm(dim=-1)
    cv   = norms.std() / norms.mean() if norms.mean() > 0 else torch.tensor(0.0)
    return iso.item(), r1.item(), cv.item()


def analyze_topology(diff_model, embeddings, encoder, device, dtype, sep_embed):
    print("\n" + "=" * 70)
    print("БЛОК 6: Topological Analysis")
    print("=" * 70)

    from src.beb_la_dii.utils.tokenizer import get_tokenizer
    tokenizer = get_tokenizer("Qwen/Qwen2.5-1.5B")

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Мама мыла раму, а папа чинил телевизор.",
        "Quantum mechanics governs subatomic particle interactions.",
        "In the beginning God created the heavens and the earth.",
        "2 + 2 = 4. The sum of angles in a triangle is 180 degrees.",
    ]

    diff_model.eval()
    all_z, all_masks = [], []
    for txt in texts:
        z, m = encode_clean_text(txt, tokenizer, embeddings, encoder, device, dtype)
        all_z.append(z); all_masks.append(m)

    max_len = max(z.shape[1] for z in all_z)
    z_batch = torch.cat([F.pad(z, (0, 0, 0, max_len - z.size(1))) for z in all_z], dim=0)
    m_batch = torch.cat([F.pad(m, (0, max_len - m.size(1)), value=0) for m in all_masks], dim=0)
    active  = z_batch[m_batch.bool()]
    iso0, r0, cv0 = compute_topological_metrics(active)
    print(f"  [z_clean baseline] Isotropy={iso0:.4f} Rank1={r0:.4f} NormCV={cv0:.4f}")

    snap = {}

    def mk_hook(idx):
        def hook(m, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            snap[idx] = x.detach().float()
        return hook

    h_last  = diff_model.dus.layers[-1].register_forward_hook(mk_hook(39))
    h_mid   = diff_model.dus.layers[19].register_forward_hook(mk_hook(19))
    h_early = diff_model.dus.layers[4].register_forward_hook(mk_hook(4))

    for t_val in [0.1, 0.5, 0.9]:
        with torch.no_grad():
            _ = diff_model(z_batch, m_batch,
                           torch.tensor([t_val], device=device),
                           sep_embed=sep_embed)
        for layer_idx, label in [(4, "L4 "), (19, "L19"), (39, "L39")]:
            if layer_idx not in snap:
                continue
            active_h = snap[layer_idx][m_batch.bool()]
            iso, r1, cv = compute_topological_metrics(active_h)
            flag = " *** COLLAPSE ***" if r1 > 0.8 else (" ! rank1 high" if r1 > 0.5 else "")
            print(f"  t={t_val} | {label}: Iso={iso:.4f} R1={r1:.4f} CV={cv:.4f}{flag}")

        # Collapse check между текстами
        means = []
        for b in range(len(texts)):
            mb = m_batch[b].bool()
            if snap.get(39) is not None and snap[39][b][mb].shape[0] > 0:
                means.append(snap[39][b][mb].mean(dim=0))
        if len(means) >= 3:
            cos01 = F.cosine_similarity(means[0].unsqueeze(0), means[1].unsqueeze(0)).item()
            cos02 = F.cosine_similarity(means[0].unsqueeze(0), means[2].unsqueeze(0)).item()
            cos12 = F.cosine_similarity(means[1].unsqueeze(0), means[2].unsqueeze(0)).item()
            print(f"  t={t_val} | L39 cross-text cos: (0↔1)={cos01:.4f} (0↔2)={cos02:.4f} (1↔2)={cos12:.4f}")

    h_last.remove(); h_mid.remove(); h_early.remove()


# ─────────────────────────────────────────────
# 7. Identity Preservation
# ─────────────────────────────────────────────
def analyze_identity(diff_model, embeddings, encoder, device, dtype, sep_embed):
    print("\n" + "=" * 70)
    print("БЛОК 7: Identity Preservation — Cos(pred, clean)")
    print("=" * 70)

    from src.beb_la_dii.utils.tokenizer import get_tokenizer
    tokenizer = get_tokenizer("Qwen/Qwen2.5-1.5B")

    texts = [
        ("English", "The quick brown fox jumps over the lazy dog."),
        ("Russian", "Мама мыла раму, а папа чинил телевизор."),
        ("Science", "Quantum mechanics governs subatomic particle interactions."),
    ]

    diff_model.eval()
    t_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    print(f"  {'Text':<10} | " + " | ".join(f"t={t:>3.1f}" for t in t_list))
    print(f"  {'-'*10}-+-" + "-+-".join(["-"*7] * len(t_list)))

    for label, txt in texts:
        z_clean, attn_mask = encode_clean_text(txt, tokenizer, embeddings, encoder, device, dtype)
        mask_b = attn_mask.bool()
        row = []
        for t_val in t_list:
            with torch.no_grad():
                pred = diff_model(z_clean, attn_mask,
                                  torch.tensor([t_val], device=device),
                                  sep_embed=sep_embed)
            cs = (pred[mask_b] * z_clean[mask_b]).sum(dim=-1).mean().item()
            row.append(cs)
        print(f"  {label:<10} | " + " | ".join(f"{v:>7.4f}" for v in row))

    print("\n  Ожидаемо: t=0.1 → ~1.0 (Identity Gate); t=0.9 → реальный денойзинг")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float32
    print(f"Device: {device}")
    print(f"Чекпоинт: {CKPT_PATH}")

    if not os.path.exists(CKPT_PATH):
        print(f"[FATAL] Файл не найден: {CKPT_PATH}")
        return

    state = load_checkpoint_and_inspect(CKPT_PATH)
    print(f"  Ключи чекпоинта: {list(state.keys())}")
    print(f"  Шаг: {state.get('step', '?')}")

    analyze_metrics_history(state)
    analyze_weight_norms(state)

    print("\n[*] Загрузка моделей...")
    tokenizer, embeddings, encoder, decoder, diff_model, lm_head_weight, sep_embed = \
        load_models(device, dtype, state)
    diff_model.eval()

    analyze_adaln_sensitivity(diff_model, device)
    analyze_residual_stream(diff_model, embeddings, encoder, device, dtype, sep_embed)
    analyze_gradient_flow(diff_model, embeddings, encoder, device, dtype, sep_embed)
    analyze_topology(diff_model, embeddings, encoder, device, dtype, sep_embed)
    analyze_identity(diff_model, embeddings, encoder, device, dtype, sep_embed)

    print("\n" + "=" * 70)
    print("Диагностика завершена.")
    print("=" * 70)


if __name__ == "__main__":
    main()
