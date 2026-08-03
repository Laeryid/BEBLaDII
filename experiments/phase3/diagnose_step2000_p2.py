"""Блоки 6-7 (фикс маски для sep_embed +1 токен)"""
import os, sys, math, torch, torch.nn.functional as F

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
project_root = r"C:\Experiments\BEBLaDII"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.phase3.inspect_and_test_phase3 import load_checkpoint_and_inspect, load_models, encode_clean_text
from src.beb_la_dii.utils.loss import safe_normalize

CKPT_PATH = r"C:\Experiments\BEBLaDII\experiments\phase3\local_checkpoints\phase3_step_2000.pth"

def compute_topo(h):
    c = h - h.mean(dim=0, keepdim=True)
    _, s, _ = torch.linalg.svd(c.float(), full_matrices=False)
    iso = (s.sum()**2) / (len(s)*(s**2).sum())
    r1  = (s[0]**2) / (s**2).sum()
    norms = h.norm(dim=-1)
    cv = norms.std() / norms.mean() if norms.mean() > 0 else torch.tensor(0.0)
    return iso.item(), r1.item(), cv.item()


def analyze_topology(diff_model, embeddings, encoder, device, dtype, sep_embed):
    print("\n" + "="*70)
    print("БЛОК 6: Topological Analysis")
    print("="*70)

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
    z_batch = torch.cat([F.pad(z, (0,0,0,max_len-z.size(1))) for z in all_z], dim=0)
    m_batch = torch.cat([F.pad(m, (0,max_len-m.size(1)), value=0) for m in all_masks], dim=0)

    iso0, r0, cv0 = compute_topo(z_batch[m_batch.bool()])
    print(f"  [z_clean baseline] Iso={iso0:.4f} R1={r0:.4f} CV={cv0:.4f}")

    snap = {}
    def mk_hook(idx):
        def hook(mod, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            # sep_embed добавляет 1 токен в начало → обрезаем
            snap[idx] = x[:, 1:, :].detach().float()  # [B, T, D]
        return hook

    hooks = [
        diff_model.dus.layers[4].register_forward_hook(mk_hook(4)),
        diff_model.dus.layers[19].register_forward_hook(mk_hook(19)),
        diff_model.dus.layers[-1].register_forward_hook(mk_hook(39)),
    ]

    for t_val in [0.1, 0.5, 0.9]:
        snap.clear()
        with torch.no_grad():
            _ = diff_model(z_batch, m_batch, torch.tensor([t_val], device=device), sep_embed=sep_embed)

        for layer_idx, label in [(4, "L4 "), (19, "L19"), (39, "L39")]:
            if layer_idx not in snap: continue
            # snap[idx] shape: [B, T_orig, D] после обрезки
            # m_batch: [B, T_orig]
            active_h = snap[layer_idx][m_batch.bool()]  # [N_active, D]
            iso, r1, cv = compute_topo(active_h)
            flag = " *** COLLAPSE ***" if r1 > 0.8 else (" ! rank1 high" if r1 > 0.5 else "")
            print(f"  t={t_val} | {label}: Iso={iso:.4f} R1={r1:.4f} CV={cv:.4f}{flag}")

        means = []
        for b in range(len(texts)):
            mb = m_batch[b].bool()
            if snap.get(39) is not None and mb.sum() > 0:
                # snap[39][b] shape: [T_orig, D]
                s39_b = snap[39][b]
                if s39_b.shape[0] == mb.shape[0]:
                    means.append(s39_b[mb].mean(dim=0))
        if len(means) >= 3:
            c01 = F.cosine_similarity(means[0].unsqueeze(0), means[1].unsqueeze(0)).item()
            c02 = F.cosine_similarity(means[0].unsqueeze(0), means[2].unsqueeze(0)).item()
            c12 = F.cosine_similarity(means[1].unsqueeze(0), means[2].unsqueeze(0)).item()
            print(f"  t={t_val} | L39 cross-text cos: (0↔1)={c01:.4f} (0↔2)={c02:.4f} (1↔2)={c12:.4f}")

    for h in hooks: h.remove()


def analyze_identity(diff_model, embeddings, encoder, device, dtype, sep_embed):
    print("\n" + "="*70)
    print("БЛОК 7: Identity Preservation — Cos(pred, clean)")
    print("="*70)

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
    print(f"  {'-'*10}-+-" + "-+-".join(["-"*7]*len(t_list)))

    for label, txt in texts:
        z_clean, attn_mask = encode_clean_text(txt, tokenizer, embeddings, encoder, device, dtype)
        mask_b = attn_mask.bool()
        row = []
        for t_val in t_list:
            with torch.no_grad():
                pred = diff_model(z_clean, attn_mask, torch.tensor([t_val], device=device), sep_embed=sep_embed)
            cs = (pred[mask_b] * z_clean[mask_b]).sum(dim=-1).mean().item()
            row.append(cs)
        print(f"  {label:<10} | " + " | ".join(f"{v:>7.4f}" for v in row))

    print("\n  Ожидаемо: t=0.1 → ~1.0 (Identity Gate dominates); t=0.9 → реальный денойзинг")
    print("  Если t=0.9 < 0.5 → модель плохо денойзит высокий шум")
    print("  Если t=0.9 ≈ t=0.5 → Identity Gate 'протекает' в оба")

    # Дополнительно: подать шум вместо clean
    print("\n  Тест с реальным шумом (z_noisy → pred cos z_clean):")
    print(f"  {'Text':<10} | " + " | ".join(f"t={t:>3.1f}" for t in t_list))
    print(f"  {'-'*10}-+-" + "-+-".join(["-"*7]*len(t_list)))
    for label, txt in texts:
        z_clean, attn_mask = encode_clean_text(txt, tokenizer, embeddings, encoder, device, dtype)
        mask_b = attn_mask.bool()
        row = []
        for t_val in t_list:
            mu    = math.cos(t_val * math.pi / 2)
            sigma = math.sin(t_val * math.pi / 2)
            eps   = torch.randn_like(z_clean)
            z_noisy = safe_normalize(mu * z_clean + sigma * eps, dim=-1)
            with torch.no_grad():
                pred = diff_model(z_noisy, attn_mask, torch.tensor([t_val], device=device), sep_embed=sep_embed)
            cs = (pred[mask_b] * z_clean[mask_b]).sum(dim=-1).mean().item()
            row.append(cs)
        print(f"  {label:<10} | " + " | ".join(f"{v:>7.4f}" for v in row))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print(f"Device: {device}")
    state = load_checkpoint_and_inspect(CKPT_PATH)
    print(f"  Шаг: {state.get('step', '?')}")

    print("\n[*] Загрузка моделей...")
    tokenizer, embeddings, encoder, decoder, diff_model, lm_head_weight, sep_embed = \
        load_models(device, dtype, state)
    diff_model.eval()

    analyze_topology(diff_model, embeddings, encoder, device, dtype, sep_embed)
    analyze_identity(diff_model, embeddings, encoder, device, dtype, sep_embed)

    print("\n" + "="*70 + "\nГотово.\n" + "="*70)

if __name__ == "__main__":
    main()
