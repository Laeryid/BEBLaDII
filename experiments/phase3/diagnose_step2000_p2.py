"""Блоки 6-7 (фикс маски для sep_embed +1 токен)"""
import os, sys, math, torch, torch.nn.functional as F

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
project_root = r"C:\Experiments\BEBLaDII"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.phase3.inspect_and_test_phase3 import load_checkpoint_and_inspect, load_models
from src.beb_la_dii.utils.loss import safe_normalize

CKPT_PATH = r"C:\Experiments\BEBLaDII\experiments\phase3\local_checkpoints\phase3_step_18995.pth"

def compute_topo(h):
    c = h - h.mean(dim=0, keepdim=True)
    _, s, _ = torch.linalg.svd(c.float(), full_matrices=False)
    iso = (s.sum()**2) / (len(s)*(s**2).sum()) if len(s)>0 else torch.tensor(0.0)
    r1  = (s[0]**2) / (s**2).sum() if len(s)>0 else torch.tensor(0.0)
    norms = h.norm(dim=-1)
    cv = norms.std() / norms.mean() if norms.mean() > 0 else torch.tensor(0.0)
    return iso.item(), r1.item(), cv.item()

def analyze_topology(diff_model, tokenizer, device):
    print("\n" + "="*70)
    print("БЛОК 6: Topological Analysis")
    print("="*70)

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Мама мыла раму, а папа чинил телевизор.",
        "Quantum mechanics governs subatomic particle interactions.",
        "In the beginning God created the heavens and the earth.",
        "2 + 2 = 4. The sum of angles in a triangle is 180 degrees.",
    ]

    diff_model.eval()
    all_input_ids, all_masks = [], []
    for txt in texts:
        tok = tokenizer(txt, return_tensors="pt", add_special_tokens=False)
        all_input_ids.append(tok.input_ids[0])
        all_masks.append(tok.attention_mask[0])

    max_len = max(len(ids) for ids in all_input_ids)
    input_ids_batch = torch.stack([F.pad(ids, (0, max_len - len(ids)), value=tokenizer.pad_token_id or 0) for ids in all_input_ids]).to(device)
    m_batch = torch.stack([F.pad(m, (0, max_len - len(m)), value=0) for m in all_masks]).to(device)

    with torch.no_grad():
        qwen_embeds = diff_model.qwen_embeddings(input_ids_batch)
        z_clean, _, _ = diff_model.encoder(qwen_embeds)
        z_clean = F.normalize(z_clean.float(), dim=-1)

    iso0, r0, cv0 = compute_topo(z_clean[m_batch.bool()])
    print(f"  [z_clean baseline] Iso={iso0:.4f} R1={r0:.4f} CV={cv0:.4f}")

    snap = {}
    def mk_hook(idx):
        def hook(mod, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            snap[idx] = x[:, 1:, :].detach().float()
        return hook

    hooks = [
        diff_model.dus.layers[4].register_forward_hook(mk_hook(4)),
        diff_model.dus.layers[19].register_forward_hook(mk_hook(19)),
        diff_model.dus.layers[-1].register_forward_hook(mk_hook(39)),
    ]

    for t_val in [0.1, 0.5, 0.9]:
        snap.clear()
        with torch.no_grad():
            out = diff_model(input_ids_batch, m_batch, torch.tensor([t_val] * len(texts), device=device))

        for layer_idx, label in [(4, "L4 "), (19, "L19"), (39, "L39")]:
            if layer_idx not in snap: continue
            active_h = snap[layer_idx][m_batch.bool()]
            iso, r1, cv = compute_topo(active_h)
            flag = " *** COLLAPSE ***" if r1 > 0.8 else (" ! rank1 high" if r1 > 0.5 else "")
            print(f"  t={t_val} | {label}: Iso={iso:.4f} R1={r1:.4f} CV={cv:.4f}{flag}")

        means = []
        for b in range(len(texts)):
            mb = m_batch[b].bool()
            if snap.get(39) is not None and mb.sum() > 0:
                s39_b = snap[39][b]
                if s39_b.shape[0] == mb.shape[0]:
                    means.append(s39_b[mb].mean(dim=0))
        if len(means) >= 3:
            c01 = F.cosine_similarity(means[0].unsqueeze(0), means[1].unsqueeze(0)).item()
            c02 = F.cosine_similarity(means[0].unsqueeze(0), means[2].unsqueeze(0)).item()
            c12 = F.cosine_similarity(means[1].unsqueeze(0), means[2].unsqueeze(0)).item()
            print(f"  t={t_val} | L39 cross-text cos: (0↔1)={c01:.4f} (0↔2)={c02:.4f} (1↔2)={c12:.4f}")

    for h in hooks: h.remove()


def analyze_identity(diff_model, tokenizer, device):
    print("\n" + "="*70)
    print("БЛОК 7: Denoising Preservation — Cos(pred, clean)")
    print("="*70)

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
        tok = tokenizer(txt, return_tensors="pt", add_special_tokens=False)
        input_ids = tok.input_ids.to(device)
        attn_mask = tok.attention_mask.to(device)
        mask_b = attn_mask.bool()
        
        with torch.no_grad():
            qwen_embeds = diff_model.qwen_embeddings(input_ids)
            z_clean, _, _ = diff_model.encoder(qwen_embeds)
            z_clean = F.normalize(z_clean.float(), dim=-1)

        row = []
        for t_val in t_list:
            with torch.no_grad():
                out = diff_model(input_ids, attn_mask, torch.tensor([t_val], device=device))
                pred = out["dus_final"]
            cs = (pred[mask_b] * z_clean[mask_b]).sum(dim=-1).mean().item()
            row.append(cs)
        print(f"  {label:<10} | " + " | ".join(f"{v:>7.4f}" for v in row))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print(f"Device: {device}")
    
    if not os.path.exists(CKPT_PATH):
        print(f"[!] CKPT not found: {CKPT_PATH}")
        return
        
    state = load_checkpoint_and_inspect(CKPT_PATH)
    print(f"  Шаг: {state.get('step', '?')}")

    print("\n[*] Загрузка моделей...")
    tokenizer, diff_model, decoder, lm_head_weight = load_models(device, dtype, state)

    analyze_topology(diff_model, tokenizer, device)
    analyze_identity(diff_model, tokenizer, device)

    print("\n" + "="*70 + "\nГотово.\n" + "="*70)

if __name__ == "__main__":
    main()
