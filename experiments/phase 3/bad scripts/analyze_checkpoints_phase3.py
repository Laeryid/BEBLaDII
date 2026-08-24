import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

project_root = r"C:\Experiments\BEBLaDII"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.phase3.inspect_and_test_phase3 import load_checkpoint_and_inspect, load_models

def compute_topological_metrics(hidden_states):
    centered = hidden_states - hidden_states.mean(dim=0, keepdim=True)
    _, s, _ = torch.linalg.svd(centered.float(), full_matrices=False)
    
    isotropy = (s.sum()**2) / (len(s) * (s**2).sum()) if len(s) > 0 else torch.tensor(0.0)
    rank1_ratio = (s[0]**2) / (s**2).sum() if len(s) > 0 else torch.tensor(0.0)
    
    norms = hidden_states.norm(dim=-1)
    norm_cv = norms.std() / norms.mean() if norms.mean() > 0 else torch.tensor(0.0)
    
    return isotropy.item(), rank1_ratio.item(), norm_cv.item()

def analyze_checkpoint(ckpt_path, diff_model, tokenizer, device):
    print(f"\n{'='*80}\nAnalyzing Checkpoint: {os.path.basename(ckpt_path)}\n{'='*80}")
    state = load_checkpoint_and_inspect(ckpt_path)
    
    dus_ema = state.get("dus_ema", state.get("dus", {}))
    clean_dus = {k.replace("_orig_module.", ""): v for k, v in dus_ema.items()}
    diff_model.dus.load_state_dict(clean_dus, strict=False)
    
    for name in ["adaLN_attn", "adaLN_mlp", "t_proj", "self_cond_proj", "sep_embed"]:
        ema = state.get(f"{name}_ema", state.get(name, {}))
        if ema:
            if name == "sep_embed":
                diff_model.sep_embed.copy_(ema)
            else:
                getattr(diff_model, name).load_state_dict(ema, strict=True)
            
    diff_model.eval()
    
    # --- DIAGNOSTIC 1: AdaLN Symmetry & Sensitivity ---
    t_vals = torch.tensor([0.01, 0.5, 0.99], device=device)
    t_sin = diff_model.t_sin_embed(t_vals)
    t_emb = diff_model.t_proj(t_sin)
    
    layer_0_attn_shift, layer_0_attn_scale = diff_model.adaLN_attn[0](t_emb)
    diff_t01_t99 = (layer_0_attn_shift[0] - layer_0_attn_shift[2]).norm().item()
    
    print(f"[AdaLN] Layer 0 Shift Std (Symmetry Break Indicator): {layer_0_attn_shift.squeeze(1).std(dim=-1).mean().item():.6f}")
    print(f"[AdaLN] Layer 0 Sensitivity (t=0.01 vs t=0.99): {diff_t01_t99:.6f}")
    
    # --- DIAGNOSTIC 2: Topological Collapse ---
    text1 = "The quick brown fox jumps over the lazy dog."
    text2 = "Мама мыла раму, а папа чинил телевизор."
    text3 = "Quantum mechanics governs subatomic particle interactions."
    
    texts = [text1, text2, text3]
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
        
    active_tokens = z_clean[m_batch.bool()]
    
    iso, rank1, cv = compute_topological_metrics(active_tokens)
    print(f"[Topology Base] Isotropy: {iso:.4f} | Rank1 Ratio: {rank1:.4f} | Norm CV: {cv:.4f}")
    
    with torch.no_grad():
        out = diff_model(input_ids_batch, m_batch, torch.tensor([0.5] * len(texts), device=device))
        
    active_layer39 = out["h_39"][m_batch.bool()]
    iso39, rank1_39, cv39 = compute_topological_metrics(active_layer39)
    print(f"[Topology L39] Isotropy: {iso39:.4f} | Rank1 Ratio: {rank1_39:.4f} | Norm CV: {cv39:.4f}")
    
    h1 = out["h_39"][0][m_batch[0].bool()].mean(dim=0)
    h2 = out["h_39"][1][m_batch[1].bool()].mean(dim=0)
    cos_between_samples = F.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0)).item()
    print(f"[Collapse Check] Cosine Similarity (Sample 1 vs 2) at Layer 39: {cos_between_samples:.4f}")
    
    # --- DIAGNOSTIC 3: Denoising Preservation (True Identity) ---
    def get_identity(t_val):
        with torch.no_grad():
            out_t = diff_model(input_ids_batch[0:1], m_batch[0:1], torch.tensor([t_val], device=device))
        pred = out_t["dus_final"][0][m_batch[0].bool()]
        target = z_clean[0][m_batch[0].bool()]
        return (pred * target).sum(dim=-1).mean().item()
        
    print(f"[Identity] True Denoising Cos(Pred, Clean) at t=0.9 (High noise): {get_identity(0.9):.4f}")
    print(f"[Identity] True Denoising Cos(Pred, Clean) at t=0.5 (Mid noise): {get_identity(0.5):.4f}")
    print(f"[Identity] True Denoising Cos(Pred, Clean) at t=0.1 (Low noise): {get_identity(0.1):.4f}")
    print("\n")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    dummy_state = {}
    print("Loading models...")
    tokenizer, diff_model, decoder, lm_head_weight = load_models(device, dtype, dummy_state)
    
    checkpoint_dir = r"C:\Experiments\BEBLaDII\experiments\phase3\local_checkpoints"
    
    pth_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth") and "step_18995" in f]
    if not pth_files:
        pth_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
        
    pth_files.sort()
    
    for f in pth_files[-1:]: # Check the latest one
        ckpt_path = os.path.join(checkpoint_dir, f)
        analyze_checkpoint(ckpt_path, diff_model, tokenizer, device)

if __name__ == "__main__":
    main()
