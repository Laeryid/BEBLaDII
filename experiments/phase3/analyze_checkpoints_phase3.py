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

from experiments.phase3.inspect_and_test_phase3 import load_checkpoint_and_inspect, load_models, encode_clean_text
from src.beb_la_dii.utils.loss import safe_normalize

def compute_topological_metrics(hidden_states):
    # hidden_states: [N, D]
    # 1. Isotropy and Rank1 Ratio via SVD
    centered = hidden_states - hidden_states.mean(dim=0, keepdim=True)
    _, s, _ = torch.linalg.svd(centered.float(), full_matrices=False)
    
    isotropy = (s.sum()**2) / (len(s) * (s**2).sum()) if len(s) > 0 else torch.tensor(0.0)
    rank1_ratio = (s[0]**2) / (s**2).sum() if len(s) > 0 else torch.tensor(0.0)
    
    # 2. Norm CV (Sphericality)
    norms = hidden_states.norm(dim=-1)
    norm_cv = norms.std() / norms.mean() if norms.mean() > 0 else torch.tensor(0.0)
    
    return isotropy.item(), rank1_ratio.item(), norm_cv.item()

def analyze_checkpoint(ckpt_path, diff_model, tokenizer, embeddings, encoder, decoder, device, dtype):
    print(f"\n{'='*80}\nAnalyzing Checkpoint: {os.path.basename(ckpt_path)}\n{'='*80}")
    state = load_checkpoint_and_inspect(ckpt_path)
    
    # Загружаем веса
    dus_ema = state.get("dus_ema", state.get("dus", {}))
    clean_dus = {k.replace("_orig_module.", ""): v for k, v in dus_ema.items()}
    diff_model.dus.load_state_dict(clean_dus, strict=False)
    
    for name in ["adaLN_attn", "adaLN_mlp", "t_proj"]:
        ema = state.get(f"{name}_ema", state.get(name, {}))
        if ema:
            getattr(diff_model, name).load_state_dict(ema, strict=True)
            
    diff_model.eval()
    
    # --- DIAGNOSTIC 1: AdaLN Symmetry & Sensitivity ---
    t_vals = torch.tensor([0.01, 0.5, 0.99], device=device)
    t_sin = diff_model.t_sin_embed(t_vals)
    t_emb = diff_model.t_proj(t_sin)
    
    layer_0_attn_shift, layer_0_attn_scale = diff_model.adaLN_attn[0](t_emb)
    diff_t01_t99 = (diff_model.adaLN_attn[0](t_emb[0:1])[0] - diff_model.adaLN_attn[0](t_emb[2:3])[0]).norm().item()
    
    print(f"[AdaLN] Layer 0 Shift Std (Symmetry Break Indicator): {layer_0_attn_shift.squeeze(1).std(dim=-1).mean().item():.6f}")
    print(f"[AdaLN] Layer 0 Sensitivity (t=0.01 vs t=0.99): {diff_t01_t99:.6f}")
    
    # --- DIAGNOSTIC 2: Topological Collapse ---
    text1 = "The quick brown fox jumps over the lazy dog."
    text2 = "Мама мыла раму, а папа чинил телевизор."
    text3 = "Quantum mechanics governs subatomic particle interactions."
    
    z1, m1 = encode_clean_text(text1, tokenizer, embeddings, encoder, device, dtype)
    z2, m2 = encode_clean_text(text2, tokenizer, embeddings, encoder, device, dtype)
    z3, m3 = encode_clean_text(text3, tokenizer, embeddings, encoder, device, dtype)

    def pad_to_max(tensors, max_len, is_mask=False):
        padded = []
        for t in tensors:
            pad_len = max_len - t.size(1)
            if is_mask:
                padded.append(F.pad(t, (0, pad_len), value=0))
            else:
                padded.append(F.pad(t, (0, 0, 0, pad_len), value=0.0))
        return torch.cat(padded, dim=0)
        
    max_len = max(z1.size(1), z2.size(1), z3.size(1))
    z_batch = pad_to_max([z1, z2, z3], max_len, is_mask=False)
    m_batch = pad_to_max([m1, m2, m3], max_len, is_mask=True)
    active_tokens = z_batch[m_batch.bool()] # [N, D]
    
    iso, rank1, cv = compute_topological_metrics(active_tokens)
    print(f"[Topology Base] Isotropy: {iso:.4f} | Rank1 Ratio: {rank1:.4f} | Norm CV: {cv:.4f}")
    
    hidden_states_map = {}
    def get_hook(layer_idx):
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            hidden_states_map[layer_idx] = h.detach().float()
        return hook
        
    hook = diff_model.dus.layers[-1].register_forward_hook(get_hook(39))
    
    with torch.no_grad():
        out = diff_model(z_batch, m_batch, torch.tensor([0.5], device=device))
        
    hook.remove()
    
    active_layer39 = hidden_states_map[39][m_batch.bool()]
    iso39, rank1_39, cv39 = compute_topological_metrics(active_layer39)
    print(f"[Topology L39] Isotropy: {iso39:.4f} | Rank1 Ratio: {rank1_39:.4f} | Norm CV: {cv39:.4f}")
    
    h1 = hidden_states_map[39][0].mean(dim=0)
    h2 = hidden_states_map[39][1].mean(dim=0)
    cos_between_samples = F.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0)).item()
    print(f"[Collapse Check] Cosine Similarity (Sample 1 vs 2) at Layer 39: {cos_between_samples:.4f}")
    
    # --- DIAGNOSTIC 3: Identity Preservation (Feeding clean token) ---
    z_clean_test = z1
    with torch.no_grad():
        z_pred_09 = diff_model(z_clean_test, m1, torch.tensor([0.9], device=device))
        z_pred_05 = diff_model(z_clean_test, m1, torch.tensor([0.5], device=device))
        z_pred_01 = diff_model(z_clean_test, m1, torch.tensor([0.1], device=device))
        
    def get_identity(pred):
        return (pred[m1.bool()] * z1[m1.bool()]).sum(dim=-1).mean().item()
        
    print(f"[Identity] Cos(Pred, Clean) at t=0.9 (Low noise): {get_identity(z_pred_09):.4f}")
    print(f"[Identity] Cos(Pred, Clean) at t=0.5 (Mid noise): {get_identity(z_pred_05):.4f}")
    print(f"[Identity] Cos(Pred, Clean) at t=0.1 (High noise): {get_identity(z_pred_01):.4f}")
    print("\n")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    dummy_state = {}
    print("Loading models...")
    tokenizer, embeddings, encoder, decoder, diff_model, lm_head_weight, sep_embed = load_models(device, dtype, dummy_state)
    
    checkpoint_dir = r"C:\Experiments\BEBLaDII\experiments\phase3\local_checkpoints"
    
    # Мы ожидаем, что пользователь скачает сюда файлы с именами вроде phase3_step_1995.pth
    # Если их имена другие, скрипт может просто искать все .pth файлы
    pth_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    
    if not pth_files:
        print(f"No .pth files found in {checkpoint_dir}. Please make sure they are downloaded.")
        return
        
    # Сортируем, чтобы идти хронологически (по номерам шагов, если они есть)
    pth_files.sort()
    
    for f in pth_files:
        ckpt_path = os.path.join(checkpoint_dir, f)
        analyze_checkpoint(ckpt_path, diff_model, tokenizer, embeddings, encoder, decoder, device, dtype)

if __name__ == "__main__":
    main()
