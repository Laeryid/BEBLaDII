import os
import sys
import math
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = "C:/Experiments/BEBLaDII"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments", "phase 4"))
from evaluate_phase4_checkpoints import BEBLaDIIPhase4aEval
from transformers import AutoTokenizer

def load_dus(device):
    dus_model = BEBLaDIIPhase4aEval(embedding_model_path="Qwen/Qwen2.5-1.5B", modernbert_path="answerdotai/ModernBERT-large")
    vae_ckpt = os.path.join(PROJECT_ROOT, "experiments", "phase 1", "planB_phase1_checkpoints_phase1_vae_step_20000.pth")
    vae_st = torch.load(vae_ckpt, map_location="cpu", weights_only=False)
    if 'encoder' in vae_st: dus_model.encoder.load_state_dict(vae_st['encoder'], strict=False)
    
    phase4_ckpt = os.path.join(PROJECT_ROOT, "experiments", "phase 4", "local_checkpoints", "phase4_step_85995.pth")
    p4_st = torch.load(phase4_ckpt, map_location="cpu", weights_only=False)
    dus_ema = p4_st.get("dus_ema", p4_st.get("dus", {}))
    clean_dus = {k.replace("_orig_module.", ""): v for k, v in dus_ema.items()}
    dus_model.dus.load_state_dict(clean_dus, strict=False)
    
    dus_model.to(device)
    dus_model.eval()
    return dus_model

def safe_normalize(x, dim=-1, eps=1e-6): return F.normalize(x, p=2, dim=dim, eps=eps)

def build_latent_dict(model, device, batch_size=2048):
    with torch.no_grad():
        qwen_emb = model.qwen_embeddings.weight
        vocab_size = qwen_emb.shape[0]
        latents = []
        for i in range(0, vocab_size, batch_size):
            batch_emb = qwen_emb[i:i+batch_size].unsqueeze(1).to(device)
            z, _, _ = model.encoder(batch_emb)
            latents.append(safe_normalize(z.squeeze(1).float(), dim=-1))
        return torch.cat(latents, dim=0)

def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    dus = load_dus(device)
    latent_dict = build_latent_dict(dus, device)
    
    text = "The quick brown fox jumps over the lazy dog."
    tok = tokenizer(text, return_tensors="pt")
    input_ids = tok.input_ids.to(device)
    attn_mask = torch.ones_like(input_ids).to(device)
    B, T = input_ids.shape
    
    with torch.no_grad():
        qwen_embeds = dus.qwen_embeddings(input_ids)
        z_clean, _, _ = dus.encoder(qwen_embeds)
        z_clean = safe_normalize(z_clean, dim=-1)
        
    initial_noise = safe_normalize(torch.randn_like(z_clean), dim=-1)
    
    # Same heterogeneous initialization as previous test
    t_start = torch.rand((B, T), device=device) * 0.7 + 0.2
    t_start[0, 2] = 0.1 # brown
    t_start[0, 6] = 0.1 # the
    t_start[0, 5] = 0.6 # over
    t_start[0, 4] = 0.8437 # jumps (to match the phase 4 bad run behavior roughly)
    
    t_local = t_start.clone()
    theta_init = t_local.unsqueeze(-1) * (math.pi / 2)
    x_t = safe_normalize(torch.cos(theta_init) * z_clean + torch.sin(theta_init) * initial_noise, dim=-1)
    
    steps = 25
    dt_min = 1.0 / steps
    
    target_idx = 4 # " jumps"
    
    print("\n--- AMBIGUITY METRIC TRACKING FOR TOKEN ' jumps' (idx 4) ---")
    print(f"{'Step':<4} | {'t_loc':<5} | {'Top1 (sim)':<15} | {'Top2 (sim)':<15} | {'Top3 (sim)':<15} | {'Delta1-2':<8} | {'ConfSim':<8}")
    print("-" * 90)
    
    for i in range(steps):
        t_global = torch.tensor([max(0.0, 1.0 - i * dt_min)], device=device)
        
        with torch.no_grad():
            out_sc = dus(input_ids, attn_mask, t_global=t_global, t_reported=t_local, z_noisy_override=x_t)
            sc_est = out_sc["dus_final"].detach()
            out = dus(input_ids, attn_mask, t_global=t_global, t_reported=t_local, self_cond=sc_est, z_noisy_override=x_t)
            z_pred_raw = out["dus_final"]
            
            gate_t = torch.sin(t_local * (math.pi / 2)).unsqueeze(-1)
            z_pred = safe_normalize(gate_t * z_pred_raw + (1.0 - gate_t) * x_t, dim=-1)
            
            # Compute Ambiguity Metric
            sims = torch.matmul(z_pred[0, target_idx], latent_dict.T)
            topk_sims, topk_idx = torch.topk(sims, k=3)
            
            w1 = tokenizer.decode([topk_idx[0].item()]).strip()
            w2 = tokenizer.decode([topk_idx[1].item()]).strip()
            w3 = tokenizer.decode([topk_idx[2].item()]).strip()
            s1, s2, s3 = topk_sims[0].item(), topk_sims[1].item(), topk_sims[2].item()
            
            delta = s1 - s2
            emb_top1 = latent_dict[topk_idx[0]]
            emb_top2 = latent_dict[topk_idx[1]]
            conf_sim = (emb_top1 * emb_top2).sum().item()
            
            print(f"{i:<4} | {t_local[0, target_idx].item():.3f} | {w1:<8} ({s1:.3f}) | {w2:<8} ({s2:.3f}) | {w3:<8} ({s3:.3f}) | {delta:.4f}   | {conf_sim:.4f}")
            
            # Step math (Option 2)
            t_est = 1.0 - topk_sims[0].item()
            delta_t = torch.clamp(t_local[0, target_idx] - t_est, min=dt_min)
            t_next = torch.clamp(t_local[0, target_idx] - delta_t, min=0.0)
            
            t_local[0, target_idx] = t_next
            
            theta_now = t_local.unsqueeze(-1) * (math.pi / 2)
            theta_next_all = t_local.clone()
            theta_next_all[0, target_idx] = t_next
            theta_next_all = theta_next_all.unsqueeze(-1) * (math.pi / 2)
            
            sin_now = torch.sin(theta_now)
            sin_now = torch.where(sin_now < 1e-5, torch.ones_like(sin_now) * 1e-5, sin_now)
            w_pred = torch.sin(theta_now - theta_next_all) / sin_now
            w_cur = torch.sin(theta_next_all) / sin_now
            
            z_next = w_pred * z_pred + w_cur * x_t
            x_t = torch.where(theta_now > 1e-5, safe_normalize(z_next, dim=-1), safe_normalize(z_pred, dim=-1))

if __name__ == "__main__":
    main()