import os
import sys
import math
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = "C:/Experiments/BEBLaDII"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments", "phase 4"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments", "phase 2"))

from evaluate_phase4_checkpoints import BEBLaDIIPhase4aEval
from transformers import AutoTokenizer

def safe_normalize(x, dim=-1, eps=1e-6):
    return F.normalize(x, p=2, dim=dim, eps=eps)

def load_dus(device):
    print("Loading DUS...")
    embed_model_id = "Qwen/Qwen2.5-1.5B"
    modernbert_id = "answerdotai/ModernBERT-large"
    vae_ckpt = os.path.join(PROJECT_ROOT, "experiments", "phase 1", "planB_phase1_checkpoints_phase1_vae_step_20000.pth")
    phase4_ckpt = os.path.join(PROJECT_ROOT, "experiments", "phase 4", "local_checkpoints", "phase4_step_85995.pth")

    dus_model = BEBLaDIIPhase4aEval(embedding_model_path=embed_model_id, modernbert_path=modernbert_id)
    
    vae_st = torch.load(vae_ckpt, map_location="cpu", weights_only=False)
    if 'encoder' in vae_st:
        dus_model.encoder.load_state_dict(vae_st['encoder'], strict=False)

    p4_st = torch.load(phase4_ckpt, map_location="cpu", weights_only=False)
    dus_ema = p4_st.get("dus_ema", p4_st.get("dus", {}))
    clean_dus = {k.replace("_orig_module.", ""): v for k, v in dus_ema.items()}
    dus_model.dus.load_state_dict(clean_dus, strict=False)
    
    dus_model.to(device)
    dus_model.eval()
    return dus_model

def get_true_n(v1, v2):
    cos_sim = (v1 * v2).sum(dim=-1)
    cos_sim = torch.clamp(cos_sim, -1.0 + 1e-6, 1.0 - 1e-6)
    return (2.0 / math.pi) * torch.acos(cos_sim)

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
    
    # Let's focus entirely on token 6 ("over") which was set to t=0.60
    t_start = torch.full((B, T), 0.60, device=device)
    theta_init = t_start.unsqueeze(-1) * (math.pi / 2)
    x_start = safe_normalize(torch.cos(theta_init) * z_clean + torch.sin(theta_init) * initial_noise, dim=-1)
    
    # -------------------------------------------------------------------------
    # METHOD A: Phase 4 SLERP (Cheat metric)
    # -------------------------------------------------------------------------
    with torch.no_grad():
        true_n_start = get_true_n(x_start, z_clean)
        t_reported_A = true_n_start
        dt = 1.0 / 25.0
        t_next_A = torch.clamp(t_reported_A - dt, min=0.0)
        
        t_global_A = torch.tensor([1.0] * B, device=device)
        out_sc_A = dus(input_ids, attn_mask, t_global=t_global_A, t_reported=t_reported_A, z_noisy_override=x_start)
        sc_est_A = out_sc_A["dus_final"].detach()
        out_A = dus(input_ids, attn_mask, t_global=t_global_A, t_reported=t_reported_A, self_cond=sc_est_A, z_noisy_override=x_start)
        z_pred_raw_A = out_A["dus_final"]
        
        gate_A = torch.sin(t_reported_A * (math.pi / 2)).unsqueeze(-1)
        z_pred_A = safe_normalize(gate_A * z_pred_raw_A + (1.0 - gate_A) * x_start, dim=-1)
        
        theta_now_A = t_reported_A.unsqueeze(-1) * (math.pi / 2)
        theta_next_A = t_next_A.unsqueeze(-1) * (math.pi / 2)
        
        sin_now_A = torch.sin(theta_now_A)
        sin_now_A = torch.where(sin_now_A < 1e-5, torch.ones_like(sin_now_A) * 1e-5, sin_now_A)
        
        w_pred = torch.sin(theta_now_A - theta_next_A) / sin_now_A
        w_cur = torch.sin(theta_next_A) / sin_now_A
        
        z_next_A = w_pred * z_pred_A + w_cur * x_start
        x_next_A = safe_normalize(z_next_A, dim=-1)
        
        true_n_next_A = get_true_n(x_next_A, z_clean)

    # -------------------------------------------------------------------------
    # METHOD B: Option 2 (Adaptive DDIM with RawDProx)
    # -------------------------------------------------------------------------
    with torch.no_grad():
        t_reported_B = t_start
        t_global_B = torch.tensor([1.0] * B, device=device)
        
        out_sc_B = dus(input_ids, attn_mask, t_global=t_global_B, t_reported=t_reported_B, z_noisy_override=x_start)
        sc_est_B = out_sc_B["dus_final"].detach()
        out_B = dus(input_ids, attn_mask, t_global=t_global_B, t_reported=t_reported_B, self_cond=sc_est_B, z_noisy_override=x_start)
        z_pred_raw_B = out_B["dus_final"]
        
        gate_B = torch.sin(t_reported_B * (math.pi / 2)).unsqueeze(-1)
        z_pred_B = safe_normalize(gate_B * z_pred_raw_B + (1.0 - gate_B) * x_start, dim=-1)
        
        # RawDProx
        flat_pred_B = z_pred_B.view(-1, 1024)
        sims_B = torch.matmul(flat_pred_B, latent_dict.T)
        raw_dprox = torch.max(sims_B, dim=-1)[0].view(B, T)
        t_est_B = 1.0 - raw_dprox
        
        delta_t_B = torch.clamp(t_reported_B - t_est_B, min=0.04)
        t_next_B = torch.clamp(t_reported_B - delta_t_B, min=0.0)
        
        theta_now_B = t_reported_B.unsqueeze(-1) * (math.pi / 2)
        theta_next_B = t_next_B.unsqueeze(-1) * (math.pi / 2)
        
        # Method B's interpolation logic (DDIM with extracted noise normalization)
        extracted_noise_B = x_start - torch.cos(theta_now_B) * z_pred_B
        norm_noise_B = safe_normalize(extracted_noise_B, dim=-1)
        x_next_B_raw = torch.cos(theta_next_B) * z_pred_B + torch.sin(theta_next_B) * norm_noise_B
        x_next_B = safe_normalize(x_next_B_raw, dim=-1)
        
        true_n_next_B = get_true_n(x_next_B, z_clean)

    # -------------------------------------------------------------------------
    # METHOD C: Option 2 BUT with SLERP interpolation math
    # -------------------------------------------------------------------------
    with torch.no_grad():
        w_pred_C = torch.sin(theta_now_B - theta_next_B) / torch.clamp(torch.sin(theta_now_B), min=1e-5)
        w_cur_C = torch.sin(theta_next_B) / torch.clamp(torch.sin(theta_now_B), min=1e-5)
        z_next_C = w_pred_C * z_pred_B + w_cur_C * x_start
        x_next_C = safe_normalize(z_next_C, dim=-1)
        true_n_next_C = get_true_n(x_next_C, z_clean)

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    idx = 6 # token "over"
    
    lines = []
    lines.append("HEAD-TO-HEAD: 1-STEP DIFFUSION COMPARISON")
    lines.append(f"Token: {tokenizer.decode([input_ids[0, idx].item()])}")
    lines.append(f"Initial physical TrueN (Distance to z_clean): {true_n_start[0, idx].item():.4f}")
    lines.append(f"Initial t_reported used for model: A={t_reported_A[0, idx].item():.4f}, B={t_reported_B[0, idx].item():.4f}")
    lines.append("")
    lines.append(f"--- Z_PRED (Model Output) ---")
    cos_A = (z_pred_A[0, idx] * z_clean[0, idx]).sum().item()
    cos_B = (z_pred_B[0, idx] * z_clean[0, idx]).sum().item()
    lines.append(f"z_pred_A cosine to z_clean: {cos_A:.4f} (TrueN: {get_true_n(z_pred_A, z_clean)[0, idx].item():.4f})")
    lines.append(f"z_pred_B cosine to z_clean: {cos_B:.4f} (TrueN: {get_true_n(z_pred_B, z_clean)[0, idx].item():.4f})")
    lines.append(f"Difference between z_pred_A and z_pred_B (L2): {torch.norm(z_pred_A[0, idx] - z_pred_B[0, idx]).item():.6f}")
    lines.append("")
    lines.append(f"--- SCHEDULE CALCULATION ---")
    lines.append(f"METHOD A (Phase 4): t_next = t_reported - 0.04 = {t_next_A[0, idx].item():.4f}")
    lines.append(f"METHOD B (Option 2): 1-RawDProx = {t_est_B[0, idx].item():.4f}, delta_t = {delta_t_B[0, idx].item():.4f}, t_next = {t_next_B[0, idx].item():.4f}")
    lines.append("")
    lines.append(f"--- INTERPOLATION OUTCOME (x_next) ---")
    lines.append(f"Target: We want TrueN of x_next to be LOWER than TrueN of x_start ({true_n_start[0, idx].item():.4f}).")
    lines.append(f"METHOD A (Slerp): TrueN = {true_n_next_A[0, idx].item():.4f} (Delta TrueN: {true_n_next_A[0, idx].item() - true_n_start[0, idx].item():.4f})")
    lines.append(f"METHOD B (DDIM Normalize): TrueN = {true_n_next_B[0, idx].item():.4f} (Delta TrueN: {true_n_next_B[0, idx].item() - true_n_start[0, idx].item():.4f})")
    lines.append(f"METHOD C (Method B schedule + Slerp Math): TrueN = {true_n_next_C[0, idx].item():.4f} (Delta TrueN: {true_n_next_C[0, idx].item() - true_n_start[0, idx].item():.4f})")
    
    report_path = os.path.join(PROJECT_ROOT, "experiments", "phase 5", "local", "step_comparison.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()