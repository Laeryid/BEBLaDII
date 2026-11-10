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

def load_dus(device):
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

def safe_normalize(x, dim=-1, eps=1e-6):
    return F.normalize(x, p=2, dim=dim, eps=eps)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    dus = load_dus(device)
    
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
    
    print("t_val | Norm(z_raw) | Cos(z_pred, x_t) | Cos(z_pred, z_cl) | Cos(z_raw, z_cl)")
    print("-" * 75)
    for t_val in [1.0, 0.8, 0.6, 0.4, 0.2, 0.04]:
        t_start = torch.full((B, T), t_val, device=device)
        theta_init = t_start.unsqueeze(-1) * (math.pi / 2)
        x_start = safe_normalize(torch.cos(theta_init) * z_clean + torch.sin(theta_init) * initial_noise, dim=-1)
        
        with torch.no_grad():
            t_global = torch.tensor([1.0] * B, device=device)
            out_sc = dus(input_ids, attn_mask, t_global=t_global, t_reported=t_start, z_noisy_override=x_start)
            sc_est = out_sc["dus_final"].detach()
            out = dus(input_ids, attn_mask, t_global=t_global, t_reported=t_start, self_cond=sc_est, z_noisy_override=x_start)
            z_pred_raw = out["dus_final"]
            
            gate_t = torch.sin(t_start * (math.pi / 2)).unsqueeze(-1)
            z_pred = safe_normalize(gate_t * z_pred_raw + (1.0 - gate_t) * x_start, dim=-1)
            
            norm_raw = torch.norm(z_pred_raw[0, 6]).item()
            cos_to_xt = (z_pred[0, 6] * x_start[0, 6]).sum().item()
            cos_to_clean = (z_pred[0, 6] * z_clean[0, 6]).sum().item()
            
            z_pred_raw_normed = safe_normalize(z_pred_raw, dim=-1)
            cos_raw_to_clean = (z_pred_raw_normed[0, 6] * z_clean[0, 6]).sum().item()
            
            print(f"{t_val:.2f}  | {norm_raw:11.4f} | {cos_to_xt:16.4f} | {cos_to_clean:17.4f} | {cos_raw_to_clean:16.4f}")

if __name__ == "__main__":
    main()