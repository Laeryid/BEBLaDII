import os
import sys
import torch
import math
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = "C:/Experiments/BEBLaDII"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import importlib.util

spec = importlib.util.spec_from_file_location("evaluate_phase4", os.path.join(PROJECT_ROOT, "experiments", "phase 4", "evaluate_phase4_checkpoints.py"))
evaluate_phase4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate_phase4)
BEBLaDIIPhase4aEval = evaluate_phase4.BEBLaDIIPhase4aEval
from transformers import AutoTokenizer

def safe_normalize(x, dim=-1, eps=1e-6):
    return F.normalize(x, p=2, dim=dim, eps=eps)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading models...")
    
    embed_model_id = "Qwen/Qwen2.5-1.5B"
    modernbert_id  = "answerdotai/ModernBERT-large"
    phase4_ckpt    = os.path.join(PROJECT_ROOT, "experiments", "phase 4", "local_checkpoints", "phase4_step_85995.pth")
    vae_ckpt       = os.path.join(PROJECT_ROOT, "experiments", "phase 1", "planB_phase1_checkpoints_phase1_vae_step_20000.pth")

    dus_model = BEBLaDIIPhase4aEval(embedding_model_path=embed_model_id, modernbert_path=modernbert_id)
    vae_st = torch.load(vae_ckpt, map_location="cpu", weights_only=False)
    dus_model.encoder.load_state_dict(vae_st['encoder'], strict=False)
    
    p4_st = torch.load(phase4_ckpt, map_location="cpu", weights_only=False)
    clean_dus = {k.replace("_orig_module.", ""): v for k, v in p4_st.get("dus_ema", {}).items()}
    dus_model.dus.load_state_dict(clean_dus, strict=False)
    dus_model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(embed_model_id)
    text = "The quick brown fox jumps over the lazy dog."
    tok = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = tok.input_ids.to(device)
    attn_mask = tok.attention_mask.to(device)

    # 1. Получаем чистые латенты
    with torch.no_grad():
        qwen_embeds = dus_model.qwen_embeddings(input_ids)
        z_clean, _, _ = dus_model.encoder(qwen_embeds)
        z_clean = safe_normalize(z_clean.float(), dim=-1)

    print("\n--- ЭКСПЕРИМЕНТ 1: Норма LayerNorm vs PreNorm ---")
    # Тестируем на 3 уровнях шума
    t_vals = [0.0, 0.5, 1.0]
    for t_scalar in t_vals:
        t_global = torch.tensor([t_scalar], device=device)
        t_reported = torch.full_like(input_ids, t_scalar, dtype=torch.float)

        with torch.no_grad():
            out = dus_model(input_ids, attn_mask, t_global=t_global, t_reported=t_reported)
            h39_raw = out["h_39_raw"] # Это теперь pre_norm
            h39_norm = out["h_39"]    # Это после LayerNorm + safe_normalize
            
            # Достанем вручную то, что выходит из LayerNorm до safe_normalize
            pre_norm = h39_raw
            ln_out = dus_model.dus.final_norm(pre_norm.to(dus_model.dus.dtype)).float()
            
            print(f"t={t_scalar}:")
            print(f"  Норма pre_norm (h39_raw): {torch.norm(pre_norm, dim=-1).mean().item():.3f} (mean) | {torch.norm(pre_norm, dim=-1).std().item():.3f} (std)")
            print(f"  Норма после LayerNorm:    {torch.norm(ln_out, dim=-1).mean().item():.3f} (mean) | {torch.norm(ln_out, dim=-1).std().item():.3f} (std)")

    print("\n--- ЭКСПЕРИМЕНТ 2: Identity Gate vs Raw DUS ---")
    # Проверим, насколько точно восстанавливается z_clean при t=0 без и с Identity Gate
    t_global = torch.tensor([0.0], device=device)
    t_reported = torch.zeros_like(input_ids, dtype=torch.float)
    x_t = z_clean # При t=0 x_t == z_clean
    
    with torch.no_grad():
        # Сначала Self-Cond
        out_sc = dus_model(input_ids, attn_mask, t_global=t_global, t_reported=t_reported, z_noisy_override=x_t)
        sc_est = out_sc["dus_final"]
        
        # Боевой прогон
        out = dus_model(input_ids, attn_mask, t_global=t_global, t_reported=t_reported, self_cond=sc_est, z_noisy_override=x_t)
        z_pred_raw = out["dus_final"]
        
        gate_t = torch.sin(t_reported * (math.pi / 2)).unsqueeze(-1)
        z_pred_gated = safe_normalize(gate_t * z_pred_raw + (1.0 - gate_t) * x_t, dim=-1)

        raw_cos = (z_pred_raw * z_clean).sum(dim=-1).mean().item()
        gated_cos = (z_pred_gated * z_clean).sum(dim=-1).mean().item()
        
        print(f"Косинус DUS_raw vs z_clean при t=0:     {raw_cos:.5f}")
        print(f"Косинус GATED_DUS vs z_clean при t=0:   {gated_cos:.5f}")

if __name__ == "__main__":
    main()
