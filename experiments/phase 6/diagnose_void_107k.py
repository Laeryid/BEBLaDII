import os
import sys
import math
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer

PROJECT_ROOT = "C:/Experiments/BEBLaDII"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from evaluate_phase6_metrics import (
    BEBLaDIIPhase6Eval,
    ModernLatentDecoder,
    spherical_noise,
    safe_normalize,
    normalize_key,
)

def decode_with_void(decoder, qwen_weight, tokenizer, z, void_embed, void_thresh=0.60):
    with torch.no_grad():
        projected = decoder(z.float())
        logits = F.linear(projected, qwen_weight.float())
        pred_ids = logits.argmax(dim=-1)[0]
        
        sim_void = (z[0].float() * void_embed.to(z.device)).sum(dim=-1)
        is_void = (sim_void >= void_thresh).cpu().numpy()
        
        tokens = []
        i = 0
        T = len(pred_ids)
        while i < T:
            if is_void[i]:
                start = i
                while i < T and is_void[i]:
                    i += 1
                tokens.append(f"<|void x{i - start}|>")
            else:
                start = i
                while i < T and not is_void[i]:
                    i += 1
                chunk_text = tokenizer.decode(pred_ids[start:i], skip_special_tokens=True)
                tokens.append(chunk_text)
        return " ".join(tokens)


def main():
    device = torch.device("cpu")
    print("="*70)
    print("DIAGNOSIS ON CHECKPOINT 107000: VOID EMISSION & EXPOSURE BIAS PROBES")
    print("="*70)

    config = {
        'qwen_path': "Qwen/Qwen2.5-1.5B",
        'modernbert_path': "answerdotai/ModernBERT-large",
        'encoder_path': os.path.join(PROJECT_ROOT, "experiments", "phase 1", "planB_phase1_checkpoints_phase1_vae_step_20000.pth"),
        'phase4_path': os.path.join(PROJECT_ROOT, "experiments", "phase 4", "local_checkpoints", "phase4_step_85995.pth"),
        'phase2_decoder_path': os.path.join(PROJECT_ROOT, "experiments", "phase 2", "planB_phase2_checkpoints_decoder_step_9000.pth"),
        'sep_token_path': os.path.join(PROJECT_ROOT, "storage", "components", "sep_token.pt"),
        'void_token_path': os.path.join(PROJECT_ROOT, "storage", "components", "void_token.pt"),
        'phase6_ckpt': os.path.join(PROJECT_ROOT, "experiments", "phase 6", "checkpoints", "planB_phase6_checkpoints_phase6_ca_layers_step_107000.pth")
    }

    tokenizer = AutoTokenizer.from_pretrained(config['qwen_path'])
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    print("Loading BEBLaDIIPhase6Eval (step 107000)...")
    model = BEBLaDIIPhase6Eval(config).to(device)
    model.eval()

    print("Loading ModernLatentDecoder...")
    decoder = ModernLatentDecoder(latent_dim=1024, qwen_dim=1536, num_layers=3, dus_weights_path=config['phase4_path']).to(device)
    ckpt_dec = torch.load(config['phase2_decoder_path'], map_location=device, weights_only=False)
    st_dec = ckpt_dec.get("decoder", ckpt_dec)
    decoder.load_state_dict({normalize_key(k).replace("decoder.", ""): v for k, v in st_dec.items()}, strict=False)
    decoder.eval()

    latent_dict_path = os.path.join(PROJECT_ROOT, "BEBLaDII-planB-Phase5-Data", "latent_dict.pt")
    latent_dict = torch.load(latent_dict_path, map_location=device).float()
    void_embed = model.void_embed.to(device).float()

    # Load Q1
    ds_path = os.path.join(PROJECT_ROOT, "BEBLaDII-planB-Phase6-Data", "phase 6", "data", "train_phase6.parquet")
    df = pd.read_parquet(ds_path)
    q1 = df.iloc[0]
    query_text = str(q1['Q'])
    ref_answer_text = str(q1['A'])
    
    q_enc = tokenizer(query_text, return_tensors='pt').to(device)
    a_enc = tokenizer(ref_answer_text, return_tensors='pt').to(device)
    ref_len = a_enc.input_ids.shape[1]

    print(f"\n[Prompt Q1]: '{query_text}'")
    print(f"[Reference Answer Tokens]: {ref_len} (Canvas: 512, Tail expected VOID: {512 - ref_len} tokens)")

    # Encode Prompt
    with torch.no_grad():
        Z_prompt_12 = model.encode_prompt(q_enc.input_ids, q_enc.attention_mask)

    # -------------------------------------------------------------------------
    # PART 1: Step 0 (t = 1.0) Micro-Analysis
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("PART 1: STEP 0 (t = 1.0) MICRO-ANALYSIS")
    print("="*70)

    B = 1
    T_a = 512
    torch.manual_seed(42)
    z_canvas = safe_normalize(torch.randn(B, T_a, 1024, device=device), dim=-1)
    t_actual = torch.full((B, T_a), 1.0, device=device)
    t_global = torch.full((B,), 1.0, device=device)
    t_reported = torch.full((B, T_a), 1.0, device=device)

    with torch.no_grad():
        # 1. Conditional pass
        h39_cond, z_pred = model.forward_step(z_canvas, Z_prompt_12, q_enc.attention_mask, t_actual, t_reported, t_global=t_global)
        
        # 2. Unconditional pass (CA disabled)
        for ca in model.ca_layers.values(): ca._disable = True
        h39_uncond, z_uncond = model.forward_step(z_canvas, Z_prompt_12, q_enc.attention_mask, t_actual, t_reported, t_global=t_global)
        for ca in model.ca_layers.values(): ca._disable = False

        # Cosine similarity to void_embed
        cos_pred_void = (z_pred[0] * void_embed).sum(dim=-1)
        cos_uncond_void = (z_uncond[0] * void_embed).sum(dim=-1)
        
        # Disagreement
        cos_pred_uncond = (z_pred * z_uncond).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        disagreement = (1.0 - cos_pred_uncond) * 0.5  # [1, 512, 1]
        
        # Specificity Gate (Step-normalized, tau=0.75, temp=0.35)
        med_d = disagreement.median(dim=1, keepdim=True).values
        std_d = disagreement.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-6)
        z_score = (disagreement - med_d) / std_d
        spec_gate_normal = torch.sigmoid((z_score - 0.75) / 0.35)
        spec_gate_override = torch.ones_like(disagreement)  # Line 826 override

        # CFG Compositions
        # A) CFG = 1.0 (Pure Conditional)
        z_cfg1 = z_pred
        
        # B) Unmodulated CFG = 4.0
        z_cfg4_unmod = safe_normalize(z_uncond + 4.0 * (z_pred - z_uncond), dim=-1)
        
        # C) Specificity-Gated CFG (with line 826 override: full CFG=4.0 everywhere)
        cfg_eff_override = 1.0 + 3.0 * spec_gate_override
        z_cfg4_override = safe_normalize(z_uncond + cfg_eff_override * (z_pred - z_uncond), dim=-1)
        
        # D) Specificity-Gated CFG (WITHOUT line 826 override: modulated by spec_gate)
        cfg_eff_normal = 1.0 + 3.0 * spec_gate_normal
        z_cfg4_normal = safe_normalize(z_uncond + cfg_eff_normal * (z_pred - z_uncond), dim=-1)

        def print_stats(name, tensor):
            p_mean = tensor[:ref_len].mean().item()
            t_mean = tensor[ref_len:].mean().item()
            print(f"  {name:<36} | Prefix (0..{ref_len}): {p_mean:+.4f} | Tail ({ref_len}..512): {t_mean:+.4f}")

        print("Predictions vs void_embed at Step 0 (t=1.0):")
        print_stats("cos(z_pred, void_embed)", cos_pred_void)
        print_stats("cos(z_uncond, void_embed)", cos_uncond_void)
        print_stats("Disagreement d(z_pred, z_uncond)", disagreement[0, :, 0])
        print_stats("Spec Gate Normal (z-score)", spec_gate_normal[0, :, 0])
        print_stats("Spec Gate Override (line 826)", spec_gate_override[0, :, 0])
        
        print("\nImpact of CFG on cos(z, void_embed) at Step 0:")
        print_stats("CFG = 1.0 (Pure Conditional)", (z_cfg1[0] * void_embed).sum(dim=-1))
        print_stats("CFG = 4.0 (Unmodulated)", (z_cfg4_unmod[0] * void_embed).sum(dim=-1))
        print_stats("CFG = 4.0 (With Line 826 Override)", (z_cfg4_override[0] * void_embed).sum(dim=-1))
        print_stats("CFG = 4.0 (WITHOUT Line 826 Override)", (z_cfg4_normal[0] * void_embed).sum(dim=-1))

        # Check nearest neighbor in latent_dict for tail under CFG=1.0 vs CFG=4.0
        print("\nTail Nearest Neighbors (tokens 100..115) from latent_dict:")
        for label, z_cand in [("Pure z_pred (CFG 1.0)", z_cfg1), ("Full CFG 4.0 (Overridden)", z_cfg4_override), ("Normal Spec-CFG (No Override)", z_cfg4_normal)]:
            sims = torch.matmul(z_cand[0, 100:115], latent_dict.T)
            top_ids = sims.max(dim=-1).indices
            decoded_words = [tokenizer.decode([tid.item()]).strip() for tid in top_ids]
            print(f"  [{label}]: {' '.join(decoded_words)}")

    # -------------------------------------------------------------------------
    # PART 2: Teacher-Forced Exposure Bias Probe at t = 0.50
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("PART 2: TEACHER-FORCED EXPOSURE BIAS PROBE (at t = 0.50)")
    print("="*70)

    with torch.no_grad():
        # Prepare clean target for Q1
        qwen_embeds_a = model.qwen_embeddings(a_enc.input_ids)
        z_a_clean_orig, _, _ = model.encoder(qwen_embeds_a)
        z_a_clean_orig = safe_normalize(z_a_clean_orig.float(), dim=-1)

        # Build clean target canvas: 75 content tokens + 437 void tokens
        target_canvas = torch.zeros(1, 512, 1024, device=device)
        target_canvas[0, :ref_len] = z_a_clean_orig[0]
        target_canvas[0, ref_len:] = void_embed.view(1, 1, -1).expand(1, 512 - ref_len, -1)
        target_canvas = safe_normalize(target_canvas, dim=-1)

        t_val = 0.50
        t_act = torch.full((1, 512), t_val, device=device)
        t_glob = torch.full((1,), t_val, device=device)

        # PROBE A: Standard clean target + noise at t=0.5
        z_noisy_a = spherical_noise(target_canvas, t_act)
        sims_a = torch.matmul(z_noisy_a, latent_dict.T)
        t_rep_a = (1.0 - sims_a.max(dim=-1).values).clamp(0.0, 1.0)
        
        _, z_pred_probe_a = model.forward_step(z_noisy_a, Z_prompt_12, q_enc.attention_mask, t_act, t_rep_a, t_global=t_glob)
        cos_void_probe_a = (z_pred_probe_a[0] * void_embed).sum(dim=-1)

        print("\nProbe A: Clean Ground Truth (75 tokens) + Void Tail (437 tokens) with spherical noise t=0.50:")
        print(f"  Prefix (0..{ref_len}) cos(z_pred, target): {(z_pred_probe_a[0, :ref_len] * target_canvas[0, :ref_len]).sum(dim=-1).mean():.4f}")
        print(f"  Tail ({ref_len}..512) cos(z_pred, void_embed): {cos_void_probe_a[ref_len:].mean():.4f} (min: {cos_void_probe_a[ref_len:].min():.4f}, max: {cos_void_probe_a[ref_len:].max():.4f})")
        print(f"  Tail tokens with cos(z, void) >= 0.60: {(cos_void_probe_a[ref_len:] >= 0.60).float().mean() * 100:.1f}%")

        # PROBE B: Tail Contamination (Prefix clean, but Tail filled with repeated content words)
        contam_canvas = target_canvas.clone()
        repeats = math.ceil((512 - ref_len) / ref_len)
        contam_tail = z_a_clean_orig[0].repeat(repeats, 1)[:512 - ref_len]
        contam_canvas[0, ref_len:] = contam_tail
        contam_canvas = safe_normalize(contam_canvas, dim=-1)

        z_noisy_b = spherical_noise(contam_canvas, t_act)
        sims_b = torch.matmul(z_noisy_b, latent_dict.T)
        t_rep_b = (1.0 - sims_b.max(dim=-1).values).clamp(0.0, 1.0)

        _, z_pred_probe_b = model.forward_step(z_noisy_b, Z_prompt_12, q_enc.attention_mask, t_act, t_rep_b, t_global=t_glob)
        cos_void_probe_b = (z_pred_probe_b[0] * void_embed).sum(dim=-1)
        cos_contam_probe_b = (z_pred_probe_b[0, ref_len:] * contam_tail).sum(dim=-1)

        print("\nProbe B: Tail Contamination (Prefix clean, but Tail 75..512 has repeated content words at t=0.50):")
        print(f"  Prefix (0..{ref_len}) cos(z_pred, target): {(z_pred_probe_b[0, :ref_len] * target_canvas[0, :ref_len]).sum(dim=-1).mean():.4f}")
        print(f"  Tail ({ref_len}..512) cos(z_pred, void_embed): {cos_void_probe_b[ref_len:].mean():.4f} (min: {cos_void_probe_b[ref_len:].min():.4f}, max: {cos_void_probe_b[ref_len:].max():.4f})")
        print(f"  Tail ({ref_len}..512) cos(z_pred, contam_words): {cos_contam_probe_b.mean():.4f}")
        print(f"  Tail tokens with cos(z, void) >= 0.60: {(cos_void_probe_b[ref_len:] >= 0.60).float().mean() * 100:.1f}%")
        
        if cos_void_probe_b[ref_len:].mean() < 0.30 and cos_contam_probe_b.mean() > 0.60:
            print("  >>> [FINDING CONFIRMED] Exposure Bias: When tail contains spurious content words at t=0.5,")
            print("      the model denoises the spurious words (cos=0.6+) instead of erasing them into void (cos<0.3)!")

    # -------------------------------------------------------------------------
    # PART 3: Full 25-Step Generation Comparison with Explicit <|void|> Rendering
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("PART 3: 25-STEP GENERATION COMPARISON WITH EXPLICIT <|void|> RENDERING")
    print("="*70)

    import gc

    @torch.no_grad()
    def run_test_gen(sampler_mode, cfg_max, cfg_min, override_step0=True, seed=42):
        gc.collect()
        torch.manual_seed(seed)
        z = safe_normalize(torch.randn(1, 512, 1024, device=device), dim=-1)
        
        # Schedule
        if "hybrid" in sampler_mode:
            t_sched = [1.0]
            for step_idx in range(24):
                t_val = 0.88 - step_idx * (0.88 / 24)
                t_sched.append(max(0.04, min(1.0, round(round(t_val / 0.04) * 0.04, 4))))
        else:
            t_sched = [max(0.04, min(1.0, round((1.0 - s * (1.0 / 25)) / 0.04) * 0.04)) for s in range(25)]

        step_logs = []
        for i, t_v in enumerate(t_sched):
            t_act = torch.full((1, 512), t_v, device=device)
            t_glob = torch.full((1,), t_v, device=device)
            sims = torch.matmul(z, latent_dict.T)
            t_rep = (1.0 - sims.max(dim=-1).values).clamp(0.0, 1.0)

            # Conditional
            _, z_p = model.forward_step(z, Z_prompt_12, q_enc.attention_mask, t_act, t_rep, t_global=t_glob)
            
            # CFG
            cfg_t = cfg_min + (cfg_max - cfg_min) * ((t_v - 0.04) / 0.96) if cfg_max > cfg_min else cfg_max
            if cfg_t > 1.0:
                for ca in model.ca_layers.values(): ca._disable = True
                _, z_u = model.forward_step(z, Z_prompt_12, q_enc.attention_mask, t_act, t_rep, t_global=t_glob)
                for ca in model.ca_layers.values(): ca._disable = False

                if "specificity" in sampler_mode:
                    cos_sim = (z_p * z_u).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
                    dis = (1.0 - cos_sim) * 0.5
                    med = dis.median(dim=1, keepdim=True).values
                    std = dis.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-6)
                    
                    if i == 0 and override_step0 and dis.mean().item() > 0.20:
                        gate = torch.ones_like(dis)
                    else:
                        z_sc = (dis - med) / std
                        gate = torch.sigmoid((z_sc - 0.75) / 0.35)
                    
                    cfg_eff = 1.0 + (cfg_t - 1.0) * gate
                    z_p = safe_normalize(z_u + cfg_eff * (z_p - z_u), dim=-1)
                else:
                    z_p = safe_normalize(z_u + cfg_t * (z_p - z_u), dim=-1)

            # Track void metrics
            cos_v = (z_p[0] * void_embed).sum(dim=-1)
            void_pct = (cos_v >= 0.60).float().mean().item() * 100.0
            tail_void_cos = cos_v[ref_len:].mean().item()

            if i % 5 == 0 or i == len(t_sched) - 1:
                step_logs.append(f"[Step {i:02d} | t={t_v:.2f} | Void={void_pct:4.1f}% | TailVoidCos={tail_void_cos:+.3f}]")

            if i == len(t_sched) - 1:
                z = z_p
                break

            if i == 0 and "hybrid" in sampler_mode:
                t_next = torch.full((1, 512), t_sched[i + 1], device=device)
                z = spherical_noise(z_p, t_next)
            else:
                t_next_v = t_sched[i + 1]
                t_next = torch.full((1, 512), t_next_v, device=device)
                th_now = t_act.unsqueeze(-1) * (math.pi / 2)
                th_next = t_next.unsqueeze(-1) * (math.pi / 2)
                sin_now = torch.sin(th_now).clamp(min=1e-5)
                w_pred = torch.sin(th_now - th_next) / sin_now
                w_cur = torch.sin(th_next) / sin_now
                z_nxt = w_pred * z_p + w_cur * z
                z = torch.where(th_now > 1e-5, safe_normalize(z_nxt, dim=-1), safe_normalize(z_p, dim=-1))

        final_text = decode_with_void(decoder, model.qwen_embeddings.weight, tokenizer, z, void_embed, void_thresh=0.60)
        gc.collect()
        return step_logs, final_text

    print("\n--- Setup 1: Baseline (specificity_hybrid, CFG 4.0 -> 1.0, with Step 0 Override) ---")
    logs1, text1 = run_test_gen("specificity_hybrid", cfg_max=4.0, cfg_min=1.0, override_step0=True)
    for l in logs1: print(" ", l)
    print("  Final Decoded Text:\n ", text1[:300], "...\n")

    print("\n--- Setup 2: specificity_hybrid WITHOUT Step 0 Override (Gate active at t=1.0) ---")
    logs2, text2 = run_test_gen("specificity_hybrid", cfg_max=4.0, cfg_min=1.0, override_step0=False)
    for l in logs2: print(" ", l)
    print("  Final Decoded Text:\n ", text2[:300], "...\n")

    print("\n--- Setup 3: standard Sampler (Linear schedule, no Anchor Jump) CFG = 1.0 ---")
    logs3, text3 = run_test_gen("standard", cfg_max=1.0, cfg_min=1.0, override_step0=False)
    for l in logs3: print(" ", l)
    print("  Final Decoded Text:\n ", text3[:300], "...\n")

    print("="*70)
    print("DIAGNOSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
