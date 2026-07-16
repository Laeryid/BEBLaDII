import argparse
import os
import re
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')


project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.beb_la_dii.model.dus import DUSModel
from src.beb_la_dii.model.vae import LatentEncoder, LatentDecoder
from src.beb_la_dii.utils.loss import safe_normalize
from Lightning.train_phase3_lightning import BEBLaDIIPhase3, EMA




def load_dus_checkpoint(dus, conf_proj, path, use_ema=True):
    state = torch.load(path, map_location="cpu", weights_only=False)

    # --- ДИАГНОСТИКА: top-level ключи чекпоинта ---
    print(f"[DIAG] Checkpoint top-level keys: {list(state.keys())}")

    dus_key = "dus_ema" if use_ema else "dus"
    proj_key = "confidence_proj_ema" if use_ema else "confidence_proj"
    
    print(f"[*] Loading weights from keys: {dus_key}, {proj_key}")

    dus_state = state.get(dus_key, state)
    if "latentBERT_state_dict" in dus_state:
        dus_state = dus_state["latentBERT_state_dict"]
    elif "model_state_dict" in dus_state:
        dus_state = dus_state["model_state_dict"]

    print(f"[DIAG] DUS state_dict total keys in checkpoint: {len(dus_state)}")
    print(f"[DIAG] First 5 checkpoint DUS keys: {list(dus_state.keys())[:5]}")

    clean_dus = {}
    for k, v in dus_state.items():
        k_clean = (
            k.replace("student.model.", "")
            .replace("model.", "")
            .replace("_orig_module.", "")
            .replace("_fsdp_wrapped_module.", "")
        )
        clean_dus[k_clean] = v

    model_dus_keys = set(dus.state_dict().keys())
    matched_dus = set(clean_dus.keys()) & model_dus_keys
    missing_dus = model_dus_keys - set(clean_dus.keys())   # есть в модели, нет в чекпоинте
    extra_dus = set(clean_dus.keys()) - model_dus_keys     # есть в чекпоинте, нет в модели

    print(f"[DIAG] DUS matched:  {len(matched_dus)} / {len(model_dus_keys)}")
    print(f"[DIAG] DUS missing (in model, not in ckpt): {len(missing_dus)}")
    if missing_dus:
        print(f"[DIAG]   Examples missing: {list(missing_dus)[:5]}")
    print(f"[DIAG] DUS extra   (in ckpt, not in model): {len(extra_dus)}")
    if extra_dus:
        print(f"[DIAG]   Examples extra:   {list(extra_dus)[:5]}")
    
    if hasattr(dus, 'final_norm') and hasattr(dus.final_norm, 'weight') and dus.final_norm.weight is not None:
        import torch.nn as nn
        if "final_norm.weight" not in clean_dus:
            nn.init.ones_(dus.final_norm.weight)
            if hasattr(dus.final_norm, 'bias') and dus.final_norm.bias is not None:
                nn.init.zeros_(dus.final_norm.bias)
            print("[*] Fixed final_norm weights to Identity (1.0) to match Kaggle architecture.")

    print(f"[DIAG] First 5 cleaned checkpoint DUS keys: {list(clean_dus.keys())[:5]}")
    print(f"[DIAG] First 5 model DUS keys:              {list(model_dus_keys)[:5]}")

    dus.load_state_dict(clean_dus, strict=False)

    proj_state = state.get(proj_key, None)
    if proj_state:
        clean_proj = {k.replace("_orig_module.", "").replace("_fsdp_wrapped_module.", ""): v for k, v in proj_state.items()}
        model_keys = set(conf_proj.state_dict().keys())
        matched = set(clean_proj.keys()) & model_keys
        print(f"[*] ConfidenceProj matched keys: {len(matched)} / {len(model_keys)}")
        conf_proj.load_state_dict(clean_proj, strict=False)
        print(f"[*] Encoder, Decoder, DUS and ConfidenceProj successfully initialized.")
    else:
        print(f"[!] Warning: confidence_proj weights NOT found in {path}")


def decode_to_text(z, decoder, lm_head_weight, tokenizer):
    """Декодирует латентный вектор обратно в текст"""
    with torch.no_grad():
        dec_out = decoder(z)
        logits = F.linear(dec_out, lm_head_weight)
        token_ids = logits.argmax(dim=-1).squeeze(0)
        return tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)


def parse_bracket_phrase(phrase, tokenizer):
    clean_text = ""
    bracket_words = []
    parts = re.split(r"(\[[^\]]+\])", phrase)

    for part in parts:
        if part.startswith("[") and part.endswith("]"):
            word = part[1:-1]
            bracket_words.append(word)
            clean_text += word
        else:
            clean_text += part

    tokens = tokenizer.encode(clean_text)
    noise_indices = []
    for bw in bracket_words:
        bw_tokens = tokenizer.encode(bw, add_special_tokens=False)
        for i in range(len(tokens) - len(bw_tokens) + 1):
            if tokens[i : i + len(bw_tokens)] == bw_tokens:
                noise_indices.extend(list(range(i, i + len(bw_tokens))))
                break
    return clean_text, tokens, noise_indices


def run_test(model, tokenizer, phrase, device, eval_dtype, decoder, lm_head_weight, low_noise_amp=0.0):
    chatml_phrase = f"<|im_start|>user\n{phrase}<|im_end|>\n<|im_start|>assistant\n<|thought|>\n"
    tokens = tokenizer.encode(chatml_phrase)
    pad_len = 512 - len(tokens)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    input_ids = torch.tensor([tokens + [pad_id] * pad_len]).to(device)
    attn_mask = torch.tensor([[1] * len(tokens) + [0] * pad_len]).to(device)

    with torch.no_grad():
        fwd_outputs = model(input_ids, attention_mask=attn_mask, low_noise_amp=low_noise_amp)
        z_clean = fwd_outputs["z_clean"]
        z_noisy = fwd_outputs["z_noisy"]
        z_recovered = fwd_outputs["dus_final"]
        c_embed = fwd_outputs["c_embed"]
        noise_mask = fwd_outputs["noise_mask"]
        
        c_embed_norm = c_embed.norm(dim=-1).mean().item()
        z_clean_norm = z_clean.norm(dim=-1).mean().item()

        # Вручную прогоним DUS чтобы достать hidden_states
        x_in = z_clean.float() + c_embed
        sep_prefix = model.sep_embed.unsqueeze(0).unsqueeze(0).expand(1, 1, -1).to(x_in.dtype)
        dus_input_extended = torch.cat([sep_prefix, x_in], dim=1)
        attention_mask_extended = F.pad(attn_mask, (1, 0), value=1)
        
        with torch.autocast(device.type if device.type != 'cpu' else 'cpu', dtype=eval_dtype, enabled=(device.type != 'cpu')):
            z_clean_normed = safe_normalize(z_clean, dim=-1)
            z_noisy_normed = safe_normalize(z_noisy, dim=-1)
            c_true = torch.clamp((z_clean_normed * z_noisy_normed).sum(dim=-1), min=0.0)

            dus_outputs = model.dus(
                inputs_embeds=dus_input_extended,
                attention_mask=attention_mask_extended,
                output_hidden_states=True,
            )

            pre_norm = dus_outputs.hidden_states[-1][:, 1:, :].float()
            clean_pre_norm = pre_norm - c_embed
            dus_final_raw = model.dus.final_norm(clean_pre_norm.to(next(model.dus.final_norm.parameters()).dtype)).float()
            dus_final = safe_normalize(dus_final_raw, dim=-1)
            
            # --- PHASE 3 METRICS MATCHING WANDB ---
            B, T, D = z_clean.size()
            active_tokens = attn_mask.sum().clamp(min=1.0)
            noised_tokens = noise_mask.sum().clamp(min=1.0) if low_noise_amp > 0 else 1.0

            z_flat = dus_final.view(-1, D)
            mask_flat = attn_mask.float().view(-1, 1)
            m_state = (z_flat * mask_flat).sum(dim=0) / active_tokens
            z_centered = z_flat - m_state
            v_state = (z_centered.pow(2) * mask_flat).sum(dim=0) / active_tokens
            var_floor = 1.0 / (D * 2)
            var_loss = F.relu(var_floor - v_state).mean().item()
            
            cov = (z_centered.T @ (z_centered * mask_flat)) / active_tokens
            cov_off_diag = cov - torch.diag(torch.diag(cov))
            cov_loss = cov_off_diag.pow(2).sum().item() / D
            prior_loss = m_state.pow(2).mean().item() + var_loss + 0.1 * cov_loss

            denoise_target = safe_normalize(z_clean, dim=-1)
            cos_denoise_full = (dus_final * denoise_target).sum(dim=-1)
            cos_denoise_val = cos_denoise_full.mean().item()
            
            if low_noise_amp > 0:
                cos_denoise_on_noised = ((cos_denoise_full * noise_mask).sum() / noised_tokens).item()
            else:
                cos_denoise_on_noised = cos_denoise_val

            w_c = c_true.pow(20.0)
            identity_target = safe_normalize(z_noisy, dim=-1)
            cos_identity = (dus_final * identity_target).sum(dim=-1)
            penalty_elementwise = 1.0 - cos_identity
            identity_penalty = ((penalty_elementwise * w_c * attn_mask.float()).sum() / active_tokens).item()
            
            c_true_mean = c_true.mean().item()
            w_c_mean = w_c.mean().item()

        metrics = {
            "var_loss": var_loss,
            "cov_loss": cov_loss,
            "prior_loss": prior_loss,
            "c_true_mean": c_true_mean,
            "w_c_mean": w_c_mean,
            "identity_penalty": identity_penalty,
            "cos_denoise_on_noised": cos_denoise_on_noised,
        }

        active_z_clean = z_clean[attn_mask.bool()]
        active_z_rec = z_recovered[attn_mask.bool()]
        
        text_ae = decode_to_text(active_z_clean.to(eval_dtype).unsqueeze(0), decoder, lm_head_weight, tokenizer)
        text_dus = decode_to_text(active_z_rec.to(eval_dtype).unsqueeze(0), decoder, lm_head_weight, tokenizer)

        cos_sim = F.cosine_similarity(active_z_clean, active_z_rec, dim=-1).mean().item()

        return cos_sim, text_ae, text_dus, c_embed_norm, z_clean_norm, dus_final_raw, dus_outputs.hidden_states, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/phase 3/planB_phase3_checkpoints_phase3_step_1000.pth",
    )
    parser.add_argument(
        "--dus_init",
        type=str,
        default="experiments/phase 3/AWAKENED_WEIGHTS_FINAL.pt",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="experiments/phase 2/planB_phase2_checkpoints_decoder_step_6000.pth",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default="experiments/phase 2/planB_phase2_checkpoints_decoder_step_6000.pth",
    )
    parser.add_argument(
        "--embed_model", type=str, default="Qwen/Qwen2.5-1.5B"
    )
    parser.add_argument(
        "--sep_token", type=str, default="storage/components/sep_token.pt"
    )
    parser.add_argument(
        "--use_ema", action="store_true", default=True, help="Использовать EMA веса из чекпоинта"
    )

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device)
    # Use bfloat16 to match TPU training behavior exactly
    args = parser.parse_args()
    eval_dtype = torch.float32
    device = torch.device(args.device)
    print(f"[*] Running LOCAL evaluation on device: {device} | dtype: {eval_dtype}")

    print(f"[*] Loading Tokenizer and Embedding Model: {args.embed_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.embed_model)
    causal_model = AutoModelForCausalLM.from_pretrained(
        args.embed_model, torch_dtype=eval_dtype
    )
    embeddings = causal_model.get_input_embeddings()
    embeddings.to(device)
    embeddings.requires_grad_(False)
    lm_head_weight = causal_model.lm_head.weight.data.to(device)

    # Используем ModernLatentDecoder, так как это Фаза 2
    from src.beb_la_dii.model.vae import LatentEncoder
    from src.beb_la_dii.model.modern_decoder import ModernLatentDecoder
    
    encoder = LatentEncoder(1536, 1024).to(device).to(eval_dtype)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    decoder = ModernLatentDecoder(
        latent_dim=1024,
        qwen_dim=1536,
        num_layers=3,
        dus_weights_path=args.dus_init
    ).to(device).to(eval_dtype)
    decoder.eval()

    def load_with_stats(module, state_dict, name):
        model_keys = set(module.state_dict().keys())
        matched_keys = set(state_dict.keys()) & model_keys
        module.load_state_dict(state_dict, strict=False)
        print(f"[*] {name} loaded ({len(matched_keys)} / {len(model_keys)} keys)")

    def resolve_path(path):
        if not path:
            return path
        if path.startswith("gs://"):
            local_path = os.path.join(".", os.path.basename(path))
            if not os.path.exists(local_path):
                print(f"[*] Downloading {path} from GCS...")
                import subprocess
                subprocess.run(["gsutil", "-q", "cp", path, local_path], check=True)
            return local_path
        return path

    args.encoder = resolve_path(args.encoder)
    args.decoder = resolve_path(args.decoder)
    args.checkpoint = resolve_path(args.checkpoint)
    args.sep_token = resolve_path(args.sep_token)
    args.dus_init = resolve_path(args.dus_init)

    if os.path.exists(args.encoder):
        state = torch.load(args.encoder, map_location="cpu", weights_only=False)
        # Load Encoder
        state_dict_enc = state.get("encoder", state)
        clean_state_enc = {k.replace("encoder.", ""): v for k, v in state_dict_enc.items()}
        load_with_stats(encoder, clean_state_enc, "LatentEncoder")
    else:
        print(f"[!] Warning: Phase 1 weights not found at {args.encoder}")

    if os.path.exists(args.decoder):
        dec_state = torch.load(args.decoder, map_location="cpu", weights_only=False)
        state_dict_dec = dec_state.get("decoder", dec_state)
        clean_state_dec = {k.replace("decoder.", ""): v for k, v in state_dict_dec.items()}
        load_with_stats(decoder, clean_state_dec, "LatentDecoder")
        decoder.to(eval_dtype)
    else:
        print(f"[!] Warning: Phase 2 decoder weights not found at {args.decoder}")

    print(f"[*] Instantiating BEBLaDIIPhase3...")
    model = BEBLaDIIPhase3(
        embedding_model_path=args.embed_model,
        modernbert_path="answerdotai/ModernBERT-large",
        dus_weights=args.dus_init,
        encoder_weights=args.encoder,
        sep_token_path=args.sep_token
    ).to(device)

    # Load Phase 3 Checkpoint fully
    print(f"[*] Loading Phase 3 checkpoint weights into BEBLaDIIPhase3...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "dus" in ckpt: 
        clean_dus = {k.replace("_orig_module.", ""): v for k, v in ckpt["dus"].items()}
        model.dus.load_state_dict(clean_dus)
    if "confidence_proj" in ckpt: 
        clean_proj = {k.replace("_orig_module.", ""): v for k, v in ckpt["confidence_proj"].items()}
        model.confidence_proj.load_state_dict(clean_proj)
    
    if args.use_ema and "dus_ema" in ckpt and "confidence_proj_ema" in ckpt:
        print("[*] Applying EMA weights...")
        ema = EMA(model, decay=0.998)
        ema.shadow = {
            **{f"dus.{k}": v for k, v in ckpt["dus_ema"].items()},
            **{f"confidence_proj.{k}": v for k, v in ckpt["confidence_proj_ema"].items()}
        }
        ema.apply(model)
        
    model.eval()

    # TEST 1: No noise
    print("\n" + "=" * 50)
    print("TEST 1: No noise (Identity Check via DUS)")
    print("=" * 50)
    test_phrase = "Machine learning allows computers to solve complex problems without explicit programming."
    cos_sim1, ae_decoded1, dus_decoded1, c_embed_norm1, z_clean_norm1, dus_final1, hidden_states1, phase3_metrics1 = run_test(
        model, tokenizer, test_phrase, device, eval_dtype, decoder, lm_head_weight, low_noise_amp=0.0
    )
    
    print(f"  [DEBUG] Norm z_clean: {z_clean_norm1:.4f}")
    print(f"  [DEBUG] Norm c_embed: {c_embed_norm1:.4f}")
    print(f"  [Cos Sim to z_clean] -> {cos_sim1:.4f}")
    print(f"  [c_true_mean] -> {phase3_metrics1['c_true_mean']:.4f}")
    print(f"  [w_c_mean] -> {phase3_metrics1['w_c_mean']:.4f}")
    print(f"  [identity_penalty] -> {phase3_metrics1['identity_penalty']:.6f}")
    print(f"  [prior_loss] -> {phase3_metrics1['prior_loss']:.6f} (var: {phase3_metrics1['var_loss']:.6f}, cov: {phase3_metrics1['cov_loss']:.6f})")

    print(f"\n  [Layer-wise Analysis]")
    for i, hs in enumerate(hidden_states1):
        hs_active = hs[:, 1:, :].float()
        hs_norm = hs_active.norm(dim=-1).mean().item()
        print(f"  L{i:<9} | {hs_norm:<10.4f}")

    print("\n" + "=" * 50)
    print("TEST 2: Heavy noise (100% tokens noised) - Check Denoising")
    print("=" * 50)
    test_phrase2 = "Neural networks possess an incredible ability to generalize data from massive amounts of information."
    cos_sim2, ae_decoded2, dus_decoded2, _, _, dus_final2, _, phase3_metrics2 = run_test(
        model, tokenizer, test_phrase2, device, eval_dtype, decoder, lm_head_weight, low_noise_amp=0.5
    )
    print(f"Original:   {test_phrase2}")
    print(f"DUS out:    {dus_decoded2.strip()}")
    print(f"  [Cos Sim] -> {cos_sim2:.4f}")
    print(f"  [Cos Sim on noised tokens] -> {phase3_metrics2['cos_denoise_on_noised']:.4f}")
    print(f"  [c_true_mean] -> {phase3_metrics2['c_true_mean']:.4f}")
    print(f"  [identity_penalty] -> {phase3_metrics2['identity_penalty']:.6f}")
    print(f"  [denoise_loss (approx)] -> {1.0 - phase3_metrics2['cos_denoise_on_noised']:.6f}")


if __name__ == "__main__":
    main()
