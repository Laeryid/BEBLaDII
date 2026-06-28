import argparse
import os
import re
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.beb_la_dii.model.dus import DUSModel
from src.beb_la_dii.model.modern_decoder import ModernLatentDecoder
from src.beb_la_dii.model.vae import LatentEncoder
from src.beb_la_dii.utils.loss import safe_normalize


def recover_z(dus_final, c_embed, r_sq):
    """
    Математически точное восстановление z_clean из выхода DUS.
    Если нейросеть предсказывает y = (z+c)/||z+c||,
    то мы можем восстановить z через квадратное уравнение.
    """
    y_dot_c = (dus_final * c_embed).sum(dim=-1, keepdim=True)
    c_norm_sq = (c_embed * c_embed).sum(dim=-1, keepdim=True)
    
    discriminant = torch.clamp((y_dot_c ** 2) - c_norm_sq + r_sq, min=1e-6)
    alpha = y_dot_c + torch.sqrt(discriminant)
    
    z_rec = alpha * dus_final - c_embed
    return safe_normalize(z_rec, dim=-1) * torch.sqrt(r_sq)


def load_dus_checkpoint(dus, conf_proj, path):
    state = torch.load(path, map_location="cpu", weights_only=False)

    dus_state = state.get("dus", state)
    if "latentBERT_state_dict" in dus_state:
        dus_state = dus_state["latentBERT_state_dict"]
    elif "model_state_dict" in dus_state:
        dus_state = dus_state["model_state_dict"]

    clean_dus = {}
    for k, v in dus_state.items():
        k_clean = (
            k.replace("student.model.", "")
            .replace("model.", "")
            .replace("_orig_module.", "")
        )
        clean_dus[k_clean] = v

    dus.load_state_dict(clean_dus, strict=False)

    proj_state = state.get("confidence_proj", None)
    if proj_state:
        clean_proj = {k.replace("_orig_module.", ""): v for k, v in proj_state.items()}
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Путь к локальному чекпоинту DUS"
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default=r"experiments\phase 1\planB_phase1_checkpoints_phase1_vae_step_20000.pth",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default=r"experiments\phase 2\planB_phase2_phase2_decoder_step_8000.pth",
    )
    parser.add_argument(
        "--embed_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    )

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[*] Running LOCAL evaluation on device: {device}")

    print(f"[*] Loading Tokenizer and Embedding Model: {args.embed_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.embed_model)
    causal_model = AutoModelForCausalLM.from_pretrained(
        args.embed_model, torch_dtype=torch.bfloat16
    )
    embeddings = causal_model.get_input_embeddings()
    embeddings.to(device)
    embeddings.requires_grad_(False)
    lm_head_weight = causal_model.lm_head.weight.data.to(device)

    print("[*] Loading LatentEncoder & ModernLatentDecoder")
    encoder = LatentEncoder().to(device).to(torch.bfloat16)
    encoder.eval()

    decoder = ModernLatentDecoder(
        latent_dim=1024, qwen_dim=1536, num_layers=3, dus_weights_path=None
    ).to(device).to(torch.bfloat16)
    dus_for_dec = DUSModel.from_scratch(weights_path=None).model.to(device).to(torch.bfloat16)
    decoder.backbone = dus_for_dec
    decoder.backbone.layers = decoder.backbone.layers[-3:]
    decoder.use_modern_bert = True

    def load_with_stats(module, state_dict, name):
        model_keys = set(module.state_dict().keys())
        matched_keys = set(state_dict.keys()) & model_keys
        module.load_state_dict(state_dict, strict=False)
        print(f"[*] {name} loaded ({len(matched_keys)} / {len(model_keys)} keys)")

    if os.path.exists(args.encoder):
        state = torch.load(args.encoder, map_location="cpu", weights_only=False)
        state_dict = state.get("encoder", state)
        clean_state = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
        load_with_stats(encoder, clean_state, "LatentEncoder")
    else:
        print(f"[!] Warning: Encoder weights not found at {args.encoder}")

    if os.path.exists(args.decoder):
        state = torch.load(args.decoder, map_location="cpu", weights_only=False)
        state_dict = state.get("decoder", state)
        clean_state = {k.replace("decoder.", ""): v for k, v in state_dict.items()}
        load_with_stats(decoder, clean_state, "ModernLatentDecoder")
    else:
        print(f"[!] Warning: Decoder weights not found at {args.decoder}")

    print(f"[*] Loading DUS Model and Confidence Proj from {args.checkpoint}")
    dus_wrapper = DUSModel.from_scratch(weights_path=None)
    dus = dus_wrapper.model.to(device).to(torch.bfloat16)

    confidence_proj = (
        torch.nn.Sequential(
            torch.nn.Linear(1, 256), torch.nn.SiLU(), torch.nn.Linear(256, 1024)
        )
        .to(device)
        .to(torch.bfloat16)
    )

    load_dus_checkpoint(dus, confidence_proj, args.checkpoint)
    dus.eval()
    confidence_proj.eval()

    # TEST 1: No noise
    print("\n" + "=" * 50)
    print("TEST 1: No noise (Identity Check via DUS)")
    print("=" * 50)
    phrases = [
        "Machine learning allows computers to solve complex problems without explicit programming.",
        "The cat is sleeping on the windowsill while fluffy snow falls slowly outside.",
    ]

    for i, phrase in enumerate(phrases, 1):
        input_ids = torch.tensor([tokenizer.encode(phrase)]).to(device)
        attn_mask = torch.ones_like(input_ids).to(device)

        with torch.no_grad():
            qwen_embeds = embeddings(input_ids)
            z_clean, _, _ = encoder(qwen_embeds)

            c_true = torch.ones(
                (1, z_clean.size(1)), device=device, dtype=torch.bfloat16
            )
            c_embed_raw = confidence_proj(c_true.unsqueeze(-1))
            c_embed = safe_normalize(c_embed_raw.float(), dim=-1).to(torch.bfloat16) * 0.1
            dus_input = (z_clean + c_embed).to(torch.bfloat16)

            dus_outputs = dus(
                inputs_embeds=dus_input,
                output_hidden_states=True,
            )
            clean_pre_norm = dus_outputs.hidden_states[-1] - c_embed
            dus_out_raw = dus.final_norm(clean_pre_norm)
            dus_out_norm = safe_normalize(dus_out_raw.float(), dim=-1).to(
                torch.bfloat16
            )

            z_recovered = dus_out_norm
            
            if i == 1:
                print(f"  [DEBUG] Norm z_clean: {torch.norm(z_clean[0,0]).item():.4f}")
                print(f"  [DEBUG] Norm c_embed: {torch.norm(c_embed[0,0]).item():.4f}")
                raw_norm = dus_out_raw.float().norm(dim=-1)[0,0].item()
                print(f"  [DEBUG] Norm l40 (raw DUS): {raw_norm:.4f}")

            text_ae = decode_to_text(z_clean, decoder, lm_head_weight, tokenizer)
            text_dus = decode_to_text(z_recovered, decoder, lm_head_weight, tokenizer)

            active_z_clean = z_clean[attn_mask.bool()]
            active_z_rec = z_recovered[attn_mask.bool()]
            cos_sim = (
                F.cosine_similarity(active_z_clean, active_z_rec, dim=-1).mean().item()
            )

        print(f"\nPhrase {i}: {phrase}")
        print(f"  [AE Only] -> {text_ae}")
        print(f"  [DUS Rec] -> {text_dus}")
        print(f"  [Cos Sim] -> {cos_sim:.4f}")

    # TEST 2: Light noise
    print("\n" + "=" * 50)
    print("TEST 2: Light noise (15% tokens)")
    print("=" * 50)
    test2_phrase = "Neural networks possess an incredible ability to generalize data from massive amounts of information."
    input_ids = torch.tensor([tokenizer.encode(test2_phrase)]).to(device)
    attn_mask = torch.ones_like(input_ids).to(device)

    with torch.no_grad():
        qwen_embeds = embeddings(input_ids)
        z_clean, _, _ = encoder(qwen_embeds)

        T = z_clean.size(1)
        noise_mask = (torch.rand(1, T).to(device) < 0.15).float().unsqueeze(-1)

        noise = torch.randn_like(z_clean)
        noise = safe_normalize(noise, dim=-1) * 0.5
        z_noisy = z_clean + noise * noise_mask

        z_clean_norm = torch.norm(z_clean, p=2, dim=-1, keepdim=True)
        z_noisy = torch.where(
            noise_mask > 0, safe_normalize(z_noisy, dim=-1) * z_clean_norm, z_noisy
        ).to(torch.bfloat16)
        r_sq = z_clean_norm**2

        c_true = torch.clamp(
            F.cosine_similarity(z_clean.float(), z_noisy.float(), dim=-1), min=0.0
        ).to(torch.bfloat16)
        c_embed_raw = confidence_proj(c_true.unsqueeze(-1))
        c_embed = safe_normalize(c_embed_raw.float(), dim=-1).to(torch.bfloat16) * 0.1
        dus_input = (z_noisy + c_embed).to(torch.bfloat16)

        dus_outputs = dus(
            inputs_embeds=dus_input,
            output_hidden_states=True,
        )
        clean_pre_norm = dus_outputs.hidden_states[-1] - c_embed
        dus_out_raw = dus.final_norm(clean_pre_norm)
        dus_out_norm = safe_normalize(dus_out_raw.float(), dim=-1).to(torch.bfloat16)
        z_recovered = dus_out_norm

        text_noisy = decode_to_text(z_noisy, decoder, lm_head_weight, tokenizer)
        text_recovered = decode_to_text(z_recovered, decoder, lm_head_weight, tokenizer)

        active_z_clean = z_clean[attn_mask.bool()]
        active_z_rec = z_recovered[attn_mask.bool()]
        sim_recov = (
            F.cosine_similarity(active_z_clean, active_z_rec, dim=-1).mean().item()
        )

    print(f"Original:   {test2_phrase}")
    print(f"Noised:     {text_noisy}")
    print(f"DUS out:    {text_recovered}")
    print(f"  [Cos Sim] -> {sim_recov:.4f}")


if __name__ == "__main__":
    main()
