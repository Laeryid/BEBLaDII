import os
import sys
import math
import json
import argparse
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = "C:/Experiments/BEBLaDII"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from transformers import AutoTokenizer
from src.beb_la_dii.model.vae import LatentEncoder
from src.beb_la_dii.model.modern_decoder import ModernLatentDecoder
from src.beb_la_dii.utils.loss import safe_normalize

from evaluate_phase6_metrics import (
    BEBLaDIIPhase6Eval,
    decode_latents_to_text,
    spherical_noise,
    cosine_noise_schedule,
    normalize_key,
    validate_environment,
)


# =========================================================================
# Self-Conditioning Projection Wrapper
# =========================================================================
class SelfConditioningModule(nn.Module):
    def __init__(self, dim: int = 1024):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)

    def load_from_phase4(self, ckpt_path: str, device: torch.device):
        print(f"Loading self_cond_proj_ema from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "self_cond_proj_ema" in ckpt:
            st = ckpt["self_cond_proj_ema"]
            if isinstance(st, dict) and "weight" in st:
                self.proj.weight.data.copy_(st["weight"].to(device))
                print(f"   -> self_cond_proj_ema loaded successfully. Weight norm: {self.proj.weight.norm():.2f}")
            else:
                self.proj.weight.data.copy_(st.to(device))
                print(f"   -> self_cond_proj_ema loaded. Weight norm: {self.proj.weight.norm():.2f}")
        elif "self_cond_proj" in ckpt:
            st = ckpt["self_cond_proj"]
            w = st["weight"] if isinstance(st, dict) else st
            self.proj.weight.data.copy_(w.to(device))
            print(f"   -> self_cond_proj loaded. Weight norm: {self.proj.weight.norm():.2f}")
        else:
            print("   -> [WARN] self_cond_proj not found in checkpoint, keeping identity/init.")

    def forward(self, z_pred_prev: torch.Tensor) -> torch.Tensor:
        if z_pred_prev is None:
            return 0.0
        return self.proj(z_pred_prev.float())


# =========================================================================
# Enhanced Forward Step with Self-Conditioning Support
# =========================================================================
def forward_step_with_sc(model: BEBLaDIIPhase6Eval, x_in: torch.Tensor, Z_prompt_12: torch.Tensor,
                         mask_Q: torch.Tensor, t_actual: torch.Tensor, t_reported: torch.Tensor,
                         t_global: torch.Tensor = None, z_self_cond: torch.Tensor = None):
    B, T_a, _ = x_in.shape
    if t_global is None:
        t_global = torch.mean(t_actual, dim=-1)
    elif t_global.dim() == 0:
        t_global = t_global.view(1).expand(B)

    t_sin_global = model.t_sin_embed(t_global)
    t_emb_global = model.t_proj_global(t_sin_global)
    t_sin_token = model.t_sin_embed(t_reported)
    t_emb_token = model.t_proj_token(t_sin_token)
    cond = torch.cat([t_emb_token, t_emb_global.unsqueeze(1).expand(-1, T_a, -1)], dim=-1)
    t_emb = model.t_joint_proj(cond)

    sep_t_emb = torch.zeros(B, 1, t_emb.shape[-1], device=t_emb.device, dtype=t_emb.dtype)
    t_emb_extended = torch.cat([sep_t_emb, t_emb], dim=1)

    for layer in model.dus.layers:
        layer_to_check = layer.original_layer if hasattr(layer, "original_layer") else layer
        if hasattr(layer_to_check, "attn_norm"):
            layer_to_check.attn_norm._current_t_emb = t_emb_extended
        if hasattr(layer_to_check, "mlp_norm"):
            layer_to_check.mlp_norm._current_t_emb = t_emb_extended

    for ca in model.ca_layers.values():
        ca._current_Z_prompt = Z_prompt_12
        ca._current_mask_Q = mask_Q
        ca._current_warmup_factor = 1.0
        ca._current_t_emb = t_emb

    # Инъекция Self-Conditioning: добавляем z_self_cond во входной поток DUS,
    # не затрагивая гейтированный skip-connection (ADR 047, 048)
    dus_canvas = x_in
    if z_self_cond is not None:
        dus_canvas = dus_canvas + z_self_cond

    sep_prefix = model.sep_embed.unsqueeze(0).unsqueeze(0).expand(B, 1, -1).to(dus_canvas.dtype)
    dus_input_extended = torch.cat([sep_prefix, dus_canvas], dim=1)
    attention_mask_extended = torch.ones((B, T_a + 1), device=x_in.device, dtype=torch.long)

    dus_outputs = model.dus(
        inputs_embeds=dus_input_extended,
        attention_mask=attention_mask_extended,
        output_hidden_states=False,
    )

    pre_norm = dus_outputs.last_hidden_state[:, 1:, :].float()
    dus_final_raw = model.dus.final_norm(pre_norm.to(model.dus.dtype)).float()
    h_39 = safe_normalize(dus_final_raw, dim=-1)

    gate = torch.sin(t_global * (math.pi / 2)).view(B, 1, 1).to(h_39.dtype)
    # Skip-connection идет строго от исходного зашумленного x_in
    dus_gated = gate * h_39 + (1.0 - gate) * x_in
    dus_final = safe_normalize(dus_gated, dim=-1)
    return h_39, dus_final


# =========================================================================
# Parametric Diffusion Sampler (Tests 1, 2, 3, 4)
# =========================================================================
def run_diagnostic_generation(
    model: BEBLaDIIPhase6Eval,
    sc_module: SelfConditioningModule,
    decoder: ModernLatentDecoder,
    latent_dict: torch.Tensor,
    tokenizer: AutoTokenizer,
    query_text: str,
    device: torch.device,
    max_len: int = 128,
    steps: int = 25,
    cfg_mode: str = "baseline", # "none", "const_2.0", "bell", "baseline"
    use_sc: bool = False,
    seed: int = 42,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    q_enc = tokenizer(query_text, return_tensors='pt').to(device)
    with torch.no_grad():
        Z_prompt_12 = model.encode_prompt(q_enc.input_ids, q_enc.attention_mask)

        B = 1
        T_a = max_len
        z_canvas = safe_normalize(torch.randn(B, T_a, 1024, device=device), dim=-1)
        void_embed = model.void_embed.to(device).float() if hasattr(model, 'void_embed') else None

        # Сетка времени
        remaining_steps = max(1, steps - 1)
        t_schedule = [1.0]
        re_noise_t = 0.88
        for s in range(remaining_steps):
            t_val = re_noise_t - s * (re_noise_t / remaining_steps)
            t_val = max(0.04, min(1.0, round(round(t_val / 0.04) * 0.04, 4)))
            t_schedule.append(t_val)

        z_pred_prev = None

        for i, t_val in enumerate(t_schedule):
            t_actual = torch.full((B, T_a), t_val, device=device)
            t_global = torch.full((B,), t_val, device=device)

            sims = torch.matmul(z_canvas, latent_dict.T)
            raw_dprox, _ = sims.max(dim=-1)
            if void_embed is not None:
                sim_void = (z_canvas * void_embed.view(1, 1, -1)).sum(dim=-1)
                raw_dprox = torch.maximum(raw_dprox, sim_void)
            t_reported = (1.0 - raw_dprox).clamp(0.0, 1.0)

            # Вычисление CFG scale для шага
            if cfg_mode == "none":
                cfg_t = 1.0
            elif cfg_mode == "const_2.0":
                cfg_t = 2.0
            elif cfg_mode == "bell":
                # Bell curve: 1.0 на краях, до 2.5 в середине (t=0.5)
                cfg_t = 1.0 + 1.5 * math.sin(t_val * math.pi)
            elif cfg_mode == "baseline":
                # Линейный спуск 4.0 -> 1.0
                cfg_t = 1.0 + 3.0 * ((t_val - 0.04) / 0.96)
            else:
                cfg_t = 1.0

            # Self-Conditioning delta
            z_sc = sc_module(z_pred_prev) if (use_sc and z_pred_prev is not None) else None

            # Conditional pass
            h39, z_pred = forward_step_with_sc(
                model, z_canvas, Z_prompt_12, q_enc.attention_mask,
                t_actual, t_reported, t_global=t_global, z_self_cond=z_sc
            )

            # Unconditional pass для CFG
            if cfg_t > 1.0:
                for ca in model.ca_layers.values():
                    ca._disable = True
                _, z_uncond = forward_step_with_sc(
                    model, z_canvas, Z_prompt_12, q_enc.attention_mask,
                    t_actual, t_reported, t_global=t_global, z_self_cond=z_sc
                )
                for ca in model.ca_layers.values():
                    ca._disable = False

                z_pred = safe_normalize(z_uncond + cfg_t * (z_pred - z_uncond), dim=-1)

            z_pred_prev = z_pred.clone()

            if i == len(t_schedule) - 1:
                z_canvas = z_pred
                break

            # Slerp / Anchor Jump
            if i == 0:
                t_next_val = t_schedule[i + 1]
                t_noise = torch.full((B, T_a), t_next_val, device=device)
                z_canvas = spherical_noise(z_pred, t_noise)
            else:
                t_next_val = t_schedule[i + 1]
                t_next = torch.full((B, T_a), t_next_val, device=device)
                theta_now = t_actual.unsqueeze(-1) * (math.pi / 2)
                theta_next = t_next.unsqueeze(-1) * (math.pi / 2)
                d_theta = theta_now - theta_next
                sin_theta_now = torch.sin(theta_now).clamp(min=1e-5)
                w1 = torch.sin(theta_next) / sin_theta_now
                w2 = torch.sin(d_theta) / sin_theta_now
                z_canvas = safe_normalize(w1 * z_canvas + w2 * z_pred, dim=-1)

        final_text = decode_latents_to_text(decoder, model.qwen_embeddings.weight, tokenizer, z_canvas, void_embed=void_embed)
        return final_text


# =========================================================================
# TEST 5: Geometric Chunk Compression & DUS Processing
# =========================================================================
def pool_mean_norm(z_chunks: torch.Tensor) -> torch.Tensor:
    """
    Вариант 1: Усреднение 4 токенов патча + сферическая нормализация.
    z_chunks: [B, N, K=4, D=1024]
    """
    return safe_normalize(z_chunks.mean(dim=2), dim=-1)


def pool_symplectic_order_aware(z_chunks: torch.Tensor) -> torch.Tensor:
    """
    Вариант 2: Symplectic Order-Aware фазовое вращение координат.
    Вращаем фазу координат [0..512] и [512..1024] на угол phi_k = k * pi / 4 для токенов k in {0,1,2,3}.
    Информация о порядке слов в чанке кодируется в непрерывной фазе перед усреднением.
    z_chunks: [B, N, K=4, D=1024]
    """
    B, N, K, D = z_chunks.shape
    half_D = D // 2
    x = z_chunks[..., :half_D]
    y = z_chunks[..., half_D:]

    # phi_k для k = 0, 1, 2, 3: [0, pi/4, pi/2, 3pi/4]
    k_idx = torch.arange(K, device=z_chunks.device, dtype=z_chunks.dtype).view(1, 1, K, 1)
    phi = k_idx * (math.pi / K)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)

    # Двумерное фазовое вращение
    x_rot = x * cos_phi - y * sin_phi
    y_rot = x * sin_phi + y * cos_phi

    z_rot = torch.cat([x_rot, y_rot], dim=-1)
    return safe_normalize(z_rot.mean(dim=2), dim=-1)


def run_test_5(model: BEBLaDIIPhase6Eval, decoder: ModernLatentDecoder, tokenizer: AutoTokenizer,
               sample_text: str, device: torch.device):
    print("\n" + "=" * 80)
    print("TEST 5: 4-TOKEN CHUNK COMPRESSION & DUS LATENT ANALYSIS")
    print("=" * 80)
    print(f"Sample Text: \"{sample_text}\"")

    enc = tokenizer(sample_text, return_tensors='pt').to(device)
    input_ids = enc.input_ids
    L = input_ids.shape[1]

    # Приводим к кратности 4 (дополняем PAD токеном если нужно)
    rem = L % 4
    if rem != 0:
        pad_len = 4 - rem
        pad_ids = torch.full((1, pad_len), tokenizer.pad_token_id, device=device, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, pad_ids], dim=1)
        L = input_ids.shape[1]

    # Ограничиваем до 256 токенов (64 патча) для идеальной читаемости
    if L > 256:
        input_ids = input_ids[:, :256]
        L = 256

    N = L // 4
    print(f"Token length: {L} tokens -> {N} patches of K=4 tokens")

    void_embed = model.void_embed.to(device).float() if hasattr(model, 'void_embed') else None

    with torch.no_grad():
        # Получаем истинные латенты Z_orig через Phase 1 VAE
        qwen_embeds = model.qwen_embeddings(input_ids)
        z_orig, _, _ = model.encoder(qwen_embeds)
        z_orig = safe_normalize(z_orig.float(), dim=-1) # [1, L, 1024]

        # 0. Исходное раскодирование Z_orig декодером (ground truth Phase 2)
        dec_orig = decode_latents_to_text(decoder, model.qwen_embeddings.weight, tokenizer, z_orig, void_embed=void_embed)
        print(f"\n[0. Ground Truth Z_orig (L={L})]:\n   \"{dec_orig}\"\n")

        # Разбиваем на патчи размера 4
        z_chunks = z_orig.view(1, N, 4, 1024)

        # Пулинг А: Mean + Norm
        P_mean = pool_mean_norm(z_chunks) # [1, N, 1024]
        # Пулинг Б: Symplectic Order-Aware
        P_symp = pool_symplectic_order_aware(z_chunks) # [1, N, 1024]

        # 1. Прямое распознавание сжатых 64 токенов через декодер
        dec_p_mean = decode_latents_to_text(decoder, model.qwen_embeddings.weight, tokenizer, P_mean, void_embed=void_embed)
        dec_p_symp = decode_latents_to_text(decoder, model.qwen_embeddings.weight, tokenizer, P_symp, void_embed=void_embed)
        print(f"[1. Direct Decoded Patches (Length={N})]:")
        print(f"   [Mean+Norm Pooling] : \"{dec_p_mean}\"")
        print(f"   [Symplectic Order ] : \"{dec_p_symp}\"\n")

        # 2. Растягивание патчей повторением x4 (Length = L)
        P_mean_rep = P_mean.repeat_interleave(4, dim=1) # [1, L, 1024]
        P_symp_rep = P_symp.repeat_interleave(4, dim=1) # [1, L, 1024]

        dec_rep_mean = decode_latents_to_text(decoder, model.qwen_embeddings.weight, tokenizer, P_mean_rep, void_embed=void_embed)
        dec_rep_symp = decode_latents_to_text(decoder, model.qwen_embeddings.weight, tokenizer, P_symp_rep, void_embed=void_embed)
        print(f"[2. Patches Repeated x4 (Length={L}) Before DUS]:")
        print(f"   [Mean+Norm Repeat x4] : \"{dec_rep_mean}\"")
        print(f"   [Symplectic Repeat x4]: \"{dec_rep_symp}\"\n")

        # Подготавливаем фиктивный промпт для DUS
        dummy_prompt = "Контекст вопроса"
        q_enc = tokenizer(dummy_prompt, return_tensors='pt').to(device)
        Z_prompt_12 = model.encode_prompt(q_enc.input_ids, q_enc.attention_mask)

        # 3. DUS на t=0.0 (чистый h_39 до гейта, так как гейт sin(0)=0)
        t_0 = torch.zeros((1, L), device=device)
        h39_m_t0, _ = model.forward_step(P_mean_rep, Z_prompt_12, q_enc.attention_mask, t_0, t_0, t_global=torch.zeros((1,), device=device))
        h39_s_t0, _ = model.forward_step(P_symp_rep, Z_prompt_12, q_enc.attention_mask, t_0, t_0, t_global=torch.zeros((1,), device=device))

        dec_h39_m_t0 = decode_latents_to_text(decoder, model.qwen_embeddings.weight, tokenizer, h39_m_t0, void_embed=void_embed)
        dec_h39_s_t0 = decode_latents_to_text(decoder, model.qwen_embeddings.weight, tokenizer, h39_s_t0, void_embed=void_embed)
        print(f"[3. DUS Output h_39 at t=0.0 (Pure Backbone Reconstruction)]:")
        print(f"   [Mean+Norm -> DUS h_39] : \"{dec_h39_m_t0}\"")
        print(f"   [Symplectic -> DUS h_39]: \"{dec_h39_s_t0}\"\n")

        # 4. DUS на t=0.2 (без добавления шума)
        t_02 = torch.full((1, L), 0.2, device=device)
        h39_m_t02, gated_m_t02 = model.forward_step(P_mean_rep, Z_prompt_12, q_enc.attention_mask, t_02, t_02, t_global=torch.tensor([0.2], device=device))
        h39_s_t02, gated_s_t02 = model.forward_step(P_symp_rep, Z_prompt_12, q_enc.attention_mask, t_02, t_02, t_global=torch.tensor([0.2], device=device))

        dec_gated_m_t02 = decode_latents_to_text(decoder, model.qwen_embeddings.weight, tokenizer, gated_m_t02, void_embed=void_embed)
        dec_gated_s_t02 = decode_latents_to_text(decoder, model.qwen_embeddings.weight, tokenizer, gated_s_t02, void_embed=void_embed)
        print(f"[4. DUS Output at t=0.2 (No Noise added)]:")
        print(f"   [Mean+Norm Gated t=0.2] : \"{dec_gated_m_t02}\"")
        print(f"   [Symplectic Gated t=0.2]: \"{dec_gated_s_t02}\"\n")

        # 5. DUS на t=0.2 (со сферическим микро-шумом kappa(0.2))
        noisy_m = spherical_noise(P_mean_rep, t_02)
        noisy_s = spherical_noise(P_symp_rep, t_02)

        _, pred_m_noise = model.forward_step(noisy_m, Z_prompt_12, q_enc.attention_mask, t_02, t_02, t_global=torch.tensor([0.2], device=device))
        _, pred_s_noise = model.forward_step(noisy_s, Z_prompt_12, q_enc.attention_mask, t_02, t_02, t_global=torch.tensor([0.2], device=device))

        dec_noise_m = decode_latents_to_text(decoder, model.qwen_embeddings.weight, tokenizer, pred_m_noise, void_embed=void_embed)
        dec_noise_s = decode_latents_to_text(decoder, model.qwen_embeddings.weight, tokenizer, pred_s_noise, void_embed=void_embed)
        print(f"[5. DUS Output at t=0.2 (With Spherical Noise t=0.2)]:")
        print(f"   [Mean+Norm Noisy -> Pred] : \"{dec_noise_m}\"")
        print(f"   [Symplectic Noisy -> Pred]: \"{dec_noise_s}\"\n")
    print("=" * 80 + "\n")

    return {
        "sample_text": sample_text,
        "token_len": L,
        "num_patches": N,
        "dec_orig": dec_orig,
        "direct_patches": {
            "mean_norm": dec_p_mean,
            "symplectic": dec_p_symp
        },
        "patches_repeated_x4": {
            "mean_norm": dec_rep_mean,
            "symplectic": dec_rep_symp
        },
        "dus_h39_t0": {
            "mean_norm": dec_h39_m_t0,
            "symplectic": dec_h39_s_t0
        },
        "dus_gated_t02_no_noise": {
            "mean_norm": dec_gated_m_t02,
            "symplectic": dec_gated_s_t02
        },
        "dus_noisy_t02_pred": {
            "mean_norm": dec_noise_m,
            "symplectic": dec_noise_s
        }
    }


# =========================================================================
# Main Entry Point
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Phase 6 Isolated Diagnostics & Tests")
    parser.add_argument("--test", type=str, default="5", choices=["all", "1", "2", "3", "4", "5"],
                        help="Which test to run: 5 (compression), 1 (self-conditioning), 2 (cfg), 3 (steps), 4 (prompts), all")
    parser.add_argument("--ckpt", type=str, default="", help="Path to Phase 6 checkpoint (119k or 107k)")
    parser.add_argument("--steps", type=int, default=25, help="Number of diffusion steps for Tests 1, 2, 4")
    parser.add_argument("--max_len", type=int, default=64, help="Canvas length for diffusion generations (CPU-friendly)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_json", type=str, default="experiments/phase 6/isolated_diagnostics_results.json",
                        help="Path to JSON file for storing all test results")
    args = parser.parse_args()

    default_ckpt_119k = os.path.join(PROJECT_ROOT, "experiments", "phase 6", "checkpoints", "planB_phase6_checkpoints_phase6_ca_layers_step_119000.pth")
    default_ckpt_107k = os.path.join(PROJECT_ROOT, "experiments", "phase 6", "checkpoints", "planB_phase6_checkpoints_phase6_ca_layers_step_107000.pth")
    target_ckpt = args.ckpt if args.ckpt else (default_ckpt_119k if os.path.exists(default_ckpt_119k) else default_ckpt_107k)

    config = {
        'qwen_path': "Qwen/Qwen2.5-1.5B",
        'modernbert_path': "answerdotai/ModernBERT-large",
        'encoder_path': os.path.join(PROJECT_ROOT, "experiments", "phase 1", "planB_phase1_checkpoints_phase1_vae_step_20000.pth"),
        'phase4_path': os.path.join(PROJECT_ROOT, "experiments", "phase 4", "local_checkpoints", "phase4_step_85995.pth"),
        'phase2_decoder_path': os.path.join(PROJECT_ROOT, "experiments", "phase 2", "planB_phase2_checkpoints_decoder_step_9000.pth"),
        'sep_token_path': os.path.join(PROJECT_ROOT, "storage", "components", "sep_token.pt"),
        'void_token_path': os.path.join(PROJECT_ROOT, "storage", "components", "void_token.pt"),
        'phase6_ckpt': target_ckpt
    }

    dict_path = os.path.join(PROJECT_ROOT, "experiments", "phase 5", "local", "latent_dict.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Phase 6 Isolated Diagnostics on {device}...")
    print(f"Target Checkpoint: {target_ckpt}")

    # Загружаем существующий JSON с результатами, если есть
    out_file = os.path.join(PROJECT_ROOT, args.output_json) if not os.path.isabs(args.output_json) else args.output_json
    existing_results = {}
    if os.path.exists(out_file):
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not read existing results: {e}")

    ckpt_tag = "119k" if "119000" in target_ckpt else ("107k" if "107000" in target_ckpt else "custom")
    if ckpt_tag not in existing_results:
        existing_results[ckpt_tag] = {"ckpt_path": target_ckpt, "timestamp": datetime.datetime.now().isoformat()}

    tokenizer = AutoTokenizer.from_pretrained(config['qwen_path'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading BEBLaDIIPhase6 model...")
    model = BEBLaDIIPhase6Eval(config).to(device)

    print("Loading ModernLatentDecoder...")
    decoder = ModernLatentDecoder(latent_dim=1024, qwen_dim=1536, num_layers=3, dus_weights_path=config['phase4_path']).to(device)
    ckpt_dec = torch.load(config['phase2_decoder_path'], map_location=device, weights_only=False)
    st_dec = ckpt_dec.get("decoder", ckpt_dec)
    clean_st_dec = {normalize_key(k).replace("decoder.", ""): v for k, v in st_dec.items()}
    decoder.load_state_dict(clean_st_dec, strict=False)
    decoder.eval()

    latent_dict = torch.load(dict_path, map_location=device).float() if os.path.exists(dict_path) else None

    sc_module = SelfConditioningModule(dim=1024).to(device)
    if os.path.exists(config['phase4_path']):
        sc_module.load_from_phase4(config['phase4_path'], device)

    # -------------------------------------------------------------
    # RUN SELECTED TESTS
    # -------------------------------------------------------------
    if args.test in ["5", "all"]:
        sample_phrase = (
            "В 1861 году в Российской империи произошло эпохальное историческое событие: "
            "император Александр Второй подписал Высочайший манифест об отмене крепостного права, "
            "освободивший крестьян от вековой помещичьей зависимости и даровавший им гражданские свободы."
        )
        res_t5 = run_test_5(model, decoder, tokenizer, sample_phrase, device)
        existing_results[ckpt_tag]["test_5_compression"] = res_t5

    default_prompt = "В каком году произошла отмена крепостного права в России?"

    if args.test in ["1", "all"]:
        print("\n" + "=" * 80)
        print("TEST 1: SELF-CONDITIONING (SC=OFF vs SC=ON)")
        print("=" * 80)
        print(f"Prompt: \"{default_prompt}\" (Steps={args.steps}, Len={args.max_len})")
        text_sc_off = run_diagnostic_generation(
            model, sc_module, decoder, latent_dict, tokenizer, default_prompt, device,
            max_len=args.max_len, steps=args.steps, cfg_mode="baseline", use_sc=False, seed=args.seed
        )
        print(f"  [SC=OFF]: \"{text_sc_off}\"")

        text_sc_on = run_diagnostic_generation(
            model, sc_module, decoder, latent_dict, tokenizer, default_prompt, device,
            max_len=args.max_len, steps=args.steps, cfg_mode="baseline", use_sc=True, seed=args.seed
        )
        print(f"  [SC=ON ]: \"{text_sc_on}\"")
        print("=" * 80 + "\n")
        existing_results[ckpt_tag]["test_1_self_conditioning"] = {
            "prompt": default_prompt,
            "steps": args.steps,
            "len": args.max_len,
            "sc_off": text_sc_off,
            "sc_on": text_sc_on
        }

    if args.test in ["2", "all"]:
        print("\n" + "=" * 80)
        print("TEST 2: CFG MODES ISOLATION")
        print("=" * 80)
        cfg_res = {}
        for cfg_m in ["none", "const_2.0", "bell", "baseline"]:
            res_txt = run_diagnostic_generation(
                model, sc_module, decoder, latent_dict, tokenizer, default_prompt, device,
                max_len=args.max_len, steps=args.steps, cfg_mode=cfg_m, use_sc=False, seed=args.seed
            )
            print(f"  [CFG Mode: {cfg_m:<10}]: \"{res_txt}\"")
            cfg_res[cfg_m] = res_txt
        print("=" * 80 + "\n")
        existing_results[ckpt_tag]["test_2_cfg_modes"] = cfg_res

    if args.test in ["3", "all"]:
        print("\n" + "=" * 80)
        print("TEST 3: DIFFUSION STEPS VARIATION")
        print("=" * 80)
        steps_res = {}
        for num_s in [25, 50, 100]:
            res_txt = run_diagnostic_generation(
                model, sc_module, decoder, latent_dict, tokenizer, default_prompt, device,
                max_len=args.max_len, steps=num_s, cfg_mode="baseline", use_sc=False, seed=args.seed
            )
            print(f"  [Steps: {num_s:3d}]: \"{res_txt}\"")
            steps_res[str(num_s)] = res_txt
        print("=" * 80 + "\n")
        existing_results[ckpt_tag]["test_3_diffusion_steps"] = steps_res

    if args.test in ["4", "all"]:
        print("\n" + "=" * 80)
        print("TEST 4: DIVERSE PROMPTS EVALUATION")
        print("=" * 80)
        prompts = [
            "В каком году произошла отмена крепостного права в России?",
            "Напиши короткое стихотворение про звезды и луну.",
            "Почему небо кажется нам голубым, а закаты красными?",
            "Привет, расскажи о себе."
        ]
        prompts_res = {}
        for p in prompts:
            res_txt = run_diagnostic_generation(
                model, sc_module, decoder, latent_dict, tokenizer, p, device,
                max_len=args.max_len, steps=args.steps, cfg_mode="baseline", use_sc=False, seed=args.seed
            )
            print(f"  [Prompt: \"{p[:45]}...\"]:\n     -> \"{res_txt}\"")
            prompts_res[p] = res_txt
        print("=" * 80 + "\n")
        existing_results[ckpt_tag]["test_4_diverse_prompts"] = prompts_res

    # Сохранение итогового файла
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(existing_results, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVED] All diagnostic test results successfully saved to: {out_file}\n")


if __name__ == "__main__":
    main()
