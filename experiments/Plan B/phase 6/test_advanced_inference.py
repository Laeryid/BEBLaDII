import os
import sys
import math
import datetime
import importlib.util
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = "C:/Experiments/BEBLaDII"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# Импорт проверенной инфраструктуры из evaluate_phase6_metrics.py
eval_script_path = os.path.join(PROJECT_ROOT, "experiments", "phase 6", "evaluate_phase6_metrics.py")
spec = importlib.util.spec_from_file_location("eval_p6", eval_script_path)
eval_p6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_p6)

BEBLaDIIPhase6Eval = eval_p6.BEBLaDIIPhase6Eval
ModernLatentDecoder = eval_p6.ModernLatentDecoder
decode_latents_to_text = eval_p6.decode_latents_to_text
spherical_noise = eval_p6.spherical_noise
safe_normalize = eval_p6.safe_normalize
normalize_key = eval_p6.normalize_key


def run_advanced_generation(
    model,
    decoder,
    latent_dict,
    tokenizer,
    query_text: str,
    device: torch.device,
    max_len: int = 64,
    steps: int = 25,
    seed: int = 42,
    mode: str = "standard",  # 'standard', 'scheduled_cfg', 'anchor_jump', 'hybrid'
    cfg_max: float = 3.5,
    cfg_min: float = 1.0,
):
    """
    Продвинутый инференс диффузии Phase 6:
      - 'standard': постоянный CFG
      - 'scheduled_cfg': динамический CFG, высокий на t=1.0 и спадающий к t=0.04
      - 'anchor_jump': перешумление от z_pred на шаге t=1.0 (сброс паразитного скелета начального шума eps_0)
      - 'hybrid': сочетание anchor_jump на старте + scheduled_cfg
    """
    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] >>> Query: '{query_text}'")
    print(f"    Mode: {mode} | Seed: {seed} | CFG: [{cfg_max} -> {cfg_min}] | Steps: {steps}")

    q_enc = tokenizer(query_text, return_tensors='pt').to(device)

    with torch.no_grad():
        Z_prompt_12 = model.encode_prompt(q_enc.input_ids, q_enc.attention_mask)

        B = 1
        T_a = max_len
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        z_canvas = safe_normalize(torch.randn(B, T_a, 1024, device=device), dim=-1)
        dt = 1.0 / steps

        for i in range(steps):
            t_val = round((1.0 - i * dt) / 0.04) * 0.04
            t_val = max(0.04, min(1.0, round(t_val, 4)))
            t_actual = torch.full((B, T_a), t_val, device=device)
            t_global = torch.full((B,), t_val, device=device)

            sims = torch.matmul(z_canvas, latent_dict.T)
            raw_dprox, nn_ids = sims.max(dim=-1)
            t_reported = (1.0 - raw_dprox).clamp(0.0, 1.0)

            # 1. Прямой шаг с CA
            _, z_pred = model.forward_step(
                z_canvas, Z_prompt_12, q_enc.attention_mask, t_actual, t_reported, t_global=t_global
            )

            # 2. Определение динамического коэффициента CFG
            if mode in ["scheduled_cfg", "hybrid"]:
                # Синусоидальное расписание: максимум при t=1.0, минимум при t->0.0
                weight_t = math.sin(t_val * (math.pi / 2))
                current_cfg = cfg_min + (cfg_max - cfg_min) * weight_t
            else:
                current_cfg = cfg_max

            # 3. Classifier-Free Guidance (CFG)
            if current_cfg > 1.0:
                for ca in model.ca_layers.values():
                    ca._disable = True
                _, z_uncond = model.forward_step(
                    z_canvas, Z_prompt_12, q_enc.attention_mask, t_actual, t_reported, t_global=t_global
                )
                for ca in model.ca_layers.values():
                    ca._disable = False
                z_pred = safe_normalize(z_uncond + current_cfg * (z_pred - z_uncond), dim=-1)

            t_next_val = max(0.0, round((t_val - 0.04) / 0.04) * 0.04)
            t_next_val = round(t_next_val, 4)
            t_next = torch.full((B, T_a), t_next_val, device=device)

            # 4. Обновление холста
            if i == 0 and mode in ["anchor_jump", "hybrid"]:
                # Прыжок из предсказания промпта: перешумляем вокруг z_pred на уровне t_next
                z_canvas = spherical_noise(z_pred, t_next)
            else:
                # Классический Slerp
                theta_now = t_actual.unsqueeze(-1) * (math.pi / 2)
                theta_next = t_next.unsqueeze(-1) * (math.pi / 2)
                sin_now = torch.sin(theta_now)
                sin_now = torch.where(sin_now < 1e-5, torch.ones_like(sin_now) * 1e-5, sin_now)
                w_pred = torch.sin(theta_now - theta_next) / sin_now
                w_cur = torch.sin(theta_next) / sin_now

                z_next = w_pred * z_pred + w_cur * z_canvas
                z_canvas = torch.where(theta_now > 1e-5, safe_normalize(z_next, dim=-1), safe_normalize(z_pred, dim=-1))

        # Финальное декодирование
        final_text = decode_latents_to_text(decoder, model.qwen_embeddings.weight, tokenizer, z_canvas)
        final_sims = torch.matmul(z_canvas, latent_dict.T)
        _, final_nn_ids = final_sims.max(dim=-1)
        final_nn_text = tokenizer.decode(final_nn_ids[0], skip_special_tokens=True)

        print(f"--- [ModernLatentDecoder]: {final_text}")
        print(f"--- [LatentDict NN      ]: {final_nn_text}\n")
        return final_text, final_nn_text


def main():
    target_ckpt = os.path.join(
        PROJECT_ROOT, "experiments", "phase 6", "checkpoints", "planB_phase6_checkpoints_phase6_ca_layers_step_49000.pth"
    )

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
    if not os.path.exists(dict_path):
        dict_path = os.path.join(PROJECT_ROOT, "BEBLaDII-planB-Phase5-Data", "latent_dict.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading models on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(config['qwen_path'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = BEBLaDIIPhase6Eval(config).to(device)

    decoder = ModernLatentDecoder(
        latent_dim=1024,
        qwen_dim=1536,
        num_layers=3,
        dus_weights_path=config['phase4_path']
    ).to(device)
    ckpt_dec = torch.load(config['phase2_decoder_path'], map_location=device, weights_only=False)
    st_dec = ckpt_dec.get("decoder", ckpt_dec)
    clean_st_dec = {normalize_key(k).replace("decoder.", ""): v for k, v in st_dec.items()}
    decoder.load_state_dict(clean_st_dec, strict=False)
    decoder.eval()

    latent_dict = torch.load(dict_path, map_location=device).float()

    q_sky = "Explain why the sky is blue using simple words."
    q_arch = "How does your generation mechanism contrast with traditional next-token prediction models?"

    experiments = [
        # 1. Проверка влияния seed при фиксированном CFG 2.5
        ("EXP 1: Sky - Seed 123 (Проверка смены синтаксического скелета)", q_sky, 123, "standard", 2.5, 1.0),
        ("EXP 2: Sky - Seed 777 (Проверка смены синтаксического скелета)", q_sky, 777, "standard", 2.5, 1.0),
        # 2. Time-Scheduled CFG: усиление влияния на старте (t=1.0) с плавным снижением
        ("EXP 3: Sky - Scheduled CFG (4.0 -> 1.0)", q_sky, 42, "scheduled_cfg", 4.0, 1.0),
        ("EXP 4: Sky - Scheduled CFG (5.0 -> 1.2)", q_sky, 42, "scheduled_cfg", 5.0, 1.2),
        # 3. Anchor Jump на шаге t=1.0 (сброс eps_0 и старт от z_pred)
        ("EXP 5: Sky - Anchor Jump (CFG 2.5)", q_sky, 42, "anchor_jump", 2.5, 1.0),
        ("EXP 6: Sky - Hybrid (Anchor Jump + Scheduled CFG 4.0 -> 1.0)", q_sky, 42, "hybrid", 4.0, 1.0),
        # 4. Проверка на вопросе об архитектуре
        ("EXP 7: Arch - Hybrid (Anchor Jump + Scheduled CFG 4.0 -> 1.0)", q_arch, 42, "hybrid", 4.0, 1.0),
    ]

    print("\n" + "="*80)
    print("STARTING ADVANCED INFERENCE EXPERIMENTS")
    print("="*80)

    for desc, q, s, m, c_max, c_min in experiments:
        print(f"\n>>> Running: {desc}")
        run_advanced_generation(
            model, decoder, latent_dict, tokenizer, q, device,
            max_len=64, steps=25, seed=s, mode=m, cfg_max=c_max, cfg_min=c_min
        )

    print("\n[ALL INFERENCE EXPERIMENTS COMPLETED]")


if __name__ == "__main__":
    main()
