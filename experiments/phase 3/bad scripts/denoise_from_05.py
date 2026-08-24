import torch
import math
from transformers import AutoTokenizer
from inspect_and_test_phase3 import load_checkpoint_and_inspect, load_models, spherical_noise

def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    norm = torch.linalg.norm(x, dim=dim, keepdim=True)
    return x / torch.clamp(norm, min=eps)

def test_denoising(diff_model, decoder, tokenizer, text, start_t=0.5, steps=10, device="cpu"):
    print(f"\n=======================================================")
    print(f"[*] Исходный текст (target): '{text}'")
    
    # 1. Encode true text to latent space
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
    with torch.no_grad():
        qwen_embeds = diff_model.qwen_embeddings(inputs.input_ids)
        z_clean, _, _ = diff_model.encoder(qwen_embeds)
    
    z_clean = safe_normalize(z_clean.float(), dim=-1)
    
    # 2. Add spherical noise
    x_t = spherical_noise(z_clean, torch.tensor([start_t], device=device))
    x_t = safe_normalize(x_t, dim=-1)
    
    T_tokens = z_clean.shape[1]
    dummy_input = torch.zeros(1, T_tokens, dtype=torch.long, device=device)
    m_batch = torch.ones(1, T_tokens, dtype=torch.long, device=device)
    
    t_steps = torch.linspace(start_t, 0.0, steps + 1, device=device)
    
    print("\n[*] Процесс очистки (декодирование предсказанного x0 на каждом шаге):")
    for i in range(steps):
        t_curr = t_steps[i]
        t_next = t_steps[i+1]
        
        with torch.no_grad():
            out = diff_model(dummy_input, m_batch, torch.tensor([t_curr], device=device), z_noisy_override=x_t)
            
        pred_x0 = out["dus_final"]
        
        # Decode current prediction
        with torch.no_grad():
            logits = decoder(pred_x0.float())
        predicted_ids = logits.argmax(dim=-1)[0]
        decoded_text = tokenizer.decode(predicted_ids, skip_special_tokens=True)
        
        # Calculate cosine similarity with true target
        cos_sim = torch.nn.functional.cosine_similarity(pred_x0, z_clean, dim=-1).mean().item()
        
        print(f"t={t_curr.item():.2f} | Cos={cos_sim:.2f} | {decoded_text}")
        
        # DDIM extrapolation
        mu_t = torch.cos(t_curr * math.pi / 2)
        sigma_t = torch.sin(t_curr * math.pi / 2)
        
        eps = (x_t - mu_t * pred_x0) / torch.clamp(sigma_t, min=1e-5)
        eps = safe_normalize(eps, dim=-1)
        
        mu_next = torch.cos(t_next * math.pi / 2)
        sigma_next = torch.sin(t_next * math.pi / 2)
        
        x_t = mu_next * pred_x0 + sigma_next * eps
        x_t = safe_normalize(x_t, dim=-1)
        
    print("=======================================================")

def main():
    device = "cpu"
    dtype = torch.float32
    ckpt_path = r"C:\Experiments\BEBLaDII\experiments\phase3\local_checkpoints\phase3_step_18995.pth"
    
    print("[*] Загрузка чекпоинта и моделей...")
    state = load_checkpoint_and_inspect(ckpt_path)
    tokenizer, diff_model, decoder, _ = load_models(device, dtype, {})
    
    dus_ema = state.get("dus_ema", state.get("dus", {}))
    clean_dus = {k.replace("_orig_module.", ""): v for k, v in dus_ema.items()}
    diff_model.dus.load_state_dict(clean_dus, strict=False)
    
    for name in ["adaLN_attn", "adaLN_mlp", "t_proj", "self_cond_proj"]:
        ema = state.get(f"{name}_ema", state.get(name, {}))
        if ema:
            getattr(diff_model, name).load_state_dict(ema, strict=True)
            
    diff_model.eval()
    decoder.eval()
    
    test_texts = [
        "Машинное обучение позволяет компьютерам решать сложные задачи без явного программирования.",
        "Архитектура диффузионных моделей основана на марковских цепях и добавлении шума."
    ]
    
    for text in test_texts:
        test_denoising(diff_model, decoder, tokenizer, text, start_t=0.5, steps=10, device=device)
        
if __name__ == "__main__":
    main()
