import torch
import math
from transformers import AutoTokenizer
from inspect_and_test_phase3 import load_checkpoint_and_inspect, load_models

def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    norm = torch.linalg.norm(x, dim=dim, keepdim=True)
    return x / torch.clamp(norm, min=eps)

def generate_text_from_noise(diff_model, decoder, tokenizer, T_tokens=20, steps=50, device="cpu"):
    print(f"\n[*] Начинаем генерацию из шума (T={T_tokens}, шагов={steps})...")
    
    # 1. Стартуем с чистого сферического шума (t=1.0)
    x_t = safe_normalize(torch.randn(1, T_tokens, 1024, device=device), dim=-1)
    
    # 2. Создаем dummy input_ids и mask для интерфейса модели
    # Сами input_ids не используются, так как мы передаем z_noisy_override
    dummy_input = torch.zeros(1, T_tokens, dtype=torch.long, device=device)
    m_batch = torch.ones(1, T_tokens, dtype=torch.long, device=device)
    
    # DDIM расписание (от 1.0 до 0.0)
    t_steps = torch.linspace(1.0, 0.0, steps + 1, device=device)
    
    for i in range(steps):
        t_curr = t_steps[i]
        t_next = t_steps[i+1]
        
        with torch.no_grad():
            out = diff_model(dummy_input, m_batch, torch.tensor([t_curr], device=device), z_noisy_override=x_t)
        
        pred_x0 = out["dus_final"]
        
        # DDIM extrapolation
        mu_t = torch.cos(t_curr * math.pi / 2)
        sigma_t = torch.sin(t_curr * math.pi / 2)
        
        # Извлекаем шум
        eps = (x_t - mu_t * pred_x0) / torch.clamp(sigma_t, min=1e-5)
        eps = safe_normalize(eps, dim=-1)
        
        # Шагаем к t_next
        mu_next = torch.cos(t_next * math.pi / 2)
        sigma_next = torch.sin(t_next * math.pi / 2)
        
        x_t = mu_next * pred_x0 + sigma_next * eps
        x_t = safe_normalize(x_t, dim=-1)
        
        if (i+1) % 10 == 0:
            print(f"  Шаг {i+1}/{steps} (t={t_curr.item():.2f} -> {t_next.item():.2f})")
            
    # Декодируем результат (x_t сейчас при t=0, что эквивалентно pred_x0)
    print("\n[*] Декодируем латентный вектор...")
    with torch.no_grad():
        logits = decoder(x_t.float())
    
    predicted_ids = logits.argmax(dim=-1)[0]
    text = tokenizer.decode(predicted_ids, skip_special_tokens=True)
    return text

def main():
    device = "cpu"
    dtype = torch.float32
    ckpt_path = r"C:\Experiments\BEBLaDII\experiments\phase3\local_checkpoints\phase3_step_18995.pth"
    
    state = load_checkpoint_and_inspect(ckpt_path)
    
    # Инициализация моделей (с загрузкой checkpoint весов Phase 2 и VAE)
    tokenizer, diff_model, decoder, _ = load_models(device, dtype, {})
    
    # Загрузка весов из чекпоинта
    print("\n[*] Загрузка весов чекпоинта Phase 3...")
    dus_ema = state.get("dus_ema", state.get("dus", {}))
    clean_dus = {k.replace("_orig_module.", ""): v for k, v in dus_ema.items()}
    diff_model.dus.load_state_dict(clean_dus, strict=False)
    
    for name in ["adaLN_attn", "adaLN_mlp", "t_proj", "self_cond_proj"]:
        ema = state.get(f"{name}_ema", state.get(name, {}))
        if ema:
            getattr(diff_model, name).load_state_dict(ema, strict=True)
            print(f"[*] {name} загружен.")
            
    diff_model.eval()
    decoder.eval()
    
    # Генерируем несколько мыслей
    for attempt in range(3):
        print(f"\n==================== Попытка {attempt+1} ====================")
        text = generate_text_from_noise(diff_model, decoder, tokenizer, T_tokens=25, steps=50, device=device)
        print("\nСгенерированный текст:")
        print(f'"{text}"')
        print("==========================================================")

if __name__ == "__main__":
    main()
