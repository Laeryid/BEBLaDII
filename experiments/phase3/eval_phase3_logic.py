import os
import re
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Добавляем корень проекта в sys.path
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.beb_la_dii.model.vae import LatentEncoder
from src.beb_la_dii.model.modern_decoder import ModernLatentDecoder
from src.beb_la_dii.model.dus import DUSModel
from src.beb_la_dii.utils.loss import safe_normalize

def recover_z(dus_final, c_embed):
    """
    Математически точное восстановление z_clean из выхода DUS.
    Таргет при обучении: dus_final = safe_normalize(z_clean + c_embed).
    Поскольку ||z_clean|| = 1, решаем уравнение для сферы:
    z_clean = alpha * dus_final - c_embed, где ||z_clean|| = 1.
    """
    y_dot_c = (dus_final * c_embed).sum(dim=-1, keepdim=True)
    c_norm_sq = (c_embed * c_embed).sum(dim=-1, keepdim=True)
    
    discriminant = torch.clamp((y_dot_c ** 2) - c_norm_sq + 1.0, min=1e-6)
    alpha = y_dot_c + torch.sqrt(discriminant)
    
    z_rec = alpha * dus_final - c_embed
    return safe_normalize(z_rec, dim=-1)

def load_dus_checkpoint(dus, conf_proj, path):
    state = torch.load(path, map_location="cpu")
    
    dus_state = state.get("dus_ema", state.get("dus", state))
    if "latentBERT_state_dict" in dus_state:
        dus_state = dus_state["latentBERT_state_dict"]
    elif "model_state_dict" in dus_state:
        dus_state = dus_state["model_state_dict"]
        
    clean_dus = {}
    for k, v in dus_state.items():
        k_clean = k.replace("student.model.", "").replace("model.", "").replace("_orig_module.", "")
        clean_dus[k_clean] = v
        
    dus.load_state_dict(clean_dus, strict=False)
    
    proj_state = state.get("confidence_proj_ema", state.get("confidence_proj", {}))
    if proj_state:
        clean_proj = {k.replace("_orig_module.", ""): v for k, v in proj_state.items()}
        conf_proj.load_state_dict(clean_proj, strict=False)
        print(f"[*] DUS and ConfidenceProj weights loaded from {path} (EMA preferred)")
    else:
        print(f"[!] Warning: confidence_proj weights NOT found in {path}")

def decode_to_text(z, decoder, lm_head_weight, tokenizer):
    """Декодирует латентный вектор обратно в текст"""
    with torch.no_grad():
        # z: (1, T, 1024) -> dec_out: (1, T, 1536)
        dec_out = decoder(z)
        
        # Проекция в токены
        logits = F.linear(dec_out, lm_head_weight) # (1, T, V)
        token_ids = logits.argmax(dim=-1).squeeze(0) # (T,)
        
        text = tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)
        return text

def parse_bracket_phrase(phrase, tokenizer):
    """
    Парсит фразу с маркерами [], возвращает:
    - чистую строку
    - индексы токенов, которые соответствуют словам в скобках
    """
    clean_text = ""
    bracket_words = []
    
    # Находим все слова в скобках
    parts = re.split(r'(\[[^\]]+\])', phrase)
    
    for part in parts:
        if part.startswith('[') and part.endswith(']'):
            word = part[1:-1]
            bracket_words.append(word)
            clean_text += word
        else:
            clean_text += part
            
    # Токенизируем чистый текст
    tokens = tokenizer.encode(clean_text)
    
    # Ищем индексы зашумляемых токенов
    noise_indices = []
    # Для упрощения: просто токенизируем слова и ищем их в последовательности
    for bw in bracket_words:
        bw_tokens = tokenizer.encode(bw, add_special_tokens=False)
        # Ищем подпоследовательность
        for i in range(len(tokens) - len(bw_tokens) + 1):
            if tokens[i:i+len(bw_tokens)] == bw_tokens:
                noise_indices.extend(list(range(i, i+len(bw_tokens))))
                break
                
    return clean_text, tokens, noise_indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Путь к чекпоинту DUS (например, 10000.pth)")
    parser.add_argument("--encoder", type=str, default="checkpoints/phase1/encoder.pth", help="Путь к весам LatentEncoder")
    parser.add_argument("--decoder", type=str, default="checkpoints/phase1/decoder.pth", help="Путь к весам LatentDecoder")
    parser.add_argument("--embed_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="Модель для эмбеддингов")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"[*] Loading Tokenizer and Embedding Model: {args.embed_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.embed_model)
    causal_model = AutoModelForCausalLM.from_pretrained(args.embed_model, torch_dtype=torch.bfloat16)
    embeddings = causal_model.get_input_embeddings()
    embeddings.to(device)
    embeddings.requires_grad_(False)
    lm_head_weight = causal_model.lm_head.weight.data.to(device)

    print("[*] Loading LatentEncoder & ModernLatentDecoder")
    encoder = LatentEncoder().to(device).to(torch.bfloat16)
    encoder.eval()
    
    decoder = ModernLatentDecoder(dus_weights_path=None)
    dus_for_dec = DUSModel.from_scratch(weights_path=None).model
    decoder.backbone = dus_for_dec
    decoder.backbone.layers = decoder.backbone.layers[-3:]
    decoder.use_modern_bert = True
    decoder = decoder.to(device).to(torch.bfloat16)
    
    def load_with_stats(module, state_dict, name):
        model_keys = set(module.state_dict().keys())
        matched_keys = set(state_dict.keys()) & model_keys
        missing, unexpected = module.load_state_dict(state_dict, strict=False)
        print(f"[*] {name} loaded:")
        print(f"    - Matched parameters: {len(matched_keys)} / {len(model_keys)}")
        if missing:
            print(f"    - Missing keys (first 5): {missing[:5]}")
        if unexpected:
            print(f"    - Unexpected keys in checkpoint (first 5): {unexpected[:5]}")

    if os.path.exists(args.encoder):
        state = torch.load(args.encoder, map_location="cpu")
        load_with_stats(encoder, state.get("encoder", state), "LatentEncoder")
    else:
        print(f"[!] Warning: Encoder weights not found at {args.encoder}")

    if os.path.exists(args.decoder):
        state = torch.load(args.decoder, map_location="cpu")
        load_with_stats(decoder, state.get("decoder", state), "LatentDecoder")
    else:
        print(f"[!] Warning: Decoder weights not found at {args.decoder}")

    encoder.eval()
    decoder.eval()

    print(f"[*] Loading DUS Model and Confidence Proj from {args.checkpoint}")
    dus_wrapper = DUSModel.from_scratch(weights_path=None)
    dus = dus_wrapper.model.to(device).to(torch.bfloat16)
    
    confidence_proj = torch.nn.Sequential(
        torch.nn.Linear(1, 256),
        torch.nn.SiLU(),
        torch.nn.Linear(256, 1024)
    ).to(device).to(torch.bfloat16)
    
    load_dus_checkpoint(dus, confidence_proj, args.checkpoint)
    dus.eval()
    confidence_proj.eval()
    
    # ---------------------------------------------------------
    # ТЕСТ 1: Без шума (5 фраз)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("ТЕСТ 1: Без шума (Проверка Identity через DUS)")
    print("="*50)
    phrases = [
        "Machine learning allows computers to solve complex problems without explicit programming.",
        "It rained heavily last night, and huge puddles formed on the street.",
        "The singularity is a hypothetical moment in the future when technological growth becomes uncontrollable.",
        "The cat is sleeping on the windowsill while fluffy snow falls slowly outside.",
        "Transformer architecture revolutionized the field of natural language processing."
    ]
    
    for i, phrase in enumerate(phrases, 1):
        input_ids = torch.tensor([tokenizer.encode(phrase)]).to(device)
        attn_mask = torch.ones_like(input_ids).to(device)
        
        with torch.no_grad():
            qwen_embeds = embeddings(input_ids)
            z_clean, _, _ = encoder(qwen_embeds)
            
            c_true = torch.ones((1, z_clean.size(1)), device=device, dtype=torch.bfloat16)
            c_embed = confidence_proj(c_true.unsqueeze(-1))
            dus_input = z_clean + c_embed
            
            dus_out_raw = dus(inputs_embeds=dus_input, attention_mask=attn_mask).last_hidden_state
            dus_out_norm = safe_normalize(dus_out_raw.float(), dim=-1).to(torch.bfloat16)
            
            z_recovered = recover_z(dus_out_norm.float(), c_embed.float()).to(torch.bfloat16)
            
            # Декодируем z_clean (чистый автоэнкодер) и z_recovered
            text_ae = decode_to_text(z_clean, decoder, lm_head_weight, tokenizer)
            text_dus = decode_to_text(z_recovered, decoder, lm_head_weight, tokenizer)
            
            # Считаем метрики расстояния (только для токенов без учета pad)
            active_z_clean = z_clean[attn_mask.bool()]
            active_z_rec = z_recovered[attn_mask.bool()]
            
            cos_sim = F.cosine_similarity(active_z_clean, active_z_rec, dim=-1).mean().item()
            l2_dist = F.pairwise_distance(active_z_clean.float(), active_z_rec.float()).mean().item()
            
        print(f"\nФраза {i}: {phrase}")
        print(f"  [AutoEncoder Only] -> {text_ae}")
        print(f"  [AutoEncoder + DUS]-> {text_dus}")
        print(f"  [Metrics DUS vs AE]-> Cosine Similarity: {cos_sim:.4f} | L2 Distance: {l2_dist:.4f}")

    # ---------------------------------------------------------
    # ТЕСТ 2: Легкий шум (15% токенов случайный сдвиг)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("ТЕСТ 2: Легкий шум (как на обучении, 15% токенов)")
    print("="*50)
    test2_phrase = "Neural networks possess an incredible ability to generalize data from massive amounts of information."
    input_ids = torch.tensor([tokenizer.encode(test2_phrase)]).to(device)
    attn_mask = torch.ones_like(input_ids).to(device)
    
    with torch.no_grad():
        qwen_embeds = embeddings(input_ids)
        z_clean, _, _ = encoder(qwen_embeds)
        
        T = z_clean.size(1)
        noise_mask = (torch.rand(1, T).to(device) < 0.15).float().unsqueeze(-1)
        
        noise = torch.randn_like(z_clean)
        noise = safe_normalize(noise, dim=-1) * 0.5 # low_noise_amp
        z_noisy = z_clean + noise * noise_mask
        
        # Восстанавливаем норму
        z_clean_norm = torch.norm(z_clean, p=2, dim=-1, keepdim=True)
        z_noisy = torch.where(
            noise_mask > 0,
            safe_normalize(z_noisy, dim=-1) * z_clean_norm,
            z_noisy
        )
        
        c_true = torch.clamp(F.cosine_similarity(z_clean.float(), z_noisy.float(), dim=-1), min=0.0).to(torch.bfloat16)
        c_embed = confidence_proj(c_true.unsqueeze(-1))
        dus_input = z_noisy + c_embed
        
        dus_out_raw = dus(inputs_embeds=dus_input, attention_mask=attn_mask).last_hidden_state
        dus_out_norm = safe_normalize(dus_out_raw.float(), dim=-1).to(torch.bfloat16)
        z_recovered = recover_z(dus_out_norm.float(), c_embed.float()).to(torch.bfloat16)
        
        text_noisy = decode_to_text(z_noisy, decoder, lm_head_weight, tokenizer)
        text_recovered = decode_to_text(z_recovered, decoder, lm_head_weight, tokenizer)
        
        active_z_clean = z_clean[attn_mask.bool()]
        active_z_noisy = z_noisy[attn_mask.bool()]
        active_z_rec = z_recovered[attn_mask.bool()]
        
        # Сравниваем с z_clean (идеалом)
        sim_noisy = F.cosine_similarity(active_z_clean, active_z_noisy, dim=-1).mean().item()
        sim_recov = F.cosine_similarity(active_z_clean, active_z_rec, dim=-1).mean().item()
        l2_noisy = F.pairwise_distance(active_z_clean.float(), active_z_noisy.float()).mean().item()
        l2_recov = F.pairwise_distance(active_z_clean.float(), active_z_rec.float()).mean().item()
        
    print(f"Оригинал:   {test2_phrase}")
    print(f"Зашумлено:  {text_noisy}")
    print(f"  [Noisy vs Ideal]   -> Cosine: {sim_noisy:.4f} | L2: {l2_noisy:.4f}")
    print(f"DUS выход:  {text_recovered}")
    print(f"  [DUS rec vs Ideal] -> Cosine: {sim_recov:.4f} | L2: {l2_recov:.4f}")

    # ---------------------------------------------------------
    # ТЕСТ 3: Тяжелый шум (глухой блэкаут слов) + Итеративная диффузия
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("ТЕСТ 3: Тяжелый шум (полная замена случайными векторами) + 5 шагов диффузии")
    print("="*50)
    test3_phrase_raw = "The capital of France is the city of [Paris], which is famous for its unique [architecture] and history."
    
    clean_text, tokens, noise_indices = parse_bracket_phrase(test3_phrase_raw, tokenizer)
    input_ids = torch.tensor([tokens]).to(device)
    attn_mask = torch.ones_like(input_ids).to(device)
    
    print(f"Оригинал (шаблон): {test3_phrase_raw}")
    print(f"Индексы для глухого шума: {noise_indices}")
    
    with torch.no_grad():
        qwen_embeds = embeddings(input_ids)
        z_clean, _, _ = encoder(qwen_embeds)
        
        z_heavy = z_clean.clone()
        z_clean_norm = torch.norm(z_clean, p=2, dim=-1, keepdim=True)
        
        for idx in noise_indices:
            # Заменяем на абсолютно случайную точку на сфере того же радиуса
            rand_vec = safe_normalize(torch.randn(1, 1, 1024).to(device), dim=-1)
            # Извлекаем радиус конкретного токена
            r = z_clean_norm[0, idx, 0]
            z_heavy[0, idx, :] = rand_vec.squeeze() * r
            
        text_heavy = decode_to_text(z_heavy, decoder, lm_head_weight, tokenizer)
        print(f"\nСам шум (итерация 0): {text_heavy}")
        
        # Диффузия (5 итераций)
        z_curr = z_heavy.clone()
        for step in range(1, 6):
            c_true = torch.clamp(F.cosine_similarity(z_clean.float(), z_curr.float(), dim=-1), min=0.0).to(torch.bfloat16)
            c_embed = confidence_proj(c_true.unsqueeze(-1))
            dus_input = z_curr + c_embed
            
            dus_out_raw = dus(inputs_embeds=dus_input, attention_mask=attn_mask).last_hidden_state
            dus_out_norm = safe_normalize(dus_out_raw.float(), dim=-1).to(torch.bfloat16)
            z_rec = recover_z(dus_out_norm.float(), c_embed.float()).to(torch.bfloat16)
            
            text_step = decode_to_text(z_rec, decoder, lm_head_weight, tokenizer)
            
            # Сравнение с эталоном z_clean
            active_z_clean = z_clean[attn_mask.bool()]
            active_z_rec = z_rec[attn_mask.bool()]
            cos_sim = F.cosine_similarity(active_z_clean, active_z_rec, dim=-1).mean().item()
            l2_dist = F.pairwise_distance(active_z_clean.float(), active_z_rec.float()).mean().item()
            
            print(f"Итерация {step}: {text_step} (Cos: {cos_sim:.4f}, L2: {l2_dist:.4f})")
            
            # Для следующей итерации используем выход текущей
            z_curr = z_out

if __name__ == "__main__":
    main()
