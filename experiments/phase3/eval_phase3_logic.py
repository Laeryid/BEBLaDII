import os
import re
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Добавляем корень проекта в sys.path
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.beb_la_dii.model.vae import LatentEncoder, LatentDecoder
from src.beb_la_dii.model.dus import DUSModel
from src.beb_la_dii.utils.loss import safe_normalize

def load_dus_checkpoint(model, path):
    state = torch.load(path, map_location="cpu")
    if "latentBERT_state_dict" in state:
        state = state["latentBERT_state_dict"]
    elif "model_state_dict" in state:
        state = state["model_state_dict"]
    
    clean_state = {}
    for k, v in state.items():
        k_clean = k.replace("student.model.", "").replace("model.", "")
        clean_state[k_clean] = v
        
    model.load_state_dict(clean_state, strict=False)
    print(f"[*] DUS weights loaded from {path}")

def decode_to_text(z, decoder, embed_matrix, tokenizer):
    """Декодирует латентный вектор обратно в текст"""
    with torch.no_grad():
        # z: (1, T, 1024) -> dec_out: (1, T, 1536)
        dec_out = decoder(z)
        
        # embed_matrix: (V, 1536)
        # Нормализуем для косинусного сходства
        dec_out_norm = F.normalize(dec_out.squeeze(0), p=2, dim=-1) # (T, 1536)
        embed_norm = F.normalize(embed_matrix, p=2, dim=-1) # (V, 1536)
        
        # Считаем сходство: (T, 1536) @ (1536, V) -> (T, V)
        sim = torch.matmul(dec_out_norm, embed_norm.T)
        token_ids = sim.argmax(dim=-1) # (T,)
        
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
    parser.add_argument("--embed_model", type=str, default="Qwen/Qwen2.5-1.5B", help="Модель для эмбеддингов")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"[*] Loading Tokenizer and Embedding Model: {args.embed_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.embed_model)
    embed_model = AutoModel.from_pretrained(args.embed_model, torch_dtype=torch.float32)
    embeddings = embed_model.get_input_embeddings()
    embeddings.to(device)
    embeddings.requires_grad_(False)
    embed_matrix = embeddings.weight.data

    print("[*] Loading LatentEncoder & LatentDecoder")
    encoder = LatentEncoder().to(device)
    decoder = LatentDecoder().to(device)
    
    if os.path.exists(args.encoder):
        state = torch.load(args.encoder, map_location="cpu")
        encoder.load_state_dict(state.get("encoder", state), strict=False)
    else:
        print(f"[!] Warning: Encoder weights not found at {args.encoder}")

    if os.path.exists(args.decoder):
        state = torch.load(args.decoder, map_location="cpu")
        decoder.load_state_dict(state.get("decoder", state), strict=False)
    else:
        print(f"[!] Warning: Decoder weights not found at {args.decoder}")

    encoder.eval()
    decoder.eval()

    print(f"[*] Loading DUS Model from {args.checkpoint}")
    dus_wrapper = DUSModel.from_scratch(weights_path=None)
    load_dus_checkpoint(dus_wrapper.model, args.checkpoint)
    dus = dus_wrapper.model.to(device)
    dus.eval()
    
    # ---------------------------------------------------------
    # ТЕСТ 1: Без шума (5 фраз)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("ТЕСТ 1: Без шума (Проверка Identity через DUS)")
    print("="*50)
    phrases = [
        "Машинное обучение позволяет компьютерам решать сложные задачи.",
        "Вчера вечером прошел сильный дождь, и на улице образовались огромные лужи.",
        "Сингулярность — это гипотетический момент в будущем, когда развитие станет неуправляемым.",
        "Кот спит на подоконнике, пока за окном медленно падает пушистый снег.",
        "Архитектура трансформеров произвела революцию в области обработки естественного языка."
    ]
    
    for i, phrase in enumerate(phrases, 1):
        input_ids = torch.tensor([tokenizer.encode(phrase)]).to(device)
        attn_mask = torch.ones_like(input_ids).to(device)
        
        with torch.no_grad():
            qwen_embeds = embeddings(input_ids)
            z_clean, _, _ = encoder(qwen_embeds)
            dus_out = dus(inputs_embeds=z_clean, attention_mask=attn_mask).last_hidden_state
            
            # Декодируем z_clean (чистый автоэнкодер) и dus_out
            text_ae = decode_to_text(z_clean, decoder, embed_matrix, tokenizer)
            text_dus = decode_to_text(dus_out, decoder, embed_matrix, tokenizer)
            
        print(f"\nФраза {i}: {phrase}")
        print(f"  [AutoEncoder Only] -> {text_ae}")
        print(f"  [AutoEncoder + DUS]-> {text_dus}")

    # ---------------------------------------------------------
    # ТЕСТ 2: Легкий шум (15% токенов случайный сдвиг)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("ТЕСТ 2: Легкий шум (как на обучении, 15% токенов)")
    print("="*50)
    test2_phrase = "Нейросети обладают невероятной способностью к обобщению данных из огромных массивов информации."
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
        
        dus_out = dus(inputs_embeds=z_noisy, attention_mask=attn_mask).last_hidden_state
        
        text_noisy = decode_to_text(z_noisy, decoder, embed_matrix, tokenizer)
        text_recovered = decode_to_text(dus_out, decoder, embed_matrix, tokenizer)
        
    print(f"Оригинал:   {test2_phrase}")
    print(f"Зашумлено:  {text_noisy}")
    print(f"DUS выход:  {text_recovered}")

    # ---------------------------------------------------------
    # ТЕСТ 3: Тяжелый шум (глухой блэкаут слов) + Итеративная диффузия
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("ТЕСТ 3: Тяжелый шум (полная замена случайными векторами) + 5 шагов диффузии")
    print("="*50)
    test3_phrase_raw = "Столица Франции — город [Париж], который знаменит своей уникальной [архитектурой] и историей."
    
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
            
        text_heavy = decode_to_text(z_heavy, decoder, embed_matrix, tokenizer)
        print(f"\nСам шум (итерация 0): {text_heavy}")
        
        # Диффузия (5 итераций)
        z_curr = z_heavy.clone()
        for step in range(1, 6):
            z_out = dus(inputs_embeds=z_curr, attention_mask=attn_mask).last_hidden_state
            
            text_step = decode_to_text(z_out, decoder, embed_matrix, tokenizer)
            print(f"Итерация {step}: {text_step}")
            
            # Для следующей итерации используем выход текущей
            z_curr = z_out

if __name__ == "__main__":
    main()
