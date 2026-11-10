import os
import sys
import math
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = "C:/Experiments/BEBLaDII"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments", "phase 4"))

from evaluate_phase4_checkpoints import BEBLaDIIPhase4aEval
from src.beb_la_dii.model.modern_decoder import ModernLatentDecoder
from transformers import AutoTokenizer, AutoModelForCausalLM

def safe_normalize(x, dim=-1, eps=1e-6):
    return F.normalize(x, p=2, dim=dim, eps=eps)

def load_model(device):
    embed_model_id = "Qwen/Qwen2.5-1.5B"
    modernbert_id  = "answerdotai/ModernBERT-large"
    phase4_ckpt    = os.path.join(PROJECT_ROOT, "experiments", "phase 4", "local_checkpoints", "phase4_step_85995.pth")
    vae_ckpt       = os.path.join(PROJECT_ROOT, "experiments", "phase 1", "planB_phase1_checkpoints_phase1_vae_step_20000.pth")

    print("Initializing BEBLaDIIPhase4aEval...")
    model = BEBLaDIIPhase4aEval(embedding_model_path=embed_model_id, modernbert_path=modernbert_id)

    print(f"Loading VAE Encoder...")
    vae_st = torch.load(vae_ckpt, map_location="cpu", weights_only=False)
    if 'encoder' in vae_st:
        model.encoder.load_state_dict(vae_st['encoder'], strict=False)

    print(f"Loading Phase 4 DUS...")
    p4_st = torch.load(phase4_ckpt, map_location="cpu", weights_only=False)
    dus_ema = p4_st.get("dus_ema", p4_st.get("dus", {}))
    clean_dus = {k.replace("_orig_module.", ""): v for k, v in dus_ema.items()}
    model.dus.load_state_dict(clean_dus, strict=False)

    model.to(device)
    model.eval()
    return model

def get_texts_from_parquet(path, col_name="text", num_samples=1000):
    try:
        df = pd.read_parquet(path)
        # Если колонка text не существует, берем первую
        if col_name not in df.columns:
            col_name = df.columns[0]
        return df[col_name].dropna().sample(n=min(num_samples, len(df))).tolist()
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []

def extract_latent_dictionary(model, device, batch_size=2048):
    """Прогоняет весь Qwen словарь через VAE для честного Target_DictProx"""
    print("Building full Latent Dictionary (151k tokens)...")
    with torch.no_grad():
        qwen_emb = model.qwen_embeddings.weight  # [151936, 1536]
        vocab_size = qwen_emb.shape[0]
        latents = []
        for i in tqdm(range(0, vocab_size, batch_size), desc="Dict encoding"):
            batch_emb = qwen_emb[i:i+batch_size].unsqueeze(1).to(device) # [B, 1, 1536]
            z, _, _ = model.encoder(batch_emb)
            latents.append(safe_normalize(z.squeeze(1).float(), dim=-1).cpu())
        
        latent_dict = torch.cat(latents, dim=0) # [151936, 1024]
        return latent_dict

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_path = os.path.join(PROJECT_ROOT, "experiments", "phase 5", "local", "unified_real_dataset.pt")
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    model = load_model(device)
    
    print("Loading Decoder and LM Head...")
    dec_ckpt = os.path.join(PROJECT_ROOT, "experiments", "phase 2", "planB_phase2_checkpoints_decoder_step_9000.pth")
    dus_weights_path = os.path.join(PROJECT_ROOT, "kaggle_upload_1_2", "AWAKENED_WEIGHTS_FINAL.pt")
    decoder = ModernLatentDecoder(latent_dim=1024, qwen_dim=1536, num_layers=3, dus_weights_path=dus_weights_path)
    dec_st = torch.load(dec_ckpt, map_location="cpu", weights_only=False)
    clean_state = {k.replace("decoder.", ""): v for k, v in dec_st.get("decoder", dec_st).items()}
    decoder.load_state_dict(clean_state, strict=False)
    decoder.to(device).eval()

    qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", torch_dtype=torch.bfloat16)
    lm_head_weight = qwen.lm_head.weight.detach().to(device)
    del qwen
    
    latent_dict = extract_latent_dictionary(model, device)
    latent_dict_gpu = latent_dict.to(device)

    # 1. Сбор текстов (Английский, Русский, Чешский)
    print("Loading multilingual datasets...")
    texts_ru = get_texts_from_parquet(os.path.join(PROJECT_ROOT, "data", "CulturaX", "data", "ru_part_00002.parquet"), num_samples=500)
    texts_cs = get_texts_from_parquet(os.path.join(PROJECT_ROOT, "data", "CulturaX", "data", "cs_part_00002.parquet"), num_samples=500)
    texts_en = get_texts_from_parquet(os.path.join(PROJECT_ROOT, "data", "open_thoughts", "data", "train-00000-of-00006.parquet"), num_samples=500)
    
    all_texts = texts_ru + texts_cs + texts_en
    print(f"Total texts loaded: {len(all_texts)}")

    seq_len = 64
    batch_size = 16
    
    all_h39_raw    = []
    all_dus_final  = []
    all_t_reported = []
    all_labels     = []

    print("Tokenizing and chunking sequences...")
    tokenized = tokenizer(all_texts, truncation=True, max_length=1024, return_attention_mask=False)["input_ids"]
    
    flat_tokens = [tok for seq in tokenized for tok in seq]
    num_chunks = len(flat_tokens) // seq_len
    input_ids_tensor = torch.tensor(flat_tokens[:num_chunks * seq_len]).view(num_chunks, seq_len)
    
    print(f"Total sequences available: {num_chunks}")
    
    # Чтобы не генерировать гигантский датасет сразу, ограничим
    max_batches = 300
    num_batches = min(num_chunks // batch_size, max_batches)

    print("Running DUS inference and calculating ground truth targets...")
    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            batch_ids = input_ids_tensor[i*batch_size : (i+1)*batch_size].to(device)
            B, T = batch_ids.shape
            attn_mask = torch.ones((B, T), device=device)

            # Создаем иерархический шум:
            # Треть батча: чистый контекст с случайным шумом (0.0 - 0.3)
            # Треть батча: грязный контекст с редкими якорями
            # Треть батча: абсолютно случайный t (0.0 - 1.0)
            
            t_base = torch.rand(B, device=device)  # [B]
            t_local = torch.rand(B, T, device=device)  # [B, T]
            
            # Смешиваем базовый шум с локальным
            t_mixed = torch.clamp(t_base.unsqueeze(-1) * 0.5 + t_local * 0.5, 0.0, 1.0)
            
            # Рандомно делаем часть токенов абсолютно чистыми (якоря t=0)
            anchor_mask = (torch.rand(B, T, device=device) > 0.85)
            t_mixed[anchor_mask] = 0.0

            # 1. Прогон Self-Conditioning
            out_sc = model(batch_ids, attn_mask, t_global=t_base, t_reported=t_mixed)
            z_noisy = out_sc["z_noisy"]
            sc_est = out_sc["dus_final"].detach()
            
            # 2. Боевой прогон с self_cond
            out = model(batch_ids, attn_mask, t_global=t_base, t_reported=t_mixed, self_cond=sc_est, z_noisy_override=z_noisy)
            h39_raw = out["h_39_raw"]    # [B, T, 1024]
            z_pred_raw = out["dus_final"] # [B, T, 1024]
            
            # 3. Identity Gate
            gate_t = torch.sin(t_mixed * (math.pi / 2)).unsqueeze(-1)
            dus_final = safe_normalize(gate_t * z_pred_raw + (1.0 - gate_t) * z_noisy, dim=-1)
            
            # --- ВЫЧИСЛЕНИЕ TARGET МЕТОК (GROUND TRUTH) ---
            
            # 1. Target_ModConf (Уверенность модели = обратная энтропия декодера)
            with torch.no_grad():
                # Прогоняем вектор (dus_final) через слой декодера и lm_head
                projected = decoder(dus_final.to(next(decoder.parameters()).dtype))
                logits = F.linear(projected.float(), lm_head_weight.float())
                probs = F.softmax(logits, dim=-1)
                
                # Энтропия H = -sum(p * log(p))
                entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
                
                # Нормализация через сигмоиду: Conf = 1 / (1 + exp(k * (H - H0)))
                # H0 = 4.0 (энтропия, соответствующая равномерному выбору из ~50 токенов, дает Conf = 0.5)
                # Если H = 1.5 (уверенный выбор из 2-3 токенов), Conf ≈ 0.92
                # Если H = 6.0 (сильный шум, выбор из ~400 токенов), Conf ≈ 0.12
                target_modconf = 1.0 / (1.0 + torch.exp(entropy - 4.0))

            # 2. Target_DictProx (Расстояние до ближайшего валидного токена из 151k словаря)
            # dus_final: [B, T, 1024], latent_dict_gpu: [151936, 1024]
            flat_pred = dus_final.view(-1, 1024)
            # Считаем косинусное сходство пакетами, чтобы не взорвать память
            max_cos = torch.zeros(flat_pred.shape[0], device=device)
            chunk_size = 10000
            for j in range(0, 151936, chunk_size):
                dict_chunk = latent_dict_gpu[j:j+chunk_size]
                # [B*T, chunk_size]
                sims = torch.matmul(flat_pred, dict_chunk.T)
                max_cos = torch.max(max_cos, torch.max(sims, dim=1)[0])
            
            target_dictprox = torch.clamp(max_cos.view(B, T), 0.0, 1.0)
            
            # 3. Target_Complexity (Обратно ModConf)
            target_complexity = 1.0 - target_modconf

            # 4. Target_Coherence
            # Пока задаем просто 1.0 для всех реальных слов, 
            # но если токен слишком сильно шумит и отдален (DProx < 0.3), считаем его галлюцинацией (0.0)
            target_coherence = torch.where(target_dictprox > 0.5, 1.0, 0.0)
            
            # 5. Target_Cryst (Интегральный)
            target_cryst = target_dictprox * target_modconf * target_coherence

            # 6. Stuck (Тупик) - DProx > 0.8, но Coherence низкая (галлюцинация)
            target_stuck = torch.where((target_dictprox > 0.8) & (target_coherence < 0.5), 1.0, 0.0)

            # Собираем метки [B, T, 6]
            labels = torch.stack([
                target_dictprox,
                target_modconf,
                target_coherence,
                target_complexity,
                target_cryst,
                target_stuck
            ], dim=-1)

            all_h39_raw.append(h39_raw.cpu())
            all_dus_final.append(dus_final.cpu())
            all_t_reported.append(t_mixed.cpu().unsqueeze(-1))
            all_labels.append(labels.cpu())

    print("Concatenating...")
    dataset = {
        "h39_raw":    torch.cat(all_h39_raw,    dim=0),
        "dus_final":  torch.cat(all_dus_final,  dim=0),
        "t_reported": torch.cat(all_t_reported, dim=0),
        "labels":     torch.cat(all_labels,     dim=0),
    }
    
    print(f"Dataset shapes:")
    for k, v in dataset.items():
        print(f"  {k}: {v.shape}")

    torch.save(dataset, output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
