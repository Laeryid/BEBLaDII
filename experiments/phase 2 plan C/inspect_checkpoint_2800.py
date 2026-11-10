import os
import sys
import torch
import torch.nn.functional as F

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = "C:/Experiments/BEBLaDII"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.beb_la_dii.model.vae import LatentEncoder
from src.beb_la_dii.utils.loss import safe_normalize

from stopword_chunk_pooler import StopwordAwareChunkPooler
from latent_decompressor_decoder import LatentDecompressorDecoder


def inspect_phrase(phrase: str, tokenizer, encoder, qwen_lm_head, pooler, decompressor, device):
    enc = tokenizer(phrase, return_tensors='pt').to(device)
    enc_ids = enc.input_ids
    attn_mask = enc.attention_mask

    L_orig = enc_ids.shape[1]
    rem = L_orig % 4
    if rem != 0:
        pad_len = 4 - rem
        enc_ids = F.pad(enc_ids, (0, pad_len), value=tokenizer.pad_token_id)
        attn_mask = F.pad(attn_mask, (0, pad_len), value=0)

    L = enc_ids.shape[1]
    N = L // 4

    tokens_list = [tokenizer.decode([tid]) for tid in enc_ids[0].tolist()]
    conn_flags = pooler.connector_mask[enc_ids[0].clamp(0, len(pooler.connector_mask) - 1)].tolist()

    with torch.no_grad():
        x_embeds = F.embedding(enc_ids, qwen_lm_head)
        z_orig, _, _ = encoder(x_embeds.float())
        z_orig = safe_normalize(z_orig.float(), dim=-1)

        # Сжатие с подавлением связок
        z_compressed, mask_N = pooler(
            z_orig, enc_ids, attention_mask=attn_mask,
            stopword_weight=0.05, noise_level=0.0, training=False
        )

        # Распаковка через чекпоинт декомпрессора
        h_decomp = decompressor(z_compressed, attention_mask_full=attn_mask, attention_mask_N=mask_N)
        logits = F.linear(h_decomp, qwen_lm_head)
        pred_ids = logits.argmax(dim=-1)[0]
        pred_text = tokenizer.decode(pred_ids[:L_orig], skip_special_tokens=True)

    print(f"\n{'='*75}")
    print(f"Исходный текст  : \"{phrase}\"")
    print(f"Длина токенов   : {L_orig} BPE -> сжато в {N} концептов (фактор 4x)")
    
    # Показываем, какие токены были отнесены к корням vs связкам в первых 3 чанках
    print("Первые чанки (сжатие 4 -> 1):")
    for ch_idx in range(min(4, N)):
        ch_toks = tokens_list[ch_idx*4 : (ch_idx+1)*4]
        ch_conn = conn_flags[ch_idx*4 : (ch_idx+1)*4]
        desc = [f"[{t}]" if not is_c else f"({t}*связка*)" for t, is_c in zip(ch_toks, ch_conn)]
        print(f"  Чанк {ch_idx+1}: {' + '.join(desc)}")

    print(f"Декомпрессор 2800: \"{pred_text}\"")
    print(f"{'='*75}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Inspecting Decompressor Checkpoint (Step 2800) on {device} ===\n")

    ckpt_path = os.path.join(PROJECT_ROOT, "experiments", "phase 2 plan C", "checkpoints", "decompressor_step_2800.pth")
    encoder_path = os.path.join(PROJECT_ROOT, "experiments", "phase 1", "planB_phase1_checkpoints_phase1_vae_step_20000.pth")
    qwen_id = "Qwen/Qwen2.5-1.5B"
    cache_dir = os.path.join(PROJECT_ROOT, "experiments", "phase 2 plan C")

    tokenizer = AutoTokenizer.from_pretrained(qwen_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("1. Initializing Pooler and Encoder...")
    pooler = StopwordAwareChunkPooler(tokenizer, cache_dir=cache_dir, chunk_size=4).to(device)
    encoder = LatentEncoder().to(device)
    if os.path.exists(encoder_path):
        ckpt_enc = torch.load(encoder_path, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt_enc.get("encoder", ckpt_enc))
    encoder.eval()

    print("2. Loading Qwen LM Head...")
    qwen_temp = AutoModelForCausalLM.from_pretrained(qwen_id, torch_dtype=torch.float32)
    qwen_lm_head = qwen_temp.lm_head.weight.detach().to(device)
    del qwen_temp

    print(f"3. Loading Decompressor from {ckpt_path}...")
    decompressor = LatentDecompressorDecoder(
        latent_dim=1024,
        qwen_dim=1536,
        num_context_layers=2,
        num_smoothing_layers=2,
        chunk_size=4
    ).to(device)

    ckpt_dec = torch.load(ckpt_path, map_location=device, weights_only=False)
    decompressor.load_state_dict(ckpt_dec["decoder"])
    decompressor.eval()
    print("   -> Checkpoint step:", ckpt_dec.get("step", "unknown"))

    # Тестируем на 3 разнородных примерах
    test_phrases = [
        # Валидационная фраза (исторический текст)
        "В 1861 году в Российской империи произошло эпохальное историческое событие: император Александр Второй подписал Высочайший манифест об отмене крепостного права.",
        
        # Короткий естественно-научный факт
        "Солнечный свет доходит до поверхности Земли примерно за восемь минут и двадцать секунд.",
        
        # Разговорная инструкция
        "Чтобы приготовить вкусный кофе, возьмите свежемолотые зерна и залейте их горячей водой."
    ]

    for p in test_phrases:
        inspect_phrase(p, tokenizer, encoder, qwen_lm_head, pooler, decompressor, device)


if __name__ == "__main__":
    main()
