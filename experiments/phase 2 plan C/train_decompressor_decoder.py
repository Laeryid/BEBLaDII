import os
import sys
import math
import random
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

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.beb_la_dii.model.vae import LatentEncoder
from src.beb_la_dii.utils.loss import safe_normalize

from stopword_chunk_pooler import StopwordAwareChunkPooler
from latent_decompressor_decoder import LatentDecompressorDecoder


def main():
    parser = argparse.ArgumentParser(description="Train Latent Decompressor Decoder (Plan C)")
    parser.add_argument("--data_path", type=str, default="BEBLaDII-planB-Phase6-Data/phase 6/data/train_phase6.parquet",
                        help="Path to training Parquet dataset")
    parser.add_argument("--qwen_path", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--encoder_path", type=str,
                        default="experiments/Plan B/phase 1/planB_phase1_checkpoints_phase1_vae_step_20000.pth")
    parser.add_argument("--dus_path", type=str,
                        default="experiments/phase 4/local_checkpoints/phase4_step_85995.pth")
    parser.add_argument("--phase2_path", type=str,
                        default="experiments/phase 2/planB_phase2_checkpoints_decoder_step_9000.pth")
    parser.add_argument("--resume_ckpt", type=str, default="",
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--output_dir", type=str, default="experiments/phase 2 plan C/checkpoints")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=256, help="Target text length (divisible by 4)")
    parser.add_argument("--lr", type=float, default=1.5e-4)  # По ADR 056 для проектора
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--log_steps", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Training Latent Decompressor Decoder (Plan C) on {device} ===")

    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = os.path.dirname(__file__)

    # 1. Токенизатор
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Модуль пулинга со стоп-словами
    print("Initializing StopwordAwareChunkPooler...")
    pooler = StopwordAwareChunkPooler(tokenizer, cache_dir=cache_dir, chunk_size=4).to(device)

    # 3. Замороженный VAE Encoder (строгая загрузка Phase 1 VAE)
    enc_path = os.path.join(PROJECT_ROOT, args.encoder_path) if not os.path.isabs(args.encoder_path) else args.encoder_path
    if not os.path.exists(enc_path):
        raise FileNotFoundError(f"[CRITICAL] Phase 1 LatentEncoder checkpoint not found at: {enc_path}")
    print(f"Loading LatentEncoder from {enc_path}...")
    encoder = LatentEncoder().to(device)
    ckpt_enc = torch.load(enc_path, map_location=device, weights_only=False)
    st_enc = ckpt_enc.get("encoder", ckpt_enc)
    encoder.load_state_dict(st_enc)
    print("   -> LatentEncoder weights loaded successfully.")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # 4. Веса Qwen Embeddings и LM Head (tied weights)
    print(f"Loading Qwen matrices from {args.qwen_path}...")
    qwen_temp = AutoModelForCausalLM.from_pretrained(args.qwen_path, torch_dtype=torch.float32)
    qwen_lm_head = qwen_temp.lm_head.weight.detach().to(device)
    del qwen_temp
    import gc; gc.collect()

    # 5. Декомпрессирующий декодер
    print("Initializing LatentDecompressorDecoder...")
    decoder = LatentDecompressorDecoder(
        latent_dim=1024,
        qwen_dim=1536,
        num_context_layers=2,
        num_smoothing_layers=2,
        chunk_size=4,
        dus_weights_path=args.dus_path if os.path.exists(args.dus_path) else None,
        phase2_weights_path=args.phase2_path if os.path.exists(args.phase2_path) else None
    ).to(device)

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    # 6. Загрузка данных
    import pandas as pd
    ds_full_path = os.path.join(PROJECT_ROOT, args.data_path) if not os.path.isabs(args.data_path) else args.data_path
    if not os.path.exists(ds_full_path):
        print(f"[ERROR] Dataset file not found at: {ds_full_path}")
        return

    print(f"Loading Parquet dataset from: {ds_full_path}")
    df = pd.read_parquet(ds_full_path)
    
    # Приоритетно берем колонку ответов 'A' (или 'answer'/'text')
    if "A" in df.columns:
        text_column = "A"
    elif "answer" in df.columns:
        text_column = "answer"
    elif "text" in df.columns:
        text_column = "text"
    else:
        text_column = df.columns[-1]

    raw_texts = df[text_column].dropna().tolist()
    print(f"Loaded {len(raw_texts)} text samples from column '{text_column}'")

    decoder.train()
    step = 0
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        print(f"Loading checkpoint to resume from: {args.resume_ckpt}...")
        ckpt = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
        st = ckpt.get("decoder", ckpt)
        decoder.load_state_dict(st)
        step = ckpt.get("step", 0)
        print(f"   -> Successfully resumed decoder weights at step {step}")

    accum_loss = 0.0
    optimizer.zero_grad()

    print(f"\nStarting training loop (Max steps: {args.max_steps}, Grad accum: {args.grad_accum})...\n")

    val_phrase = (
        "В 1861 году в Российской империи произошло эпохальное историческое событие: "
        "император Александр Второй подписал Высочайший манифест об отмене крепостного права."
    )

    while step < args.max_steps:
        # Случайный батч
        batch_samples = random.sample(raw_texts, min(args.batch_size, len(raw_texts)))
        encoded = tokenizer(
            batch_samples,
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt"
        ).to(device)

        input_ids = encoded.input_ids
        attn_mask = encoded.attention_mask

        # Гарантируем кратность 4
        T_curr = input_ids.shape[1]
        if T_curr % 4 != 0:
            pad_len = 4 - (T_curr % 4)
            input_ids = F.pad(input_ids, (0, pad_len), value=tokenizer.pad_token_id)
            attn_mask = F.pad(attn_mask, (0, pad_len), value=0)

        with torch.no_grad():
            x_emb = F.embedding(input_ids, qwen_lm_head)
            z_clean, _, _ = encoder(x_emb.float())
            z_clean = safe_normalize(z_clean.float(), dim=-1)
            # Сжатие со стоп-словами и учетом маски паддинга
            z_compressed, mask_N = pooler(
                z_clean, input_ids, attention_mask=attn_mask,
                stopword_weight=0.05, noise_level=0.80, training=True
            )

        h_decomp = decoder(z_compressed, attention_mask_full=attn_mask, attention_mask_N=mask_N)
        logits = F.linear(h_decomp, qwen_lm_head)

        # CrossEntropyLoss с маскированием PAD
        ce_raw = loss_fn(logits.view(-1, logits.shape[-1]), input_ids.view(-1))
        ce_loss = (ce_raw * attn_mask.view(-1).float()).sum() / attn_mask.view(-1).float().sum().clamp(min=1.0)

        loss_scaled = ce_loss / args.grad_accum
        loss_scaled.backward()
        accum_loss += ce_loss.item()

        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        step += 1

        if step % args.log_steps == 0:
            avg_loss = accum_loss / args.log_steps
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Step {step:05d} | CE Loss: {avg_loss:.4f}", flush=True)
            accum_loss = 0.0

        if step % args.save_steps == 0 or step == args.max_steps:
            # Валидационный декодинг
            decoder.eval()
            with torch.no_grad():
                val_enc = tokenizer(val_phrase, return_tensors='pt').to(device)
                val_ids = val_enc.input_ids
                val_mask = val_enc.attention_mask
                if val_ids.shape[1] % 4 != 0:
                    pad_len = 4 - (val_ids.shape[1] % 4)
                    val_ids = F.pad(val_ids, (0, pad_len), value=tokenizer.pad_token_id)
                    val_mask = F.pad(val_mask, (0, pad_len), value=0)

                val_x = F.embedding(val_ids, qwen_lm_head)
                val_z, _, _ = encoder(val_x.float())
                val_z_norm = safe_normalize(val_z.float(), dim=-1)
                val_z_comp, val_mask_N = pooler(val_z_norm, val_ids, attention_mask=val_mask, stopword_weight=0.05, training=False)
                
                val_h = decoder(val_z_comp, attention_mask_full=val_mask, attention_mask_N=val_mask_N)
                val_logits = F.linear(val_h, qwen_lm_head)
                val_pred_ids = val_logits.argmax(dim=-1)[0]
                val_pred_text = tokenizer.decode(val_pred_ids, skip_special_tokens=True)
                print(f"\n--- Validation at Step {step} ---", flush=True)
                print(f"Target: \"{val_phrase}\"", flush=True)
                print(f"Decomp: \"{val_pred_text}\"\n", flush=True)

            save_path = os.path.join(args.output_dir, f"decompressor_step_{step}.pth")
            torch.save({"decoder": decoder.state_dict(), "step": step}, save_path)
            print(f"Checkpoint saved to: {save_path}\n", flush=True)
            decoder.train()

    print("=== Training Complete ===", flush=True)


if __name__ == "__main__":
    main()
