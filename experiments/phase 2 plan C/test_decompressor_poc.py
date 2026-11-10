import os
import sys
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Plan C: Decompressor Decoder PoC Test on {device} ===\n")

    qwen_id = "Qwen/Qwen2.5-1.5B"
    encoder_path = os.path.join(PROJECT_ROOT, "experiments", "phase 1", "planB_phase1_checkpoints_phase1_vae_step_20000.pth")
    dus_path = os.path.join(PROJECT_ROOT, "experiments", "phase 4", "local_checkpoints", "phase4_step_85995.pth")
    phase2_path = os.path.join(PROJECT_ROOT, "experiments", "phase 2", "planB_phase2_checkpoints_decoder_step_9000.pth")
    cache_dir = os.path.join(PROJECT_ROOT, "experiments", "phase 2 plan C")

    tokenizer = AutoTokenizer.from_pretrained(qwen_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("1. Initializing StopwordAwareChunkPooler with Word-Boundary Guard...")
    pooler = StopwordAwareChunkPooler(tokenizer, cache_dir=cache_dir, chunk_size=4).to(device)
    n_conn = pooler.connector_mask.sum().item()
    print(f"   -> Connector mask ready: {n_conn} / {len(pooler.connector_mask)} tokens flagged as connectors/punct ({n_conn/len(pooler.connector_mask)*100:.2f}%)\n")

    print("2. Loading LatentEncoder (Phase 1 VAE) and Qwen Embeddings...")
    encoder = LatentEncoder().to(device)
    if os.path.exists(encoder_path):
        ckpt_enc = torch.load(encoder_path, map_location=device, weights_only=False)
        st_enc = ckpt_enc.get("encoder", ckpt_enc)
        encoder.load_state_dict(st_enc)
        print(f"   -> Loaded Phase 1 VAE Encoder from {encoder_path}")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Загружаем веса Qwen
    print("   -> Extracting Qwen LM Head & Embedding weights...")
    qwen_temp = AutoModelForCausalLM.from_pretrained(qwen_id, torch_dtype=torch.float32)
    qwen_lm_head = qwen_temp.lm_head.weight.detach().to(device)
    del qwen_temp
    import gc; gc.collect()

    print("\n3. Initializing LatentDecompressorDecoder (N -> 4N)...")
    decompressor = LatentDecompressorDecoder(
        latent_dim=1024,
        qwen_dim=1536,
        num_context_layers=2,
        num_smoothing_layers=2,
        chunk_size=4,
        dus_weights_path=dus_path,
        phase2_weights_path=phase2_path
    ).to(device)

    # 4. Тестовый пример фразы
    phrase = (
        "В 1861 году в Российской империи произошло эпохальное историческое событие: "
        "император Александр Второй подписал Высочайший манифест об отмене крепостного права, "
        "освободивший крестьян от вековой помещичьей зависимости и даровавший им гражданские свободы."
    )
    enc = tokenizer(phrase, return_tensors='pt').to(device)
    enc_ids = enc.input_ids
    attn_mask = enc.attention_mask

    L = enc_ids.shape[1]
    rem = L % 4
    if rem != 0:
        pad_len = 4 - rem
        enc_ids = F.pad(enc_ids, (0, pad_len), value=tokenizer.pad_token_id)
        attn_mask = F.pad(attn_mask, (0, pad_len), value=0)
        L = enc_ids.shape[1]

    N = L // 4
    print(f"\nTarget Text: \"{phrase}\"")
    print(f"Length: {L} tokens -> Compressed to {N} concept tokens\n")

    # Анализ распознанных связок
    tokens_list = [tokenizer.decode([tid]) for tid in enc_ids[0].tolist()]
    conn_flags = pooler.connector_mask[enc_ids[0].clamp(0, len(pooler.connector_mask)-1)].tolist()
    flagged_words = [tokens_list[i] for i, f in enumerate(conn_flags) if f]
    print(f"Detected connectors/punct in phrase ({len(flagged_words)}/{L}):")
    print(f"   {flagged_words[:16]}... (total {len(flagged_words)})\n")

    # Получаем идеальные Z_orig через VAE Encoder
    with torch.no_grad():
        x_embeds = F.embedding(enc_ids, qwen_lm_head)
        z_orig, _, _ = encoder(x_embeds.float())
        z_orig = safe_normalize(z_orig.float(), dim=-1)

    print("5. Running StopwordAware Chunk Pooling (K=4 -> 1)...")
    z_compressed, mask_N = pooler(
        z_orig, enc_ids, attention_mask=attn_mask,
        stopword_weight=0.05, noise_level=0.80, training=True
    )
    print(f"   -> Original latent shape    : {z_orig.shape}")
    print(f"   -> Compressed latent shape  : {z_compressed.shape} (N={N} concepts)")
    print(f"   -> Compressed mask_N shape  : {mask_N.shape}")

    # 6. Микро-проверка Overfit (25 итераций AdamW), чтобы доказать способность распаковки
    print("\n6. Running Micro-Overfit Check (25 optimization steps on 1 sentence)...")
    optimizer = torch.optim.AdamW(decompressor.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    decompressor.train()
    for step in range(26):
        optimizer.zero_grad()
        # Прямой проход: z_compressed [1, N, 1024] -> hidden [1, 4N, 1536]
        h_decomp = decompressor(z_compressed, attention_mask_full=attn_mask, attention_mask_N=mask_N)
        logits = F.linear(h_decomp, qwen_lm_head)  # [1, L, vocab_size]

        ce_raw = loss_fn(logits.view(-1, logits.shape[-1]), enc_ids.view(-1))
        loss = (ce_raw * attn_mask.view(-1).float()).sum() / attn_mask.view(-1).float().sum().clamp(min=1.0)
        
        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            pred_ids = logits.argmax(dim=-1)[0]
            pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
            print(f"   Step {step:02d} | CE Loss: {loss.item():.4f} | Decoded: \"{pred_text[:85]}...\"")

    print("\n=== PoC Test Completed Successfully ===")


if __name__ == "__main__":
    main()
