import os
import sys
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = "C:/Experiments/BEBLaDII"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments", "phase 4"))

from evaluate_phase4_checkpoints import BEBLaDIIPhase4aEval
from transformers import AutoTokenizer

def safe_normalize(x, dim=-1, eps=1e-6):
    return F.normalize(x, p=2, dim=dim, eps=eps)


def load_model(device):
    embed_model_id = "Qwen/Qwen2.5-1.5B"
    modernbert_id  = "answerdotai/ModernBERT-large"
    phase4_ckpt    = os.path.join(PROJECT_ROOT, "experiments", "phase 4", "local_checkpoints", "phase4_step_85995.pth")
    vae_ckpt       = os.path.join(PROJECT_ROOT, "experiments", "phase 1", "planB_phase1_checkpoints_phase1_vae_step_20000.pth")

    print("Initializing BEBLaDIIPhase4aEval...")
    model = BEBLaDIIPhase4aEval(embedding_model_path=embed_model_id, modernbert_path=modernbert_id)

    # VAE Encoder
    print(f"Loading VAE Encoder...")
    vae_st = torch.load(vae_ckpt, map_location="cpu", weights_only=False)
    if 'encoder' in vae_st:
        model.encoder.load_state_dict(vae_st['encoder'], strict=False)

    # Phase 4 DUS
    print(f"Loading Phase 4 DUS...")
    p4_st = torch.load(phase4_ckpt, map_location="cpu", weights_only=False)
    dus_ema = p4_st.get("dus_ema", p4_st.get("dus", {}))
    clean_dus = {k.replace("_orig_module.", ""): v for k, v in dus_ema.items()}
    model.dus.load_state_dict(clean_dus, strict=False)

    model.to(device)
    model.eval()
    return model


def build_dot_sequences(token_ids, dot_id, device):
    """
    Создает последовательности вида [w0, '.', w1, '.', w2, '.', ...]
    token_ids: [B, S] — S слов на последовательность
    Возвращает: input_ids [B, 2S], attn_mask [B, 2S]
    """
    B, S = token_ids.shape
    T = S * 2
    dots = torch.full((B, S), dot_id, dtype=torch.long, device=device)
    seqs = torch.empty((B, T), dtype=torch.long, device=device)
    seqs[:, 0::2] = token_ids
    seqs[:, 1::2] = dots
    mask = torch.ones((B, T), dtype=torch.long, device=device)
    return seqs, mask


def run_dus(model, input_ids, attn_mask, t_global, z_noisy_override=None):
    """
    Запускает модель и возвращает h39, z_noisy, z_clean.
    Identity Gate применяется снаружи (для инференса).
    """
    t = t_global
    if t.dim() == 0:
        t = t.unsqueeze(0).expand(input_ids.shape[0])
    out = model(input_ids, attn_mask, t_global=t, z_noisy_override=z_noisy_override)
    return out  # содержит "h_39", "z_noisy", "z_clean"


def identity_gate(h39, z_current, t_scalar):
    """Identity Gate (инференс): gate = sin(t * pi/2)"""
    gate = math.sin(t_scalar * math.pi / 2)
    return safe_normalize(gate * h39 + (1.0 - gate) * z_current, dim=-1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_path = os.path.join(PROJECT_ROOT, "experiments", "phase 5", "local", "unified_dataset.pt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Токенизатор
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    vocab_size = len(tokenizer)
    dot_id = tokenizer.encode(".")[-1]  # последний токен "." (без BOS)
    print(f"Vocab size: {vocab_size}, dot token id: {dot_id}")

    model = load_model(device)

    # Параметры генерации
    words_per_seq = 32      # 32 слова + 32 точки = 64 токена
    batch_words   = 16 * words_per_seq  # 16 последовательностей на батч

    # Накопители
    all_h39        = []
    all_dus_final  = []
    all_t_reported = []
    all_labels     = []

    print("Generating synthetic dataset (bag-of-words through DUS)...")

    with torch.no_grad():
        for offset in tqdm(range(0, vocab_size, batch_words)):
            end = min(offset + batch_words, vocab_size)
            n_words = end - offset
            if n_words < words_per_seq:
                break  # неполный последний кусок пропускаем

            actual_bs = n_words // words_per_seq
            word_ids = torch.arange(offset, offset + actual_bs * words_per_seq, device=device)
            word_ids = word_ids.view(actual_bs, words_per_seq)

            input_ids, attn_mask = build_dot_sequences(word_ids, dot_id, device)
            T = input_ids.shape[1]  # 64

            # --- Сценарий 1: Чистые якоря (t = 0.0) ---
            t0_val = 0.0
            t0 = torch.full((actual_bs,), t0_val, device=device)
            out0 = run_dus(model, input_ids, attn_mask, t0)
            h39_0     = out0["h_39"]                     # [B, T, 1024]
            z_clean_0 = out0.get("z_clean")              # [B, T, 1024]
            # При t=0 нет шума, z_current = z_clean
            dus_final_0 = identity_gate(h39_0, z_clean_0, t0_val)

            all_h39.append(h39_0.cpu())
            all_dus_final.append(dus_final_0.cpu())
            all_t_reported.append(torch.full((actual_bs, T, 1), t0_val))
            # Метки: слова разделены точками — нет грамматической связи
            # dict_proximity=1 (слова из словаря), confidence=1, coherence=0 (нет связи),
            # complexity=0 (однозначные слова), crystallization=1 (готово, t=0)
            labels0 = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0]).view(1, 1, 5).expand(actual_bs, T, 5)
            all_labels.append(labels0)

            # --- Сценарий 2: Чистый шум (t = 1.0) ---
            t1_val = 1.0
            t1 = torch.full((actual_bs,), t1_val, device=device)
            out1 = run_dus(model, input_ids, attn_mask, t1)
            h39_1     = out1["h_39"]
            z_noisy_1 = out1["z_noisy"]
            dus_final_1 = identity_gate(h39_1, z_noisy_1, t1_val)

            all_h39.append(h39_1.cpu())
            all_dus_final.append(dus_final_1.cpu())
            all_t_reported.append(torch.full((actual_bs, T, 1), t1_val))
            # Метки: чистый шум — ничего не готово, ничему не верить
            labels1 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]).view(1, 1, 5).expand(actual_bs, T, 5)
            all_labels.append(labels1)

            # --- Сценарий 3: Развилка / интерполяция (t = 0.0, но смешанный z_noisy) ---
            perm = torch.randperm(actual_bs, device=device)
            z_clean_src = out0.get("z_clean")  # берем из сценария 1
            z_interp = safe_normalize((z_clean_src + z_clean_src[perm]) / 2.0, dim=-1)

            out2 = run_dus(model, input_ids, attn_mask, t0, z_noisy_override=z_interp)
            h39_2     = out2["h_39"]
            dus_final_2 = identity_gate(h39_2, z_interp, t0_val)

            all_h39.append(h39_2.cpu())
            all_dus_final.append(dus_final_2.cpu())
            all_t_reported.append(torch.full((actual_bs, T, 1), t0_val))
            # Метки: развилка — высокая сложность, низкая кристаллизация
            labels2 = torch.tensor([0.5, 0.5, 0.0, 1.0, 0.0]).view(1, 1, 5).expand(actual_bs, T, 5)
            all_labels.append(labels2)

    print("Concatenating...")
    dataset = {
        "h39":        torch.cat(all_h39,        dim=0),
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
