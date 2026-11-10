"""
Скрипт Truncated SVD матрицы эмбеддингов Mistral-Nemo-Base-2407 (5120 -> 1024)
и генерации маски стоп-слов Tekken для Plan C Phase 1.

Результаты сохраняются в:
- C:\\Experiments\\BEBLaDII\\data\\plan C\\pretrained_embeddings_1024.pt
- C:\\Experiments\\BEBLaDII\\data\\plan C\\tekken_stopwords_mask.pt
"""

import os
import sys
import re
import time
import json
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tokenizers import Tokenizer

# Вывод UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

OUTPUT_DIR = "C:/Experiments/BEBLaDII/data/plan C"
EMBEDDINGS_OUT = os.path.join(OUTPUT_DIR, "pretrained_embeddings_1024.pt")
STOPWORDS_OUT = os.path.join(OUTPUT_DIR, "tekken_stopwords_mask.pt")

BASE_CONNECTORS = {
    # Предлоги русские
    "в", "во", "на", "с", "со", "к", "ко", "по", "из", "изо", "у", "о", "об", "обо",
    "от", "ото", "до", "за", "над", "под", "подо", "пред", "предо", "при", "про",
    "без", "безо", "для", "через", "сквозь", "между", "меж", "среди", "около", "вокруг",
    "после", "кроме", "вместо", "ради", "насчет", "вслед", "навстречу",
    # Союзы русские
    "и", "да", "а", "но", "или", "либо", "что", "чтобы", "чтоб", "как", "когда", "если",
    "то", "хотя", "пока", "будто", "словно", "точно", "потому", "поэтому", "также", "тоже",
    # Частицы русские
    "же", "ж", "бы", "б", "ли", "ль", "не", "ни", "вот", "ведь", "даже", "лишь",
    "только", "уже", "еще", "ещё", "так", "разве", "неужели",
    # Местоимения-связки и вспомогательные глаголы
    "он", "она", "оно", "они", "его", "ее", "её", "их", "ему", "ей", "им", "нем", "ней",
    "мы", "вы", "я", "ты", "меня", "тебя", "нас", "вас", "мне", "тебе", "нам", "вам",
    "свой", "своя", "свое", "своё", "свои", "себе", "себя", "собой", "собою",
    "этот", "эта", "это", "эти", "этого", "этой", "этих", "этому", "этим",
    "тот", "та", "то", "те", "того", "той", "тех", "тому", "тем",
    "весь", "вся", "все", "всё", "всего", "всей", "всех", "всему", "всеми",
    "который", "которая", "которое", "которые", "которого", "которой", "которых", "которому",
    "быть", "был", "была", "было", "были", "есть", "будет", "будут", "буду", "будем",
    # Базовые английские stopwords
    "the", "a", "an", "in", "on", "at", "to", "for", "with", "from", "of", "by", "as",
    "and", "but", "or", "so", "if", "than", "that", "this", "these", "those",
    "it", "its", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had"
}


def download_and_extract_embeddings() -> torch.Tensor:
    print("=== 1. Скачивание шарда с эмбеддингами Mistral-Nemo ===", flush=True)
    repo_id = "mistralai/Mistral-Nemo-Base-2407"
    shard_filename = "model-00001-of-00005.safetensors"

    print(f"Загрузка {shard_filename} из Hugging Face (только шард ~4.9 ГБ)...", flush=True)
    shard_path = hf_hub_download(
        repo_id=repo_id,
        filename=shard_filename,
    )
    print(f"Шард доступен по пути: {shard_path}", flush=True)

    print("Чтение тензора 'model.embed_tokens.weight'...", flush=True)
    state = load_file(shard_path, device="cpu")
    if "model.embed_tokens.weight" not in state:
        raise KeyError(f"'model.embed_tokens.weight' не найден в {shard_path}!")

    embed_tokens = state["model.embed_tokens.weight"].float()
    print(f"Извлечена матрица эмбеддингов: форма {embed_tokens.shape}, тип {embed_tokens.dtype}", flush=True)
    del state
    return embed_tokens


def compute_truncated_svd(E: torch.Tensor, target_dim: int = 1024) -> torch.Tensor:
    """
    Быстрый и численный Truncated SVD через матрицу Грама признаков G = E^T E (5120 x 5120):
    E in R^{131072 x 5120}
    G = E^T E in R^{5120 x 5120}
    G = V Lambda V^T
    E_{1024} = E @ V_{1024} in R^{131072 x 1024}
    """
    print(f"=== 2. Вычисление Truncated SVD (5120 -> {target_dim}) ===", flush=True)
    start_t = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство для SVD: {device}", flush=True)

    E_dev = E.to(device)
    print("Вычисление матрицы Грама G = E^T E (5120 x 5120)...", flush=True)
    G = torch.matmul(E_dev.t(), E_dev)  # (5120, 5120)
    G = 0.5 * (G + G.t())  # Симметризация

    print("Спектральное разложение eigh(G)...", flush=True)
    eigvals, eigvecs = torch.linalg.eigh(G)  # eigvals: возрастающие (5120,), eigvecs: (5120, 5120)

    # Топ target_dim собственных векторов (последние target_dim столбцов)
    V_top = eigvecs[:, -target_dim:]  # (5120, 1024)
    # Переворачиваем по убыванию собственных чисел
    V_top = torch.flip(V_top, dims=[-1])
    eigvals_top = torch.flip(eigvals[-target_dim:], dims=[-1])

    explained_variance_ratio = eigvals_top.sum() / eigvals.sum()
    print(f"Доля сохраненной дисперсии в топ-{target_dim}: {explained_variance_ratio.item():.4%}", flush=True)

    print(f"Проекция матрицы эмбеддингов E_1024 = E @ V_top ...", flush=True)
    E_proj = torch.matmul(E_dev, V_top)  # (131072, 1024)

    # Сохраняем в float16/bfloat16 для компактности (~268 МБ)
    E_out = E_proj.cpu().half()
    elapsed = time.time() - start_t
    print(f"SVD успешно вычислен за {elapsed:.2f} сек. Форма: {E_out.shape}", flush=True)
    return E_out


def build_tekken_stopwords_mask(vocab_size: int = 131072) -> torch.Tensor:
    print("=== 3. Построение булевой маски стоп-слов Tekken ===", flush=True)
    mask = torch.zeros(vocab_size, dtype=torch.bool)

    # 1. Загрузка токенизатора Tekken из кэша
    repo_id = "mistralai/Mistral-Nemo-Base-2407"
    tekken_file = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")
    print(f"Загрузка токенизатора из {tekken_file} ...", flush=True)
    tok = Tokenizer.from_file(tekken_file)

    punct_pattern = re.compile(r"^[\s\W_]+$")
    count_stop = 0

    for token_id in range(vocab_size):
        try:
            text = tok.id_to_token(token_id)
            if text is None:
                continue
            decoded = tok.decode([token_id], skip_special_tokens=False)
        except Exception:
            continue

        # Спецтокены
        if (text.startswith("<") and text.endswith(">")) or (text.startswith("[") and text.endswith("]")):
            mask[token_id] = True
            count_stop += 1
            continue

        # Пунктуация, пробелы, переносы строк
        if punct_pattern.match(decoded) or punct_pattern.match(text):
            mask[token_id] = True
            count_stop += 1
            continue

        clean_text = decoded.strip().lower()
        if clean_text in BASE_CONNECTORS:
            # Проверяем границу слова (начало слова)
            if decoded.startswith(" ") or text.startswith(" ") or text.startswith("Ġ"):
                mask[token_id] = True
                count_stop += 1

    print(f"Найдено стоп-слов/служебных токенов: {count_stop} из {vocab_size} ({count_stop/vocab_size:.2%})", flush=True)
    return mask


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Загрузка и SVD эмбеддингов
    if os.path.exists(EMBEDDINGS_OUT):
        print(f"[Exist] Файл эмбеддингов уже существует: {EMBEDDINGS_OUT}", flush=True)
    else:
        embed_tokens = download_and_extract_embeddings()
        E_1024 = compute_truncated_svd(embed_tokens, target_dim=1024)
        torch.save({"weight": E_1024}, EMBEDDINGS_OUT)
        print(f"[Saved] SVD-эмбеддинги сохранены в: {EMBEDDINGS_OUT} ({os.path.getsize(EMBEDDINGS_OUT)/1024/1024:.1f} МБ)", flush=True)

    # 2. Построение маски стоп-слов Tekken
    if os.path.exists(STOPWORDS_OUT):
        print(f"[Exist] Маска стоп-слов уже существует: {STOPWORDS_OUT}", flush=True)
    else:
        mask = build_tekken_stopwords_mask(vocab_size=131072)
        torch.save({"mask": mask}, STOPWORDS_OUT)
        print(f"[Saved] Маска стоп-слов сохранена в: {STOPWORDS_OUT}", flush=True)

    print("\n=== Все артефакты Plan C успешно подготовлены ===", flush=True)


if __name__ == "__main__":
    main()
