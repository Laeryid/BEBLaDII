"""
prepare_phase3_data.py — Подготовка датасета для Phase 3 (OutputProjector).

Читает локальные Parquet файлы через IndexedParquetDataset (быстро, офлайн).
Берёт выборки из Magpie, OpenThoughts и CulturaX.
Токенизирует, нарезает на чанки до max-length токенов,
добавляет якорь <|thought|> и сохраняет в Parquet.

Использование:
    .venv\\Scripts\\python.exe "experiments/phase 3/prepare_phase3_data.py" ^
        --magpie-dir   data/magpie_reasoning ^
        --ot-dir       data/open_thoughts ^
        --culturax-dir data/CulturaX/data

Порядок:
  1. Убедиться, что датасеты скачаны (scripts/download_datasets.py).
  2. Запустить этот скрипт.
  3. Загрузить результат на GCS:
       gsutil -m cp -r storage/experiments/phase3/data "gs://bebladii-datasets-us/phase 3/train_data/"
"""

import argparse
import os
import random
import sys

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer

# Поддержка двух вариантов имени пакета
try:
    from indexed_parquet import IndexedParquetDataset
except ImportError:
    from indexed_parquet_dataset import IndexedParquetDataset


# ---------------------------------------------------------------------------
# Аргументы
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir", type=str, default="storage/experiments/phase3/data"
    )
    parser.add_argument(
        "--tokenizer-id", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Итоговая длина последовательности вместе с якорём.",
    )
    parser.add_argument("--anchor", type=str, default="<|thought|>")

    # Пути к локальным датасетам
    parser.add_argument("--magpie-dir", type=str, default="data/magpie_reasoning")
    parser.add_argument("--ot-dir", type=str, default="data/open_thoughts")
    parser.add_argument("--culturax-dir", type=str, default="data/CulturaX/data")

    # Объёмы (всего ~150k чанков)
    parser.add_argument("--n-magpie", type=int, default=50000)
    parser.add_argument("--n-openthoughts", type=int, default=50000)
    parser.add_argument("--n-culturax-ru", type=int, default=25000)
    parser.add_argument("--n-culturax-cs", type=int, default=25000)

    # Фильтр: минимальная длина текста в токенах для Magpie/OT
    parser.add_argument(
        "--min-text-tokens",
        type=int,
        default=1000,
        help="Минимальная длина ответа в токенах для Magpie/OT (чтобы нарезка была осмысленной).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Сохранение чанков в Parquet
# ---------------------------------------------------------------------------


def save_chunks(
    chunks: list[list[int]],
    out_path: str,
    anchor_ids: list[int],
    max_length: int,
    desc: str,
):
    """Добавляет якорь, сохраняет в Parquet."""
    print(f"\nСохранение {desc}: {len(chunks)} чанков → {out_path}")
    max_text_len = max_length - len(anchor_ids)
    rows = []
    for token_ids in tqdm(chunks, desc="Форматирование"):
        token_ids = token_ids[:max_text_len]
        if not token_ids:
            continue
        rows.append({"input_ids": anchor_ids + token_ids})

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path)
    print(f"  Записано: {len(rows)} примеров.")


# ---------------------------------------------------------------------------
# Magpie: поле "response"
# ---------------------------------------------------------------------------


def collect_magpie_chunks(
    ds_dir: str, tokenizer, n: int, chunk_size: int, min_tokens: int
) -> list[list[int]]:
    """
    Читает Magpie локально через IndexedParquetDataset.
    Берёт поле 'response', токенизирует, нарезает на чанки по chunk_size.
    Пропускает короткие тексты (<min_tokens токенов).
    """
    print(f"\nMagpie: загрузка из {ds_dir}...")
    ds = IndexedParquetDataset.from_folder(ds_dir)
    print(f"  Всего строк: {len(ds)}")

    chunks = []
    pbar = tqdm(total=n, desc="Magpie chunks")

    for item in ds:
        if len(chunks) >= n:
            break
        text = item.get("response", "") or ""
        if not text:
            continue

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) < min_tokens:
            continue

        # Нарезаем на непересекающиеся чанки
        for start in range(0, len(token_ids) - chunk_size + 1, chunk_size):
            if len(chunks) >= n:
                break
            chunks.append(token_ids[start : start + chunk_size])
            pbar.update(1)

    pbar.close()
    print(f"  Собрано чанков: {len(chunks)}")
    return chunks


# ---------------------------------------------------------------------------
# OpenThoughts: поле "conversations[].value" (role=assistant/gpt)
# ---------------------------------------------------------------------------


def collect_ot_chunks(
    ds_dir: str, tokenizer, n: int, chunk_size: int, min_tokens: int
) -> list[list[int]]:
    """
    Читает OpenThoughts локально через IndexedParquetDataset.
    Берёт ответ ассистента из 'conversations', токенизирует, нарезает на чанки.
    """
    print(f"\nOpenThoughts: загрузка из {ds_dir}...")
    ds = IndexedParquetDataset.from_folder(ds_dir)
    print(f"  Всего строк: {len(ds)}")

    chunks = []
    pbar = tqdm(total=n, desc="OpenThoughts chunks")

    for item in ds:
        if len(chunks) >= n:
            break
        convs = item.get("conversations", []) or []
        text = ""
        for msg in convs:
            role = msg.get("from", "") or msg.get("role", "")
            if role in ("gpt", "assistant"):
                text = msg.get("value", "") or msg.get("content", "")
                break
        if not text:
            continue

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) < min_tokens:
            continue

        for start in range(0, len(token_ids) - chunk_size + 1, chunk_size):
            if len(chunks) >= n:
                break
            chunks.append(token_ids[start : start + chunk_size])
            pbar.update(1)

    pbar.close()
    print(f"  Собрано чанков: {len(chunks)}")
    return chunks


# ---------------------------------------------------------------------------
# CulturaX: поле "text", случайная выборка целых текстов
# ---------------------------------------------------------------------------


def collect_culturax_chunks(file_path: str, tokenizer, n: int,
                             chunk_size: int) -> list[list[int]]:
    """
    Читает CulturaX батчами (iter_batches) для максимальной скорости диска,
    так как поштучное чтение огромных файлов вызывает I/O bottleneck.
    """
    if not os.path.exists(file_path):
        print(f"  WARN: {file_path} не найден. Пропускаем.")
        return []

    print(f"\nCulturaX: загрузка из {file_path}...")
    
    parquet_file = pq.ParquetFile(file_path)
    total = parquet_file.metadata.num_rows
    print(f"  Всего строк: {total}")

    # Сохраняем индексы в set для быстрого поиска
    sample_n = min(n * 3, total)
    indices = set(random.sample(range(total), sample_n))

    max_chars = chunk_size * 8

    chunks = []
    pbar = tqdm(total=n, desc=f"CulturaX {os.path.basename(file_path)[:2].upper()} chunks")

    current_idx = 0
    # Читаем сразу по 10 000 строк за одно обращение к диску
    for batch in parquet_file.iter_batches(batch_size=10000, columns=["text"]):
        if len(chunks) >= n:
            break
            
        text_array = batch.column(0)
        
        for i in range(len(text_array)):
            if len(chunks) >= n:
                break
                
            if current_idx in indices:
                text = text_array[i].as_py()
                if text:
                    text = text[:max_chars]
                    token_ids = tokenizer.encode(text, add_special_tokens=False)
                    
                    if len(token_ids) >= chunk_size // 2:
                        chunks.append(token_ids[:chunk_size])
                        pbar.update(1)
                        
            current_idx += 1

    pbar.close()
    print(f"  Собрано чанков: {len(chunks)}")
    return chunks


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("Phase 3: prepare_phase3_data")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id, trust_remote_code=True)
    anchor_ids = tokenizer.encode(args.anchor, add_special_tokens=False)
    chunk_size = args.max_length - len(anchor_ids)

    print(f"  Якорь '{args.anchor}' → ids={anchor_ids} ({len(anchor_ids)} ток.)")
    print(f"  chunk_size = {chunk_size} токенов (max_length={args.max_length})")

    # --- CulturaX RU ---
    if args.n_culturax_ru > 0:
        ru_file = os.path.join(args.culturax_dir, "ru_part_00002.parquet")
        ru_chunks = collect_culturax_chunks(
            ru_file, tokenizer, args.n_culturax_ru, chunk_size
        )
        if ru_chunks:
            save_chunks(
                ru_chunks,
                os.path.join(args.out_dir, "culturax_ru.parquet"),
                anchor_ids,
                args.max_length,
                "CulturaX RU",
            )

    # --- CulturaX CS ---
    if args.n_culturax_cs > 0:
        cs_file = os.path.join(args.culturax_dir, "cs_part_00002.parquet")
        cs_chunks = collect_culturax_chunks(
            cs_file, tokenizer, args.n_culturax_cs, chunk_size
        )
        if cs_chunks:
            save_chunks(
                cs_chunks,
                os.path.join(args.out_dir, "culturax_cs.parquet"),
                anchor_ids,
                args.max_length,
                "CulturaX CS",
            )

    # --- Magpie ---
    if args.n_magpie > 0:
        magpie_chunks = collect_magpie_chunks(
            args.magpie_dir, tokenizer, args.n_magpie, chunk_size, args.min_text_tokens
        )
        if magpie_chunks:
            save_chunks(
                magpie_chunks,
                os.path.join(args.out_dir, "magpie.parquet"),
                anchor_ids,
                args.max_length,
                "Magpie",
            )

    # --- OpenThoughts ---
    if args.n_openthoughts > 0:
        ot_chunks = collect_ot_chunks(
            args.ot_dir,
            tokenizer,
            args.n_openthoughts,
            chunk_size,
            args.min_text_tokens,
        )
        if ot_chunks:
            save_chunks(
                ot_chunks,
                os.path.join(args.out_dir, "openthoughts.parquet"),
                anchor_ids,
                args.max_length,
                "OpenThoughts",
            )

    print("\n" + "=" * 60)
    print("Готово!")
    print(f"Файлы в: {os.path.abspath(args.out_dir)}")
    print("\nЗагрузка на GCS:")
    print(
        f'  gsutil -m cp -r {args.out_dir} "gs://bebladii-datasets-us/phase 3/train_data/"'
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
