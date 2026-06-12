"""
prepare_phase3_data.py — Подготовка датасета для Phase 3 (OutputProjector).
Берёт свежие выборки из Magpie, OpenThoughts и локальных файлов CulturaX.
Токенизирует, обрезает до 512 токенов, добавляет якорь <|thought|> и сохраняет в Parquet.
"""

import os
import argparse
import random
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="storage/experiments/phase3/data")
    parser.add_argument("--tokenizer-id", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--anchor", type=str, default="<|thought|>")
    parser.add_argument("--culturax-dir", type=str, default="data", help="Папка с локальными файлами CulturaX")
    
    # Объёмы для каждой части (всего ~150k)
    parser.add_argument("--n-magpie", type=int, default=50000)
    parser.add_argument("--n-openthoughts", type=int, default=50000)
    parser.add_argument("--n-culturax-ru", type=int, default=25000)
    parser.add_argument("--n-culturax-cs", type=int, default=25000)
    return parser.parse_args()

def process_and_save_tokens(token_lists, out_path, anchor_ids, max_length, desc):
    """Принимает уже готовые списки токенов, добавляет якорь и сохраняет в parquet."""
    print(f"\nСохранение {desc} ({len(token_lists)} примеров)...")
    
    processed = []
    max_text_len = max_length - len(anchor_ids)
    
    for token_ids in tqdm(token_lists, desc="Форматирование"):
        # Обрезаем (на всякий случай, хотя куски уже должны быть правильной длины)
        token_ids = token_ids[:max_text_len]
        if not token_ids:
            continue
            
        final_ids = anchor_ids + token_ids
        processed.append({"input_ids": final_ids})
        
    print(f"Запись в {out_path}...")
    table = pa.Table.from_pylist(processed)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pq.write_table(table, out_path)
    print(f"Готово: {len(processed)} примеров.")

def process_and_save_texts(texts, out_path, tokenizer, anchor_ids, max_length, desc):
    """Токенизирует сырые тексты и сохраняет (используется для CulturaX)."""
    print(f"\nОбработка {desc} ({len(texts)} примеров)...")
    token_lists = []
    for text in tqdm(texts, desc="Токенизация"):
        token_lists.append(tokenizer.encode(text, add_special_tokens=False))
    process_and_save_tokens(token_lists, out_path, anchor_ids, max_length, desc)

def get_culturax_texts(file_path, count):
    """Считывает случайные тексты из локального Parquet файла CulturaX."""
    if not os.path.exists(file_path):
        print(f"ВНИМАНИЕ: Файл {file_path} не найден! Пропускаем.")
        return []
        
    print(f"Чтение {file_path}...")
    # Читаем только колонку text для экономии памяти
    table = pq.read_table(file_path, columns=["text"])
    total_rows = table.num_rows
    
    if total_rows == 0:
        return []
        
    # Выбираем случайные индексы
    indices = random.sample(range(total_rows), min(count, total_rows))
    
    texts = []
    # Извлечение данных (медленно, если делать по одному, поэтому преобразуем нужные)
    # PyArrow ChunkedArray to python list
    text_column = table.column("text")
    for idx in tqdm(indices, desc="Сэмплирование CulturaX"):
        texts.append(text_column[idx].as_py())
        
    return texts

def extract_chunks_from_long_text(text, tokenizer, chunk_size, min_length=8000):
    """
    Токенизирует текст. Если длина > min_length, вырезает 3 куска:
    начало, середина, конец.
    Возвращает список списков токенов (от 0 до 3 элементов).
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < min_length:
        return []
        
    # Берем 3 куска
    chunk1 = tokens[0 : chunk_size]
    
    mid_idx = len(tokens) // 2 - chunk_size // 2
    chunk2 = tokens[mid_idx : mid_idx + chunk_size]
    
    end_idx = len(tokens) - chunk_size
    chunk3 = tokens[end_idx : len(tokens)]
    
    return [chunk1, chunk2, chunk3]

def get_magpie_chunks(count, tokenizer, chunk_size):
    """Ищет длинные рассуждения в Magpie и нарезает их на куски."""
    print("Загрузка Magpie-Reasoning-V2 (поиск длинных >8k токенов)...")
    ds = load_dataset("Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B", split="train", streaming=True)
    
    chunks = []
    iterator = iter(ds)
    docs_processed = 0
    
    pbar = tqdm(total=count, desc="Сбор кусков Magpie")
    while len(chunks) < count:
        try:
            item = next(iterator)
            docs_processed += 1
            response = item.get("response", "")
            
            new_chunks = extract_chunks_from_long_text(response, tokenizer, chunk_size)
            if new_chunks:
                # Берём столько, сколько не хватает до count
                needed = count - len(chunks)
                chunks.extend(new_chunks[:needed])
                pbar.update(len(new_chunks[:needed]))
                
        except StopIteration:
            print("Достигнут конец датасета!")
            break
            
    pbar.close()
    print(f"Просмотрено документов: {docs_processed}. Собрано кусков: {len(chunks)}.")
    return chunks

def get_openthoughts_chunks(count, tokenizer, chunk_size):
    """Ищет длинные рассуждения в OpenThoughts и нарезает их на куски."""
    print("Загрузка OpenThoughts-114k (поиск длинных >8k токенов)...")
    ds = load_dataset("open-thoughts/OpenThoughts-114k", split="train", streaming=True)
    
    chunks = []
    iterator = iter(ds)
    docs_processed = 0
    
    pbar = tqdm(total=count, desc="Сбор кусков OpenThoughts")
    while len(chunks) < count:
        try:
            item = next(iterator)
            docs_processed += 1
            
            convs = item.get("conversations", [])
            for msg in convs:
                if msg.get("from") == "gpt" or msg.get("role") == "assistant":
                    val = msg.get("value", "")
                    
                    new_chunks = extract_chunks_from_long_text(val, tokenizer, chunk_size)
                    if new_chunks:
                        needed = count - len(chunks)
                        chunks.extend(new_chunks[:needed])
                        pbar.update(len(new_chunks[:needed]))
                    break
        except StopIteration:
            print("Достигнут конец датасета!")
            break
            
    pbar.close()
    print(f"Просмотрено документов: {docs_processed}. Собрано кусков: {len(chunks)}.")
    return chunks

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("Инициализация токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id, trust_remote_code=True)
    anchor_ids = tokenizer.encode(args.anchor, add_special_tokens=False)
    print(f"Якорь '{args.anchor}' → {anchor_ids}")
    
    # CulturaX обрабатываем как сырые тексты
    if args.n_culturax_ru > 0:
        ru_file = os.path.join(args.culturax_dir, "ru_part_00002.parquet")
        ru_texts = get_culturax_texts(ru_file, args.n_culturax_ru)
        if ru_texts:
            process_and_save_texts(ru_texts, os.path.join(args.out_dir, "culturax_ru.parquet"), 
                                   tokenizer, anchor_ids, args.max_length, "CulturaX RU")
                             
    if args.n_culturax_cs > 0:
        cs_file = os.path.join(args.culturax_dir, "cs_part_00002.parquet")
        cs_texts = get_culturax_texts(cs_file, args.n_culturax_cs)
        if cs_texts:
            process_and_save_texts(cs_texts, os.path.join(args.out_dir, "culturax_cs.parquet"), 
                                   tokenizer, anchor_ids, args.max_length, "CulturaX CS")
                             
    # Magpie и OpenThoughts — ищем длинные и нарезаем кусками
    chunk_size = args.max_length - len(anchor_ids)
    
    if args.n_magpie > 0:
        magpie_chunks = get_magpie_chunks(args.n_magpie, tokenizer, chunk_size)
        if magpie_chunks:
            process_and_save_tokens(magpie_chunks, os.path.join(args.out_dir, "magpie.parquet"), 
                                    anchor_ids, args.max_length, "Magpie")
                             
    if args.n_openthoughts > 0:
        ot_chunks = get_openthoughts_chunks(args.n_openthoughts, tokenizer, chunk_size)
        if ot_chunks:
            process_and_save_tokens(ot_chunks, os.path.join(args.out_dir, "openthoughts.parquet"), 
                                    anchor_ids, args.max_length, "OpenThoughts")
                             
    print("\nПодготовка данных Phase 3 успешно завершена!")
    print(f"Все файлы сохранены в: {args.out_dir}")

if __name__ == "__main__":
    main()
