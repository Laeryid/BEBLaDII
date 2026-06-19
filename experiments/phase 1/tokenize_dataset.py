import os
import glob
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

def process_parquet(file_path, output_path, tokenizer, max_length=4096):
    if os.path.exists(output_path):
        print(f"Skipping {file_path}, output already exists.")
        return

    df = pd.read_parquet(file_path)
    
    if "text" not in df.columns:
        print(f"Skipping {file_path}, no 'text' column.")
        return

    print(f"Tokenizing {file_path}...")
    input_ids_list = []
    
    # Пакетная токенизация для скорости
    texts = df["text"].tolist()
    batch_size = 1000
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        encoded = tokenizer(batch_texts, truncation=True, max_length=max_length)
        input_ids_list.extend(encoded["input_ids"])
        
    df["input_ids"] = input_ids_list
    # Удаляем сырой текст, чтобы сэкономить место
    df = df.drop(columns=["text"])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--max_length", type=int, default=4096)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    
    parquet_files = glob.glob(os.path.join(args.input_dir, "**", "*.parquet"), recursive=True)
    
    for file in parquet_files:
        rel_path = os.path.relpath(file, args.input_dir)
        out_file = os.path.join(args.output_dir, rel_path)
        process_parquet(file, out_file, tokenizer, args.max_length)
        
    print(f"Готово! Сохранено в {args.output_dir}")

if __name__ == "__main__":
    main()
