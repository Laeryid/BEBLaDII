import os
import pandas as pd
import glob
import shutil
import numpy as np

def get_parquet_files(folder):
    return glob.glob(os.path.join(folder, "**", "*.parquet"), recursive=True)

def extract_text_from_row(row):
    # Если есть поле response, берем его
    if 'response' in row and isinstance(row['response'], str) and row['response'].strip():
        return row['response']
    
    # Если данные в формате ShareGPT (conversations)
    if 'conversations' in row and isinstance(row['conversations'], (list, tuple, np.ndarray)):
        for msg in row['conversations']:
            if isinstance(msg, dict):
                role = msg.get('from', msg.get('role', ''))
                if role in ('gpt', 'assistant'):
                    return msg.get('value', msg.get('content', ''))
    return ""

def process_conversational_dataset(folder, n):
    files = get_parquet_files(folder)
    texts = []
    
    for f in files:
        if len(texts) >= n: break
        try:
            df = pd.read_parquet(f)
            # Применяем извлечение ко всем строкам
            for _, row in df.iterrows():
                if len(texts) >= n: break
                val = extract_text_from_row(row)
                if val and len(val) >= 100:
                    texts.append(val[:4000]) # грубая обрезка по символам (хватит на ~800 токенов)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    return pd.DataFrame({'text': texts})

def process_culturax(file_path, n):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return pd.DataFrame({'text': []})
        
    df = pd.read_parquet(file_path, columns=['text'])
    df = df[df['text'].str.len() >= 100]
    
    if len(df) > n:
        df = df.sample(n=n, random_state=42)
        
    df['text'] = df['text'].str[:4000]
    return df[['text']]

def main():
    print("Processing Magpie...")
    df_magpie = process_conversational_dataset("data/magpie_reasoning", 50000)
    print(f"Magpie: {len(df_magpie)} records")

    print("Processing OpenThoughts...")
    df_ot = process_conversational_dataset("data/open_thoughts", 50000)
    print(f"OpenThoughts: {len(df_ot)} records")

    print("Processing CulturaX RU...")
    df_ru = process_culturax("data/CulturaX/data/ru_part_00002.parquet", 25000)
    print(f"CulturaX RU: {len(df_ru)} records")

    print("Processing CulturaX CS...")
    df_cs = process_culturax("data/CulturaX/data/cs_part_00002.parquet", 25000)
    print(f"CulturaX CS: {len(df_cs)} records")

    print("Merging datasets...")
    df_all = pd.concat([df_magpie, df_ot, df_ru, df_cs], ignore_index=True)
    
    if len(df_all) == 0:
        print("No data extracted! Check source folders.")
        return
        
    print("Shuffling...")
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

    val_size = int(len(df_all) * 0.05)
    df_val = df_all.iloc[:val_size]
    df_train = df_all.iloc[val_size:]

    out_dir = r"C:\Experiments\BEBLaDII\BEBLaDII-planB-Phase3-Data\phase 3\train_data\data"
    os.makedirs(out_dir, exist_ok=True)
    
    # Очистка целевой папки
    print("Clearing output directory...")
    for old_f in glob.glob(os.path.join(out_dir, "*")):
        if os.path.isfile(old_f):
            os.remove(old_f)
        elif os.path.isdir(old_f):
            shutil.rmtree(old_f)

    print(f"Saving Train: {len(df_train)} records")
    df_train.to_parquet(os.path.join(out_dir, "train.parquet"), index=False)

    print(f"Saving Val: {len(df_val)} records")
    df_val.to_parquet(os.path.join(out_dir, "val.parquet"), index=False)
    
    print("Done! Ready for Kaggle upload.")

if __name__ == "__main__":
    main()
