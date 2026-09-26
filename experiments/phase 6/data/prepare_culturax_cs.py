import os
import sys
import random
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.beb_la_dii.utils.tokenizer import get_tokenizer

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "BEBLaDII-planB-Phase6-Data", "phase 6", "train_data", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CS_PROMPTS = [
    "Pokračujte prosím v této myšlence:\n",
    "Dokončete následující text:\n",
    "Prosím, navažte na tento začátek:\n",
    "Jak by tento text mohl pokračovat?\n",
    "Napište pokračování tohoto textu:\n",
    "Dokončete myšlenku:\n",
    "Doplňte zbývající část textu:\n",
    "Pokračujte ve psaní:\n"
]

def main():
    file_path = os.path.join(PROJECT_ROOT, "data", "CulturaX", "data", "cs_part_00002.parquet")
    print(f"Loading {file_path}...")
    
    # Загружаем паркет (1.5 ГБ)
    df = pd.read_parquet(file_path, columns=["text"])
    
    # Берем с конца, чтобы избежать пересечения с первыми батчами Phase 1
    df = df.iloc[-200000:]
    
    tokenizer = get_tokenizer()
    valid_records = []
    
    print("Filtering texts...")
    for text in tqdm(df["text"]):
        if not isinstance(text, str):
            continue
            
        text = text.strip()
        # Ищем первую точку для разделения на Q и A
        # Чтобы Q не был слишком коротким, ищем точку после 40 символов
        split_idx = text.find(". ", 40)
        if split_idx == -1:
            continue
            
        q_raw = text[:split_idx + 1].strip()
        a_raw = text[split_idx + 1:].strip()
        
        if not q_raw or not a_raw:
            continue
            
        # Токенизируем A для проверки длины
        tokens_a = tokenizer.encode(a_raw, add_special_tokens=False)
        length = len(tokens_a)
        
        if 50 <= length <= 500:
            prompt_prefix = random.choice(CS_PROMPTS)
            q = f"{prompt_prefix}\n{q_raw}"
            
            valid_records.append({
                "Q": q,
                "A": a_raw,
                "length": length
            })
            
            if len(valid_records) >= 10000:
                break
                
    out_path = os.path.join(OUTPUT_DIR, "data_culturax_cs.parquet")
    pd.DataFrame(valid_records).to_parquet(out_path, index=False)
    print(f"Done! Saved {len(valid_records)} to {out_path}")

if __name__ == "__main__":
    main()
