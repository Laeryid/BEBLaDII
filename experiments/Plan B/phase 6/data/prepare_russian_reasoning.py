import os
import sys
import glob
import pandas as pd
from tqdm import tqdm
import re

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.beb_la_dii.utils.tokenizer import get_tokenizer

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "BEBLaDII-planB-Phase6-Data", "phase 6", "train_data", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_output(text):
    match = re.search(r'<output>(.*?)</output>', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def main():
    input_dir = os.path.join(PROJECT_ROOT, "data", "russian-reasoning")
    parquet_files = glob.glob(os.path.join(input_dir, "**", "*.parquet"), recursive=True)
    
    if not parquet_files:
        print(f"No parquet files found in {input_dir}")
        return
        
    print(f"Found {len(parquet_files)} parquet files. Loading...")
    
    tokenizer = get_tokenizer()
    valid_records = []
    
    for f in parquet_files:
        df = pd.read_parquet(f)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(f)}"):
            conv = row.get("conversation")
            
            # В pandas массивы структур обычно читаются как list of dicts (в numpy/pyarrow formats)
            # Если это numpy array, преобразуем в list.
            if hasattr(conv, "tolist"):
                conv = conv.tolist()
                
            if not conv or len(conv) < 2:
                continue
                
            user_msg = next((msg for msg in conv if msg.get("role") == "user"), None)
            if not user_msg:
                continue
            q = user_msg.get("content", "").strip()
            
            assistant_msg = next((msg for msg in conv if msg.get("role") == "assistant"), None)
            if not assistant_msg:
                continue
                
            a_full = assistant_msg.get("content", "")
            a = extract_output(a_full)
            
            if not q or not a:
                continue
                
            tokens = tokenizer.encode(a, add_special_tokens=False)
            length = len(tokens)
            
            if 50 <= length <= 500:
                valid_records.append({
                    "Q": q,
                    "A": a,
                    "length": length
                })
                
    out_path = os.path.join(OUTPUT_DIR, "data_russian_reasoning.parquet")
    pd.DataFrame(valid_records).to_parquet(out_path, index=False)
    print(f"Done! Saved {len(valid_records)} to {out_path}")

if __name__ == "__main__":
    main()
