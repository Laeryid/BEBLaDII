import os
import sys
import pandas as pd
from tqdm import tqdm
from huggingface_hub import hf_hub_download

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.beb_la_dii.utils.tokenizer import get_tokenizer

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "BEBLaDII-planB-Phase6-Data", "phase 6", "train_data", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("Downloading StackExchange Philosophy...")
    local_file = hf_hub_download(
        repo_id="mlfoundations-dev/stackexchange_philosophy",
        filename="data/train-00000-of-00001.parquet",
        repo_type="dataset"
    )
    
    print(f"Loaded {local_file}")
    df = pd.read_parquet(local_file)
    print(f"Total rows: {len(df)}")
    
    tokenizer = get_tokenizer()
    valid_records = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Filtering Philosophy"):
        q = row.get("instruction", "")
        a = row.get("completion", "")
        
        if not isinstance(q, str) or not isinstance(a, str):
            continue
            
        q = q.strip()
        a = a.strip()
        
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
            
    out_path = os.path.join(OUTPUT_DIR, "data_philosophy.parquet")
    pd.DataFrame(valid_records).to_parquet(out_path, index=False)
    print(f"Done! Saved {len(valid_records)} to {out_path}")

if __name__ == "__main__":
    main()
