import os
import sys
import glob
import pandas as pd
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.beb_la_dii.utils.tokenizer import get_tokenizer

def main():
    tokenizer = get_tokenizer()
    
    input_dir = os.path.join(PROJECT_ROOT, "data", "open_thoughts", "data")
    output_dir = os.path.join(PROJECT_ROOT, "BEBLaDII-planB-Phase6-Data", "phase 6", "train_data", "data")
    os.makedirs(output_dir, exist_ok=True)
    
    parquet_files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    
    total_processed = 0
    valid_records = []
    
    for file_idx, file_path in enumerate(parquet_files):
        print(f"Processing file {file_idx+1}/{len(parquet_files)}: {os.path.basename(file_path)}")
        df = pd.read_parquet(file_path)
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Filtering OpenThoughts"):
            conversations = row.get("conversations", None)
            if conversations is None or len(conversations) < 2:
                continue
                
            q_text = conversations[0].get("value", "")
            a_full = conversations[1].get("value", "")
            
            if not isinstance(q_text, str) or not isinstance(a_full, str):
                continue
                
            if "<|begin_of_solution|>" not in a_full:
                continue
                
            parts = a_full.split("<|begin_of_solution|>")
            sol_text = parts[-1].replace("<|end_of_solution|>", "").strip()
            
            if not sol_text:
                continue
                
            tokens = tokenizer.encode(sol_text, add_special_tokens=False)
            length = len(tokens)
            
            if 50 <= length <= 500:
                valid_records.append({
                    "Q": q_text.strip(),
                    "A": sol_text,
                    "length": length
                })
                
        total_processed += len(df)
        
    print("=" * 40)
    print(f"Finished processing OpenThoughts-114k.")
    print(f"Total processed: {total_processed}")
    print(f"Total saved: {len(valid_records)}")
    
    if valid_records:
        out_df = pd.DataFrame(valid_records)
        out_path = os.path.join(output_dir, "data_openthoughts.parquet")
        out_df.to_parquet(out_path, index=False)
        print(f"Saved all records to {out_path}")

if __name__ == "__main__":
    main()
