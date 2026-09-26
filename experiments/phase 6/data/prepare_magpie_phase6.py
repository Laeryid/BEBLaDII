import os
import sys
import glob
import pandas as pd
from tqdm import tqdm
import re

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.beb_la_dii.utils.tokenizer import get_tokenizer

def main():
    tokenizer = get_tokenizer()
    
    input_dir = os.path.join(PROJECT_ROOT, "data", "magpie_reasoning", "data")
    output_dir = os.path.join(PROJECT_ROOT, "BEBLaDII-planB-Phase6-Data", "phase 6", "train_data", "data")
    os.makedirs(output_dir, exist_ok=True)
    
    parquet_files = glob.glob(os.path.join(input_dir, "*.parquet"))
    
    total_processed = 0
    valid_records = []
    
    for file_idx, file_path in enumerate(parquet_files):
        print(f"Processing file {file_idx+1}/{len(parquet_files)}: {os.path.basename(file_path)}")
        df = pd.read_parquet(file_path)
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Filtering records"):
            instruction = row.get("instruction", "")
            response = row.get("response", "")
            
            if not isinstance(response, str) or not isinstance(instruction, str):
                continue
                
            parts = response.split("Final Answer")
            if len(parts) < 2:
                continue
            
            final_answer_raw = parts[-1]
            final_answer_clean = re.sub(r'^[\*\s\n\:]+', '', final_answer_raw)
            
            if not final_answer_clean.strip():
                continue
                
            tokens = tokenizer.encode(final_answer_clean, add_special_tokens=False)
            length = len(tokens)
            
            if 50 <= length <= 500:
                valid_records.append({
                    "Q": instruction,
                    "A": final_answer_clean,
                    "length": length
                })
        
        total_processed += len(df)
        
    print("=" * 40)
    print(f"Finished processing Magpie-Reasoning-V2.")
    print(f"Total processed: {total_processed}")
    print(f"Total saved: {len(valid_records)}")
    
    if valid_records:
        out_df = pd.DataFrame(valid_records)
        out_path = os.path.join(output_dir, "data_magpie.parquet")
        out_df.to_parquet(out_path, index=False)
        print(f"Saved all records to {out_path}")

if __name__ == "__main__":
    main()
