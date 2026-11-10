import os
import sys
import glob
import json
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.beb_la_dii.utils.tokenizer import get_tokenizer

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "BEBLaDII-planB-Phase6-Data", "phase 6", "train_data", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_json(args):
    f, q_key, a_key = args
    with open(f, "r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except:
            return None
            
    if isinstance(data, list):
        data = data[0]
        
    q = data.get(q_key, "")
    a = data.get(a_key, "")
    if not q or not a:
        return None
        
    q = str(q).strip()
    a = str(a).strip()
    return q, a

def process_camel_folder(folder_name, q_key, a_key, out_filename):
    tokenizer = get_tokenizer()
    input_dir = os.path.join(PROJECT_ROOT, "data", folder_name)
    json_files = glob.glob(os.path.join(input_dir, "**", "*.json"), recursive=True)
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
        
    print(f"\nLoading {folder_name} ({len(json_files)} files) with threads...")
    
    args_list = [(f, q_key, a_key) for f in json_files]
    loaded = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        for res in tqdm(executor.map(load_json, args_list), total=len(args_list), desc=f"Loading {folder_name}"):
            if res is not None:
                loaded.append(res)
                
    print(f"Loaded {len(loaded)} pairs. Now tokenizing...")
    valid_records = []
    
    for q, a in tqdm(loaded, desc=f"Tokenizing {folder_name}"):
        tokens = tokenizer.encode(a, add_special_tokens=False)
        length = len(tokens)
        
        if 50 <= length <= 500:
            valid_records.append({"Q": q, "A": a, "length": length})
                
    print(f"Done. Processed: {len(loaded)}, Saved: {len(valid_records)}")
    if valid_records:
        out_path = os.path.join(OUTPUT_DIR, out_filename)
        pd.DataFrame(valid_records).to_parquet(out_path, index=False)
        print(f"Saved to {out_path}")

def process_loong(folder_name, out_filename):
    tokenizer = get_tokenizer()
        
    input_dir = os.path.join(PROJECT_ROOT, "data", folder_name)
    parquet_files = glob.glob(os.path.join(input_dir, "**", "*.parquet"), recursive=True)
    if not parquet_files:
        print(f"No Parquet files found in {input_dir}")
        return
        
    print(f"\nProcessing {folder_name} ({len(parquet_files)} files)...")
    valid_records = []
    total = 0
    
    for f in parquet_files:
        df = pd.read_parquet(f)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Parsing {os.path.basename(f)}"):
            q = row.get("question", "")
            rationale = row.get("rationale", "")
            
            if not isinstance(q, str) or not isinstance(rationale, str):
                continue
                
            q = q.strip()
            a = rationale.strip()
            
            if not q or not a:
                continue
                
            q = q + "\n\nProvide a Python code to solve this problem:"
                
            tokens = tokenizer.encode(a, add_special_tokens=False)
            length = len(tokens)
            
            if 50 <= length <= 500:
                valid_records.append({
                    "Q": q,
                    "A": a,
                    "length": length
                })
            total += 1
            
    print(f"Done. Processed: {total}, Saved: {len(valid_records)}")
    if valid_records:
        out_path = os.path.join(OUTPUT_DIR, out_filename)
        pd.DataFrame(valid_records).to_parquet(out_path, index=False)
        print(f"Saved to {out_path}")

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    process_camel_folder("camel_math", "message_1", "message_2", "data_camel_math.parquet")
    process_camel_folder("camel_physics", "message_1", "message_2", "data_camel_physics.parquet")
    process_loong("loong", "data_camel_loong.parquet")

if __name__ == "__main__":
    main()
