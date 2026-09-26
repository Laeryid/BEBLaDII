import os
import sys
import json
import zipfile
import glob
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR   = os.path.join(PROJECT_ROOT, "data")

def main():
    for name in ["math", "physics"]:
        zip_path = os.path.join(DATA_DIR, f"{name}.zip")
        out_dir = os.path.join(DATA_DIR, f"camel_{name}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(out_dir)
            
        json_files = glob.glob(os.path.join(out_dir, "**", "*.json"), recursive=True)
        jsonl_files = glob.glob(os.path.join(out_dir, "**", "*.jsonl"), recursive=True)
        
        print(f"\n{name.upper()} Structure:")
        print(f"JSON: {len(json_files)}, JSONL: {len(jsonl_files)}")
        if jsonl_files:
            with open(jsonl_files[0], encoding="utf-8") as f:
                data = json.loads(f.readline())
                print(f"Keys: {list(data.keys())}")
                print(f"Sample: {str(data)[:300]}")

if __name__ == "__main__":
    main()
