"""
Download camel-ai/math and camel-ai/physics from HuggingFace (zip archives),
extract, inspect structure, and prepare filtered Parquet files for Phase 6.
"""
import os
import sys
import json
import zipfile
import glob
import pandas as pd
from tqdm import tqdm
from huggingface_hub import hf_hub_download

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.beb_la_dii.utils.tokenizer import get_tokenizer

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "BEBLaDII-planB-Phase6-Data", "phase 6", "train_data", "data")
DATA_DIR   = os.path.join(PROJECT_ROOT, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SOURCES = [
    {"repo": "camel-ai/math",    "file": "math.zip",    "out_dir": "camel_math",    "out_parquet": "data_camel_math.parquet"},
    {"repo": "camel-ai/physics", "file": "physics.zip", "out_dir": "camel_physics", "out_parquet": "data_camel_physics.parquet"},
]


def download_and_extract(repo, filename, extract_to):
    print(f"Downloading {repo}/{filename}...")
    local_zip = hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset",
                                local_dir=extract_to)
    print(f"Extracting to {extract_to}...")
    with zipfile.ZipFile(local_zip, "r") as z:
        z.extractall(extract_to)
    print("Done.")
    return extract_to


def inspect_structure(extract_dir):
    """Print first file found and its structure."""
    json_files = glob.glob(os.path.join(extract_dir, "**", "*.json"), recursive=True)
    jsonl_files = glob.glob(os.path.join(extract_dir, "**", "*.jsonl"), recursive=True)
    csv_files = glob.glob(os.path.join(extract_dir, "**", "*.csv"), recursive=True)
    print(f"  JSON files: {len(json_files)}, JSONL: {len(jsonl_files)}, CSV: {len(csv_files)}")
    
    # Show first record from first file found
    for f in (json_files + jsonl_files + csv_files)[:1]:
        print(f"  Sample file: {f}")
        with open(f, encoding="utf-8") as fh:
            if f.endswith(".jsonl"):
                data = json.loads(fh.readline())
            elif f.endswith(".json"):
                data = json.load(fh)
                if isinstance(data, list):
                    data = data[0]
            else:
                import csv
                reader = csv.DictReader(fh)
                data = next(reader)
        print(f"  Keys: {list(data.keys())}")
        for k, v in data.items():
            val = str(v)[:150]
            print(f"    {k}: {val}")
        break
    return json_files, jsonl_files, csv_files


def load_all_records(json_files, jsonl_files, csv_files):
    records = []
    for f in json_files:
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, list):
                records.extend(data)
            else:
                records.append(data)
    for f in jsonl_files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    for f in csv_files:
        df = pd.read_csv(f)
        records.extend(df.to_dict("records"))
    return records


def filter_records(records, tokenizer, q_key, a_key):
    valid = []
    for rec in tqdm(records, desc="Filtering"):
        q = rec.get(q_key, "")
        a = rec.get(a_key, "")
        if not isinstance(q, str) or not isinstance(a, str):
            continue
        q, a = q.strip(), a.strip()
        if not q or not a:
            continue
        tokens = tokenizer.encode(a, add_special_tokens=False)
        if 50 <= len(tokens) <= 500:
            valid.append({"Q": q, "A": a, "length": len(tokens)})
    return valid


def main():
    tokenizer = get_tokenizer()
    
    for src in SOURCES:
        extract_dir = os.path.join(DATA_DIR, src["out_dir"])
        os.makedirs(extract_dir, exist_ok=True)
        
        download_and_extract(src["repo"], src["file"], extract_dir)
        
        print(f"\n=== Inspecting {src['repo']} ===")
        json_files, jsonl_files, csv_files = inspect_structure(extract_dir)
        
        print("\nLoading all records...")
        records = load_all_records(json_files, jsonl_files, csv_files)
        print(f"Total records loaded: {len(records)}")
        
        if not records:
            print("No records found! Skipping.")
            continue
        
        # Detect Q/A keys from first record
        first = records[0]
        keys = list(first.keys())
        print(f"Available keys: {keys}")
        
        # Common key patterns for CAMEL datasets
        q_key = next((k for k in keys if k.lower() in ("question", "problem", "instruction", "input")), keys[0])
        a_key = next((k for k in keys if k.lower() in ("solution", "answer", "response", "output")), keys[1] if len(keys) > 1 else keys[0])
        print(f"Using Q={q_key!r}, A={a_key!r}")
        
        valid = filter_records(records, tokenizer, q_key, a_key)
        print(f"Saved: {len(valid)} / {len(records)}")
        
        if valid:
            out_path = os.path.join(OUTPUT_DIR, src["out_parquet"])
            pd.DataFrame(valid).to_parquet(out_path, index=False)
            print(f"Written to {out_path}")
        print()


if __name__ == "__main__":
    main()
