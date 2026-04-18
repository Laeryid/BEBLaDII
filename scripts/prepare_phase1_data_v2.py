import os
import shutil
import json
from pathlib import Path
import torch
from tqdm import tqdm

try:
    from indexed_parquet_dataset import IndexedParquetDataset
except ImportError:
    from indexed_parquet import IndexedParquetDataset

import sys
import codecs

# Fix for Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

# Import our tokenizer getter
# Add project root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from beb_la_dii.utils.tokenizer import get_tokenizer

def apply_format(item, dtype):
    if item is None: return ""
    
    if dtype == 'raw':
        return item.get('text', '') or ""
    elif dtype == 'magpie':
        inst = item.get('instruction', '') or ""
        resp = item.get('response', '') or ""
        return f"<|im_start|>user\n{inst}<|im_end|>\n<|im_start|>assistant\n<|thought|>\n{resp}<|im_end|>"
    elif dtype == 'sharegpt':
        system = item.get('system') or ""
        convs = item.get('conversations') or []
        text = ""
        if system:
            text += f"<|im_start|>system\n{system}<|im_end|>\n"
        
        if not isinstance(convs, (list, tuple)):
            return text
            
        for i, msg in enumerate(convs):
            if not isinstance(msg, dict): continue
            role = "user" if msg.get('from') == 'human' else "assistant"
            content = msg.get('value', '') or ""
            text += f"<|im_start|>{role}\n"
            if role == "assistant" and i == 1: 
                text += f"<|thought|>\n"
            text += f"{content}<|im_end|>\n"
        return text
    return str(item)

def prepare_data_v2():
    print("Initializing tokenizer...")
    tokenizer = get_tokenizer()
    
    root = Path(".")
    staging = root / "kaggle_upload_1_3"
    staging_data = staging / "data"
    
    # Plans for data
    data_plans = [
        # Awakening (Sample & Truncate)
        {"name": "CulturaX_cs_Awakening", "target": "Awakening", "path": "data/CulturaX", "pattern": "cs_part_00000.parquet", "count": 45000, "type": "raw", "action": "truncate"},
        {"name": "CulturaX_ru_Awakening", "target": "Awakening", "path": "data/CulturaX", "pattern": "ru_part_00000.parquet", "count": 45000, "type": "raw", "action": "truncate"},
        {"name": "magpie_awakening", "target": "Awakening", "path": "data/magpie_reasoning", "count": 10000, "type": "magpie", "action": "truncate"},
        
        # Reasoning (Filter < 4096 for reasoning, specific bounds for CulturaX)
        {"name": "magpie_reasoning", "target": "Reasoning", "path": "data/magpie_reasoning", "count": 80000, "type": "magpie", "action": "filter"},
        {"name": "open_thoughts", "target": "Reasoning", "path": "data/open_thoughts", "count": None, "type": "sharegpt", "action": "filter"},
        {"name": "CulturaX_cs_Reasoning", "target": "Reasoning", "path": "data/CulturaX", "pattern": "cs_part_00000.parquet", "count": 15000, "type": "raw", "action": "filter"},
        {"name": "CulturaX_ru_Reasoning", "target": "Reasoning", "path": "data/CulturaX", "pattern": "ru_part_00000.parquet", "count": 15000, "type": "raw", "action": "filter"},
    ]
    
    for plan in data_plans:
        src_path = root / plan["path"]
        if not src_path.exists():
            print(f"Skipping {plan['name']}: {src_path} not found.")
            continue
            
        print(f"\n--- Processing: {plan['name']} ({plan['target']}) ---")
        pattern = plan.get('pattern', '*.parquet')
        ds = IndexedParquetDataset.from_folder(src_path, pattern=pattern)
        
        dest_path = staging_data / plan["target"] / f"{plan['name']}.parquet"
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        if plan["action"] == "filter":
            is_culturax = "CulturaX" in plan["name"]
            
            if is_culturax:
                print(f"Filtering {plan['name']} for length 3000-4000 tokens...")
                filter_fn = lambda batch: [3000 <= x["token_count"] <= 4000 for x in batch]
            else:
                print(f"Filtering {plan['name']} for length < 4096 tokens...")
                filter_fn = lambda batch: [0 < x["token_count"] < 4096 for x in batch]

            # We add a column 'token_count' using map
            def calc_len_batch(batch):
                texts = [apply_format(item, plan["type"]) for item in batch]
                # Fast batch token counting
                batch_results = tokenizer.batch_encode_plus(
                    texts, 
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False
                )
                for item, tokens in zip(batch, batch_results["input_ids"]):
                    item["token_count"] = len(tokens)
                return batch
            
            # Use batch versions for 10-100x speedup
            ds_mapped = ds.map_batches(calc_len_batch, batch_size=512)
            ds_filtered = ds_mapped.filter_batches(
                filter_fn, 
                batch_size=512,
                show_progress=True
            )
            
            if plan.get("count"):
                print(f"Starting clone with filter and limit {plan['count']} (using .limit() for speed)...")
                # Using .limit() instead of .sample() to avoid a full scan for length counting
                ds_final = ds_filtered.limit(plan["count"])
            else:
                print(f"Starting clone for all filtered records...")
                ds_final = ds_filtered
                
            ds_final.clone(dest_path, optimize_by_reorder=True)
            
        elif plan["action"] == "truncate":
            print(f"Truncating {plan['name']} to 500 tokens...")
            # Sample first to save work
            if len(ds) > plan["count"]:
                ds_sampled = ds.sample(n=plan["count"])
            else:
                ds_sampled = ds
            
            def truncate_text(item):
                # We need to truncate the actual field that the trainee will use.
                # For 'raw', it's 'text'. For others, it's instruction/response/conversations.
                # But truncation is better done on the 'final' string or we just truncate the original fields.
                # User: "оставим просто первые 500 токенов ... будем обрезать".
                
                if plan["type"] == 'raw':
                    text = item.get('text', '')
                    tokens = tokenizer.encode(text, add_special_tokens=False)[:500]
                    item['text'] = tokenizer.decode(tokens)
                elif plan["type"] == 'magpie':
                    # Truncate response mostly, or both. Let's truncate the whole joined text?
                    # No, we need to keep the structure. Let's just truncate the response field if it's there.
                    resp = item.get('response', '')
                    tokens = tokenizer.encode(resp, add_special_tokens=False)[:500]
                    item['response'] = tokenizer.decode(tokens)
                elif plan["type"] == 'sharegpt':
                    # Truncate each conversation content
                    convs = item.get('conversations', [])
                    if isinstance(convs, list):
                        for msg in convs:
                            content = msg.get('value', '')
                            tokens = tokenizer.encode(content, add_special_tokens=False)[:500]
                            msg['value'] = tokenizer.decode(tokens)
                return item
            
            ds_final = ds_sampled.map(truncate_text)
            ds_final.clone(dest_path, optimize_by_reorder=True)

    print("\nData preparation complete!")

if __name__ == "__main__":
    prepare_data_v2()
