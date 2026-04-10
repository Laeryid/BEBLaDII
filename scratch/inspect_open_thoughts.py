import pandas as pd
import glob
import sys
import json

sys.stdout.reconfigure(encoding='utf-8')

file_path = "data/open_thoughts/data/train-00000-of-00006.parquet"
df = pd.read_parquet(file_path, engine='pyarrow')

print(f"File: {file_path}")
print("Structure of first conversation:")
conv = df.iloc[0]['conversations']
print(json.dumps(conv.tolist() if hasattr(conv, 'tolist') else list(conv), indent=2, ensure_ascii=False))
