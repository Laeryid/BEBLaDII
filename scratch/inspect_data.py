import pandas as pd
import glob
import os
import sys

# Обеспечиваем вывод в UTF-8
sys.stdout.reconfigure(encoding='utf-8')

paths = [
    "data/magpie_reasoning/data/*.parquet",
    "data/open_thoughts/data/*.parquet"
]

for pattern in paths:
    files = glob.glob(pattern)
    if files:
        df = pd.read_parquet(files[0], engine='pyarrow')
        print(f"File: {files[0]}")
        print(f"Columns: {df.columns.tolist()}")
        # Берем только текстовые колонки для примера, чтобы избежать проблем с выводом
        print(f"Sample (first 200 chars): {str(df.iloc[0].to_dict())[:200]}...")
        print("-" * 20)
    else:
        print(f"No files found for pattern: {pattern}")
