import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

file_path = "data/CulturaX/ru_part_00000.parquet"
df = pd.read_parquet(file_path, engine='pyarrow')

print(f"File: {file_path}")
print(f"Columns: {df.columns.tolist()}")
print(f"Sample: {str(df.iloc[0].to_dict())[:200]}...")
