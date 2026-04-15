from indexed_parquet_dataset import IndexedParquetDataset
import os
import pandas as pd

# Create dummy parquet
df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
df.to_parquet("test_map.parquet")

ds = IndexedParquetDataset.from_folder(".", pattern="test_map.parquet")
print("Original:", ds[0])

def my_map(item):
    item["c"] = item["a"] * 10
    return item

ds_mapped = ds.map(my_map)
print("Mapped:", ds_mapped[0])

# Cleanup
os.remove("test_map.parquet")
if os.path.exists("test_map.parquet.idx"): os.remove("test_map.parquet.idx")
if os.path.exists(".indexed_parquet_cache"): 
    import shutil
    shutil.rmtree(".indexed_parquet_cache")
