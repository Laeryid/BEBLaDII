import os
import sys
from pathlib import Path

# Fix for Windows console encoding
if sys.platform == 'win32':
    import codecs
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

# Add project root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from indexed_parquet_dataset import IndexedParquetDataset
except ImportError:
    from indexed_parquet import IndexedParquetDataset

def main():
    target_path = Path(ROOT_DIR) / "kaggle_upload_1_3" / "data" / "Reasoning"
    
    file_name = "magpie_reasoning.parquet"
    full_path = target_path / file_name
    backup_path = target_path / "magpie_reasoning_backup.parquet"
    
    if not full_path.exists():
        print(f"Error: {full_path} not found.")
        return
        
    print(f"Loading {full_path}...")
    ds = IndexedParquetDataset.from_folder(target_path, pattern=file_name)
    
    total = len(ds)
    print(f"Total records in current file: {total}")
    
    target_count = 80000
    if total <= target_count:
        print(f"Dataset already has <= {target_count} records. Exiting.")
        return
        
    print(f"Sampling {target_count} records...")
    # Используем встроенный метод sample
    ds_sampled = ds.sample(target_count)
    
    sampled_path = target_path / "magpie_reasoning_sampled.parquet"
    if sampled_path.exists():
        sampled_path.unlink()
        
    print(f"Writing sampled dataset to {sampled_path}...")
    ds_sampled.clone(sampled_path, optimize_by_reorder=True)
    
    # CRITICAL: Close handles before renaming on Windows
    print("Closing dataset handles...")
    if hasattr(ds, '_file_handles'):
        ds._file_handles.clear()
    if hasattr(ds_sampled, '_file_handles'):
        ds_sampled._file_handles.clear()
    
    import gc
    gc.collect()
    
    print(f"Renaming original to {backup_path.name}...")
    if backup_path.exists():
        backup_path.unlink()
    
    try:
        full_path.rename(backup_path)
    except PermissionError as e:
        print(f"Failed to rename due to PermissionError: {e}")
        print("Attempting to wait and retry...")
        import time
        time.sleep(2)
        gc.collect()
        full_path.rename(backup_path)
    
    print("Moving sampled file to original name...")
    sampled_path.rename(full_path)
    
    print("Done! Можем использовать данные для обучения.")

if __name__ == "__main__":
    main()
