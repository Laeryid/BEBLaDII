import os
import shutil
from indexed_parquet_dataset import IndexedParquetDataset

def main():
    src_dir = r"c:\Experiments\BEBLaDII\kaggle_upload_1_3\data\Reasoning"
    train_dir = os.path.join(src_dir, "train")
    val_dir = os.path.join(src_dir, "val")
    backup_dir = os.path.join(src_dir, "_source_backup")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(backup_dir, exist_ok=True)

    print("Loading dataset from folder...")
    # Указываем явно, чтобы не зацепить подпапки, если они уже есть
    ds = IndexedParquetDataset.from_folder(src_dir, pattern="*.parquet")
    
    total_samples = len(ds)
    print(f"Total samples found: {total_samples}")
    
    if total_samples == 0:
        print("Error: No samples found. Check if parquet files are in the root Reasoning folder.")
        return

    print("Performing train_test_split (test_size=0.1, seed=42)...")
    train_ds, val_ds = ds.train_test_split(test_size=0.1, shuffle=True, seed=42)

    print(f"Saving train set ({len(train_ds)} samples) to {train_dir}...")
    train_ds.to_parquet(os.path.join(train_dir, "train_reasoning.parquet"))

    print(f"Saving val set ({len(val_ds)} samples) to {val_dir}...")
    val_ds.to_parquet(os.path.join(val_dir, "val_reasoning.parquet"))

    print("Closing datasets and cleaning up memory...")
    # На всякий случай удаляем объекты, которые держат открытыми файлы
    del train_ds
    del val_ds
    del ds
    
    import gc
    gc.collect()

    print("Moving original files to backup...")
    moved_count = 0
    for f in os.listdir(src_dir):
        full_path = os.path.join(src_dir, f)
        if f.endswith(".parquet") and os.path.isfile(full_path):
            try:
                shutil.move(full_path, os.path.join(backup_dir, f))
                print(f"Moved: {f}")
                moved_count += 1
            except Exception as e:
                print(f"Failed to move {f}: {e}")
    
    print(f"Process finished. Moved {moved_count} files.")

if __name__ == "__main__":
    main()
