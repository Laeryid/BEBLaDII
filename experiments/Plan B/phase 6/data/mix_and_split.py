import os
import sys
import glob
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, "BEBLaDII-planB-Phase6-Data", "phase 6", "data")
VAL_SIZE = 2000

def main():
    # Ищем все файлы data_*.parquet
    files = sorted(glob.glob(os.path.join(DATA_DIR, "data_*.parquet")))
    if not files:
        print(f"No data_*.parquet files found in {DATA_DIR}!")
        return

    print(f"Found {len(files)} dataset files:")
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        # Ensure canonical columns
        if 'Q' in df.columns and 'A' in df.columns:
            if 'length' not in df.columns:
                df['length'] = df['A'].apply(lambda x: len(str(x).split()))
            df = df[['Q', 'A', 'length']]
        print(f" - {os.path.basename(f)}: {len(df)} rows")
        dfs.append(df)

    # Склеиваем
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal combined rows: {len(combined_df)}")

    # Перемешиваем
    print("Shuffling...")
    shuffled_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Разбиваем на train / val
    val_df = shuffled_df.iloc[:VAL_SIZE]
    train_df = shuffled_df.iloc[VAL_SIZE:]

    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")

    # Сохраняем
    train_path = os.path.join(DATA_DIR, "train_phase6.parquet")
    val_path = os.path.join(DATA_DIR, "val_phase6.parquet")
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    
    print("\nDone! Saved:")
    print(train_path)
    print(val_path)

if __name__ == "__main__":
    main()
