import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

def extract_from_conv(conv):
    if isinstance(conv, (list, np.ndarray)):
        return "\n\n".join([str(m.get("value", "")) for m in conv if isinstance(m, (dict, pd.Series))])
    return ""

def main():
    src_dir = r"c:\Experiments\BEBLaDII\kaggle_upload_1_3\data\_reasoning_source_backup"
    output_base = r"c:\Experiments\BEBLaDII\kaggle_upload_1_3\data\Reasoning"
    
    train_dir = os.path.join(output_base, "train")
    val_dir = os.path.join(output_base, "val")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    all_dfs = []
    files = [f for f in os.listdir(src_dir) if f.endswith(".parquet")]
    
    for f in files:
        path = os.path.join(src_dir, f)
        print(f"Processing: {f}...")
        df = pd.read_parquet(path)
        
        if "CulturaX" in f:
            df = df[["text"]]
        elif "magpie" in f:
            print(f"  - Mapping magpie columns...")
            df["text"] = df["instruction"].fillna("") + "\n\n" + df["response"].fillna("")
            df = df[["text"]]
        elif "open_thoughts" in f:
            print(f"  - Extracting from 'conversations' for open_thoughts...")
            df["text"] = df["conversations"].apply(extract_from_conv)
            df = df[["text"]]
        else:
            if "text" in df.columns:
                df = df[["text"]]
            else:
                print(f"  - WARNING: Skipping {f}, no 'text' column and no known mapping.")
                continue
        
        # Очистка
        df = df[df["text"].str.strip() != ""]
        df = df.dropna(subset=["text"])
        
        print(f"  - Valid samples: {len(df)}")
        all_dfs.append(df)

    if not all_dfs:
        print("Error: No data loaded!")
        return

    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal combined samples: {len(full_df)}")

    # Split
    train_df, val_df = train_test_split(full_df, test_size=0.1, random_state=42, shuffle=True)

    # Save
    print(f"Saving train ({len(train_df)})...")
    train_df.to_parquet(os.path.join(train_dir, "train_reasoning.parquet"), index=False)
    
    print(f"Saving val ({len(val_df)})...")
    val_df.to_parquet(os.path.join(val_dir, "val_reasoning.parquet"), index=False)

    print("\nSuccess! Dataset is reassembled with all sources.")

if __name__ == "__main__":
    main()
