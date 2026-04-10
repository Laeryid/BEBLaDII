
import os
from huggingface_hub import snapshot_download

def download_dataset(repo_id, local_dir):
    print(f"Downloading {repo_id} to {local_dir}...")
    # Use snapshot_download to get only parquet files
    # allow_patterns ensures we don't download everything (like git history or other formats)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=["*.parquet"],
        local_dir_use_symlinks=False  # Avoid windows symlink issues
    )
    print(f"Done with {repo_id}")

def main():
    # Dataset 1: Magpie Reasoning V2
    download_dataset(
        "Magpie-Align/Magpie-Reasoning-V2-250K-CoT-QwQ",
        "data/magpie_reasoning"
    )
    
    # Dataset 2: OpenThoughts 114k
    download_dataset(
        "open-thoughts/OpenThoughts-114k",
        "data/open_thoughts"
    )
    
    print("\nAll downloads completed successfully!")

if __name__ == "__main__":
    main()
