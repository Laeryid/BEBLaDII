
import os
from huggingface_hub import HfApi

def check_repo():
    repo_id = "Magpie-Align/Magpie-Reasoning-V2-250K-CoT-QwQ"
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        print(f"Files in {repo_id}:")
        for f in files:
            print(f"  - {f}")
        
        info = api.dataset_info(repo_id)
        print(f"Total size: {info.siblings[0].size if hasattr(info.siblings[0], 'size') else 'unknown'}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_repo()
