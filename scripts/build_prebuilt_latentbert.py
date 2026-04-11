import os
import sys
import shutil
from pathlib import Path

# Fix python paths
ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT / "src"))

# Temporary cache for Hugging Face
TMP_CACHE = ROOT / "storage" / "tmp_hf_cache"

def build():
    try:
        from beb_la_dii.model.dus import create_latentbert
    except ImportError as e:
        print(f"Import Error: {e}")
        return

    target_dir = ROOT / "storage" / "prebuilt" / "latentBERT" / "v1.0"
    target_dir.mkdir(parents=True, exist_ok=True)
    TMP_CACHE.mkdir(parents=True, exist_ok=True)
    
    # Force HF to use temp cache
    os.environ["HF_HOME"] = str(TMP_CACHE)
    os.environ["TRANSFORMERS_CACHE"] = str(TMP_CACHE)
    
    print(f"--- Starting local build of 40-layer skeleton ---")
    print(f"Temp cache: {TMP_CACHE}")
    
    try:
        # Create model via DUS
        model = create_latentbert(
            model_id="answerdotai/ModernBERT-large",
            target_layers=40
        )
        
        print(f"Saving skeleton to {target_dir}...")
        model.save_pretrained(target_dir)
        print("--- Build finished successfully! ---")
        
    finally:
        # Cleanup base model (as requested by USER)
        if TMP_CACHE.exists():
            print(f"Cleaning up temp cache (deleting base model)...")
            shutil.rmtree(TMP_CACHE)
            print("Cache cleared.")

if __name__ == "__main__":
    build()
