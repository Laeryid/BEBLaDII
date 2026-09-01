import os
import sys
import time
import torch

# Добавляем src в PYTHONPATH
sys.path.insert(0, os.path.abspath("src"))

from beb_la_dii.utils.data import get_dataloader

def test():
    print("Initializing dataloader...")
    try:
        dl = get_dataloader(
            stage='reasoning',  # используем reasoning или awakening
            batch_size=64,
            max_length=512,
            split='train',
            val_ratio=0.0
        )
    except Exception as e:
        print(f"Error initializing dataloader: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Dataloader initialized. Starting iteration...")
    
    start_time = None
    samples_processed = 0
    
    try:
        for i, batch in enumerate(dl):
            if i == 0:
                print(f"First batch received! Input shape: {batch['input_ids'].shape}")
                start_time = time.time()
            else:
                samples_processed += batch['input_ids'].shape[0]
            
            if i % 10 == 0 and i > 0:
                elapsed = time.time() - start_time
                print(f"Batch {i}, throughput: {samples_processed / elapsed:.2f} samples/sec")
                
            if i >= 50:
                print("Successfully tested 50 batches.")
                break
    except Exception as e:
        print(f"Error during iteration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Запускаем с freeze_support для Windows multi-processing
    import multiprocessing
    multiprocessing.freeze_support()
    test()
