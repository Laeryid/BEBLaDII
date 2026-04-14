import torch
import numpy as np
from tqdm.auto import tqdm
from src.beb_la_dii.utils.data import get_dataloader
from src.beb_la_dii.utils.tokenizer import get_tokenizer
import os

def main():
    print("Loading tokenizer and dataset...")
    tokenizer = get_tokenizer()
    
    try:
        loader = get_dataloader(stage='reasoning', batch_size=1, max_length=10, split='train')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
        
    # In get_dataloader, random_split returns a Subset.
    # So `loader.dataset` is a Subset, and `loader.dataset.dataset` is DistillationDataset.
    try:
        subset = loader.dataset
        base_ds = subset.dataset
    except Exception as e:
        print(f"Error accessing base dataset: {e}")
        return
        
    print(f"Total samples in reasoning train split: {len(subset)}")
    
    # We will sample 1000 items to estimate the distribution
    num_samples = min(1000, len(subset))
    indices = np.random.choice(len(subset), num_samples, replace=False)
    
    lengths = []
    
    for idx in tqdm(indices, desc="Evaluating lengths"):
        original_idx = subset.indices[idx]
        
        text = ""
        for m in base_ds.index_map:
            if m['start'] <= original_idx < m['end']:
                item = m['ds'][original_idx - m['start']]
                text = base_ds._apply_mapper(item, m['type'])
                break
        
        if not text:
            continue
            
        # Encode to check length without padding/truncation restrictions
        tokens = tokenizer.encode(text)
        lengths.append(len(tokens))
        
    if not lengths:
        print("No valid texts retrieved.")
        return
        
    lengths = np.array(lengths)
    
    print("\n--- Token Length Distribution (Reasoning Data) ---")
    print(f"Processed samples:   {len(lengths)}")
    print(f"Mean:                {np.mean(lengths):.1f}")
    print(f"Median (50%):        {np.percentile(lengths, 50):.0f}")
    print(f"75th percentile:     {np.percentile(lengths, 75):.0f}")
    print(f"90th percentile:     {np.percentile(lengths, 90):.0f}")
    print(f"95th percentile:     {np.percentile(lengths, 95):.0f}")
    print(f"99th percentile:     {np.percentile(lengths, 99):.0f}")
    print(f"Maximum:             {np.max(lengths)}")

if __name__ == '__main__':
    main()
