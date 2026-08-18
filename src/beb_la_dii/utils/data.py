import torch
from torch.utils.data import Dataset, DataLoader, random_split
from indexed_parquet_dataset import IndexedParquetDataset
from .tokenizer import get_tokenizer
import os

class DistillationDataset(Dataset):
    """
    Universal dataset for distillation based on IndexedParquetDataset.
    Assumes data is already pre-processed into clean text format (Clean Text Diffusion).
    """
    def __init__(self, tokenizer, data_configs, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.datasets = []
        self.total_samples = 0
        self.index_map = []
        
        current_offset = 0
        
        for config in data_configs:
            path = config['path']
            if not os.path.exists(path):
                print(f"Warning: Path {path} not found. Skipping.")
                continue
                
            pattern = config.get('pattern', '*.parquet')
            print(f" -> Loading {path} (pattern: {pattern})...")
            
            ds = IndexedParquetDataset.from_folder(path, pattern=pattern, auto_fill=True)
            
            if 'count' in config:
                n = min(config['count'], len(ds))
                ds = ds.sample(n=n)
            elif 'ratio' in config:
                n = int(len(ds) * config['ratio'])
                ds = ds.sample(n=n)
            
            self.datasets.append(ds)
            print(f"    Loaded {len(ds)} samples")
            
            self.index_map.append({
                'start': current_offset,
                'end': current_offset + len(ds),
                'ds': ds
            })
            current_offset += len(ds)
            
        self.total_samples = current_offset
        print(f"Initialized combined dataset: {self.total_samples} samples total.")

    def _apply_mapper(self, item):
        if item is None: return ""
        
        # Simple extraction of text since we moved to pre-processed clean text
        text = item.get('text', '') or ""
        
        # --- BULLETPROOF FALLBACK ---
        if not text.strip() and isinstance(item, dict):
            vals = [str(v) for k, v in item.items() if isinstance(v, str) and len(str(v)) > 10]
            if vals:
                text = "\n".join(vals)
            else:
                text = str(item)
                
        return text

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        for m in self.index_map:
            if m['start'] <= idx < m['end']:
                item = m['ds'][idx - m['start']]
                text = self._apply_mapper(item)
                break
        else:
            return None

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }

def get_dataloader(stage='awakening', batch_size=1, max_length=512, split='train', val_ratio=0.05, data_dir='data'):
    tokenizer = get_tokenizer()
    stage_capitalized = stage.capitalize() if stage.lower() in ['awakening', 'reasoning'] else stage
    stage_path = os.path.join(data_dir, stage_capitalized)
    
    dataset = None
    
    # helper to check if a specific parquet file exists in a directory
    def check_split_file(directory, split_name):
        f = os.path.join(directory, f"{split_name}.parquet")
        if os.path.exists(f) and os.path.isfile(f):
            return f
        return None

    # First, look for unified split files (train.parquet / val.parquet)
    # Check stage_path first, then fallback to data_dir
    target_dir = stage_path if os.path.exists(stage_path) else data_dir
    split_file = check_split_file(target_dir, split)
    
    if split_file:
        print(f"Found dedicated split file: {split_file}")
        configs = [{'path': target_dir, 'pattern': f"{split}.parquet"}]
        dataset = DistillationDataset(tokenizer, configs, max_length=max_length)
        val_ratio = 0  # Pre-split, no need to split again
    
    # If no split files, fallback to old logic of loading everything and random splitting
    elif os.path.exists(target_dir):
        print(f"No specific '{split}.parquet' found. Loading all parquet files from {target_dir}")
        configs = []
        for item in sorted(os.listdir(target_dir)):
            if item.endswith('.parquet'):
                configs.append({'path': target_dir, 'pattern': item})
                
        if configs:
            dataset = DistillationDataset(tokenizer, configs, max_length=max_length)
        else:
            print(f"Warning: No valid data found in {target_dir}.")
    
    # Absolute fallback to local predefined sets
    if dataset is None or len(dataset) == 0:
        print(f"Using default local configs for stage: {stage}")
        if stage == 'awakening':
            configs = [{'path': 'data/CulturaX', 'count': 100000}]
        else:
            configs = [{'path': 'data/magpie_reasoning', 'count': 100000}]
        dataset = DistillationDataset(tokenizer, configs, max_length=max_length)
        
    # Split logic
    if val_ratio > 0:
        val_size = int(len(dataset) * val_ratio)
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(
            dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        dataset = train_ds if split == 'train' else val_ds
        
    sampler = None
    shuffle = (split == 'train')
    
    import torch.distributed as dist
    try:
        import torch_xla.core.xla_model as xm
        is_xla = True
    except ImportError:
        is_xla = False

    if is_xla:
        from torch.utils.data.distributed import DistributedSampler
        try:
            world_size = xm.xrt_world_size()
            rank = xm.get_ordinal()
            if world_size > 1:
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=shuffle
                )
                shuffle = False
        except Exception as e:
            print(f"Warning: Failed to initialize DistributedSampler: {e}")

    import multiprocessing as mp
    ctx = mp.get_context('spawn') if hasattr(mp, 'get_context') else None
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        sampler=sampler,
        num_workers=4,
        multiprocessing_context=ctx,
        pin_memory=False
    )
