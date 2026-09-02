import torch
from torch.utils.data import IterableDataset, DataLoader
from indexed_parquet_dataset import IndexedParquetDataset
from .tokenizer import get_tokenizer
import os

class DistillationDataset(IterableDataset):
    """
    Universal dataset for distillation based on IndexedParquetDataset.
    Uses IterableDataset with .shard() and .shuffle(chunk_size) for multi-worker I/O optimization.
    """
    def __init__(self, tokenizer, data_configs, max_length=512, split='train', val_ratio=0.0):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.datasets = []
        self.total_samples = 0

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
                ds = ds.sample(n=n, seed=42)
            elif 'ratio' in config:
                n = int(len(ds) * config['ratio'])
                ds = ds.sample(n=n, seed=42)

            # Handle dynamic validation split if pre-split files are not used
            if val_ratio > 0:
                ds = ds.shuffle(seed=42) # Global deterministic shuffle before split
                val_size = int(len(ds) * val_ratio)
                train_size = len(ds) - val_size
                if split == 'train':
                    ds = ds.select(slice(0, train_size))
                else:
                    ds = ds.select(slice(train_size, None))

            # Apply file-group-aware chunked shuffle for training
            if split == 'train':
                ds = ds.shuffle(seed=None, rg_buffer=4)

            self.datasets.append(ds)
            print(f"    Loaded {len(ds)} samples for '{split}'")
            self.total_samples += len(ds)

        print(f"Initialized combined {split} dataset: {self.total_samples} samples total.")

    def _apply_mapper(self, item):
        if item is None: return ""
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

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        for ds in self.datasets:
            # Delegate sharding to IndexedParquetDataset for multi-worker safety
            if num_workers > 1:
                n = len(ds)
                chunk_size = (n + num_workers - 1) // num_workers
                start = worker_id * chunk_size
                end = min(start + chunk_size, n)
                sharded_ds = ds.select(slice(start, end))
            else:
                sharded_ds = ds

            for item in sharded_ds:
                text = self._apply_mapper(item)
                if not text:
                    continue

                encoding = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                yield {
                    "input_ids": encoding["input_ids"].squeeze(0),
                    "attention_mask": encoding["attention_mask"].squeeze(0)
                }

def get_dataloader(stage='awakening', batch_size=1, max_length=512, split='train', val_ratio=0.05, data_dir='data'):
    tokenizer = get_tokenizer()
    stage_capitalized = stage.capitalize() if stage.lower() in ['awakening', 'reasoning'] else stage
    stage_path = os.path.join(data_dir, stage_capitalized)

    dataset = None

    def check_split_file(directory, split_name):
        f = os.path.join(directory, f"{split_name}.parquet")
        if os.path.exists(f) and os.path.isfile(f):
            return f
        return None

    target_dir = stage_path if os.path.exists(stage_path) else data_dir
    split_file = check_split_file(target_dir, split)

    if split_file:
        print(f"Found dedicated split file: {split_file}")
        configs = [{'path': target_dir, 'pattern': f"{split}.parquet"}]
        dataset = DistillationDataset(tokenizer, configs, max_length=max_length, split=split, val_ratio=0.0)
    elif os.path.exists(target_dir):
        print(f"No specific '{split}.parquet' found. Loading all parquet files from {target_dir}")
        configs = [{'path': target_dir, 'pattern': item} for item in sorted(os.listdir(target_dir)) if item.endswith('.parquet')]
        if configs:
            dataset = DistillationDataset(tokenizer, configs, max_length=max_length, split=split, val_ratio=val_ratio)
        else:
            print(f"Warning: No valid data found in {target_dir}.")

    if dataset is None or len(dataset) == 0:
        print(f"Using default local configs for stage: {stage}")
        configs = [{'path': 'data/CulturaX', 'count': 100000}] if stage == 'awakening' else [{'path': 'data/magpie_reasoning', 'count': 100000}]
        dataset = DistillationDataset(tokenizer, configs, max_length=max_length, split=split, val_ratio=val_ratio)

    import multiprocessing as mp
    ctx = mp.get_context('spawn') if hasattr(mp, 'get_context') else None

    # Note: IterableDataset does not support shuffle=True or custom samplers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        multiprocessing_context=ctx,
        pin_memory=False,
        prefetch_factor=8,
        persistent_workers=True
    )
