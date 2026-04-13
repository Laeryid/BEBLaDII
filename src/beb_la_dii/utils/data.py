import torch
from torch.utils.data import Dataset, DataLoader
from indexed_parquet_dataset import IndexedParquetDataset
from .tokenizer import get_tokenizer
import os

class DistillationDataset(Dataset):
    """
    Universal dataset for distillation based on IndexedParquetDataset.
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
            
            self.index_map.append({
                'start': current_offset,
                'end': current_offset + len(ds),
                'ds': ds,
                'type': config['type']
            })
            current_offset += len(ds)
            
        self.total_samples = current_offset
        print(f"Initialized combined dataset: {self.total_samples} samples.")

    def _apply_mapper(self, item, dtype):
        if dtype == 'raw':
            return item.get('text', '')
        elif dtype == 'magpie':
            inst = item.get('instruction', '')
            resp = item.get('response', '')
            return f"<|im_start|>user\n{inst}<|im_end|>\n<|im_start|>assistant\n<|thought|>\n{resp}<|im_end|>"
        elif dtype == 'sharegpt':
            system = item.get('system', '')
            convs = item.get('conversations', [])
            text = ""
            if system:
                text += f"<|im_start|>system\n{system}<|im_end|>\n"
            for i, msg in enumerate(convs):
                role = "user" if msg['from'] == 'human' else "assistant"
                content = msg['value']
                text += f"<|im_start|>{role}\n"
                if role == "assistant" and i == 1: 
                    text += f"<|thought|>\n"
                text += f"{content}<|im_end|>\n"
            return text
        return str(item)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        for m in self.index_map:
            if m['start'] <= idx < m['end']:
                item = m['ds'][idx - m['start']]
                text = self._apply_mapper(item, m['type'])
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

def get_dataloader(stage='awakening', batch_size=1, max_length=512):
    tokenizer = get_tokenizer()
    # Нормализуем имя стадии для путей (Awakening / Reasoning)
    stage_capitalized = stage.capitalize() if stage.lower() in ['awakening', 'reasoning'] else stage
    stage_path = os.path.join('data', stage_capitalized)
    
    # 1. Если есть папка стадии (Kaggle или подготовленный локальный запуск)
    if os.path.exists(stage_path):
        print(f"Loading merged dataset from: {stage_path}")
        # Рекурсивно сканируем подпапки
        try:
            from indexed_parquet import IndexedParquetDataset
        except ImportError:
            from indexed_parquet_dataset import IndexedParquetDataset
            
        dataset_obj = IndexedParquetDataset.from_folder(stage_path)
        # Нам нужно обернуть это в DistillationDataset для маппинга текстов
        # Для простоты мы можем создать конфиги на лету из подпапок
        # Сканируем содержимое папки (файлы или подпапки)
        configs = []
        print(f"DEBUG DATA: Scanning directory: {stage_path}")
        try:
            items = os.listdir(stage_path)
            print(f"DEBUG DATA: Found items: {items}")
        except Exception as e:
            print(f"DEBUG DATA: Error listing directory: {e}")
            items = []
        
        for item in items:
            item_path = os.path.join(stage_path, item)
            
            # Определяем тип данных по имени
            dtype = 'raw'
            name_lower = item.lower()
            if 'magpie' in name_lower: dtype = 'magpie'
            elif 'open_thoughts' in name_lower or 'sharegpt' in name_lower: dtype = 'sharegpt'
            
            # В Kaggle это могут быть симлинки, проверяем существование и расширение
            is_parquet = item.endswith('.parquet')
            
            if is_parquet:
                # Если это файл в корне, используем родительскую папку + паттерн имени файла
                print(f"DEBUG DATA: Registering file: {item} as {dtype}")
                configs.append({'path': stage_path, 'type': dtype, 'pattern': item})
            elif os.path.isdir(item_path):
                # Если это подпапка, используем её как базовый путь
                print(f"DEBUG DATA: Registering folder: {item} as {dtype}")
                configs.append({'path': item_path, 'type': dtype})
        
        if not configs:
            print(f"Warning: No valid data found in {stage_path}. Falling back to default local configs.")
            # Используем флаг 'default', чтобы избежать бесконечной рекурсии
            return get_dataloader(stage='default', batch_size=batch_size, max_length=max_length)
            
        dataset = DistillationDataset(tokenizer, configs, max_length=max_length)
    else:
        # 2. Стандартная локальная логика
        print(f"Using default local configs for stage: {stage}")
        if stage == 'awakening':
            configs = [
                {'path': 'data/CulturaX', 'type': 'raw', 'count': 90000},
                {'path': 'data/magpie_reasoning', 'type': 'magpie', 'count': 10000}
            ]
        else:
            configs = [
                {'path': 'data/magpie_reasoning', 'type': 'magpie', 'count': 50000},
                {'path': 'data/open_thoughts', 'type': 'sharegpt', 'count': 50000}
            ]
            
        dataset = DistillationDataset(tokenizer, configs, max_length=max_length)
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
