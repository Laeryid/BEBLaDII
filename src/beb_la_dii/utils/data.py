from datasets import load_dataset, concatenate_datasets
import torch
from torch.utils.data import Dataset, DataLoader
from .tokenizer import get_tokenizer

class DistillationDataset(Dataset):
    """
    Датасет для дистилляции рассуждений.
    Объединяет несколько источников и форматирует их для Qwen Tokenizer.
    """
    def __init__(self, tokenizer, max_length=512, limit=100000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print("Загрузка датасетов...")
        # 1. Magpie (Reasoning)
        magpie = load_dataset("Magpie-Align/Magpie-Reasoning-150K", split="train", streaming=True)
        
        # 2. Cosmopedia (Optional/Academic)
        # 3. CulturX (RU/CS)
        
        # Для Фазы 1 ограничимся Magpie как основным источником рассуждений
        self.data = []
        count = 0
        for item in magpie:
            # Форматируем в чат-шаблон Qwen
            # Magpie обычно имеет 'instruction' и 'response' (который содержит рассуждение)
            instruction = item.get('instruction', '')
            response = item.get('response', '')
            
            text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n<|thought|>\n{response}<|im_end|>"
            self.data.append(text)
            count += 1
            if count >= limit:
                break
                
        print(f"Загружено {len(self.data)} примеров.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
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

def get_dataloader(batch_size=1, max_length=512, limit=1000):
    tokenizer = get_tokenizer()
    dataset = DistillationDataset(tokenizer, max_length=max_length, limit=limit)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    # Тест
    loader = get_dataloader(batch_size=2, limit=10)
    for batch in loader:
        print(f"Batch input_ids shape: {batch['input_ids'].shape}")
        break
