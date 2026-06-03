import os
import torch
import torch.nn as nn
from transformers import AutoConfig
from huggingface_hub import hf_hub_download
import json
from safetensors.torch import load_file

def main():
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    out_dir = r"C:\Experiments\BEBLaDII\storage\components\model\teacher_embedder"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Fetching config for {model_id}...")
    config = AutoConfig.from_pretrained(model_id)
    
    # 1. Загружаем индексный файл safetensors, чтобы найти нужный фрагмент
    print("Fetching safetensors index...")
    index_path = hf_hub_download(repo_id=model_id, filename="model.safetensors.index.json")
    
    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)
        
    weight_map = index_data.get("weight_map", {})
    embed_key = "model.embed_tokens.weight"
    
    if embed_key not in weight_map:
        raise ValueError(f"Could not find {embed_key} in safetensors index.")
        
    target_file = weight_map[embed_key]
    print(f"Embeddings found in {target_file}. Downloading ONLY this chunk (saves space!)...")
    
    # 2. Скачиваем ТОЛЬКО файл с эмбеддингами (обычно ~2-4 ГБ), а не всю модель (15 ГБ)
    chunk_path = hf_hub_download(repo_id=model_id, filename=target_file)
    print(f"Loaded chunk: {chunk_path}")
    
    # 3. Извлекаем веса
    print("Extracting embedding weights...")
    tensors = load_file(chunk_path)
    embed_weight = tensors[embed_key]
    
    # 4. Создаем независимый nn.Embedding
    embedder = nn.Embedding(config.vocab_size, config.hidden_size)
    assert embedder.weight.shape == embed_weight.shape, f"Shape mismatch: {embedder.weight.shape} vs {embed_weight.shape}"
    
    embedder.weight.data.copy_(embed_weight)
    
    # Переводим в bfloat16, чтобы сэкономить место (будет ~1 ГБ) и соответствовать Phase 1/2
    embedder.to(torch.bfloat16)
    
    # 5. Сохраняем как самостоятельный компонент
    out_path = os.path.join(out_dir, "weights.pt")
    torch.save(embedder.state_dict(), out_path)
    
    config_dict = {
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "dtype": "bfloat16",
        "source_model": model_id
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
        
    print(f"Successfully saved standalone Teacher Embedder to {out_dir}")
    print(f"Component weight file size: {os.path.getsize(out_path) / (1024**2):.2f} MB")

if __name__ == "__main__":
    main()
