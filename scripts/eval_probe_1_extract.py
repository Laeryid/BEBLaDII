import sys
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.beb_la_dii.model.dus import create_latentbert
from src.beb_la_dii.model.projectors import InputProjector

def mean_pooling(token_embeddings, attention_mask):
    """
    Усредняет токены с учетом маски внимания.
    token_embeddings: (B, T, D)
    attention_mask: (B, T)
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Загрузка Teacher Embedder
    print("Loading Teacher Embedder...")
    embedder_path = r"C:\Experiments\BEBLaDII\storage\components\model\teacher_embedder\weights.pt"
    teacher_embedder = nn.Embedding(152064, 3584, dtype=torch.bfloat16)
    teacher_embedder.load_state_dict(torch.load(embedder_path, map_location="cpu"))
    teacher_embedder.to(device)
    teacher_embedder.eval()
    
    # 2. Загрузка токенизатора учителя
    print("Loading Teacher Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 3. Инициализация Студента и Проектора
    print("Initializing Student and InputProjector...")
    student = create_latentbert("answerdotai/ModernBERT-large", target_layers=40)
    input_projector = InputProjector()
    
    # 4. Загрузка весов из чекпоинта
    ckpt_path = r"C:\Users\hp\Downloads\latest_checkpoint.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"]
    
    student_dict = {}
    proj_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace("_orig_module.", "").replace("module.", "")
        if clean_k.startswith("student.model."):
            student_dict[clean_k.replace("student.model.", "")] = v
        elif clean_k.startswith("student."):
            student_dict[clean_k.replace("student.", "")] = v
        elif clean_k.startswith("model."):
            student_dict[clean_k.replace("model.", "", 1)] = v
        elif clean_k.startswith("input_projector."):
            proj_dict[clean_k.replace("input_projector.", "")] = v
        else:
            student_dict[clean_k] = v

    student.load_state_dict(student_dict, strict=False)
    input_projector.load_state_dict(proj_dict, strict=False)
    
    student.to(device).eval()
    input_projector.to(device).eval()
    
    # 5. Подготовка датасета
    print("Loading SNLI dataset...")
    dataset = load_dataset("snli")
    
    # Фильтруем примеры без консенсуса (label == -1)
    train_ds = dataset['train'].filter(lambda x: x['label'] != -1)
    test_ds = dataset['test'].filter(lambda x: x['label'] != -1)
    
    # Берем сабсет, чтобы не ждать часами. 50k train, 5k test достаточно для зондирования.
    train_ds = train_ds.select(range(min(50000, len(train_ds))))
    test_ds = test_ds.select(range(min(5000, len(test_ds))))
    
    def extract_features(ds, batch_size=32):
        all_features = []
        all_labels = []
        
        for i in tqdm(range(0, len(ds), batch_size), desc="Extracting"):
            batch = ds[i:i+batch_size]
            premises = batch['premise']
            hypotheses = batch['hypothesis']
            labels = batch['label']
            
            def encode_texts(texts):
                inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                
                with torch.no_grad():
                    teacher_embeds = teacher_embedder(input_ids)
                    z, _, _ = input_projector(teacher_embeds.to(torch.float32))
                    outputs = student(
                        inputs_embeds=z,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )
                    l40_states = outputs.hidden_states[40] # (B, T, 1024)
                    pooled = mean_pooling(l40_states, attention_mask)
                return pooled
            
            u = encode_texts(premises)
            v = encode_texts(hypotheses)
            
            # SentenceBERT features: [u, v, |u - v|]
            uv_diff = torch.abs(u - v)
            features = torch.cat([u, v, uv_diff], dim=1) # (B, 3072)
            
            all_features.append(features.cpu())
            all_labels.extend(labels)
            
        return torch.cat(all_features, dim=0), torch.tensor(all_labels)

    print("Extracting Train features...")
    X_train, y_train = extract_features(train_ds)
    print("Extracting Test features...")
    X_test, y_test = extract_features(test_ds)
    
    # Сохраняем на диск
    out_dir = r"C:\Experiments\BEBLaDII\storage\experiments\eval"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "snli_features.pt")
    
    torch.save({
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test
    }, out_path)
    
    print(f"✅ Features saved to {out_path}")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

if __name__ == "__main__":
    main()
