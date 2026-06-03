import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.beb_la_dii.model.dus import create_latentbert
from src.beb_la_dii.model.projectors import InputProjector
from transformers import AutoTokenizer

def evaluate_texts(name, texts, teacher_embedder, tokenizer, student, input_projector, device):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        teacher_embeds = teacher_embedder(input_ids)
        z, mu, logvar = input_projector(teacher_embeds.to(torch.float32))
        outputs = student(
            inputs_embeds=z,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
    
    l40_states = outputs.hidden_states[40]
    mask = attention_mask.bool()
    active_tokens = l40_states[mask]
    norms = active_tokens.norm(dim=-1)
    
    # Сравниваем biased (как в обучении) и unbiased (стандартный std)
    cv_unbiased = (norms.std(unbiased=True) / norms.mean()).item()
    cv_biased = (norms.std(unbiased=False) / norms.mean()).item()
    
    print(f"\n--- Results for: {name} ---")
    print(f"Tokens count (N)       : {active_tokens.size(0)}")
    print(f"Mean norm              : {norms.mean().item():.4f}")
    print(f"Std (biased)           : {norms.std(unbiased=False).item():.4f}")
    print(f"CV (biased / train)    : {cv_biased:.6f}")
    print(f"CV (unbiased / local)  : {cv_unbiased:.6f}")
    return cv_biased

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Загрузка Teacher Embedder
    print("Loading Teacher Embedder...")
    embedder_path = r"C:\Experiments\BEBLaDII\storage\components\model\teacher_embedder\weights.pt"
    teacher_embedder = nn.Embedding(152064, 3584, dtype=torch.bfloat16)
    teacher_embedder.load_state_dict(torch.load(embedder_path, map_location="cpu"))
    teacher_embedder.to(device).eval()
    
    # 2. Загрузка токенизатора учителя
    print("Loading Teacher Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    
    # 3. Инициализация Студента и Проектора
    print("Initializing Student and InputProjector...")
    student = create_latentbert("answerdotai/ModernBERT-large", target_layers=40)
    input_projector = InputProjector()
    
    # 4. Загрузка весов из чекпоинта
    ckpt_path = r"C:\Experiments\BEBLaDII\storage\experiments\20260601 Phase + Reasoning adr 27\checkpoints_latest_checkpoint.pt"
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
    
    # Тест 1: Английский технический текст
    texts_en = [
        "In the realm of latent space models, evaluating the variance of token norms is a crucial step.",
        "Deep learning architectures such as ModernBERT use deeply up-scaled layers to align with teacher models.",
        "The quick brown fox jumps over the lazy dog and tests the spherical manifold."
    ]
    
    # Тест 2: Русский язык (другие токенные паттерны)
    texts_ru = [
        "Проверка латентного пространства модели на сферичность является важной задачей.",
        "Мы хотим убедиться, что метрики не были искусственно подкручены для красивого отчета.",
        "Этот тест запускает полностью независимый расчет на совершенно других данных."
    ]
    
    # Тест 3: Абсолютный шум / гиббериш
    texts_noise = [
        "asdffds qwertyuuiop zxccvvbnnm 1234567890 hello world test",
        "random random random random random random random random random",
        "!!! ??? @@@ ### $$$ %%% ^^^ &&& *** ((()))"
    ]
    
    evaluate_texts("Technical English", texts_en, teacher_embedder, tokenizer, student, input_projector, device)
    evaluate_texts("Casual Russian", texts_ru, teacher_embedder, tokenizer, student, input_projector, device)
    evaluate_texts("Random Gibberish / Noise", texts_noise, teacher_embedder, tokenizer, student, input_projector, device)

if __name__ == "__main__":
    main()
