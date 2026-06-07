import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.beb_la_dii.model.dus import create_latentbert
from src.beb_la_dii.model.projectors import InputProjector
from transformers import AutoTokenizer

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
    
    # 2. Загрузка токенизатора учителя (чтобы токены были корректными)
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
    
    print(f"Sample keys from ckpt: {list(state_dict.keys())[:5]}")
    
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

    print(f"Extracted {len(student_dict)} keys for student, {len(proj_dict)} keys for projector.")
    
    student.load_state_dict(student_dict, strict=False)
    input_projector.load_state_dict(proj_dict, strict=False)
    
    student.to(device).eval()
    input_projector.to(device).eval()
    
    # 5. Подготовка текста
    texts = [
        "In the realm of latent space models, evaluating the variance of token norms is a crucial step.",
        "Deep learning architectures such as ModernBERT use deeply up-scaled layers to align with teacher models.",
        "The quick brown fox jumps over the lazy dog and tests the spherical manifold."
    ]
    
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    print("Running forward pass (Teacher Embeddings -> Projector -> Student)...")
    with torch.no_grad():
        # A) Эмбеддинги Учителя
        teacher_embeds = teacher_embedder(input_ids) # (B, T, 3584)
        
        # B) Проектор (возвращает z, mu, logvar)
        z, mu, logvar = input_projector(teacher_embeds.to(torch.float32))
        
        # C) Студент (используя inputs_embeds, минуя свой word_embeddings)
        outputs = student(
            inputs_embeds=z,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
    
    # 6. Расчет CV на слое 40
    l40_states = outputs.hidden_states[40] # shape: (B, T, D)
    mask = attention_mask.bool()
    
    active_tokens = l40_states[mask] # shape: (N, D)
    norms = active_tokens.norm(dim=-1)
    
    cv = norms.std() / norms.mean()
    print("\n" + "=" * 60)
    print(f"Calculated CHEAT-FREE norm_cv_l40_raw : {cv.item():.5f}")
    print(f"Mean norm                             : {norms.mean().item():.3f}")
    print(f"Std deviation of norms                : {norms.std().item():.3f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
