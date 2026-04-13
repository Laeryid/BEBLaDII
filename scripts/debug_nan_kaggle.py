import torch
import os
import sys
from tqdm import tqdm

# Добавляем путь к проекту
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.beb_la_dii.model.assembler import ModelAssembler
from src.beb_la_dii.utils.data import get_dataloader
from experiments.train_phase1_kaggle import setup_kaggle, build_weights_map

def debug_nan():
    print("=== DEBUG: NAN SOURCE LOCATOR ===")
    
    # 1. Настройка окружения
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resource_ds = setup_kaggle()
    weights_map = build_weights_map()
    
    # 2. Сборка системы
    print("\n[1/4] Assembling system...")
    
    # Определение путей (дублируем логику из train_phase1_kaggle для автономности дебаг-скрипта)
    KAG_RES = "/kaggle/input/datasets/bogdanbuliakov/bebladii-resources"
    VERSION = "v1.0"
    
    if os.path.exists(KAG_RES):
        student_base_id = os.path.join(KAG_RES, "prebuilt/latentBERT", VERSION)
        components_root = os.path.join(KAG_RES, "components")
    else:
        student_base_id = os.path.join("storage/prebuilt/latentBERT", VERSION)
        components_root = "storage/components"
    
    weights_map = build_weights_map() # Использует дефолт или может быть расширен
    # Но так как мы хотим Kaggle пути в weights_map тоже:
    from experiments.train_phase1_kaggle import build_weights_map as build_kag_weights
    weights_map = build_kag_weights(components_root=components_root)

    assembler = ModelAssembler()
    distiller = assembler.assemble_phase1_distiller(
        version=VERSION,
        student_base_id=student_base_id,
        weights_map=weights_map
    )
    distiller.eval() # Режим оценки для чистоты
    
    # 3. Загрузка одного батча
    print("\n[2/4] Loading real data batch...")
    dataloader = get_dataloader(stage='awakening', batch_size=2, max_length=512)
    batch = next(iter(dataloader))
    
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    print(f"Batch loaded. Sequence length: {input_ids.shape[1]}")
    
    # 4. Поэтапный запуск с замером статистики
    print("\n[3/4] Step-by-step Execution Diagnostics:")
    
    def print_stats(tensor, name):
        if tensor is None:
            print(f"  {name:25}: NONE")
            return
        t = tensor.detach().float()
        has_nan = torch.isnan(t).any().item()
        has_inf = torch.isinf(t).any().item()
        status = "!!! NAN !!!" if has_nan else ("!!! INF !!!" if has_inf else "OK")
        print(f"  {name:25}: {status} | Mean: {t.mean():.4f} | Std: {t.std():.4f} | Max: {t.max():.4f}")

    with torch.no_grad():
        # --- A. Teacher ---
        print("\n--- Phase A: Teacher (DeepSeek 4-bit) ---")
        teacher_outputs = distiller.teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        print_stats(teacher_outputs.hidden_states[0], "Teacher Embeddings")
        print_stats(teacher_outputs.hidden_states[14], "Teacher Layer 14")
        print_stats(teacher_outputs.hidden_states[21], "Teacher Layer 21")
        print_stats(teacher_outputs.hidden_states[28], "Teacher Layer 28")

        # --- B. Projector ---
        print("\n--- Phase B: InputProjector (3584 -> 1024) ---")
        t_embeds = teacher_outputs.hidden_states[0].to(distiller.student_device)
        student_in = distiller.input_projector(t_embeds)
        print_stats(student_in, "Projected Input")

        # --- C. Student ---
        print("\n--- Phase C: Student (ModernBERT DUS) ---")
        student_outputs = distiller.student(
            inputs_embeds=student_in,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        print_stats(student_outputs.hidden_states[0], "Student In-Embeds")
        print_stats(student_outputs.hidden_states[20], "Student Layer 20")
        print_stats(student_outputs.hidden_states[40], "Student Layer 40")
        print_stats(student_outputs.logits, "Student Logits (Tokens)")

        # --- D. Final Projections ---
        print("\n--- Phase D: FeatureProjectors (1024 -> 3584) ---")
        for idx in [20, 30, 40]:
            h = student_outputs.hidden_states[idx]
            proj = distiller.feature_projectors[str(idx)](h)
            print_stats(proj, f"Projected State {idx}")

    print("\n=== DEBUG COMPLETE ===")

if __name__ == "__main__":
    debug_nan()
