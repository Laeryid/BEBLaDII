import os
import torch
from torch.optim import AdamW
from tqdm import tqdm
import wandb
import shutil

# Импорты проекта
from src.beb_la_dii.model.distiller import ReasoningDistiller
from src.beb_la_dii.model.assembler import ModelAssembler
from src.beb_la_dii.utils.tokenizer import get_tokenizer
from src.beb_la_dii.utils.loss import DistillationLoss
from src.beb_la_dii.utils.data import get_dataloader
from src.beb_la_dii.utils.experiment_tracker import ExperimentTracker

# Константы
BASE_MODEL_NAME = "answerdotai/ModernBERT-large"
TEACHER_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MAX_LENGTH = 512
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
EPOCHS = 1
LEARNING_RATE = 5e-5
STAGE = 'awakening' # 'awakening' для Stage 1, 'reasoning' для Stage 2
VERSION = "v1.0"

# Настройка окружения
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def setup_kaggle():
    """Настройка путей для Kaggle: симлинки данных из Input в ./data/"""
    if not os.path.exists("/kaggle/input"):
        return None
        
    print("Обнаружена среда Kaggle. Настройка путей...")
    
    # Создаем папку data в текущей рабочей директории
    os.makedirs("data", exist_ok=True)
    
    # Ищем наш датасет с ресурсами
    input_base = "/kaggle/input"
    resource_ds = None
    
    for ds_name in os.listdir(input_base):
        if "bebladii" in ds_name.lower():
            resource_ds = os.path.join(input_base, ds_name)
            break
            
    if not resource_ds:
        print("WARN: Датасет bebladii-resources не найден в /kaggle/input")
        return None

    # 1. Симлинки для данных
    ds_data_path = os.path.join(resource_ds, "data")
    if os.path.exists(ds_data_path):
        for folder in os.listdir(ds_data_path):
            src = os.path.join(ds_data_path, folder)
            dst = os.path.join("data", folder)
            if not os.path.exists(dst):
                print(f"Создание симлинка: {dst} -> {src}")
                os.symlink(src, dst)
    
    if os.path.exists("data"):
        print(f"Содержимое папки data после настройки: {os.listdir('data')}")
        
    return resource_ds

def train():
    # 0. Автоматическая настройка для Kaggle
    resource_ds = setup_kaggle()
    
    # 1. WandB Log In
    if os.environ.get("WANDB_API_KEY"):
        wandb.init(project="beb-la-dii-phase1", name=f"latentbert-{STAGE}-{VERSION}")
    else:
        print("WANDB_API_KEY не найден, логирование в wandb отключено.")
        
    # 2. Инициализация моделей через Assembler
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Инициализация системы через Assembler на {device}...")
    
    # Реестр в составе Ассемблера теперь сам поищет веса в resource_ds
    alt_roots = [resource_ds] if resource_ds else None
    assembler = ModelAssembler(alt_roots=alt_roots)
    
    distiller = assembler.assemble_phase1_distiller(
        teacher_id=TEACHER_NAME,
        version=VERSION,
        device_map="auto"
    )
    
    # 2.5 Инициализация трекера экспериментов
    hyperparams = {
        "base_model": BASE_MODEL_NAME,
        "teacher": TEACHER_NAME,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "target_layers": 40,
        "version": VERSION
    }
    tracker = ExperimentTracker(project_root=".", stage=STAGE)
    # 3. Настройка градиентов
    # Замораживаем всё, затем включаем нужное
    for param in distiller.parameters():
        param.requires_grad = False
        
    # Включаем градиенты для latentBERT и Проекторов
    for param in distiller.student.parameters():
        param.requires_grad = True
        
    for param in distiller.input_projector.parameters():
        param.requires_grad = True
        
    # Feature projectors хранятся в nn.ModuleDict в ReasoningDistiller
    for proj in distiller.feature_projectors.values():
        for param in proj.parameters():
            param.requires_grad = True
            
    print(f"Обучаемых параметров: {sum(p.numel() for p in distiller.parameters() if p.requires_grad):,}")
    
    # 4. Подготовка данных
    print(f"Подготовка данных для стадии: {STAGE}...")
    dataloader = get_dataloader(stage=STAGE, batch_size=BATCH_SIZE, max_length=MAX_LENGTH)
    
    # Сбор метаданных датасетов для воспроизводимости
    dataset_metadata = []
    for entry in dataloader.dataset.index_map:
        ds = entry['ds']
        ds_info = {
            "type": entry['type'],
            "start": entry['start'],
            "end": entry['end'],
            "count": entry['end'] - entry['start'],
            "internal_index": ds.index if hasattr(ds, 'index') else None,
            "internal_indices": ds.indices if hasattr(ds, 'indices') else None,
        }
        dataset_metadata.append(ds_info)
    
    tracker.log_metadata(hyperparams, dataset_metadata)
    
    # 5. Оптимизатор и Лосс
    optimizer = AdamW(filter(lambda p: p.requires_grad, distiller.parameters()), lr=LEARNING_RATE)
    criterion = DistillationLoss()
    
    # 6. Тренировочный цикл
    distiller.train()
    progress_bar = tqdm(dataloader, desc=f"Training Phase 1 ({STAGE})")
    
    accum_loss = 0.0
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Форвард
        # ReasoningDistiller возвращает (projected_student_states, teacher_targets)
        student_states, teacher_targets = distiller(input_ids, attention_mask)
        
        # Расчет лосса
        loss = criterion(student_states, teacher_targets)
        loss = loss / GRAD_ACCUM_STEPS
        
        # Бэквард
        loss.backward()
        accum_loss += loss.item()
        
        # Шаг оптимизатора
        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            # Обрезаем градиенты для стабильности в 40-слойной модели
            torch.nn.utils.clip_grad_norm_(distiller.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Логирование
            avg_loss = accum_loss * GRAD_ACCUM_STEPS
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
            if wandb.run:
                wandb.log({
                    "loss": avg_loss, 
                    "step": step,
                    "lr": LEARNING_RATE
                })
            
            # Локальное логирование
            tracker.log_step(step, {"loss": avg_loss, "lr": LEARNING_RATE})
            accum_loss = 0.0
            
    # 7. Сохранение результатов
    print(f"Сохранение финальных результатов через трекер...")
    state_to_save = {
        "latentBERT_state_dict": distiller.student.state_dict(),
        "input_projector": distiller.input_projector.state_dict(),
        "feature_projectors": distiller.feature_projectors.state_dict(),
        "config": hyperparams
    }
    
    # Сохраняем финальный чекпоинт с состоянием оптимизатора
    tracker.save_checkpoint(
        state_to_save, 
        optimizer_state=optimizer.state_dict(), 
        name=f"latentbert_{STAGE}_final"
    )
    
    tracker.finish()
    print(f"Дистилляция стадии {STAGE} успешно завершена!")

if __name__ == "__main__":
    train()
