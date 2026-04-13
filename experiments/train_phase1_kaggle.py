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

# Путь к вашему датасету на Kaggle
KAGGLE_RESOURCES_DATASET = "/kaggle/input/bebladii-resources" 
# Если Kaggle использует полный путь с именем пользователя:
KAGGLE_RESOURCES_DATASET_ALT = "/kaggle/input/datasets/bogdanbuliakov/bebladii-resources"

# Настройка окружения
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def setup_kaggle():
    """Настройка путей для Kaggle: симлинки данных и пребилтов."""
    if not os.path.exists("/kaggle/input"):
        return None
        
    print("Обнаружена среда Kaggle. Настройка путей...")
    
    # 1. Поиск базовой папки ресурсов
    resource_ds = None
    input_base = "/kaggle/input"
    
    # Проверяем приоритетные пути
    for path in [KAGGLE_RESOURCES_DATASET, KAGGLE_RESOURCES_DATASET_ALT]:
        if os.path.exists(path):
            resource_ds = path
            break
            
    # Если не нашли по прямым путям — ищем перебором
    if not resource_ds:
        print(f"Поиск ресурсов в {input_base}...")
        for root, dirs, files in os.walk(input_base, topdown=True):
            if "bebladii-resources" in root.lower():
                resource_ds = root
                break
            # Ограничиваем глубину поиска для скорости
            if root.count(os.sep) - input_base.count(os.sep) > 2:
                del dirs[:] 
                
    if not resource_ds:
        print(f"WARN: Ресурсы bebladii не найдены. Доступные папки в /kaggle/input: {os.listdir(input_base)}")
        return None

    print(f"Используются ресурсы из: {resource_ds}")

    # 2. Симлинки для данных (в папку data/)
    os.makedirs("data", exist_ok=True)
    ds_data_path = os.path.join(resource_ds, "data")
    if os.path.exists(ds_data_path):
        for folder in os.listdir(ds_data_path):
            src = os.path.join(ds_data_path, folder)
            dst = os.path.join("data", folder)
            if not os.path.exists(dst):
                print(f"Symlink: {dst} -> {src}")
                os.symlink(src, dst)
    
    # 3. Симлинки для пребилтов и компонентов (в папку storage/)
    os.makedirs("storage", exist_ok=True)
    for folder in ["prebuilt", "components"]:
        src = os.path.join(resource_ds, folder)
        dst = os.path.join("storage", folder)
        if os.path.exists(src):
            if not os.path.exists(dst):
                print(f"Symlink: {dst} -> {src}")
                os.symlink(src, dst)

    if os.path.exists("data"):
        print(f"Содержимое папки data после настройки: {os.listdir('data')}")
        
    return resource_ds

def build_weights_map():
    """
    Строит карту весов component_id -> path.
    Всегда использует storage/components (на Kaggle это симлинк).
    Если файл не найден по пути, компонент инициализируется случайно.
    """
    components_root = "storage/components"

    def _w(comp_type, comp_id):
        return os.path.join(components_root, comp_type, comp_id, VERSION, "weights.pt")

    return {
        "latentBERT":         _w("model",     "latentBERT"),
        "qwen_to_bert_input": _w("projector", "qwen_to_bert_input"),
        "feat_proj_20":       _w("projector", "feat_proj_20"),
        "feat_proj_30":       _w("projector", "feat_proj_30"),
        "feat_proj_40":       _w("projector", "feat_proj_40"),
    }


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

    weights_map = build_weights_map()
    print("Weights map:")
    for k, v in weights_map.items():
        status = "[found]" if os.path.exists(v) else "[random init]"
        print(f"  {k}: {status} {v}")

    assembler = ModelAssembler()
    distiller = assembler.assemble_phase1_distiller(
        teacher_id=TEACHER_NAME,
        version=VERSION,
        weights_map=weights_map,
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
    
    # 5. Оптимизатор, Лосс и Скалер для FP16
    optimizer = AdamW(filter(lambda p: p.requires_grad, distiller.parameters()), lr=LEARNING_RATE)
    criterion = DistillationLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    # 6. Тренировочный цикл
    distiller.train()
    progress_bar = tqdm(dataloader, desc=f"Training Phase 1 ({STAGE})")
    
    accum_loss = 0.0
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Форвард с автокастом для стабильности FP16
        with torch.cuda.amp.autocast():
            # ReasoningDistiller возвращает (projected_student_states, teacher_targets)
            student_states, teacher_targets = distiller(input_ids, attention_mask)
            
            # Расчет лосса с учетом маски
            loss = criterion(student_states, teacher_targets, attention_mask=attention_mask)
            loss = loss / GRAD_ACCUM_STEPS
        
        # Проверка на NaN
        if torch.isnan(loss):
            print(f"WARN: NaN loss detected at step {step}. Skipping batch.")
            optimizer.zero_grad()
            continue

        # Бэквард с масштабированием
        scaler.scale(loss).backward()
        accum_loss += loss.item()
        
        # Шаг оптимизатора
        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            # Unscale для корректного clipping градиентов
            scaler.unscale_(optimizer)
            
            # Обрезаем градиенты для стабильности в 40-слойной модели
            torch.nn.utils.clip_grad_norm_(distiller.parameters(), max_norm=1.0)
            
            # Шаг через скалер
            scaler.step(optimizer)
            scaler.update()
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
