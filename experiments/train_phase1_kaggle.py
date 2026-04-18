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
MAX_LENGTH = 4096
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
EPOCHS = 1
LEARNING_RATE = 5e-5
STAGE = 'reasoning' # 'awakening' для Stage 1, 'reasoning' для Stage 2
VAL_EVERY_STEPS = 100  # Валидация каждые 100 шагов
VAL_MAX_SAMPLES = 1000 # Ограничение кол-ва сэмплов для валидации
VERSION = "v1.0"
WARMUP_STEPS = 1000

# Кастомный путь к весам (например, из другого датасета Kaggle)
CUSTOM_STUDENT_WEIGHTS_PATH = None 

# Путь к вашему датасету на Kaggle
KAGGLE_RESOURCES_DATASET = "/kaggle/input/bebladii-resources" 
# Если Kaggle использует полный путь с именем пользователя:
KAGGLE_RESOURCES_DATASET_ALT = "/kaggle/input/datasets/bogdanbuliakov/bebladii-resources"

# Настройка окружения
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

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

def build_weights_map(components_root="storage/components"):
    """
    Строит карту весов component_id -> path.
    Если путь не найден, компонент инициализируется случайно.
    """
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
        
    # 2. Определение путей к весам и базовым моделям
    # Приоритет Kaggle абсолютным путям (предотвращает проблемы с симлинками в transformers)
    KAG_RES = "/kaggle/input/datasets/bogdanbuliakov/bebladii-resources"
    
    if os.path.exists(KAG_RES):
        print(f"Использование прямых путей Kaggle из {KAG_RES}")
        student_base_id = os.path.join(KAG_RES, "prebuilt/latentBERT", VERSION)
        components_root = os.path.join(KAG_RES, "components")
    else:
        student_base_id = os.path.join("storage/prebuilt/latentBERT", VERSION)
        if not os.path.exists(student_base_id):
            student_base_id = BASE_MODEL_NAME
        components_root = "storage/components"

    weights_map = build_weights_map(components_root=components_root)
    print("Weights map:")
    for k, v in weights_map.items():
        status = "[found]" if os.path.exists(v) else "[random init]"
        print(f"  {k}: {status} {v}")

    assembler = ModelAssembler()
    distiller = assembler.assemble_phase1_distiller(
        teacher_id=TEACHER_NAME,
        student_base_id=student_base_id,
        version=VERSION,
        weights_map=weights_map,
        device_map={"": 0}, # Учитель строго на нулевой GPU
        student_device="cuda:1" # Студент строго на первой GPU
    )
    
    # Оптимизации памяти для длинных последовательностей
    if hasattr(distiller.student.model, "gradient_checkpointing_enable"):
        print("Enabling Gradient Checkpointing for Student...")
        distiller.student.model.gradient_checkpointing_enable()
        distiller.student.model.config.use_cache = False
    
    # 2.2 Загрузка кастомных весов, если указаны
    if CUSTOM_STUDENT_WEIGHTS_PATH and os.path.exists(CUSTOM_STUDENT_WEIGHTS_PATH):
        print(f"Загрузка кастомных весов студента из {CUSTOM_STUDENT_WEIGHTS_PATH}...")
        try:
            ckpt = torch.load(CUSTOM_STUDENT_WEIGHTS_PATH, map_location='cpu')
            if isinstance(ckpt, dict) and "latentBERT_state_dict" in ckpt:
                distiller.student.load_state_dict(ckpt["latentBERT_state_dict"])
                if "input_projector" in ckpt:
                    distiller.input_projector.load_state_dict(ckpt["input_projector"])
                if "feature_projectors" in ckpt:
                    distiller.feature_projectors.load_state_dict(ckpt["feature_projectors"])
                print("Успешно: Загружен полный чекпоинт (Student + Projectors)")
            else:
                distiller.student.load_state_dict(ckpt)
                print("Успешно: Загружен state_dict студента")
        except Exception as e:
            print(f"ОШИБКА загрузки кастомных весов: {e}")

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
    train_loader = get_dataloader(stage=STAGE, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, split='train')
    val_loader = get_dataloader(stage=STAGE, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, split='val')
    
    # Сбор метаданных датасетов для воспроизводимости
    dataset_metadata = []
    # Извлекаем оригинальный датасет из Subset
    base_dataset = train_loader.dataset.dataset if isinstance(train_loader.dataset, torch.utils.data.Subset) else train_loader.dataset
    for entry in base_dataset.index_map:
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
    progress_bar = tqdm(train_loader, desc=f"Training Phase 1 ({STAGE})")
    
    accum_loss = 0.0
    accum_mse = 0.0
    accum_cosine = 0.0
    accum_kl = 0.0
    accum_metrics = {} # Для послойных метрик
    
    best_val_loss = float('inf')
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress_bar):
        try:
            input_ids = batch["input_ids"].to(distiller.student_device)
            attention_mask = batch["attention_mask"].to(distiller.student_device)
            
            # KL Annealing: от 0.0 до 1e-4 за 500 макро-шагов
            macro_step = step // GRAD_ACCUM_STEPS
            current_beta = min(1e-4, 1e-4 * (macro_step / 500.0))

            # Форвард с автокастом для стабильности FP16
            with torch.amp.autocast('cuda'):
                # ReasoningDistiller возвращает (projected_student_states, teacher_targets, mu, logvar)
                student_states, teacher_targets, mu, logvar = distiller(input_ids, attention_mask)
                
                # Расчет лосса с учетом маски
                loss_mask = attention_mask.to(distiller.student_device)
                loss, loss_metrics = criterion(student_states, teacher_targets, attention_mask=loss_mask, mu=mu, logvar=logvar, beta=current_beta)
                
                
                # Масштабирование для накопления градиентов
                loss = loss / GRAD_ACCUM_STEPS
            
            # Проверка на NaN
            if torch.isnan(loss):
                print(f"WARN: NaN loss detected at step {step}. Skipping batch.")
                optimizer.zero_grad()
                continue

            # Бэквард с масштабированием
            scaler.scale(loss).backward()
            
            # Накопление метрик для логирования
            accum_loss += loss.item()
            accum_mse += loss_metrics["mse"] / GRAD_ACCUM_STEPS
            accum_cosine += loss_metrics["cosine"] / GRAD_ACCUM_STEPS
            if "kl" in loss_metrics: accum_kl += loss_metrics["kl"] / GRAD_ACCUM_STEPS
            
            # Накопление послойных метрик
            for k, v in loss_metrics.items():
                if k.startswith("l") and ("_mse" in k or "_cos" in k):
                    accum_metrics[k] = accum_metrics.get(k, 0.0) + v / GRAD_ACCUM_STEPS
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n[OOM] Out of memory at step {step}. Cleaning cache and skipping...")
                for p in distiller.parameters(): 
                    if p.grad is not None: p.grad = None
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                optimizer.zero_grad() # Сброс накопленных градиентов при ошибке
                accum_loss = 0.0
                accum_mse = 0.0
                accum_cosine = 0.0
                accum_kl = 0.0
                accum_metrics = {}
                continue
            else:
                raise e
        
        # Шаг оптимизатора
        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            # --- ВРУЧНУЮ WARMUP ---
            macro_step = (step + 1) // GRAD_ACCUM_STEPS
            if macro_step <= WARMUP_STEPS:
                lr_scale = macro_step / WARMUP_STEPS
                for pg in optimizer.param_groups:
                    pg['lr'] = LEARNING_RATE * lr_scale
            
            # Unscale для корректного clipping градиентов
            scaler.unscale_(optimizer)
            
            # Обрезаем градиенты и получаем норму
            grad_norm = torch.nn.utils.clip_grad_norm_(distiller.parameters(), max_norm=1.0)
            
            # Шаг через скалер
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Логирование
            avg_loss = accum_loss # accum_loss уже сумма поделенных на GRAD_ACCUM_STEPS
            avg_mse = accum_mse
            avg_cos = accum_cosine
            avg_kl = accum_kl
            current_lr = optimizer.param_groups[0]['lr']
            
            progress_bar.set_postfix({
                "loss": f"{avg_loss:.4f}", 
                "mse": f"{avg_mse:.4f}", 
                "kl": f"{avg_kl:.4f}",
                "gn": f"{grad_norm:.2f}"
            })
            
            if wandb.run:
                log_dict = {
                    "loss": avg_loss, 
                    "mse": avg_mse,
                    "cosine": avg_cos,
                    "kl": avg_kl,
                    "beta": current_beta,
                    "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                    "step": step,
                    "lr": current_lr
                }
                # Добавляем послойные метрики
                for k, v in accum_metrics.items():
                    log_dict[f"train/{k}"] = v
                    
                wandb.log(log_dict)
            
            # Локальное логирование
            tracker.log_step(step, {
                "loss": avg_loss, 
                "mse": avg_mse, 
                "cosine": avg_cos, 
                "kl": avg_kl,
                "grad_norm": grad_norm,
                "lr": current_lr
            })
            accum_loss = 0.0
            accum_mse = 0.0
            accum_cosine = 0.0
            accum_kl = 0.0
            accum_metrics = {}
            
            # --- ВАЛИДАЦИЯ ---
            if (step + 1) % VAL_EVERY_STEPS == 0:
                print(f"\n--- Running Validation (Step {step+1}) ---")
                distiller.eval()
                val_loss_sum = 0.0
                val_mse_sum = 0.0
                val_cos_sum = 0.0
                val_steps = 0
                max_val_steps = VAL_MAX_SAMPLES // BATCH_SIZE
                
                with torch.no_grad():
                    for v_step, v_batch in enumerate(val_loader):
                        if v_step >= max_val_steps: break
                        
                        v_input_ids = v_batch['input_ids'].to(distiller.student_device)
                        v_mask = v_batch['attention_mask'].to(distiller.student_device)
                        
                        # Loss mask must be on student device
                        v_loss_mask = v_mask.to(distiller.student_device)
                        
                        # Forward pass in eval mode outputs expected shapes
                        v_student_states, v_teacher_targets, v_mu, v_logvar = distiller(v_input_ids, v_mask)
                        
                        v_loss, v_metrics = criterion(v_student_states, v_teacher_targets, attention_mask=v_loss_mask, mu=v_mu, logvar=v_logvar, beta=current_beta)
                        
                        val_loss_sum += v_loss.item()
                        val_mse_sum += v_metrics["mse"].item()
                        val_cos_sum += v_metrics["cosine"].item()
                        val_steps += 1
                
                avg_val_loss = val_loss_sum / val_steps if val_steps > 0 else 0
                avg_val_mse = val_mse_sum / val_steps if val_steps > 0 else 0
                avg_val_cos = val_cos_sum / val_steps if val_steps > 0 else 0
                
                print(f"[{step+1}] Validation - Loss: {avg_val_loss:.4f}, MSE: {avg_val_mse:.4f}, Cos: {avg_val_cos:.4f}")
                
                # --- СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ ---
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    print(f"New Best Model detected! (val_loss: {best_val_loss:.4f}) Saving...")
                    best_state = {
                        "latentBERT_state_dict": distiller.student.state_dict(),
                        "input_projector": distiller.input_projector.state_dict(),
                        "feature_projectors": distiller.feature_projectors.state_dict(),
                        "config": hyperparams,
                        "step": step,
                        "val_loss": best_val_loss
                    }
                    tracker.save_checkpoint(best_state, name="phase1_best")
                    if wandb.run:
                        wandb.log({"best_val_loss": best_val_loss}, commit=False)
                
                if wandb.run:
                    wandb.log({
                        "val_loss": avg_val_loss, 
                        "val_mse": avg_val_mse,
                        "val_cosine": avg_val_cos,
                        "step": step
                    })
                
                distiller.train() # Возврат в режим обучения
            
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
