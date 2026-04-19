# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # BEB-La-DII: Phase 1 Training (Reasoning)
# Этот блокнот оптимизирован для второй стадии (Reasoning). Он включает в себя:
# 1. Автоматическую настройку зеркальной структуры путей.
# 2. Инсталляцию зависимостей.
# 3. Загрузку весов Awakening из кастомного датасета.
# 4. **[NEW]** Поддержку μ-VAE головы и KL-дивергенции.
# 5. **[NEW]** Сохранение наилучшей модели по валидационному лоссу.

# %%
# 0. ПОЛУЧЕНИЕ И ОБНОВЛЕНИЕ ИСХОДНОГО КОДА
import os, sys, shutil
from pathlib import Path

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

REPO_URL = "https://github.com/Laeryid/BEBLaDII"
REPO_NAME = "BEBLaDII"

os.chdir("/kaggle/working/")

if not os.path.exists(REPO_NAME):
    print(f"Клонирование репозитория {REPO_URL}...")
    # !git clone {REPO_URL}
else:
    print(f"Репозиторий {REPO_NAME} уже присутствует. Проверка обновлений...")
    # !cd {REPO_NAME} && git pull

if os.path.exists(REPO_NAME) and REPO_NAME not in os.getcwd():
    os.chdir(REPO_NAME)
    print(f"Рабочая директория: {os.getcwd()}")

# %%
# 1. УСТАНОВКА ЗАВИСИМОСТЕЙ И ПУТЕЙ
import subprocess
def install_packages():
    packages = ["transformers==4.57.2", "indexed-parquet-dataset", "optimum-intel[openvino]", "wandb", "accelerate", "bitsandbytes", "jupytext"]
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install_packages()

root_str = os.getcwd()
if root_str not in sys.path: sys.path.insert(0, root_str)
print(f"Корень проекта: {root_str}")

# %%
import os, torch, json, wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm.auto import tqdm

try:
    from src.beb_la_dii.model.assembler import ModelAssembler
    from src.beb_la_dii.utils.loss import DistillationLoss
    from src.beb_la_dii.utils.data import get_dataloader
    from src.beb_la_dii.utils.experiment_tracker import ExperimentTracker
    from src.beb_la_dii.utils.tokenizer import get_tokenizer
    print("Модули проекта успешно импортированы.")
except ImportError as e: print(f"Ошибка импорта: {e}")

def smart_load_weights(model, path, strict=False):
    """
    Ультра-надежная загрузка весов. 
    Поддерживает плоские и вложенные структуры проекторов, 
    игнорирует teacher.* и корректно матчит student.model.model.*
    """
    if not os.path.exists(path):
        print(f"[WARN] Файл не найден: {path}")
        return False
        
    print(f"--- [LOADER] Загрузка весов из {path} ---")
    ckpt = torch.load(path, map_location='cpu')
    
    # 1. Формируем максимально плоский словарь из чекпоинта
    flat_ckpt = {}
    if isinstance(ckpt, dict) and any(k in ckpt for k in ["latentBERT_state_dict", "state_dict", "model_state_dict"]):
        print("[LOADER] Распаковка многокомпонентного чекпоинта...")
        if "model_state_dict" in ckpt:
            # Новый формат: уже содержит полные ключи 'student.', 'input_projector.' и т.д.
            flat_ckpt = ckpt["model_state_dict"]
        else:
            # Старый формат: только веса студента, нужно добавить префикс
            model_sd = ckpt.get("latentBERT_state_dict", ckpt.get("state_dict", {}))
            for k, v in model_sd.items(): flat_ckpt[f"student.{k}"] = v
        
        # Обработка проекторов (поддержка и плоских, и вложенных структур)
        for p_name in ["input_projector", "feature_projectors"]:
            if p_name in ckpt:
                p_sd = ckpt[p_name]
                if isinstance(p_sd, dict):
                    for k, v in p_sd.items():
                        # Если k - это уже словарь (вложенный), распаковываем глубже
                        if isinstance(v, dict):
                            for inner_k, inner_v in v.items():
                                flat_ckpt[f"{p_name}.{k}.{inner_k}"] = inner_v
                        else:
                            flat_ckpt[f"{p_name}.{k}"] = v
    else: flat_ckpt = ckpt

    # 2. Получаем целевые ключи модели (исключая учителя)
    model_state = model.state_dict()
    target_keys = {k: v for k, v in model_state.items() if not k.startswith("teacher.")}
    
    new_state = {}
    matched_groups = {"student": 0, "input_projector": 0, "feature_projectors": 0}

    def clean_prefix(key):
        k = key
        for p in ["student.", "model.", "distiller.", "input_projector.", "feature_projectors."]:
            while k.startswith(p): k = k[len(p):]
        return k

    # 3. Сопоставление
    for k, v in flat_ckpt.items():
        if k in target_keys: # Прямое совпадение
            new_state[k] = v
            g = k.split(".")[0]
            if g in matched_groups: matched_groups[g] += 1
            continue
            
        # Умный матчинг по суффиксу
        ck = clean_prefix(k)
        if len(ck) <= 5: continue
        
        for tk in target_keys.keys():
            if tk.endswith(ck):
                new_state[tk] = v
                g = tk.split(".")[0]
                if g in matched_groups: matched_groups[g] += 1
                break
    
    if not new_state:
        print("!!! ОШИБКА: Ни один тензор не загружен.")
        return False

    msg = model.load_state_dict(new_state, strict=strict)
    print(f"[LOADER] Загружено: Student={matched_groups['student']}, Input={matched_groups['input_projector']}, Feature={matched_groups['feature_projectors']}")
    print(f"[LOADER] Итог: {msg}")
    
    del ckpt, flat_ckpt, new_state
    torch.cuda.empty_cache()
    return True

# %%
# --- PATH CONFIGURATION ---
VERSION = "v1.0"

# Пути к датасетам на Kaggle

KAGGLE_DATA_DATASET = "/kaggle/input/datasets/bogdanbuliakov/bebladii-phase1-data"
KAGGLE_MODEL_DATASET = "/kaggle/input/datasets/bogdanbuliakov/bebladii-resources"

# Пути для обратной совместимости и локального запуска
RESOURCES_PATH = KAGGLE_MODEL_DATASET if os.path.exists(KAGGLE_MODEL_DATASET) else "./storage"
DATA_PATH = KAGGLE_DATA_DATASET if os.path.exists(KAGGLE_DATA_DATASET) else "."

KAGGLE_TEACHER_MODEL = "/kaggle/input/models/deepseek-ai/deepseek-r1/transformers/deepseek-r1-distill-qwen-7b/2"
TEACHER_NAME = KAGGLE_TEACHER_MODEL if os.path.exists(KAGGLE_TEACHER_MODEL) else "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# Hyperparameters
BASE_MODEL_NAME = "answerdotai/ModernBERT-large"
MAX_LENGTH = 4096
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
STAGE = 'reasoning'
VAL_EVERY_STEPS = 200
VAL_MAX_SAMPLES = 100
LEARNING_RATE = 5e-5
EPOCHS = 1
WARMUP_STEPS = 50

# VAE & Validation Settings
BETA_MAX = 0.0001
best_val_loss = float('inf')

RESUME_RUN = False
RESUME_PATH = "/kaggle/working/RESUME_PHASE1_STEP_X.pt"

CUSTOM_STUDENT_WEIGHTS_PATH = "/kaggle/input/datasets/bogdanbuliakov/bebladii-phase1-awakaned-weights/AWAKENED_WEIGHTS_FINAL.pt"
# CUSTOM_STUDENT_WEIGHTS_PATH = "/kaggle/working/BEBLaDII/storage/experiments/20260418_155301_reasoning/checkpoints/BEST_MODEL.pt"

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# %%
def setup_mirrored_kaggle(data_path, resources_path):
    """Настройка зеркальной структуры симлинков с принудительной перезаписью при конфликтах"""
    if not os.path.exists("/kaggle/input"): return
    print("--- Зеркальная настройка ресурсов Kaggle ---")
    
    def force_symlink(src, dst):
        if os.path.lexists(dst):
            try:
                if os.path.islink(dst): os.unlink(dst)
                elif os.path.isdir(dst): shutil.rmtree(dst)
                else: os.remove(dst)
            except Exception: pass

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            os.symlink(src, dst)
        except FileExistsError:
            # Вторая попытка, если ФС сообщила об ошибке после удаления
            try:
                if os.path.islink(dst): os.unlink(dst)
                elif os.path.isdir(dst): shutil.rmtree(dst)
                else: os.remove(dst)
                os.symlink(src, dst)
            except Exception as e: print(f"Ошибка при создании симлинка {dst}: {e}")

    # Маппинг для данных
    if os.path.exists(data_path):
        sub_data = os.path.join(data_path, "data")
        src_root = sub_data if os.path.exists(sub_data) else data_path
        os.makedirs("data", exist_ok=True)
        for item in os.listdir(src_root):
            force_symlink(os.path.join(src_root, item), os.path.join("data", item))
            print(f"Symlink (Data): data/{item} -> {item}")
    
    # Маппинг для модели/компонентов
    if os.path.exists(resources_path):
        mappings = [("components", "storage/components"), ("prebuilt", "storage/prebuilt")]
        for src_name, dst_path in mappings:
            src = os.path.join(resources_path, src_name)
            if os.path.exists(src):
                force_symlink(src, dst_path)
                print(f"Symlink (Model): {dst_path} -> {src_name}")

setup_mirrored_kaggle(DATA_PATH, RESOURCES_PATH)

# %%
# СБОРКА МОДЕЛИ И НАСТРОЙКА ГРАДИЕНТОВ
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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Инициализация на {device}...")

# Определение путей
if os.path.exists(KAGGLE_MODEL_DATASET):
    student_base_id = os.path.join(KAGGLE_MODEL_DATASET, "prebuilt/latentBERT", VERSION)
    components_root = os.path.join(KAGGLE_MODEL_DATASET, "components")
else:
    student_base_id = os.path.join("storage/prebuilt/latentBERT", VERSION)
    if not os.path.exists(student_base_id): student_base_id = BASE_MODEL_NAME
    components_root = "storage/components"

weights_map = build_weights_map(components_root=components_root)
assembler = ModelAssembler()
distiller = assembler.assemble_phase1_distiller(
    teacher_id=TEACHER_NAME, 
    student_base_id=student_base_id,
    version=VERSION,
    weights_map=weights_map,
    device_map={"": 0},
    student_device="cuda:1"
)

# Оптимизации памяти для длинных последовательностей
if hasattr(distiller.student.model, 'gradient_checkpointing_enable'):
    print('Enabling Gradient Checkpointing for Student...')
    distiller.student.model.gradient_checkpointing_enable()
    distiller.student.model.config.use_cache = False

# --- ЗАГРУЗКА КАСТОМНЫХ ВЕСОВ --- 
if RESUME_RUN:
    print("[INIT] RESUME_RUN=True: Загрузка весов отложена до инициализации оптимизатора...")
elif CUSTOM_STUDENT_WEIGHTS_PATH and os.path.exists(CUSTOM_STUDENT_WEIGHTS_PATH):
    smart_load_weights(distiller, CUSTOM_STUDENT_WEIGHTS_PATH, strict=False)

# 1. Замораживаем всё
for p in distiller.parameters(): p.requires_grad = False

# 2. Размораживаем студента и проекторы
for p in distiller.student.parameters(): p.requires_grad = True
for p in distiller.input_projector.parameters(): p.requires_grad = True
for proj in distiller.feature_projectors.values():
    for p in proj.parameters(): p.requires_grad = True

print(f"Обучаемых параметров: {sum(p.numel() for p in distiller.parameters() if p.requires_grad):,}")

# %%
# ПОДГОТОВКА ОБУЧЕНИЯ
from kaggle_secrets import UserSecretsClient

tokenizer = get_tokenizer()
train_loader = get_dataloader(stage=STAGE, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, split='train')
val_loader = get_dataloader(stage=STAGE, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, split='val')
optimizer = AdamW(filter(lambda p: p.requires_grad, distiller.parameters()), lr=LEARNING_RATE)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=2, eta_min=1e-7)
criterion = DistillationLoss(cos_weight=20.0)
# Добавляем скейлер для Mixed Precision
scaler = torch.amp.GradScaler('cuda')
tracker = ExperimentTracker(project_root=".", stage=STAGE)

global_step = 0
offset_epoch = 0

if RESUME_RUN and RESUME_PATH and os.path.exists(RESUME_PATH):
    print(f"--- [RESUME] Восстановление из {RESUME_PATH} ---")
    ckpt = torch.load(RESUME_PATH, map_location='cpu')
    
    if 'model_state_dict' in ckpt:
        distiller.load_state_dict(ckpt['model_state_dict'], strict=False)
        print("Веса модели загружены (non-strict).")
    
    if 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print("Состояние оптимизатора загружено.")
    else:
        print("[WARN] optimizer_state_dict отсутствует в чекпоинте!")
        
    if 'scaler_state_dict' in ckpt:
        try:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
            print("Состояние GradScaler загружено.")
        except Exception as e:
            print(f"[WARN] Ошибка загрузки скейлера: {e}")

    if 'scheduler_state_dict' in ckpt:
        try:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            print("Состояние планировщика загружено.")
        except Exception as e:
            print(f"[WARN] Ошибка загрузки планировщика: {e}")
            
    best_val_loss = ckpt.get('best_val_loss', float('inf'))
    global_step = ckpt.get('step', -1) + 1
    offset_epoch = ckpt.get('epoch', 0)
    
    print(f"[RESUME] Продолжаем с Эпохи {offset_epoch}, Глобальный шаг {global_step}, Best Val Loss: {best_val_loss}")
    del ckpt
    torch.cuda.empty_cache()

try:
    user_secrets = UserSecretsClient()
    wandb_key = user_secrets.get_secret("WANDB_API_KEY")
    if wandb_key: os.environ["WANDB_API_KEY"] = wandb_key
except Exception: pass

if os.environ.get("WANDB_API_KEY"):
    wandb.init(project="BEBLaDII", name=f"phase1-{STAGE}")

# %%
# ЦИКЛ ОБУЧЕНИЯ
distiller.train()
print(f"--- Запуск обучения Фазы 1 ({STAGE}) ---")

for epoch in range(offset_epoch, EPOCHS):
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    accum_loss = 0.0
    accum_mse = 0.0
    accum_cosine = 0.0
    accum_kl = 0.0
    accum_metrics = {} # Для послойных метрик
    
    # Сброс градиентов перед началом эпохи
    optimizer.zero_grad()
    
    for step_idx, batch in enumerate(progress_bar):
        global_batch_idx = global_step
        global_step += 1
        try:
            # 1. Используем distiller.student_device вместо глобального device.
            input_ids = batch['input_ids'].to(distiller.student_device).to(torch.long)
            mask = batch['attention_mask'].to(distiller.student_device)
            
            # Расчет текущей BETA (линейный отжиг)
            macro_step_total = (global_batch_idx + 1) // GRAD_ACCUM_STEPS
            current_beta = min(BETA_MAX, BETA_MAX * (macro_step_total / (WARMUP_STEPS or 1)))
            
            with torch.amp.autocast('cuda'):
                student_states, teacher_targets, mu, logvar = distiller(input_ids, mask)
                loss_mask = mask.to(distiller.student_device)
                loss, loss_metrics = criterion(
                    student_states, teacher_targets, 
                    attention_mask=loss_mask, 
                    mu=mu, logvar=logvar, beta=current_beta
                )
                loss = loss / GRAD_ACCUM_STEPS
            
            # 2. Мягкая защита от NaN.
            if torch.isnan(loss):
                print(f"\n[WARN] NaN loss detected at global step {global_batch_idx}. Skipping batch.")
                optimizer.zero_grad()
                continue
            
            # 3. Scale Backward
            scaler.scale(loss).backward()
            
            accum_loss += loss.item()
            accum_mse += loss_metrics.get('mse', 0.0) / GRAD_ACCUM_STEPS
            accum_cosine += loss_metrics.get('cosine', 0.0) / GRAD_ACCUM_STEPS
            accum_kl += loss_metrics.get('kl', 0.0) / GRAD_ACCUM_STEPS
            
            for k, v in loss_metrics.items():
                if k.startswith("l") and ("_mse" in k or "_cos" in k):
                    accum_metrics[k] = accum_metrics.get(k, 0.0) + v / GRAD_ACCUM_STEPS
            
            # 4. Шаг оптимизатора
            if (global_batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                # 5. Unscale перед клиппингом
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(distiller.parameters(), 1.0)
                
                # 6. Безопасный шаг через скейлер
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
                avg_loss = accum_loss
                avg_mse = accum_mse
                avg_kl = accum_kl
                current_lr = scheduler.get_last_lr()[0]
                
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "kl": f"{avg_kl:.6f}", "gn": f"{grad_norm:.2f}"})
                
                if wandb.run: 
                    log_dict = {
                        "train/loss": avg_loss, 
                        "train/mse": avg_mse,
                        "train/cosine": accum_cosine,
                        "train/kl": avg_kl,
                        "tech/beta": current_beta,
                        "tech/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                        "tech/step": global_batch_idx,
                        "tech/lr": current_lr,
                        "tech/scaler": scaler.get_scale()
                    }
                    for k, v in accum_metrics.items(): log_dict[f"train/layers/{k}"] = v
                    wandb.log(log_dict)
                
                accum_loss = 0.0; accum_mse = 0.0; accum_cosine = 0.0; accum_kl = 0.0; accum_metrics = {}
            
            # --- ВАЛИДАЦИЯ ---
            if (global_batch_idx + 1) % VAL_EVERY_STEPS == 0:
                print(f"\n--- Валидация (Шаг {global_batch_idx+1}) ---")
                distiller.eval()
                val_loss_sum = 0.0
                val_mse_sum = 0.0
                val_cos_sum = 0.0
                val_kl_sum = 0.0
                val_layers_sums = {}
                val_steps = 0
                max_val_steps = VAL_MAX_SAMPLES // BATCH_SIZE
                
                with torch.no_grad():
                    for v_step, v_batch in enumerate(val_loader):
                        if v_step >= max_val_steps: break
                        v_input_ids = v_batch['input_ids'].to(distiller.student_device).to(torch.long)
                        v_mask = v_batch['attention_mask'].to(distiller.student_device)
                        
                        with torch.amp.autocast('cuda'):
                            v_st, v_tgt, v_mu, v_logvar = distiller(v_input_ids, v_mask)
                            v_loss_msk = v_mask.to(distiller.student_device)
                            # Для валидации используем текущую beta
                            v_loss, v_metrics = criterion(
                                v_st, v_tgt, 
                                attention_mask=v_loss_msk, 
                                mu=v_mu, logvar=v_logvar, beta=current_beta
                            )
                        
                        val_loss_sum += v_loss.item()
                        val_mse_sum += v_metrics.get("mse", 0.0)
                        val_cos_sum += v_metrics.get("cosine", 0.0)
                        val_kl_sum += v_metrics.get("kl", 0.0)
                        
                        for k, v in v_metrics.items():
                            if k.startswith("l") and ("_mse" in k or "_cos" in k):
                                val_layers_sums[k] = val_layers_sums.get(k, 0.0) + v
                        val_steps += 1
                
                avg_val_loss = val_loss_sum / val_steps if val_steps > 0 else 0
                avg_val_mse = val_mse_sum / val_steps if val_steps > 0 else 0
                avg_val_cos = val_cos_sum / val_steps if val_steps > 0 else 0
                avg_val_kl = val_kl_sum / val_steps if val_steps > 0 else 0
                
                print(f"[{global_batch_idx+1}] Validation - Loss: {avg_val_loss:.4f}, MSE: {avg_val_mse:.4f}, Cosine: {avg_val_cos:.4f}, KL: {avg_val_kl:.6f}")
                
                if wandb.run:
                    val_log = {
                        "val/loss": avg_val_loss, 
                        "val/mse": avg_val_mse,
                        "val/cosine": avg_val_cos,
                        "val/kl": avg_val_kl,
                        "step": global_batch_idx
                    }
                    for k, v in val_layers_sums.items():
                        val_log[f"val/layers/{k}"] = v / val_steps
                    wandb.log(val_log)
                
                # --- СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ ---
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    print(f"[SAVING BEST] New best val_loss: {best_val_loss:.4f}. Saving BEST_MODEL...")
                    ckpt_dict = {
                        'epoch': epoch,
                        'step': global_batch_idx,
                        'model_state_dict': distiller.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_loss': best_val_loss,
                        'current_beta': current_beta
                    }
                    tracker.save_checkpoint(ckpt_dict, name='BEST_MODEL')
                
                distiller.train()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n[OOM] Step {global_batch_idx}: Cleaning cache...")
                for p in distiller.parameters():
                    if p.grad is not None: p.grad = None
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                optimizer.zero_grad()
                accum_loss = 0.0; accum_mse = 0.0; accum_cosine = 0.0; accum_kl = 0.0; accum_metrics = {}
                continue
            else: 
                print(f"Ошибка RuntimeError: {e}")
                continue
        except Exception as e: 
            print(f"Критическая ошибка: {e}")
            continue

    # Сохранение после каждой эпохи
    ckpt_dict = {
        'epoch': epoch,
        'step': global_step - 1,
        'model_state_dict': distiller.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'current_beta': current_beta if 'current_beta' in locals() else None 
    }
    tracker.save_checkpoint(ckpt_dict, name=f"phase1_{STAGE}_epoch_{epoch}")

print("Обучение завершено!")


# %%
# %% [markdown]
# ### Emergency Resume Checkpoint
# Эта ячейка сохраняет полное состояние для бесшовного продолжения обучения.

# %%
import torch
import os

# Путь для сохранения (корень рабочей директории Kaggle)
# Рекомендую именовать с указанием текущего шага
resume_save_path = f"/kaggle/working/RESUME_PHASE1_STEP_{global_step}.pt"

print(f"Подготовка к сохранению полного чекпоинта на шаге {global_step}...")

try:
    checkpoint = {
        'epoch': epoch,
        'step': global_step - 1,
        'model_state_dict': distiller.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        # Сохраняем и доп. параметры, если есть
        'current_beta': current_beta if 'current_beta' in locals() else None 
    }
    
    torch.save(checkpoint, resume_save_path)
    
    # Также создаем "удобную" копию без номера шага для скриптов
    # torch.save(checkpoint, "/kaggle/working/RESUME_LATEST.pt")
    
    print("="*50)
    print(f"УСПЕШНО СОХРАНЕНО: {resume_save_path}")
    print(f"Размер файла: {os.path.getsize(resume_save_path) / 1024 / 1024:.2f} MB")
    print("Теперь можно смело скачивать этот файл или фиксировать версию датасета.")
    print("="*50)
    
except Exception as e:
    print(f"Ошибка при сохранении: {e}")

