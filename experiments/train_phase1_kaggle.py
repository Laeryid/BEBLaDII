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
    """Умная загрузка весов с обработкой префиксов и вложенных словарей"""
    if not os.path.exists(path):
        print(f"[WARN] Файл не найден: {path}")
        return False
        
    ckpt = torch.load(path, map_location='cpu')
    
    # Извлекаем state_dict если он обернут
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt: state_dict = ckpt["state_dict"]
        elif "latentBERT_state_dict" in ckpt: state_dict = ckpt["latentBERT_state_dict"]
        else: state_dict = ckpt
    else: state_dict = ckpt

    model_state = model.state_dict()
    new_state = {}
    
    # Сопоставляем ключи
    for k, v in state_dict.items():
        if k in model_state:
            new_state[k] = v
        else:
            # Пробуем убрать/добавить префиксы
            clean_key = k
            for prefix in ["student.", "distiller.", "model."]:
                if clean_key.startswith(prefix): clean_key = clean_key[len(prefix):]
            
            matched = False
            for mk in model_state.keys():
                if mk.endswith(clean_key) or clean_key.endswith(mk):
                    new_state[mk] = v
                    matched = True
                    break
    
    msg = model.load_state_dict(new_state, strict=strict)
    print(f"Загружено {len(new_state)} тензоров из {len(state_dict)}. Matching: {msg}")
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
LEARNING_RATE = 1e-5
EPOCHS = 1
WARMUP_STEPS = 200

# VAE & Validation Settings
BETA_MAX = 0.00001
best_val_loss = float('inf')

CUSTOM_STUDENT_WEIGHTS_PATH = "/kaggle/working/BEST_MODEL.pt"

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
if CUSTOM_STUDENT_WEIGHTS_PATH and os.path.exists(CUSTOM_STUDENT_WEIGHTS_PATH):
    print(f"--- Загрузка весов из {CUSTOM_STUDENT_WEIGHTS_PATH} ---")
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
criterion = DistillationLoss()
# Добавляем скейлер для Mixed Precision
scaler = torch.amp.GradScaler('cuda')
tracker = ExperimentTracker(project_root=".", stage=STAGE)

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

for epoch in range(EPOCHS):
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    accum_loss = 0.0
    accum_mse = 0.0
    accum_cosine = 0.0
    accum_kl = 0.0
    accum_metrics = {} # Для послойных метрик
    
    # Сброс градиентов перед началом эпохи
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress_bar):
        try:
            # 1. Используем distiller.student_device вместо глобального device.
            input_ids = batch['input_ids'].to(distiller.student_device).to(torch.long)
            mask = batch['attention_mask'].to(distiller.student_device)
            
            # Расчет текущей BETA (линейный отжиг)
            macro_step_total = (step + 1) // GRAD_ACCUM_STEPS
            current_beta = min(BETA_MAX, BETA_MAX * (macro_step_total / (WARMUP_STEPS * 10 or 1)))
            
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
                print(f"\n[WARN] NaN loss detected at step {step}. Skipping batch.")
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
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                macro_step = (step + 1) // GRAD_ACCUM_STEPS
                if macro_step <= WARMUP_STEPS:
                    lr_scale = macro_step / WARMUP_STEPS
                    for pg in optimizer.param_groups: 
                        pg['lr'] = LEARNING_RATE * lr_scale
                
                # 5. Unscale перед клиппингом
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(distiller.parameters(), 1.0)
                
                # 6. Безопасный шаг через скейлер
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                avg_loss = accum_loss
                avg_mse = accum_mse
                avg_kl = accum_kl
                current_lr = optimizer.param_groups[0]['lr']
                
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "kl": f"{avg_kl:.6f}", "gn": f"{grad_norm:.2f}"})
                
                if wandb.run: 
                    log_dict = {
                        "loss": avg_loss, 
                        "mse": avg_mse,
                        "cosine": accum_cosine,
                        "kl": avg_kl,
                        "beta": current_beta,
                        "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                        "step": step,
                        "lr": current_lr
                    }
                    for k, v in accum_metrics.items(): log_dict[f"train/{k}"] = v
                    wandb.log(log_dict)
                
                accum_loss = 0.0; accum_mse = 0.0; accum_cosine = 0.0; accum_kl = 0.0; accum_metrics = {}
            
            # --- ВАЛИДАЦИЯ ---
            if (step + 1) % VAL_EVERY_STEPS == 0:
                print(f"\n--- Валидация (Шаг {step+1}) ---")
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
                
                print(f"[{step+1}] Validation - Loss: {avg_val_loss:.4f}, MSE: {avg_val_mse:.4f}, Cosine: {avg_val_cos:.4f}, KL: {avg_val_kl:.6f}")
                
                if wandb.run:
                    val_log = {
                        "val/loss": avg_val_loss, 
                        "val/mse": avg_val_mse,
                        "val/cosine": avg_val_cos,
                        "val/kl": avg_val_kl,
                        "step": step
                    }
                    for k, v in val_layers_sums.items():
                        val_log[f"val/layers/{k}"] = v / val_steps
                    wandb.log(val_log)
                
                # --- СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ ---
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    print(f"[SAVING BEST] New best val_loss: {best_val_loss:.4f}. Saving BEST_MODEL...")
                    # Сохраняем весь дистиллер (единый стандарт)
                    tracker.save_checkpoint(distiller.state_dict(), name='BEST_MODEL')
                
                distiller.train()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n[OOM] Step {step}: Cleaning cache...")
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
    tracker.save_checkpoint(distiller.state_dict(), name=f"phase1_{STAGE}_epoch_{epoch}")

print("Обучение завершено!")

