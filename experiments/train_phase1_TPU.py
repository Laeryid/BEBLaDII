import os, sys, shutil, subprocess, re, torch, json, wandb
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm.auto import tqdm

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False

# Модули проекта
try:
    from src.beb_la_dii.model.assembler import ModelAssembler
    from src.beb_la_dii.utils.loss import DistillationLoss
    from src.beb_la_dii.utils.data import get_dataloader
    from src.beb_la_dii.utils.experiment_tracker import ExperimentTracker
    from src.beb_la_dii.utils.tokenizer import get_tokenizer
except ImportError:
    print("Ошибка: Модули проекта не найдены. Убедитесь, что вы в корне репозитория.")

# --- КОНФИГУРАЦИЯ ---
VERSION = "v1.0"
GCS_DATA_BUCKET = "gs://bebladii-datasets"
GCS_WEIGHTS_BUCKET = "gs://bebladii-weigths"
GCS_CHECKPOINT_BUCKET = "gs://bebladii-weigths"
RESOURCES_PATH = "./storage"
DATA_PATH = "./data"
TEACHER_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
BASE_MODEL_NAME = "answerdotai/ModernBERT-large"

# Гиперпараметры
MAX_LENGTH = 4096
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 1
STAGE = 'reasoning'
VAL_EVERY_STEPS = 200
VAL_MAX_SAMPLES = 100
LEARNING_RATE = 5e-5
EPOCHS = 1
WARMUP_STEPS = 50
SPOT_SAVE_EVERY = 200
BETA_MAX = 0.0001

def get_secret_safe(key_name):
    if key_name in os.environ: return os.environ[key_name]
    # Для TPU VM обычно используются переменные окружения или gcloud secrets
    return None

def smart_load_weights(model, path, strict=False):
    if not os.path.exists(path): return False
    print(f"--- [LOADER] Загрузка весов из {path} ---")
    ckpt = torch.load(path, map_location='cpu')
    flat_ckpt = {}
    if isinstance(ckpt, dict) and any(k in ckpt for k in ["latentBERT_state_dict", "state_dict", "model_state_dict"]):
        if "model_state_dict" in ckpt: flat_ckpt = ckpt["model_state_dict"]
        else:
            model_sd = ckpt.get("latentBERT_state_dict", ckpt.get("state_dict", {}))
            for k, v in model_sd.items(): flat_ckpt[f"student.{k}"] = v
        for p_name in ["input_projector", "feature_projectors"]:
            if p_name in ckpt:
                p_sd = ckpt[p_name]
                if isinstance(p_sd, dict):
                    for k, v in p_sd.items():
                        if isinstance(v, dict):
                            for inner_k, inner_v in v.items(): flat_ckpt[f"{p_name}.{k}.{inner_k}"] = inner_v
                        else: flat_ckpt[f"{p_name}.{k}"] = v
    else: flat_ckpt = ckpt
    
    model_state = model.state_dict()
    target_keys = {k: v for k, v in model_state.items() if not k.startswith("teacher.")}
    new_state = {}
    
    def clean_prefix(key):
        k = key
        for p in ["student.", "model.", "distiller.", "input_projector.", "feature_projectors."]:
            while k.startswith(p): k = k[len(p):]
        return k

    for k, v in flat_ckpt.items():
        if k in target_keys: new_state[k] = v
        else:
            ck = clean_prefix(k)
            if len(ck) <= 5: continue
            for tk in target_keys.keys():
                if tk.endswith(ck):
                    new_state[tk] = v
                    break
    model.load_state_dict(new_state, strict=strict)
    return True

def _mp_fn(index, flags):
    resume_run = flags.get('resume_run', False)
    resume_path = flags.get('resume_path', None)
    custom_student_weights_path = flags.get('custom_student_weights_path', None)
    
    best_val_loss = float('inf')
    global_step = 0

    device = xm.xla_device() if XLA_AVAILABLE else "cpu"
    print(f"[{index}] Инициализация на {device}...")

    # Сборка модели
    assembler = ModelAssembler()
    distiller = assembler.assemble_phase1_distiller(
        teacher_id=TEACHER_NAME, 
        student_base_id=BASE_MODEL_NAME, # Для TPU v6e лучше начинать с базы или пребилта
        version=VERSION,
        weights_map={}, # На TPU v6e веса грузим через smart_load_weights ниже
        device_map={"": device},
        student_device=device
    )

    if hasattr(distiller.student.model, 'gradient_checkpointing_enable'):
        distiller.student.model.gradient_checkpointing_enable()

    # Загрузка весов
    if resume_run and resume_path:
        ckpt = torch.load(resume_path, map_location='cpu')
        distiller.load_state_dict(ckpt['model_state_dict'], strict=False)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        global_step = ckpt.get('step', -1) + 1
        print(f"[{index}] Восстановлено с шага {global_step}")
    elif custom_student_weights_path:
        smart_load_weights(distiller, custom_student_weights_path)

    # Обучаемые параметры
    for p in distiller.parameters(): p.requires_grad = False
    for p in distiller.student.parameters(): p.requires_grad = True
    for p in distiller.input_projector.parameters(): p.requires_grad = True
    for proj in distiller.feature_projectors.values():
        for p in proj.parameters(): p.requires_grad = True

    train_loader = get_dataloader(stage=STAGE, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, split='train')
    val_loader = get_dataloader(stage=STAGE, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, split='val')
    
    if XLA_AVAILABLE:
        train_loader = pl.MpDeviceLoader(train_loader, device)
        val_loader = pl.MpDeviceLoader(val_loader, device)

    optimizer = AdamW(filter(lambda p: p.requires_grad, distiller.parameters()), lr=LEARNING_RATE)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=2, eta_min=1e-7)
    criterion = DistillationLoss(cos_weight=20.0)
    tracker = ExperimentTracker(project_root=".", stage=STAGE)

    if xm.is_master_ordinal():
        wandb_key = get_secret_safe("WANDB_API_KEY")
        if wandb_key: wandb.init(project="BEBLaDII", name=f"tpu-phase1-{STAGE}")

    # Цикл обучения
    distiller.train()
    for epoch in range(EPOCHS):
        progress_bar = tqdm(train_loader, disable=not xm.is_master_ordinal())
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids']
            mask = batch['attention_mask']
            
            # Linear beta warmup
            current_beta = min(BETA_MAX, BETA_MAX * (global_step / (WARMUP_STEPS or 1)))
            
            student_states, teacher_targets, mu, logvar = distiller(input_ids, mask)
            loss, metrics = criterion(student_states, teacher_targets, mask, mu, logvar, beta=current_beta)
            
            loss.backward()
            xm.optimizer_step(optimizer)
            scheduler.step()
            
            global_step += 1
            if xm.is_master_ordinal():
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step})
                if wandb.run: wandb.log({"train/loss": loss.item(), "step": global_step, "beta": current_beta})

            # Сохранение
            if global_step % SPOT_SAVE_EVERY == 0 and xm.is_master_ordinal():
                print(f"\n[SAVE] Шаг {global_step}...")
                ckpt = {'step': global_step, 'model_state_dict': distiller.state_dict(), 'best_val_loss': best_val_loss}
                torch.save(ckpt, f"RESUME_TPU_STEP_{global_step}.pt")
                subprocess.run(["gsutil", "cp", f"RESUME_TPU_STEP_{global_step}.pt", f"{GCS_CHECKPOINT_BUCKET}/checkpoints/"])

if __name__ == "__main__":
    print("--- [TPU STARTUP] Подготовка ресурсов ---")
    
    # 1. Синхронизация данных
    os.makedirs(DATA_PATH, exist_ok=True)
    subprocess.run(["gsutil", "-m", "rsync", "-r", f"{GCS_DATA_BUCKET}/", DATA_PATH], check=True)
    
    # 2. Поиск чекпоинта
    flags = {'resume_run': False, 'resume_path': None}
    try:
        res = subprocess.run(["gsutil", "ls", f"{GCS_CHECKPOINT_BUCKET}/checkpoints/RESUME_TPU_STEP_*.pt"], capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            latest = sorted(res.stdout.strip().split('\n'))[-1]
            local = os.path.basename(latest)
            subprocess.run(["gsutil", "cp", latest, local], check=True)
            flags['resume_run'] = True
            flags['resume_path'] = local
    except: pass

    # 3. Запуск
    if XLA_AVAILABLE:
        os.environ["PJRT_DEVICE"] = "TPU"
        xmp.spawn(_mp_fn, args=(flags,), nprocs=None, start_method="spawn")
    else:
        _mp_fn(0, flags)
