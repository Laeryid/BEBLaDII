import os, sys
import torch._dynamo

# Отключаем Dynamo/Inductor, так как он конфликтует с XLA (особенно в ModernBERT)
torch._dynamo.disable()
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

# Пытаемся прочитать ключ W&B из файла
if os.path.exists("/home/hp/wandb_key.txt"):
    with open("/home/hp/wandb_key.txt", "r") as f:
        _key = f.read().strip()
        if _key:
            import wandb
            wandb.login(key=_key)
            os.environ["WANDB_API_KEY"] = _key

# 1. УСТАНОВКА ПЕРЕМЕННЫХ
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["XLA_USE_BF16"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# Для v6e-4 критично указать топологию для PJRT
os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "2,2,1" 
os.environ["TPU_NUM_DEVICES"] = "4"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.runtime as xr
import torch_xla.experimental.xla_sharding as xs
from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel as FSDP
import numpy as np
import subprocess, re
from tqdm.auto import tqdm

# Включаем SPMD режим до любых операций XLA
xr.use_spmd()

# ХАК: monkey-patch torch.xla для починки gradient checkpointing
if not hasattr(torch, "xla"):
    class DummyXLA:
        @staticmethod
        def get_rng_state(*args, **kwargs): return torch.tensor(0)
        @staticmethod
        def set_rng_state(*args, **kwargs): pass
    torch.xla = DummyXLA()

# Добавляем путь к src
sys.path.append(os.getcwd())

def shard_output(output, mesh):
    # FSDP требует этот callable для не-тензорных выходов (tuple/dict)
    # Мы просто возвращаем его как есть, так как SPMD прокидывает sharding автоматически
    return None


def train():
    # Импорты модулей проекта
    from src.beb_la_dii.model.assembler import ModelAssembler
    from src.beb_la_dii.utils.loss import DistillationLoss
    from src.beb_la_dii.utils.data import get_dataloader

    # Определяем наше ядро (0, 1, 2 или 3)
    device = xm.xla_device()
    rank = xm.get_local_ordinal()
    print(f"[{rank}] Запущено на ядре: {device}")

    # Сборка модели
    assembler = ModelAssembler()
    distiller = assembler.assemble_phase1_distiller(
        teacher_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        student_base_id="answerdotai/ModernBERT-large",
        version="v1.0", weights_map={}, device_map={"": device}, student_device=device
    )
    # Настройка SPMD Mesh
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (num_devices, 1)
    device_ids = np.array(range(num_devices))
    mesh = xs.Mesh(device_ids, mesh_shape, ('fsdp', 'model'))
    xs.set_global_mesh(mesh)
    
    # Включаем градиентный чекпоинтинг для экономии памяти TPU (иначе OOM на seq_len=4096)
    # ОТКЛЮЧЕНО: GC в XLA вызывает параллельную рематериализацию 20 матриц ModernBERT!
    # if hasattr(distiller.student.model, 'gradient_checkpointing_enable'):
    #     distiller.student.model.gradient_checkpointing_enable(
    #         gradient_checkpointing_kwargs={'preserve_rng_state': False, 'use_reentrant': True}
    #     )
    #     if rank == 0:
    #         print("--- [RANK 0] Gradient Checkpointing ВКЛЮЧЕН (XLA-safe) ---")

    # Загрузка последнего чекпоинта (если есть) ДО обертки FSDP
    # Явная заморозка Учителя, чтобы XLA не строил графы активаций для него
    for name, param in distiller.named_parameters():
        if "teacher" in name:
            param.requires_grad = False

    ckpt_path = "latest_checkpoint.pt"
    if not os.path.exists(ckpt_path) and os.path.exists("AWAKENED_WEIGHTS_FINAL.pt"):
        ckpt_path = "AWAKENED_WEIGHTS_FINAL.pt"
        if rank == 0: print(f"--- [INIT] Используем базовые веса пробуждения: {ckpt_path} ---")

    ckpt = None
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        # Очищаем префикс _orig_module. и исключаем веса учителя из загрузки
        # Поддерживаем как плоский state_dict, так и вложенный в 'model_state_dict'
        raw_sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        cleaned_sd = {}
        for k, v in raw_sd.items():
            new_k = k.replace("_orig_module.", "")
            if "teacher" not in new_k:
                cleaned_sd[new_k] = v
                
        incompatible_keys = distiller.load_state_dict(cleaned_sd, strict=False)
        if rank == 0: 
            print(f"--- [RESUME] Веса загружены из {ckpt_path} ---")
            if len(incompatible_keys.missing_keys) > 0:
                print(f"--- [RESUME WARNING] Missing keys (first 10): {incompatible_keys.missing_keys[:10]}")

    # Оборачиваем модель в SpmdFullyShardedDataParallel
    def auto_wrap_policy(module, recurse, unwrapped_params, **kwargs):
        cls_name = module.__class__.__name__
        return any(name in cls_name for name in ["ModernBertLayer", "ModernBertBlock", "FeatureProjector", "InputProjector"])

    distiller = FSDP(
        distiller,
        mesh=mesh,
        auto_wrap_policy=auto_wrap_policy,
        shard_output=shard_output
    )
    if rank == 0: print("--- [FSDP] Модель успешно обернута (SPMD) ---")

    # Настройка оптимизатора и планировщика
    from transformers.optimization import Adafactor
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, distiller.parameters()), 
        lr=1e-4, scale_parameter=False, relative_step=False, warmup_init=False,
        clip_threshold=1.0
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=1, eta_min=2e-5)
    criterion = DistillationLoss(cos_weight=20.0)

    global_step = 0
    start_epoch = 0
    wandb_run_id = None

    # Восстановление состояний (ПОСЛЕ обертки FSDP)
    if ckpt:
        # 1. Оптимизатор
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                if rank == 0: print("--- [RESUME] Состояние оптимизатора восстановлено ---")
            except Exception as e:
                if rank == 0: print(f"--- [RESUME WARNING] Ошибка оптимизатора: {e} ---")
        
        # 2. Планировщик
        if 'scheduler_state_dict' in ckpt:
            try:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                if rank == 0: print("--- [RESUME] Состояние планировщика восстановлено ---")
            except Exception as e:
                if rank == 0: print(f"--- [RESUME WARNING] Ошибка планировщика: {e} ---")
        
        # 3. Шаги и Эпоха
        global_step = ckpt.get('global_step', 0)
        start_epoch = ckpt.get('epoch', 0)
        wandb_run_id = ckpt.get('wandb_run_id', None)
        if rank == 0: print(f"--- [RESUME] Продолжаем: Эпоха {start_epoch}, Шаг {global_step} ---")

    # Данные
    train_loader = get_dataloader(stage='reasoning', batch_size=4, max_length=2048, split='train', val_ratio=0.0)
    val_loader = get_dataloader(stage='reasoning', batch_size=16, max_length=2048, split='val', val_ratio=0.0)
    accumulation_steps = 4

    if rank == 0:
        wandb_kwargs = {
            "project": "BEBLaDII",
            "name": "tpu-v6e-spmd",
            "resume": "allow",
            "id": wandb_run_id
        }
        try:
            wandb.init(**wandb_kwargs)
        except Exception as e:
            print(f"--- [WANDB ERROR] Ошибка: {e}. Пробуем форсировать онлайн... ---")
            wandb.init(**wandb_kwargs, mode="online")
        
        # Сохраняем текущий ID для будущих чекпоинтов
        wandb_run_id = wandb.run.id

    # Обучение
    distiller.train()
    
    for epoch in range(start_epoch, 10):
        # Рассчитываем, сколько батчей пропустить внутри текущей эпохи
        batches_per_epoch = len(train_loader)
        batches_to_skip = global_step % batches_per_epoch if epoch == start_epoch else 0
        
        progress_bar = tqdm(train_loader, disable=(rank != 0), initial=batches_to_skip, total=batches_per_epoch, desc=f"Epoch {epoch}")
        
        if rank == 0 and batches_to_skip > 0:
            print(f"--- [RESUME] Пропускаем {batches_to_skip} батчей в эпохе {epoch}... ---")
            
        optimizer.zero_grad()
        for i, batch in enumerate(progress_bar):
            if i < batches_to_skip:
                continue
                
            # Ручной перенос на устройство и SPMD Data Parallel шардинг
            for k, v in batch.items():
                v = v.to(device)
                xs.mark_sharding(v, mesh, ('fsdp',) + (None,) * (v.dim() - 1))
                batch[k] = v
            
            student_states, teacher_targets, mu, logvar = distiller(batch['input_ids'], batch['attention_mask'])
            loss, loss_metrics = criterion(student_states, teacher_targets, batch['attention_mask'], mu, logvar, beta=0.0001)
            
            loss = loss / accumulation_steps
            loss.backward()
            
            # Очистка памяти
            del student_states, teacher_targets, mu, logvar
            
            if (global_step + 1) % accumulation_steps == 0:
                xm.optimizer_step(optimizer, barrier=True)
                scheduler.step()
                optimizer.zero_grad()
                
                # Сохранение и валидация каждые 500 шагов оптимизатора
                if ((global_step + 1) // accumulation_steps) % 500 == 0:
                    xm.mark_step()
                    if rank == 0: print(f"\n--- [RANK 0] Снапшот на шаге {global_step}... ---")
                    
                    full_sd = distiller.state_dict()
                    trainable_sd = {k: v for k, v in full_sd.items() if "teacher" not in k}
                    
                    save_data = {
                        'model_state_dict': trainable_sd,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'global_step': global_step,
                        'epoch': epoch,
                        'wandb_run_id': wandb_run_id
                    }
                    local_ckpt_name = f"ckpt_{global_step}.pt"
                    xm.save(save_data, local_ckpt_name)
                    
                    if rank == 0:
                        try:
                            import shutil
                            shutil.copy(local_ckpt_name, "latest_checkpoint.pt")
                            subprocess.run(["gsutil", "cp", local_ckpt_name, "gs://bebladii-weigths/checkpoints/"], check=True)
                            subprocess.run(["gsutil", "cp", "latest_checkpoint.pt", "gs://bebladii-weigths/checkpoints/"], check=True)
                            
                            prev_step = global_step - (500 * accumulation_steps)
                            prev_ckpt = f"ckpt_{prev_step}.pt"
                            if os.path.exists(prev_ckpt): os.remove(prev_ckpt)
                        except Exception as e:
                            print(f"--- [GCS ERROR] {e} ---")

                    # Валидация
                    distiller.eval()
                    val_loss_sum, val_steps = 0.0, 0
                    val_metrics_sums = {}
                    max_val_steps = 100
                    
                    with torch.no_grad():
                        for v_step, v_batch in enumerate(val_loader):
                            if v_step >= max_val_steps: break
                            for k, v in v_batch.items():
                                v = v.to(device)
                                xs.mark_sharding(v, mesh, ('fsdp',) + (None,) * (v.dim() - 1))
                                v_batch[k] = v
                                
                            v_st, v_tgt, v_mu, v_logvar = distiller(v_batch['input_ids'], v_batch['attention_mask'])
                            v_loss, v_metrics = criterion(v_st, v_tgt, v_batch['attention_mask'], v_mu, v_logvar, beta=0.0001)
                            
                            val_loss_sum += v_loss.item()
                            for k, val in v_metrics.items():
                                val_metrics_sums[k] = val_metrics_sums.get(k, 0.0) + (val.item() if torch.is_tensor(val) else val)
                            val_steps += 1
                            xm.mark_step()
                    
                    if rank == 0:
                        avg_val_loss = val_loss_sum / val_steps
                        val_log = {"val/loss": avg_val_loss, "global_step": global_step}
                        for k, v_sum in val_metrics_sums.items():
                            val_log[f"val/{k}"] = v_sum / val_steps
                        wandb.log(val_log, step=global_step)
                    
                    distiller.train()
            
            xm.mark_step()
            global_step += 1
            
            if rank == 0 and global_step % 20 == 0:
                lr = optimizer.param_groups[0]['lr']
                log_dict = {"train/loss": loss.item() * accumulation_steps, "train/lr": lr, "global_step": global_step}
                for k, v in loss_metrics.items():
                    log_dict[f"train/{k}"] = v.item() if torch.is_tensor(v) else v
                wandb.log(log_dict, step=global_step)
                progress_bar.set_postfix({"loss": f"{log_dict['train/loss']:.4f}", "step": global_step})

if __name__ == "__main__":
    import time
    rank = int(os.environ.get("LOCAL_RANK", 0))

    if rank == 0:
        print("--- [RANK 0] Подготовка ресурсов ---")
        os.makedirs("./data", exist_ok=True)
        subprocess.run(["gsutil", "-m", "rsync", "-r", "gs://bebladii-datasets/data/", "./data"], check=True)

        # Скачивание чекпоинтов
        try:
            # 1. Проверяем свежий чекпоинт
            res = subprocess.run(["gsutil", "ls", "gs://bebladii-weigths/checkpoints/latest_checkpoint.pt"], capture_output=True, text=True)
            if res.returncode == 0:
                subprocess.run(["gsutil", "cp", "gs://bebladii-weigths/checkpoints/latest_checkpoint.pt", "latest_checkpoint.pt"], check=True)
            
            # 2. Скачиваем базовые веса Awakening (всегда, как запасной вариант)
            if not os.path.exists("latest_checkpoint.pt"):
                print("--- [RANK 0] Свежий чекпоинт не найден, скачиваем AWAKENED_WEIGHTS_FINAL.pt ---")
                subprocess.run(["gsutil", "cp", "gs://bebladii-weigths/kaggle_upload_1_2/AWAKENED_WEIGHTS_FINAL.pt", "AWAKENED_WEIGHTS_FINAL.pt"], check=True)
        except Exception as e:
            print(f"--- [RANK 0] Ошибка загрузки весов: {e} ---")
        
        with open("/tmp/resources_prepared.flag", "w") as f: f.write("ok")
    else:
        while not os.path.exists("/tmp/resources_prepared.flag"):
            time.sleep(2)

    import torch_xla.core.xla_model as xm
    xm.rendezvous("init_done")
    train()
