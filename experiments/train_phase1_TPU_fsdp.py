import os, sys
import torch._dynamo

# Отключаем Dynamo/Inductor, так как он конфликтует с XLA (особенно в ModernBERT)
torch._dynamo.disable()
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

# Устанавливаем ключ W&B напрямую
os.environ["WANDB_API_KEY"] = "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"

# 1. УСТАНОВКА ПЕРЕМЕННЫХ
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["XLA_USE_BF16"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
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

def custom_auto_wrap_policy(module, recurse, unwrapped_params, **kwargs):
    cls_name = module.__class__.__name__
    if any(name in cls_name for name in [
        "Qwen2DecoderLayer", "ModernBertLayer", "ModernBertBlock",
        "FeatureProjector", "InputProjector", "Qwen2RMSNorm", 
        "Embedding", "Linear", "LayerNorm"
    ]):
        return True
    return False

def custom_shard_output(output, mesh):
    import torch_xla.experimental.xla_sharding as xs
    # выход имеет вид: (student_states, teacher_targets, mu, logvar)
    student_states, teacher_targets, mu, logvar = output
    
    # Шардируем выходные состояния ученика (batch_size находится на 0-й оси)
    if isinstance(student_states, dict):
        for v in student_states.values():
            if v is not None:
                xs.mark_sharding(v, mesh, ('fsdp',) + (None,) * (v.dim() - 1))
                
    # Шардируем целевые состояния учителя
    if isinstance(teacher_targets, dict):
        for v in teacher_targets.values():
            if v is not None:
                xs.mark_sharding(v, mesh, ('fsdp',) + (None,) * (v.dim() - 1))
                
    if mu is not None:
        xs.mark_sharding(mu, mesh, ('fsdp',) + (None,) * (mu.dim() - 1))
    if logvar is not None:
        xs.mark_sharding(logvar, mesh, ('fsdp',) + (None,) * (logvar.dim() - 1))
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
    if hasattr(distiller.student.model, 'gradient_checkpointing_enable'):
        # Для XLA критически важно отключить preserve_rng_state, так как torch.utils.checkpoint
        # пытается вызвать getattr(torch, "xla"), что приводит к AttributeError.
        distiller.student.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'preserve_rng_state': False}
        )
        if rank == 0:
            print("--- [RANK 0] Gradient Checkpointing ВКЛЮЧЕН (XLA-safe) ---")

    # Загрузка последнего чекпоинта (если есть) ДО обертки FSDP
    if os.path.exists("latest_checkpoint.pt"):
        ckpt = torch.load("latest_checkpoint.pt", map_location='cpu')
        incompatible_keys = distiller.load_state_dict(ckpt['model_state_dict'], strict=False)
        if rank == 0: 
            print(f"--- [RESUME] Чекпоинт загружен ---")
            if len(incompatible_keys.missing_keys) > 0:
                print(f"--- [RESUME WARNING] Missing keys (first 10): {incompatible_keys.missing_keys[:10]}")
            if len(incompatible_keys.unexpected_keys) > 0:
                print(f"--- [RESUME WARNING] Unexpected keys (first 10): {incompatible_keys.unexpected_keys[:10]}")

    # Оборачиваем модель в SpmdFullyShardedDataParallel
    distiller = FSDP(
        distiller,
        mesh=mesh,
        auto_wrap_policy=custom_auto_wrap_policy,
        shard_output=custom_shard_output
    )
    if rank == 0: print("--- [FSDP] Модель успешно обернута в XlaFullyShardedDataParallel ---")

    # Настройка оптимизатора и планировщика (согласно ADR 002)
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, distiller.parameters()), lr=5e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=2, eta_min=1e-7)
    criterion = DistillationLoss(cos_weight=20.0)

    global_step = 0
    # Загрузка полного состояния (если есть) ДО обертки FSDP (для весов) и ПОСЛЕ создания оптимизатора
    if os.path.exists("latest_checkpoint.pt"):
        # Загружаем на CPU, чтобы не забивать HBM
        ckpt = torch.load("latest_checkpoint.pt", map_location='cpu')
        if rank == 0:
            print(f"--- [DEBUG] Ключи в чекпоинте: {list(ckpt.keys())} ---")
        
        # 1. Веса
        incompatible_keys = distiller.load_state_dict(ckpt['model_state_dict'], strict=False)
        if rank == 0: 
            print(f"--- [RESUME] Веса загружены ---")
            if incompatible_keys.missing_keys: print(f"Missing: {len(incompatible_keys.missing_keys)}")
        
        # 2. Оптимизатор
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                if rank == 0: print("--- [RESUME] Состояние оптимизатора восстановлено ---")
            except Exception as e:
                if rank == 0: print(f"--- [RESUME WARNING] Ошибка оптимизатора: {e} ---")
        
        # 3. Планировщик
        if 'scheduler_state_dict' in ckpt:
            try:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                if rank == 0: print("--- [RESUME] Состояние планировщика восстановлено ---")
            except Exception as e:
                if rank == 0: print(f"--- [RESUME WARNING] Ошибка планировщика: {e} ---")
        
        # 4. Глобальный шаг
        if 'global_step' in ckpt:
            global_step = ckpt['global_step']
            if rank == 0: print(f"--- [RESUME] Продолжаем с шага {global_step} ---")
        else:
            # Если ключа нет (бывает в Kaggle-снапшотах), ставим 5400 вручную
            global_step = 5400
            if rank == 0: print(f"--- [RESUME WARNING] 'global_step' не найден, форсируем 5400 ---")

    # Данные
    train_loader = get_dataloader(stage='reasoning', batch_size=4, max_length=4096, split='train')
    val_loader = get_dataloader(stage='reasoning', batch_size=4, max_length=4096, split='val')
    # Строго без MpDeviceLoader, иначе возникает дедлок с PyArrow при чтении Parquet!

    if rank == 0:
        try:
            wandb.init(project="BEBLaDII", name="tpu-v6e-spmd", resume="allow")
        except Exception as e:
            print(f"--- [WANDB ERROR] Ошибка инициализации: {e} ---")
            print("--- Переключение W&B в offline режим ---")
            wandb.init(project="BEBLaDII", name="tpu-v6e-spmd", mode="offline", resume="allow")

    # Обучение
    distiller.train()
    
    for epoch in range(1):
        progress_bar = tqdm(train_loader, disable=(rank != 0), desc=f"Epoch {epoch}")
        
        if rank == 0:
            print(f"--- [RANK 0] Начинаем итерации. ---")
            
        optimizer.zero_grad()
        for batch in progress_bar:
            # Ручной перенос на устройство и SPMD Data Parallel шардинг
            for k, v in batch.items():
                v = v.to(device)
                # Критично: сообщаем XLA, что батч нужно разрезать по ядрам ('fsdp')
                xs.mark_sharding(v, mesh, ('fsdp', None))
                batch[k] = v
            
            student_states, teacher_targets, mu, logvar = distiller(batch['input_ids'], batch['attention_mask'])
            loss, loss_metrics = criterion(student_states, teacher_targets, batch['attention_mask'], mu, logvar, beta=0.0001)
            
            loss.backward()
            
            # Для SPMD FSDP используется стандартный optimizer.step()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            xm.mark_step()
            
            global_step += 1
            
            if rank == 0:
                lr = optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step, "lr": f"{lr:.1e}"})
                
                if global_step % 10 == 0:
                    log_dict = {
                        "train/loss": loss.item(),
                        "train/lr": lr,
                        "global_step": global_step,
                    }
                    for k, v in loss_metrics.items():
                        val = v.item() if torch.is_tensor(v) else v
                        if k.startswith("l") and ("_mse" in k or "_cos" in k):
                            log_dict[f"train/layers/{k}"] = val
                        else:
                            log_dict[f"train/{k}"] = val
                    wandb.log(log_dict, step=global_step)
                
            # Сохранение (раз в 500 шагов, так как модель тяжелая)
            if global_step % 500 == 0:
                xm.mark_step() # Синхронизация перед сохранением
                if rank == 0: print(f"--- [RANK 0] Подготовка чекпоинта на шаге {global_step}... ---")
                
                # Сохраняем веса, оптимизатор и планировщик
                save_data = {
                    'model_state_dict': distiller.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'global_step': global_step
                }
                xm.save(save_data, "latest_checkpoint.pt") # master_only=True внутри xm.save по умолчанию для файлов
                
                if rank == 0:
                    print(f"--- [SAVE] Чекпоинт сохранен. Отправка в GCS... ---")
                    # Отправляем в фоновом режиме
                    subprocess.Popen(["gsutil", "cp", "latest_checkpoint.pt", f"gs://bebladii-weigths/checkpoints/ckpt_{global_step}.pt"])
                    subprocess.Popen(["gsutil", "cp", "latest_checkpoint.pt", "gs://bebladii-weigths/checkpoints/latest_checkpoint.pt"])

            if global_step % 500 == 0:
                if rank == 0: print(f"\n--- [RANK 0] Валидация (Шаг {global_step}) ---")
                distiller.eval()
                val_loss_sum = 0.0
                val_steps = 0
                max_val_steps = 50
                
                with torch.no_grad():
                    for v_step, v_batch in enumerate(val_loader):
                        if v_step >= max_val_steps: break
                        for k, v in v_batch.items():
                            v = v.to(device)
                            xs.mark_sharding(v, mesh, ('fsdp', None))
                            v_batch[k] = v
                            
                        v_st, v_tgt, v_mu, v_logvar = distiller(v_batch['input_ids'], v_batch['attention_mask'])
                        v_loss, _ = criterion(v_st, v_tgt, v_batch['attention_mask'], v_mu, v_logvar, beta=0.0001)
                        xm.mark_step()
                        val_loss_sum += v_loss.item()
                        val_steps += 1
                
                if rank == 0:
                    avg_val_loss = val_loss_sum / val_steps if val_steps > 0 else 0
                    print(f"[{global_step}] Validation - Loss: {avg_val_loss:.4f}")
                    wandb.log({"val/loss": avg_val_loss, "global_step": global_step})
                
                distiller.train()

if __name__ == "__main__":
    import time
    rank = int(os.environ.get("LOCAL_RANK", 0))

    if rank == 0:
        print("--- [RANK 0] Подготовка ресурсов ---")
        if not os.path.exists("src"):
            subprocess.run(["git", "init"], check=True)
            subprocess.run(["git", "remote", "add", "origin", "https://github.com/Laeryid/BEBLaDII"], check=True)
            subprocess.run(["git", "pull", "origin", "main"], check=True)
        
        os.makedirs("./data", exist_ok=True)
        subprocess.run(["gsutil", "-m", "rsync", "-r", "gs://bebladii-datasets/data/", "./data"], check=True)

        # Скачиваем чекпоинт
        try:
            # Сначала проверяем TPU чекпоинт
            res = subprocess.run(["gsutil", "ls", "gs://bebladii-weigths/checkpoints/latest_checkpoint.pt"], capture_output=True, text=True)
            if res.returncode == 0 and res.stdout.strip():
                print("--- [RANK 0] Найден TPU чекпоинт. Скачиваем... ---")
                subprocess.run(["gsutil", "cp", "gs://bebladii-weigths/checkpoints/latest_checkpoint.pt", "latest_checkpoint.pt"], check=True)
            else:
                # Если TPU чекпоинта нет, скачиваем Kaggle чекпоинт
                print("--- [RANK 0] TPU чекпоинт не найден. Ищем Kaggle чекпоинт... ---")
                res = subprocess.run(["gsutil", "ls", "gs://bebladii-weigths/kaggle_upload_1_4/RESUME_PHASE1_STEP_5400.pt"], capture_output=True, text=True)
                if res.returncode == 0 and res.stdout.strip():
                    print("--- [RANK 0] Скачиваем Kaggle чекпоинт RESUME_PHASE1_STEP_5400.pt ---")
                    subprocess.run(["gsutil", "cp", "gs://bebladii-weigths/kaggle_upload_1_4/RESUME_PHASE1_STEP_5400.pt", "latest_checkpoint.pt"], check=True)
        except Exception as e:
            print(f"--- [RANK 0] Ошибка при загрузке чекпоинта: {e} ---")
        
        # Сигнализируем, что загрузка завершена
        with open("/tmp/resources_prepared.flag", "w") as f: f.write("ok")
    else:
        # Процессы 1-3 ждут, пока процесс 0 скачает файлы, ПЕРЕД инициализацией XLA
        print(f"--- [RANK {rank}] Ожидание загрузки ресурсов... ---")
        while not os.path.exists("/tmp/resources_prepared.flag"):
            time.sleep(2)

    # Инициализируем распределенное окружение для torchrun (привязка PJRT к топологии torchrun)
    # Для SPMD FSDPv2 инициализация process_group с xla backend больше не нужна!
    # import torch.distributed as dist
    # import torch_xla.distributed.xla_backend
    # dist.init_process_group("xla", init_method="xla://")

    # Инициализируем XLA только после скачивания, чтобы избежать проблем с fork() и gsutil
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    
    # Синхронизация на уровне XLA (на всякий случай)
    xm.rendezvous("init_done")

    # Запускаем само обучение
    train()
