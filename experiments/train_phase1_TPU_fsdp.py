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

    # Оборачиваем модель в SpmdFullyShardedDataParallel
    distiller = FSDP(
        distiller,
        mesh=mesh,
        auto_wrap_policy=custom_auto_wrap_policy,
    )
    if rank == 0: print("--- [FSDP] Модель успешно обернута в XlaFullyShardedDataParallel ---")

    # Загрузка последнего чекпоинта (если есть)
    if os.path.exists("latest_checkpoint.pt"):
        ckpt = torch.load("latest_checkpoint.pt", map_location='cpu')
        # Для FSDP load_state_dict работает прозрачно, если стейт был сохранен так же (консолидирован)
        distiller.load_state_dict(ckpt['model_state_dict'], strict=False)
        if rank == 0: print(f"--- [RESUME] Чекпоинт загружен ---")

    # Настройка
    from torch.optim import AdamW
    optimizer = AdamW(filter(lambda p: p.requires_grad, distiller.parameters()), lr=5e-5)
    criterion = DistillationLoss(cos_weight=20.0)

    # Данные
    train_loader = get_dataloader(stage='reasoning', batch_size=4, max_length=4096, split='train')
    val_loader = get_dataloader(stage='reasoning', batch_size=4, max_length=4096, split='val')
    # Строго без MpDeviceLoader, иначе возникает дедлок с PyArrow при чтении Parquet!

    if rank == 0:
        try:
            wandb.init(project="BEBLaDII", name="tpu-v6e-torchrun")
        except Exception as e:
            print(f"--- [WANDB ERROR] Ошибка инициализации: {e} ---")
            print("--- Переключение W&B в offline режим ---")
            wandb.init(project="BEBLaDII", name="tpu-v6e-torchrun", mode="offline")

    # Обучение
    distiller.train()
    global_step = 0
    gradient_accumulation_steps = 8 # Глобальный батч будет 2 * 4 (ядра) * 8 = 64
    
    for epoch in range(1):
        progress_bar = tqdm(train_loader, disable=(rank != 0), desc=f"Epoch {epoch}")
        
        if rank == 0:
            print(f"--- [RANK 0] Начинаем итерации. Градиенты обновляются каждые {gradient_accumulation_steps} шагов. ---")
            
        optimizer.zero_grad()
        for batch in progress_bar:
            # Ручной перенос на устройство (без MpDeviceLoader)
            batch = {k: v.to(device) for k, v in batch.items()}
            
            if rank == 0 and global_step < 3: print(f"\n--- [RANK 0] Шаг {global_step+1}: Forward pass... ---")
            
            student_states, teacher_targets, mu, logvar = distiller(batch['input_ids'], batch['attention_mask'])
            loss, loss_metrics = criterion(student_states, teacher_targets, batch['attention_mask'], mu, logvar, beta=0.0001)
            
            # FSDP в XLA может падать при накоплении градиентов без no_sync(). 
            # Отключаем gradient_accumulation_steps для стабильности.
            loss.backward()
            
            # Для SPMD FSDP используется стандартный optimizer.step()
            optimizer.step()
            optimizer.zero_grad()
            xm.mark_step()
            
            global_step += 1

            if rank == 0:
                # Метрика лосса 
                final_loss = loss.item()
                progress_bar.set_postfix({"loss": f"{final_loss:.4f}", "step": global_step})
                
                log_dict = {
                    "train/loss": final_loss,
                    "global_step": global_step,
                    "tech/beta": 0.0001,
                    "tech/lr": 5e-5
                }
                for k, v in loss_metrics.items():
                    val = v.item() if torch.is_tensor(v) else v
                    if k.startswith("l") and ("_mse" in k or "_cos" in k):
                        log_dict[f"train/layers/{k}"] = val
                    else:
                        log_dict[f"train/{k}"] = val
                        
                wandb.log(log_dict)
                
            if global_step % 200 == 0:
                if rank == 0: print(f"\n--- [RANK 0] Валидация (Шаг {global_step}) ---")
                distiller.eval()
                val_loss_sum = 0.0
                val_mse_sum = 0.0
                val_cos_sum = 0.0
                val_kl_sum = 0.0
                val_layers_sums = {}
                val_steps = 0
                max_val_steps = 100 // 2
                
                with torch.no_grad():
                    for v_step, v_batch in enumerate(val_loader):
                        if v_step >= max_val_steps: break
                        v_batch = {k: v.to(device) for k, v in v_batch.items()}
                        v_st, v_tgt, v_mu, v_logvar = distiller(v_batch['input_ids'], v_batch['attention_mask'])
                        v_loss, v_metrics = criterion(v_st, v_tgt, v_batch['attention_mask'], v_mu, v_logvar, beta=0.0001)
                        xm.mark_step()
                        
                        val_loss_sum += v_loss.item()
                        val_mse_sum += v_metrics.get("mse", 0.0).item() if torch.is_tensor(v_metrics.get("mse", 0.0)) else v_metrics.get("mse", 0.0)
                        val_cos_sum += v_metrics.get("cosine", 0.0).item() if torch.is_tensor(v_metrics.get("cosine", 0.0)) else v_metrics.get("cosine", 0.0)
                        val_kl_sum += v_metrics.get("kl", 0.0).item() if torch.is_tensor(v_metrics.get("kl", 0.0)) else v_metrics.get("kl", 0.0)
                        
                        for k, v in v_metrics.items():
                            if k.startswith("l") and ("_mse" in k or "_cos" in k):
                                val = v.item() if torch.is_tensor(v) else v
                                val_layers_sums[k] = val_layers_sums.get(k, 0.0) + val
                        val_steps += 1
                
                if rank == 0:
                    avg_val_loss = val_loss_sum / val_steps if val_steps > 0 else 0
                    avg_val_mse = val_mse_sum / val_steps if val_steps > 0 else 0
                    avg_val_cos = val_cos_sum / val_steps if val_steps > 0 else 0
                    avg_val_kl = val_kl_sum / val_steps if val_steps > 0 else 0
                    
                    print(f"[{global_step}] Validation - Loss: {avg_val_loss:.4f}, MSE: {avg_val_mse:.4f}, Cosine: {avg_val_cos:.4f}, KL: {avg_val_kl:.6f}")
                    
                    val_log = {
                        "val/loss": avg_val_loss,
                        "val/mse": avg_val_mse,
                        "val/cosine": avg_val_cos,
                        "val/kl": avg_val_kl,
                        "global_step": global_step
                    }
                    for k, v in val_layers_sums.items():
                        val_log[f"val/layers/{k}"] = v / val_steps
                    wandb.log(val_log)
                    
                # Сохраняем веса корректно для XLA (вызывается на всех ядрах для синхронизации, пишет только rank 0)
                xm.save({'model_state_dict': distiller.state_dict()}, "latest_checkpoint.pt", master_only=True)
                
                if rank == 0:
                    # Отправляем в фон (Popen), чтобы rank 0 не отставал от остальных ядер
                    subprocess.Popen(["gsutil", "cp", "latest_checkpoint.pt", "gs://bebladii-weigths/checkpoints/"])
                    
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
    import torch.distributed as dist
    import torch_xla.distributed.xla_backend
    dist.init_process_group("xla", init_method="xla://")

    # Инициализируем XLA только после скачивания, чтобы избежать проблем с fork() и gsutil
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    
    # Синхронизация на уровне XLA (на всякий случай)
    xm.rendezvous("init_done")

    # Запускаем само обучение
    train()
