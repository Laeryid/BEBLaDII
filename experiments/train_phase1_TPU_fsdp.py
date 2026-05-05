import os, sys
import torch._dynamo

# Отключаем Dynamo/Inductor, так как он конфликтует с XLA (особенно в ModernBERT)
torch._dynamo.disable()
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

# Устанавливаем ключ W&B напрямую
os.environ["WANDB_API_KEY"] = "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"
os.environ["WANDB_START_METHOD"] = "thread"

# 1. УСТАНОВКА ПЕРЕМЕННЫХ
os.environ["PJRT_DEVICE"] = "TPU"
# os.environ["XLA_USE_BF16"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
wandb.require("core")
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
        "FeatureProjector", "InputProjector"
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

def debug_model_norms(model, tag, rank):
    """Выводит нормы весов для отладки загрузки и FSDP."""
    if rank != 0: return
    # Получаем state_dict (в FSDP это может вызвать gather, но для пары слоев это не критично)
    sd = model.state_dict()
    
    # Ищем веса через фильтрацию
    layers_to_check = [20, 30, 39]
    print(f"--- [DEBUG NORMS] {tag} ---")
    
    for l_idx in layers_to_check:
        # MLP Weight
        wo_keys = [k for k in sd.keys() if f"student.model.layers.{l_idx}.mlp.Wo.weight" in k]
        wo_norm = torch.norm(sd[wo_keys[0]].float()).item() if wo_keys else -1.0
        
        # LayerNorm Weight (Gamma)
        ln_keys = [k for k in sd.keys() if f"student.model.layers.{l_idx}.attn_norm.weight" in k]
        ln_norm = torch.norm(sd[ln_keys[0]].float()).item() if ln_keys else -1.0
        
        print(f"Layer {l_idx}: Wo Norm = {wo_norm:.2f}, LN Norm = {ln_norm:.2f} (Ref: 32.0)")

    # Input Projector
    ip_keys = [k for k in sd.keys() if "input_projector.proj.0.weight" in k]
    ip_norm = torch.norm(sd[ip_keys[0]].float()).item() if ip_keys else -1.0
    print(f"InputProjector Norm: {ip_norm:.2f}")

    # Feature Projector
    p_keys = [k for k in sd.keys() if "feature_projectors.40.proj.2.weight" in k]
    p_norm = torch.norm(sd[p_keys[0]].float()).item() if p_keys else -1.0
    print(f"Projector L40 Norm: {p_norm:.2f}")

def train():
    # Импорты модулей проекта
    from src.beb_la_dii.model.assembler import ModelAssembler
    from src.beb_la_dii.utils.loss import DistillationLoss
    from src.beb_la_dii.utils.data import get_dataloader

    # Определяем наше ядро (0, 1, 2 или 3)
    device = xm.xla_device()
    rank = xm.get_local_ordinal()
    print(f"[{rank}] Запущено на ядре: {device}")

    # Сборка модели с использованием локального 40-слойного пребилта
    student_prebuilt = "storage/prebuilt/latentBERT/v1.0"
    assembler = ModelAssembler()
    distiller = assembler.assemble_phase1_distiller(
        teacher_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        student_base_id=student_prebuilt,
        version="v1.0", weights_map=None, device_map={"": device}, student_device=device
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

    # Оборачиваем модель в SpmdFullyShardedDataParallel
    # 0. Нормы ДО загрузки
    debug_model_norms(distiller, "BEFORE LOADING", rank)

    if os.path.exists("latest_checkpoint.pt"):
        ckpt = torch.load("latest_checkpoint.pt", map_location='cpu')
        
        # Очищаем префиксы и приводим к единому формату .model
        cleaned_sd = {}
        for k, v in ckpt['model_state_dict'].items():
            new_k = k.replace("_orig_module.", "")
            # Схлопываем двойной .model.model в одинарный .model (совместимость с Kaggle 5400)
            new_k = new_k.replace("student.model.model.", "student.model.")
            
            if "teacher" not in new_k:
                cleaned_sd[new_k] = v
                
        incompatible_keys = distiller.load_state_dict(cleaned_sd, strict=False)
        if rank == 0: 
            print(f"--- [RESUME] Чекпоинт загружен (Robust key mapping applied) ---")
            if len(incompatible_keys.missing_keys) > 0:
                print(f"--- [RESUME WARNING] Missing keys (first 10): {incompatible_keys.missing_keys[:10]}")

        # 1. Нормы ПОСЛЕ загрузки (CPU state)
        debug_model_norms(distiller, "AFTER LOADING (CPU)", rank)

    # Оборачиваем модель в SpmdFullyShardedDataParallel
    distiller = FSDP(
        distiller,
        mesh=mesh,
        auto_wrap_policy=custom_auto_wrap_policy,
        shard_output=custom_shard_output
    )
    if rank == 0: print("--- [FSDP] Модель успешно обернута в XlaFullyShardedDataParallel ---")
    
    # 2. Нормы ПОСЛЕ FSDP (Sharded/TPU state)
    debug_model_norms(distiller, "AFTER FSDP WRAP (TPU)", rank)

    # Настройка оптимизатора и планировщика (согласно ADR 002)
    from transformers.optimization import Adafactor
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    
    # Используем Adafactor для HBM оптимизации
    # clip_threshold=1.0 заменяет внешний clip_grad_norm_, разгружая компилятор XLA
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, distiller.parameters()), 
        lr=5e-5, scale_parameter=False, relative_step=False, warmup_init=False,
        clip_threshold=1.0
    )
    # Сохраняем начальный LR для корректного локального вармапа
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = 5e-5

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=2, eta_min=1e-7)
    criterion = DistillationLoss(cos_weight=20.0)

    global_step = 0
    # Загрузка состояния оптимизатора и планировщика (ПОСЛЕ обертки FSDP)
    if os.path.exists("latest_checkpoint.pt"):
        # Используем уже загруженный в память ckpt (или загружаем заново, если он был удален)
        ckpt = torch.load("latest_checkpoint.pt", map_location='cpu')
        if rank == 0:
            print(f"--- [DEBUG] Ключи в чекпоинте: {list(ckpt.keys())} ---")
        
        # Веса загружать здесь НЕЛЬЗЯ, так как модель уже обернута в FSDP, 
        # и это сломает шардирование (xs.mark_sharding), что приведет к OOM!
        
        # 2. Оптимизатор
        if 'optimizer_state_dict' in ckpt:
            try:
                # Временно отключаем загрузку оптимизатора при смене AdamW -> Adafactor
                # optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                if rank == 0: print("--- [RESUME WARNING] Загрузка оптимизатора пропущена (переход на Adafactor) ---")
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

    # Данные (Используем SPMD: батч 4 будет разрезан на 4 ядра по 1 примеру)
    train_loader = get_dataloader(stage='reasoning', batch_size=4, max_length=4096, split='train')
    val_loader = get_dataloader(stage='reasoning', batch_size=4, max_length=4096, split='val')
    accumulation_steps = 1
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
    
    # Собираем список "взрывоопасных" параметров ОДИН РАЗ перед циклом
    explosive_params = []
    for name, param in distiller.named_parameters():
        if param.requires_grad:
            # Студент (слои после 30) или проектор 40-го слоя
            if (".layers." in name and any(f".layers.{i}." in name for i in range(31, 40))) or \
                ("feature_projectors.40" in name):
                explosive_params.append(param)
    
    if rank == 0:
        print(f"--- [INFO] Выборочный клиппинг включен для {len(explosive_params)} тензоров (слои > 30) ---")

    local_step = 0
    warmup_steps = 200
    
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
                xs.mark_sharding(v, mesh, ('fsdp',) + (None,) * (v.dim() - 1))
                batch[k] = v
            
            student_states, teacher_targets, mu, logvar = distiller(batch['input_ids'], batch['attention_mask'])
            loss, loss_metrics = criterion(student_states, teacher_targets, batch['attention_mask'], mu, logvar, beta=0.0001)
            
            # Делим лосс на шаги накопления
            loss = loss / accumulation_steps
            
            # ВАЖНО: Удаляем огромные тензоры до градиентного шага и компиляции графа!
            del student_states, teacher_targets, mu, logvar
            import gc; gc.collect()
            
            loss.backward()
            
            if (global_step + 1) % accumulation_steps == 0:
                # ВЫБОРОЧНЫЙ КЛИППИНГ (для обхода RESOURCE_EXHAUSTED sflag)
                if explosive_params:
                    torch.nn.utils.clip_grad_norm_(explosive_params, 1.0)
                
                # Шаг оптимизатора (с встроенным глобальным clip_threshold=1.0)
                xm.optimizer_step(optimizer, barrier=True)
                scheduler.step()
                
                # Локальный warmup множитель для предотвращения Optimizer Shock (ADR 002 update)
                # Даже если мы продолжаем с шага 14500, нам нужен плавный вход на первые 200 локальных шагов
                if local_step < warmup_steps:
                    lr_multiplier = (local_step + 1) / warmup_steps
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group.get('initial_lr', 5e-5) * lr_multiplier
                
                optimizer.zero_grad()
                local_step += 1
            
            # Извлекаем скаляры, чтобы не держать тензоры в HBM как выходы графа
            if rank == 0 and (global_step + 1) % 10 == 0:
                loss_scalars = {k: (v.item() if torch.is_tensor(v) else v) for k, v in loss_metrics.items()}
                loss_val = loss.item()
            else:
                loss_scalars = {}
                loss_val = loss.item() if rank == 0 else 0.0
                
            del loss_metrics, loss
            
            xm.mark_step()
            
            global_step += 1
            
            if rank == 0:
                lr = optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({"loss": f"{loss_val:.4f}", "step": global_step, "lr": f"{lr:.1e}"})
                
                if global_step % 10 == 0:
                    log_dict = {
                        "train/loss": loss_val,
                        "train/lr": lr,
                        "global_step": global_step,
                    }
                    for k, val in loss_scalars.items():
                        if k.startswith("l") and ("_mse" in k or "_cos" in k):
                            log_dict[f"train/layers/{k}"] = val
                        else:
                            log_dict[f"train/{k}"] = val
                    wandb.log(log_dict, step=global_step)
                
            # Сохранение (раз в 500 шагов, так как модель тяжелая)
            if global_step % 500 == 0:
                xm.mark_step() # Синхронизация перед сохранением
                if rank == 0: print(f"--- [RANK 0] Подготовка чекпоинта на шаге {global_step}... ---")
                
                # Сохраняем ТОЛЬКО веса Ученика (чекпоинт ~1ГБ вместо 15ГБ)
                full_sd = distiller.state_dict()
                trainable_sd = {k: v for k, v in full_sd.items() if "teacher" not in k}
                
                save_data = {
                    'model_state_dict': trainable_sd,
                    'scheduler_state_dict': scheduler.state_dict(),
                    'global_step': global_step
                }
                local_ckpt_name = f"ckpt_{global_step}.pt"
                xm.save(save_data, local_ckpt_name)
                
                if rank == 0:
                    import shutil
                    # Обновляем latest_checkpoint.pt локально
                    shutil.copy(local_ckpt_name, "latest_checkpoint.pt")
                    
                    # Удаляем предыдущий локальный чекпоинт, чтобы не забивать диск TPU
                    prev_ckpt = f"ckpt_{global_step - 500}.pt"
                    if os.path.exists(prev_ckpt):
                        os.remove(prev_ckpt)
                        
                    print(f"--- [SAVE] Легкий чекпоинт {local_ckpt_name} сохранен. Отправка в GCS... ---")
                    # Отправляем синхронно (без Popen), чтобы избежать зависаний XLA процессов
                    subprocess.run(f"gsutil -m cp {local_ckpt_name} gs://bebladii-weigths/checkpoints/ && gsutil -m cp {local_ckpt_name} gs://bebladii-weigths/checkpoints/latest_checkpoint.pt", shell=True, check=False)

            if global_step % 500 == 0:
                if rank == 0: print(f"\n--- [RANK 0] Валидация (Шаг {global_step}) ---")
                distiller.eval()
                val_loss_sum = None
                val_metrics_sums = {}
                val_steps = 0
                max_val_steps = 50
                
                with torch.no_grad():
                    for v_step, v_batch in enumerate(val_loader):
                        if v_step >= max_val_steps: break
                        for k, v in v_batch.items():
                            v = v.to(device)
                            xs.mark_sharding(v, mesh, ('fsdp',) + (None,) * (v.dim() - 1))
                            v_batch[k] = v
                            
                        v_st, v_tgt, v_mu, v_logvar = distiller(v_batch['input_ids'], v_batch['attention_mask'])
                        v_loss, v_metrics = criterion(v_st, v_tgt, v_batch['attention_mask'], v_mu, v_logvar, beta=0.0001)
                        
                        # Аккумулируем тензоры напрямую на TPU (без блокирующих .item())
                        v_loss_det = v_loss.detach()
                        if val_loss_sum is None:
                            val_loss_sum = v_loss_det
                        else:
                            val_loss_sum += v_loss_det
                            
                        for k, v in v_metrics.items():
                            v_det = v.detach() if torch.is_tensor(v) else torch.tensor(v, device=device)
                            if k not in val_metrics_sums:
                                val_metrics_sums[k] = v_det
                            else:
                                val_metrics_sums[k] += v_det
                        
                        # Удаляем тензоры до компиляции
                        del v_st, v_tgt, v_mu, v_logvar, v_loss, v_metrics, v_loss_det
                        
                        xm.mark_step()
                        val_steps += 1
                
                if rank == 0:
                    # Извлекаем все метрики на CPU ровно один раз
                    avg_val_loss = (val_loss_sum.item() / val_steps) if val_steps > 0 and val_loss_sum is not None else 0
                    print(f"[{global_step}] Validation - Loss: {avg_val_loss:.4f}")
                    
                    val_log_dict = {"val/loss": avg_val_loss, "global_step": global_step}
                    for k, v_sum in val_metrics_sums.items():
                        avg_val = (v_sum.item() / val_steps) if torch.is_tensor(v_sum) else (v_sum / val_steps)
                        if k.startswith("l") and ("_mse" in k or "_cos" in k):
                            val_log_dict[f"val/layers/{k}"] = avg_val
                        else:
                            val_log_dict[f"val/{k}"] = avg_val
                    
                    wandb.log(val_log_dict, step=global_step)
                
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

        # Скачиваем пребилт 40-слойной модели (структура)
        os.makedirs("storage/prebuilt/latentBERT/v1.0", exist_ok=True)
        print("--- [RANK 0] Скачивание пребилта 40-слойной модели... ---")
        subprocess.run(["gsutil", "-m", "rsync", "-r", "gs://bebladii-weigths/prebuilt/latentBERT/v1.0/", "storage/prebuilt/latentBERT/v1.0/"], check=True)

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
