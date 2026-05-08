import os, sys
import torch._dynamo
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import subprocess, re, time, json, shutil

# Отключаем Dynamo для XLA
torch._dynamo.disable()

def setup_env():
    """Настройка переменных окружения для TPU v6e"""
    os.environ["PJRT_DEVICE"] = "TPU"
    os.environ["XLA_USE_BF16"] = "1"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    # Для v6e-4 (4 TPU)
    os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "2,2,1" 
    os.environ["TPU_NUM_DEVICES"] = "4"
    os.environ["XLA_USE_SPMD"] = "1" # Явно включаем через переменную

def setup_wandb(rank, run_id=None):
    """Изолированная инициализация WandB"""
    key_path = "/home/hp/wandb_key.txt"
    if not os.path.exists(key_path):
        return None

    with open(key_path, "r") as f:
        _key = f.read().strip()
    
    if not _key:
        return None

    os.environ["WANDB_API_KEY"] = _key
    
    if rank == 0:
        import wandb
        wandb.login(key=_key)
        return wandb
    return None

# ХАК: Dummy XLA для сред без установленного torch_xla
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
    # Импорты XLA только внутри функции для чистоты глобального пространства
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import torch_xla.experimental.xla_sharding as xs
    from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel as FSDP
    
    # Импорты модулей проекта
    from src.beb_la_dii.model.assembler import ModelAssembler
    from src.beb_la_dii.utils.loss import DistillationLoss
    from src.beb_la_dii.utils.data import get_dataloader

    # Определяем наше ядро
    device = xm.xla_device()
    rank = xm.get_local_ordinal()
    
    # Настройка WandB только для Rank 0
    wandb = setup_wandb(rank)
    
    if rank == 0:
        print(f"--- [RANK 0] Инициализация на TPU (v6e)... ---")

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
    
    # Загрузка последнего чекпоинта (если есть) ДО обертки FSDP
    for name, param in distiller.named_parameters():
        if "teacher" in name:
            param.requires_grad = False

    def load_awakening_weights(model, sd, rank=0):
        """
        Специализированная загрузка для AWAKENED_WEIGHTS_FINAL.pt.
        Распаковывает вложенные словари: latentBERT_state_dict, input_projector, feature_projectors.
        """
        model_sd = model.state_dict()
        new_sd = {}
        matched = 0
        
        # Маппинг ключей из файла на атрибуты ReasoningDistiller
        # AWAKENED файл: {latentBERT_state_dict: {...}, input_projector: {...}, ...}
        # Наша модель: {student.model...., input_projector...., feature_projectors....}
        
        # 1. Студент (ModernBERT)
        if "latentBERT_state_dict" in sd:
            l_sd = sd["latentBERT_state_dict"]
            for k, v in l_sd.items():
                # В файле ключи 'model.layers...', в модели 'student.model.layers...'
                target_k = k.replace("model.", "student.model.", 1)
                if target_k in model_sd:
                    new_sd[target_k] = v
                    matched += 1
                elif f"student.{k}" in model_sd:
                    new_sd[f"student.{k}"] = v
                    matched += 1
        
        # 2. Input Projector
        if "input_projector" in sd:
            ip_sd = sd["input_projector"]
            for k, v in ip_sd.items():
                target_k = f"input_projector.{k}"
                if target_k in model_sd:
                    new_sd[target_k] = v
                    matched += 1
                    
        # 3. Feature Projectors
        if "feature_projectors" in sd:
            fp_sd = sd["feature_projectors"]
            for k, v in fp_sd.items():
                target_k = f"feature_projectors.{k}"
                if target_k in model_sd:
                    new_sd[target_k] = v
                    matched += 1

        model.load_state_dict(new_sd, strict=False)
        if rank == 0:
            print(f"--- [INIT] Awakening Load: Matched {matched} params ---")

    def smart_load_weights(model, sd, rank=0):
        """
        Умная загрузка весов для обычных чекпоинтов: сопоставляет ключи по суффиксам.
        """
        model_sd = model.state_dict()
        new_sd = {}
        matched = 0
        total_trainable = 0
        
        clean_sd = {}
        for k, v in sd.items():
            clean_k = k.replace("_orig_module.", "").replace("module.", "")
            clean_sd[clean_k] = v

        for k, v in model_sd.items():
            if "teacher" in k: continue
            total_trainable += 1
            if k in clean_sd:
                new_sd[k] = clean_sd[k]
                matched += 1
            else:
                for ck in clean_sd.keys():
                    if k.endswith(ck) or ck.endswith(k):
                        new_sd[k] = clean_sd[ck]
                        matched += 1
                        break
            
        model.load_state_dict(new_sd, strict=False)
        if rank == 0:
            print(f"--- [INIT] Smart Load: Matched {matched}/{total_trainable} trainable params ---")

    ckpt_path = "latest_checkpoint.pt"
    if not os.path.exists(ckpt_path) and os.path.exists("AWAKENED_WEIGHTS_FINAL.pt"):
        ckpt_path = "AWAKENED_WEIGHTS_FINAL.pt"
        if rank == 0: print(f"--- [INIT] Используем базовые веса пробуждения: {ckpt_path} ---")

    ckpt = None
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        raw_sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        
        if ckpt_path == "AWAKENED_WEIGHTS_FINAL.pt":
            load_awakening_weights(distiller, raw_sd, rank=rank)
        else:
            smart_load_weights(distiller, raw_sd, rank=rank)
            
        if rank == 0: 
            print(f"--- [RESUME] Веса загружены из {ckpt_path} ---")

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
    # T_0=2000, eta_min=1e-6 согласно плану
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2000, T_mult=1, eta_min=1e-6)
    criterion = DistillationLoss()

    global_step = 0
    start_epoch = 0
    wandb_run_id = None

    if ckpt:
        # Загружаем прогресс только если это полноценный чекпоинт обучения, а не базовые веса
        is_full_checkpoint = (ckpt_path == "latest_checkpoint.pt")
        
        if is_full_checkpoint:
            if 'optimizer_state_dict' in ckpt:
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                except Exception as e:
                    if rank == 0: print(f"--- [RESUME WARNING] Ошибка оптимизатора: {e} ---")
            
            if 'scheduler_state_dict' in ckpt:
                try:
                    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                except Exception as e:
                    if rank == 0: print(f"--- [RESUME WARNING] Ошибка планировщика: {e} ---")
            
            global_step = ckpt.get('global_step', 0)
            start_epoch = ckpt.get('epoch', 0)
            wandb_run_id = ckpt.get('wandb_run_id', None)
        else:
            if rank == 0:
                print("--- [INIT] Базовые веса загружены. Начинаем обучение с 0 шага. ---")

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
            import wandb
            wandb.init(**wandb_kwargs)
            wandb_run_id = wandb.run.id
        except Exception as e:
            if rank == 0: print(f"--- [WANDB ERROR] {e} ---")

    # Обучение
    distiller.train()
    
    for epoch in range(start_epoch, 10):
        batches_per_epoch = len(train_loader)
        batches_to_skip = global_step % batches_per_epoch if epoch == start_epoch else 0
        progress_bar = tqdm(train_loader, disable=(rank != 0), initial=batches_to_skip, total=batches_per_epoch, desc=f"Epoch {epoch}")
        
        optimizer.zero_grad()
        for i, batch in enumerate(progress_bar):
            if i < batches_to_skip: continue
                
            for k, v in batch.items():
                v = v.to(device)
                xs.mark_sharding(v, mesh, ('fsdp',) + (None,) * (v.dim() - 1))
                batch[k] = v
            
            # Targeted Loss Masking support
            actual_mask = batch['loss_mask'] if 'loss_mask' in batch else batch['attention_mask']
            
            # Global KL-Annealing (Resume-safe)
            warmup_steps = 1000.0
            current_beta = min(0.0001, 0.0001 * (global_step / warmup_steps))

            student_states, teacher_targets, mu, logvar = distiller(batch['input_ids'], batch['attention_mask'])
            loss, loss_metrics = criterion(student_states, teacher_targets, actual_mask, mu, logvar, beta=current_beta)
            
            loss = loss / accumulation_steps
            loss.backward()
            
            del student_states, teacher_targets, mu, logvar
            
            if (global_step + 1) % accumulation_steps == 0:
                xm.optimizer_step(optimizer, barrier=True)
                scheduler.step()
                
                # Manual Warmup (up to 1000 global steps)
                if global_step <= 1000:
                    warmup_factor = max(0.01, global_step / 1000.0)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * warmup_factor
                
                optimizer.zero_grad()
                
                if ((global_step + 1) // accumulation_steps) % 500 == 0:
                    xm.mark_step()
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
                            # Локальное сохранение
                            shutil.copy(local_ckpt_name, "latest_checkpoint.pt")
                            
                            # Отправка в GCS
                            subprocess.run(["gsutil", "cp", local_ckpt_name, "gs://bebladii-weigths/checkpoints/"], check=True)
                            subprocess.run(["gsutil", "cp", "latest_checkpoint.pt", "gs://bebladii-weigths/checkpoints/"], check=True)
                            
                            # Работа с логами тренировки
                            if os.path.exists("history.jsonl"):
                                th_ver = f"history_{global_step}.jsonl"
                                shutil.copy("history.jsonl", th_ver)
                                subprocess.run(["gsutil", "cp", "history.jsonl", "gs://bebladii-weigths/checkpoints/history.jsonl"], check=True)
                                subprocess.run(["gsutil", "cp", th_ver, f"gs://bebladii-weigths/checkpoints/{th_ver}"], check=True)
                                os.remove(th_ver)

                            # Работа с логами валидации
                            if os.path.exists("history_val.jsonl"):
                                h_ver = f"history_val_{global_step}.jsonl"
                                shutil.copy("history_val.jsonl", h_ver)
                                subprocess.run(["gsutil", "cp", "history_val.jsonl", "gs://bebladii-weigths/checkpoints/history_val.jsonl"], check=True)
                                subprocess.run(["gsutil", "cp", h_ver, f"gs://bebladii-weigths/checkpoints/{h_ver}"], check=True)
                                os.remove(h_ver)

                            # Очистка старых локальных чекпоинтов (храним только последние 500 шагов)
                            prev_step = global_step - (500 * accumulation_steps)
                            prev_ckpt = f"ckpt_{prev_step}.pt"
                            if os.path.exists(prev_ckpt): os.remove(prev_ckpt)
                        except Exception as e:
                            print(f"--- [GCS ERROR] {e} ---")
                    
                    # Барьер 1: Синхронизация после работы с GCS
                    xm.rendezvous("gcs_sync_done")

                    distiller.eval()
                    val_loss_sum, val_steps = 0.0, 0
                    val_metrics_sums = {}
                    max_val_steps = 50 
                    
                    with torch.no_grad():
                        for v_step, v_batch in enumerate(val_loader):
                            if v_step >= max_val_steps: break
                            for k, v in v_batch.items():
                                v = v.to(device)
                                xs.mark_sharding(v, mesh, ('fsdp',) + (None,) * (v.dim() - 1))
                                v_batch[k] = v
                                
                            # Validation also uses targeted mask if available
                            v_actual_mask = v_batch['loss_mask'] if 'loss_mask' in v_batch else v_batch['attention_mask']
                            
                            v_st, v_tgt, v_mu, v_logvar = distiller(v_batch['input_ids'], v_batch['attention_mask'])
                            v_loss, v_metrics = criterion(v_st, v_tgt, v_actual_mask, v_mu, v_logvar, beta=0.0001)
                            
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
                        import wandb
                        wandb.log(val_log, step=global_step)
                        print(f"--- [VAL] Step {global_step}: Loss {avg_val_loss:.4f} ---")
                    
                    # Барьер 2: Синхронизация после валидации
                    xm.rendezvous("validation_done")
                    distiller.train()
            
            if (i + 1) % accumulation_steps == 0:
                global_step += 1
                if xm.is_master_ordinal() and global_step % 20 == 0:
                    import wandb
                    log_dict = {
                        "train/loss": loss.item() * accumulation_steps, 
                        "train/lr": optimizer.param_groups[0]['lr'],
                        "global_step": global_step
                    }
                    for k, v in loss_metrics.items():
                        log_dict[f"train/{k}"] = v.item() if torch.is_tensor(v) else v
                    
                    wandb.log(log_dict, step=global_step)
                    
                    # Сохранение в локальный файл истории
                    with open("history.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_dict, ensure_ascii=False) + "\n")
                    
                    if global_step % 50 == 0:
                        print(f"--- [LOG] Step {global_step}: Loss {log_dict['train/loss']:.4f} ---")

            xm.mark_step()

if __name__ == "__main__":
    setup_env() 
    
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    
    xr.use_spmd()
    
    rank = int(os.environ.get("LOCAL_RANK", 0))

    if rank == 0:
        print("--- [RANK 0] Подготовка ресурсов ---")
        os.makedirs("./data", exist_ok=True)
        try:
            subprocess.run(["gsutil", "-m", "rsync", "-r", "gs://bebladii-datasets/data/", "./data"], check=True)
            # 2. Поиск и скачивание весов
            for weight_file in ["latest_checkpoint.pt", "AWAKENED_WEIGHTS_FINAL.pt"]:
                res = subprocess.run(["gsutil", "ls", f"gs://bebladii-weigths/checkpoints/{weight_file}"], capture_output=True, text=True)
                if res.returncode == 0:
                    print(f"--- [RANK 0] Загрузка {weight_file} из GCS ---")
                    subprocess.run(["gsutil", "cp", f"gs://bebladii-weigths/checkpoints/{weight_file}", weight_file], check=True)

            # 3. Синхронизация истории
            for h_file in ["history.jsonl", "history_val.jsonl"]:
                res_h = subprocess.run(["gsutil", "ls", f"gs://bebladii-weigths/checkpoints/{h_file}"], capture_output=True, text=True)
                if res_h.returncode == 0:
                    print(f"--- [RANK 0] Загрузка {h_file} из CS ---")
                    subprocess.run(["gsutil", "cp", f"gs://bebladii-weigths/checkpoints/{h_file}", h_file], check=True)
                else:
                    with open(h_file, "w") as f: pass
        except Exception as e:
            print(f"--- [RANK 0] Ошибка загрузки ресурсов: {e} ---")
        
        with open("/tmp/resources_prepared.flag", "w") as f: f.write("ok")
    else:
        while not os.path.exists("/tmp/resources_prepared.flag"):
            time.sleep(1)

    xm.rendezvous("init_done")
    train()
