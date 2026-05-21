import os, sys
import torch._dynamo
import torch
import torch.nn as nn
import torch.nn.functional as F
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

def calculate_isotropy(x):
    """
    Вычисляет изотропию латентного пространства через сингулярные числа.
    I(X) = [sum(sigma_i)]^2 / [d * sum(sigma_i^2)]
    """
    if x.ndim == 3: # (B, T, D) -> (B*T, D)
        x = x.reshape(-1, x.size(-1))
    
    # Центрируем данные
    x = x - x.mean(dim=0, keepdim=True)
    
    try:
        # SVD в bfloat16 может быть нестабилен, переходим в float32
        _, s, _ = torch.linalg.svd(x.float(), full_matrices=False)
        isotropy = (s.sum()**2) / (len(s) * (s**2).sum() + 1e-8)
        return isotropy.item()
    except Exception:
        return 0.0

def calculate_neighbor_recall(student_h, teacher_h, k=5):
    """
    Насколько топ-k соседей в пространстве ученика совпадают с учителем.
    Работает на уровне одного батча (B*T примеров).
    """
    if student_h.ndim == 3:
        student_h = student_h.reshape(-1, student_h.size(-1))
        teacher_h = teacher_h.reshape(-1, teacher_h.size(-1))
        
    # Берем случайное подмножество, если токенов слишком много (для скорости)
    num_samples = min(256, student_h.size(0))
    indices = torch.randperm(student_h.size(0))[:num_samples]
    s_sub = F.normalize(student_h[indices].float(), dim=-1)
    t_sub = F.normalize(teacher_h[indices].float(), dim=-1)
    
    # Матрицы сходства (N x N)
    s_sim = s_sub @ s_sub.T
    t_sim = t_sub @ t_sub.T
    
    # Топ-k индексов (исключая самого себя, поэтому k+1)
    s_topk = s_sim.topk(k + 1, dim=-1).indices[:, 1:]
    t_topk = t_sim.topk(k + 1, dim=-1).indices[:, 1:]
    
    # Подсчет пересечений
    recall = 0.0
    for i in range(num_samples):
        s_set = set(s_topk[i].tolist())
        t_set = set(t_topk[i].tolist())
        recall += len(s_set.intersection(t_set)) / k
        
    return recall / num_samples

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
    from src.beb_la_dii.data.mask_variants import make_mask_variants, MASK_WEIGHTS

    # Определяем наше ядро
    device = xm.xla_device()
    rank = xm.get_local_ordinal()
    
    # Настройка WandB только для Rank 0
    wandb = setup_wandb(rank)
    
    if rank == 0:
        print(f"--- [RANK 0] Инициализация на TPU (v6e)... ---")

    first_log_done = False # Флаг для отладки логирования

    # Сборка модели
    assembler = ModelAssembler()
    distiller = assembler.assemble_phase1_distiller(
        teacher_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        student_base_id="answerdotai/ModernBERT-large",
        version="v1.0", weights_map={}, device_map={"": device}, student_device=device
    )
    
    # Настройка SPMD Mesh
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (num_devices, 1) # Это уже кортеж
    device_ids = np.array(range(num_devices))
    mesh = xs.Mesh(device_ids, mesh_shape, ('fsdp', 'model'))
    xs.set_global_mesh(mesh)
    
    # Загрузка последнего чекпоинта (если есть) ДО обертки FSDP
    # 1. Замораживаем учителя полностью
    for param in distiller.teacher.parameters():
        param.requires_grad = False
    distiller.teacher.eval()

    # 2. Размораживаем студента полностью
    for param in distiller.student.parameters():
        param.requires_grad = True

    # 3. Размораживаем FeatureProjectors полностью (включая скейлы)
    for param in distiller.feature_projectors.parameters():
        param.requires_grad = True

    # 4. Размораживаем InputProjector полностью
    for param in distiller.input_projector.parameters():
        param.requires_grad = True

    if rank == 0:
        trainable_params = sum(p.numel() for p in distiller.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in distiller.parameters() if not p.requires_grad)
        print(f"--- [INIT] Trainable parameters: {trainable_params:,} ---")
        print(f"--- [INIT] Frozen parameters: {frozen_params:,} ---")

    def load_awakening_weights(model, sd, rank=0):
        """
        Специализированная загрузка для AWAKENED_WEIGHTS_FINAL.pt.
        Если чекпоинт в плоском FSDP-формате (ключи с _orig_module.) — делегируем в smart_load_weights.
        Иначе распаковываем вложенные словари: latentBERT_state_dict, input_projector, feature_projectors.
        """
        # Определяем формат чекпоинта по первому ключу
        first_key = next(iter(sd.keys()), "")
        if first_key.startswith("_orig_module.") or first_key.startswith("module."):
            if rank == 0:
                print(f"--- [INIT] Detected flat FSDP checkpoint, delegating to smart_load_weights ---")
            return smart_load_weights(model, sd, rank=rank)
        
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
                # Пробуем разные варианты префиксов
                possible_keys = [
                    k.replace("model.", "student.model.", 1),
                    f"student.{k}",
                    k if k.startswith("student.") else None
                ]
                for target_k in filter(None, possible_keys):
                    if target_k in model_sd:
                        new_sd[target_k] = v
                        matched += 1
                        break
        
        # 2. Input Projector
        if "input_projector" in sd:
            ip_sd = sd["input_projector"]
            for k, v in ip_sd.items():
                target_k = f"input_projector.{k}" if not k.startswith("input_projector.") else k
                if target_k in model_sd:
                    new_sd[target_k] = v
                    matched += 1
                    
        # 3. Feature Projectors
        if "feature_projectors" in sd:
            fp_sd = sd["feature_projectors"]
            for k, v in fp_sd.items():
                target_k = f"feature_projectors.{k}" if not k.startswith("feature_projectors.") else k
                if target_k in model_sd:
                    new_sd[target_k] = v
                    matched += 1

        model.load_state_dict(new_sd, strict=False)
        if rank == 0:
            print(f"--- [INIT] Awakening Load: Matched {matched} params ---")
            if matched == 0:
                print(f"DEBUG: Keys in checkpoint: {list(sd.keys())[:5]}")
                if "latentBERT_state_dict" in sd:
                    print(f"DEBUG: Sample student keys: {list(sd['latentBERT_state_dict'].keys())[:5]}")
                print(f"DEBUG: Sample model keys: {list(model_sd.keys())[:5]}")

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

    # Оборачиваем в FSDP только тяжелые слои ModernBERT
    def auto_wrap_policy(module, recurse, unwrapped_params, **kwargs):
        cls_name = module.__class__.__name__
        return any(name in cls_name for name in ["ModernBertLayer", "ModernBertBlock"])

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
    criterion = DistillationLoss(cos_weight=0.0)  # ADR-018: отказ от косинуса абсолютных векторов

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
            current_beta = ckpt.get('current_beta', 0.0) # Загружаем сохраненную beta
        else:
            if rank == 0:
                print("--- [INIT] Базовые веса загружены. Начинаем обучение с 0 шага. ---")

    # Данные
    train_loader = get_dataloader(stage='reasoning', batch_size=1, max_length=2048, split='train', val_ratio=0.0)
    val_loader = get_dataloader(stage='reasoning', batch_size=1, max_length=2048, split='val', val_ratio=0.0)
    accumulation_steps = 16

    if rank == 0:
        wandb_kwargs = {
            "project": "BEBLaDII",
            "name": f"tpu-v6e-spmd-resumed-{global_step}"
        }
        try:
            import wandb
            wandb.init(**wandb_kwargs)
            wandb_run_id = wandb.run.id
            if rank == 0: 
                print(f"--- [WANDB] Started NEW run: {wandb_run_id} ---")
        except Exception as e:
            if rank == 0: print(f"--- [WANDB ERROR] {e} ---")

    # Обучение
    distiller.train()
    
    for epoch in range(start_epoch, 10):
        batches_per_epoch = len(train_loader)
        
        # Обнуляем пропуск батчей, чтобы не ждать 9 часов. Данные начнутся с начала эпохи,
        # но LR, Beta и Gamma будут считаться для global_step = 4500.
        batches_to_skip = 0 
        
        progress_bar = tqdm(train_loader, disable=(rank != 0), initial=batches_to_skip, total=batches_per_epoch, desc=f"Epoch {epoch}")
        
        optimizer.zero_grad()
        for i, batch in enumerate(progress_bar):
            if i < batches_to_skip: continue
                
            for k, v in batch.items():
                v = v.to(device)
                batch[k] = v

            # Anti-Phase Beta Scheduler (ADR-011)
            import math
            rel_step = global_step % 2000
            BETA_MAX = 0.1
            warmup_steps = 1000.0
            warmup_factor = min(1.0, global_step / warmup_steps)
            current_beta = (BETA_MAX * (1 - math.cos(2 * math.pi * rel_step / 2000)) / 2) * warmup_factor

            # --- [Trajectory-Aware Distillation] ---
            # 1. Генерация 4 вариантов масок
            variants = make_mask_variants(batch)
            
            # 2. Объединяем в один большой батч (4*B, T)
            B_orig = batch['input_ids'].shape[0]
            T = batch['input_ids'].shape[1]
            
            # Interleaving: [S0V0, S0V1, S0V2, S0V3, S1V0...]
            # Это гарантирует, что все маски одного семпла попадут на одно XLA-устройство при шардинге
            v_input_ids = torch.stack([v["input_ids"] for v in variants], dim=1).view(-1, T)
            v_attn_mask = torch.stack([v["attention_mask"] for v in variants], dim=1).view(-1, T)
            
            # Учителю нужен только 1 (полный) вариант, чтобы избежать OOM
            t_input_ids = variants[3]["input_ids"]
            t_attn_mask = variants[3]["attention_mask"]
            
            # SPMD Sharding для объединенного батча
            xs.mark_sharding(v_input_ids, mesh, ('fsdp', None))
            xs.mark_sharding(v_attn_mask, mesh, ('fsdp', None))
            # Если B_orig не кратно количеству устройств (например, 1 < 4), то реплицируем тензор учителя
            t_shard = ('fsdp', None) if B_orig % xr.global_runtime_device_count() == 0 else (None, None)
            xs.mark_sharding(t_input_ids, mesh, t_shard)
            xs.mark_sharding(t_attn_mask, mesh, t_shard)
            
            # 3. Единый forward проход
            v_student_states, v_teacher_targets, v_mu, v_logvar, v_raw_st = distiller(
                input_ids=v_input_ids, 
                attention_mask=v_attn_mask,
                teacher_input_ids=t_input_ids,
                teacher_attention_mask=t_attn_mask
            )
            
            # 4. Разделяем выходы обратно на 4 варианта с помощью view
            # Поскольку батч interleaved, h[:, idx] оставляет вычисления локальными на TPU ядрах
            s_variants = []
            t_variants = []
            for idx in range(4):
                s_variants.append({l: h.view(B_orig, 4, T, -1)[:, idx] for l, h in v_student_states.items()})
                t_variants.append({l: h.view(B_orig, 4, T, -1)[:, idx] for l, h in v_teacher_targets.items()})
            
            # 5. Вычисление L_state (взвешенная сумма по всем маскам)
            total_l_state = 0
            loss_metrics = {}
            for idx in range(4):
                v_mu_p = v_mu.view(B_orig, 4, T, -1)[:, idx] if v_mu is not None else None
                v_logvar_p = v_logvar.view(B_orig, 4, T, -1)[:, idx] if v_logvar is not None else None
                v_raw_p = {l: h.view(B_orig, 4, T, -1)[:, idx] for l, h in v_raw_st.items()} if v_raw_st is not None else None
                
                l_state, m_state = criterion(
                    s_variants[idx], t_variants[idx], variants[idx]["attention_mask"],
                    mu=v_mu_p, logvar=v_logvar_p, beta=current_beta,
                    raw_student_states=v_raw_p, lambda_prior=0.1
                )
                total_l_state += MASK_WEIGHTS[idx] * l_state
                
                # Логируем метрики полной маски (idx=3)
                if idx == 3:
                    for k, v in m_state.items():
                        loss_metrics[f"full_{k}"] = v
            
            # 6. Вычисление L_delta (Semantic Gradients)
            # gamma растет с обучением (0.3 -> 0.5 за 10к шагов)
            current_gamma = min(0.5, 0.3 + (global_step / 10000.0) * 0.2)
            total_l_delta = 0
            for idx in range(3): # Переходы 0->1, 1->2, 2->3
                l_delta, m_delta = criterion.compute_delta_loss(
                    s_variants[idx], s_variants[idx+1],
                    t_variants[idx], t_variants[idx+1],
                    variants[idx]["attention_mask"],
                    variants[idx+1]["attention_mask"]
                )
                total_l_delta += l_delta
                # Логируем каждый переход (t01, t12, t23)
                for k, v in m_delta.items():
                    loss_metrics[f"{k}_t{idx}{idx+1}"] = v
            
            loss = total_l_state + current_gamma * total_l_delta
            
            # 7. Balance Regularization (ADR-011)
            loss_bal = distiller.compute_balance_loss(lambda_balance=1.0)
            loss = loss + loss_bal
            
            loss_metrics["l_state_total"] = total_l_state.detach()
            loss_metrics["l_delta_total"] = total_l_delta.detach()
            loss_metrics["gamma"] = current_gamma
            loss_metrics["balance_reg"] = loss_bal.detach()
            
            loss = loss / accumulation_steps
            loss.backward()
            
            del v_student_states, v_teacher_targets, v_mu, v_logvar, s_variants, t_variants
            
            if (i + 1) % accumulation_steps == 0:
                xm.optimizer_step(optimizer, barrier=True)
                scheduler.step()
                
                # Manual Warmup (up to 1000 global steps)
                current_optim_step = global_step + 1
                if current_optim_step <= warmup_steps:
                    lr_warmup_factor = max(0.01, current_optim_step / warmup_steps)
                    # Корректно устанавливаем LR относительно начального значения
                    for idx_p, param_group in enumerate(optimizer.param_groups):
                        param_group['lr'] = scheduler.base_lrs[idx_p] * lr_warmup_factor
                
                optimizer.zero_grad()
                
                if current_optim_step % 500 == 0:
                    xm.mark_step()
                    full_sd = distiller.state_dict()
                    trainable_sd = {k: v for k, v in full_sd.items() if "teacher" not in k}
                    
                    save_data = {
                        'model_state_dict': trainable_sd,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'global_step': current_optim_step,
                        'epoch': epoch,
                        'wandb_run_id': wandb_run_id,
                        'current_beta': current_beta
                    }
                    local_ckpt_name = f"ckpt_{current_optim_step}.pt"
                    xm.save(save_data, local_ckpt_name)
                    
                    if rank == 0:
                        try:
                            # Локальное сохранение
                            shutil.copy(local_ckpt_name, "latest_checkpoint.pt")
                            
                            # Отправка в GCS (возвращено к синхронному для стабильности графов)
                            subprocess.run(["gsutil", "-q", "cp", local_ckpt_name, "gs://bebladii-weigths/checkpoints/"], check=True)
                            subprocess.run(["gsutil", "-q", "cp", "latest_checkpoint.pt", "gs://bebladii-weigths/checkpoints/"], check=True)
                            
                            # Работа с логами тренировки
                            if os.path.exists("history.jsonl"):
                                th_ver = f"history_{current_optim_step}.jsonl"
                                shutil.copy("history.jsonl", th_ver)
                                subprocess.run(["gsutil", "cp", "history.jsonl", "gs://bebladii-weigths/checkpoints/history.jsonl"], check=True)
                                subprocess.run(["gsutil", "cp", th_ver, f"gs://bebladii-weigths/checkpoints/{th_ver}"], check=True)
                                os.remove(th_ver)

                            # Работа с логами валидации
                            if os.path.exists("history_val.jsonl"):
                                h_ver = f"history_val_{current_optim_step}.jsonl"
                                shutil.copy("history_val.jsonl", h_ver)
                                subprocess.run(["gsutil", "cp", "history_val.jsonl", "gs://bebladii-weigths/checkpoints/history_val.jsonl"], check=True)
                                subprocess.run(["gsutil", "cp", h_ver, f"gs://bebladii-weigths/checkpoints/{h_ver}"], check=True)
                                os.remove(h_ver)

                            # Очистка старых локальных чекпоинтов (храним только последние 500 шагов)
                            prev_step = current_optim_step - (500 * accumulation_steps)
                            prev_ckpt = f"ckpt_{prev_step}.pt"
                            if os.path.exists(prev_ckpt): os.remove(prev_ckpt)
                        except Exception as e:
                            print(f"--- [GCS ERROR] {e} ---")
                    
                    # ПРИНУДИТЕЛЬНЫЙ сброс графа перед барьером, чтобы Rank 0 не тормозил
                    xm.mark_step()
                    # Барьер 1: Синхронизация после работы с GCS
                    xm.rendezvous("gcs_sync_done")

                # Валидация каждые 50 шагов (учащаем с 100)
                if (current_optim_step + 10) % 50 == 0:
                    distiller.eval()
                    val_loss_sum, val_steps = 0.0, 0
                    val_metrics_sums = {}
                    val_heavy_metrics = {}
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
                            
                            v_st, v_tgt, v_mu, v_logvar, v_raw = distiller(v_batch['input_ids'], v_batch['attention_mask'])
                            v_loss, v_metrics = criterion(
                                v_st, v_tgt, v_actual_mask, 
                                v_mu, v_logvar, beta=0.0001,
                                raw_student_states=v_raw, lambda_prior=0.1
                            )
                            
                            val_loss_sum += v_loss.item()
                            for k, val in v_metrics.items():
                                val_metrics_sums[k] = val_metrics_sums.get(k, 0.0) + (val.item() if torch.is_tensor(val) else val)
                            
                            # Дополнительные метрики для l40 (считаем на первом батче для скорости)
                            if v_step == 0:
                                xm.mark_step()
                                if rank == 0: print("--- [VAL] Calculating heavy metrics (l40)... ---")
                                
                                # Переносим на CPU для стабильности SVD и Recall
                                if rank == 0: print("    -> Transferring to CPU...")
                                v_actual_mask_cpu = v_actual_mask.detach().cpu().bool()
                                
                                # BERT-пространство студента (1024d)
                                l40_raw_cpu = v_raw[40].detach().cpu()[v_actual_mask_cpu]
                                # Проектированное пространство студента (3584d, Qwen)
                                l40_projected_cpu = v_st[40].detach().cpu()[v_actual_mask_cpu]
                                # Пространство учителя (3584d, Qwen)
                                l40_teacher_cpu = v_tgt[40].detach().cpu()[v_actual_mask_cpu]
                                
                                if rank == 0: print("    -> Calculating isotropy (SVD)...")
                                val_heavy_metrics["l40_isotropy"] = calculate_isotropy(l40_raw_cpu)
                                val_heavy_metrics["l40_projected_isotropy"] = calculate_isotropy(l40_projected_cpu)
                                
                                if rank == 0: print("    -> Calculating neighbor recall...")
                                val_heavy_metrics["l40_neighbor_recall"] = calculate_neighbor_recall(l40_raw_cpu, l40_teacher_cpu)
                                
                                if rank == 0: print("    -> Calculating noise sensitivity...")
                                def compute_noise_sensitivity(latent, sigmas=(0.1, 0.3, 0.5, 1.0)):
                                    results = {}
                                    for sigma in sigmas:
                                        noise = torch.randn_like(latent) * sigma
                                        noisy = latent + noise
                                        cos = F.cosine_similarity(latent, noisy, dim=-1)
                                        results[f"l40_ns_sigma_{str(sigma).replace('.','_')}"] = (1 - cos).mean().item()
                                    return results
                                
                                val_heavy_metrics.update(compute_noise_sensitivity(l40_raw_cpu))

                                # Метрика дрейфа (для нового проектора) в BERT-пространстве
                                # Так как тензор теперь (N, D), считаем mean и var по нулевому измерению
                                mu_l40 = l40_raw_cpu.mean(dim=0)
                                var_l40 = l40_raw_cpu.var(dim=0, unbiased=False)
                                val_heavy_metrics["l40_mu_drift"] = mu_l40.pow(2).mean().item()
                                val_heavy_metrics["l40_var_drift"] = (var_l40 - 1.0).pow(2).mean().item()
                                if rank == 0: print("--- [VAL] Heavy metrics done. ---")

                            val_steps += 1
                            xm.mark_step()
                    
                    if rank == 0:
                        avg_val_loss = val_loss_sum / val_steps
                        val_log = {"val/loss": avg_val_loss, "global_step": current_optim_step}
                        for k, v_sum in val_metrics_sums.items():
                            val_log[f"val/{k}"] = v_sum / val_steps
                        for k, val in val_heavy_metrics.items():
                            val_log[f"val/{k}"] = val
                        import wandb
                        wandb.log(val_log, step=current_optim_step)
                        print(f"--- [VAL] Step {current_optim_step}: Loss {avg_val_loss:.4f} ---")
                        
                        # Сохранение в локальный файл истории валидации
                        with open("history_val.jsonl", "a", encoding="utf-8") as f:
                            f.write(json.dumps(val_log, ensure_ascii=False) + "\n")
                    
                    # Барьер 2: Синхронизация после валидации
                    xm.rendezvous("validation_done")
                    distiller.train()
                
                # Update global_step at the very end of accumulation block
                global_step = current_optim_step

                if xm.is_master_ordinal() and global_step % 20 == 0:
                    import wandb
                    log_dict = {
                        "train/loss": loss.item() * accumulation_steps, 
                        "train/lr": optimizer.param_groups[0]['lr'],
                        "train/beta": current_beta,
                        "global_step": global_step
                    }
                    for k, v in loss_metrics.items():
                        log_dict[f"train/{k}"] = v.item() if torch.is_tensor(v) else v
                    
                    # Мониторинг скейлов: output_scale и residual_scale
                    for name, param in distiller.named_parameters():
                        if "output_scale" in name or "residual_scale" in name:
                            for proj_key in ["20", "30", "40"]:
                                if f".{proj_key}." in name or f"_{proj_key}." in name:
                                    s_type = "out" if "output_scale" in name else "res"
                                    log_dict[f"train/scale_{s_type}_l{proj_key}"] = param.detach().float().mean().item()
                                    break
                    
                    wandb.log(log_dict, step=global_step)
                    
                    # Сохранение в локальный файл истории
                    with open("history.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_dict, ensure_ascii=False) + "\n")
                    
                    # Обновление прогресс-бара (вместо принта)
                    progress_bar.set_postfix({
                        "loss": f"{log_dict['train/loss']:.4f}",
                        "kl": f"{log_dict.get('train/kl', 0):.4f}",
                        "beta": f"{current_beta:.6f}"
                    })

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
                # Пытаемся найти файл в checkpoints/ (для latest_checkpoint) 
                # или в kaggle_upload_1_2/ (для AWAKENED)
                gcs_path = f"gs://bebladii-weigths/checkpoints/{weight_file}"
                if weight_file == "AWAKENED_WEIGHTS_FINAL.pt":
                    gcs_path = f"gs://bebladii-weigths/kaggle_upload_1_2/{weight_file}"

                res = subprocess.run(["gsutil", "ls", gcs_path], capture_output=True, text=True)
                if res.returncode == 0:
                    print(f"--- [RANK 0] Загрузка {weight_file} из {gcs_path} ---")
                    subprocess.run(["gsutil", "cp", gcs_path, weight_file], check=True)

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
