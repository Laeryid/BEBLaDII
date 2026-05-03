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
import wandb
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import subprocess, re
from tqdm.auto import tqdm

# Добавляем путь к src
sys.path.append(os.getcwd())

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
    # Включаем градиентный чекпоинтинг для экономии памяти TPU (иначе OOM на seq_len=4096)
    if hasattr(distiller.student.model, 'gradient_checkpointing_enable'):
        distiller.student.model.gradient_checkpointing_enable()
        if rank == 0:
            print("--- [RANK 0] Gradient Checkpointing ВКЛЮЧЕН ---")

    # Загрузка последнего чекпоинта (если есть)
    if os.path.exists("latest_checkpoint.pt"):
        ckpt = torch.load("latest_checkpoint.pt", map_location='cpu')
        distiller.load_state_dict(ckpt['model_state_dict'], strict=False)
        if rank == 0: print(f"--- [RESUME] Чекпоинт загружен ---")

    # Настройка
    from torch.optim import AdamW
    optimizer = AdamW(filter(lambda p: p.requires_grad, distiller.parameters()), lr=5e-5)
    criterion = DistillationLoss(cos_weight=20.0)

    # Данные
    train_loader = get_dataloader(stage='reasoning', batch_size=4, max_length=4096, split='train')
    train_loader = pl.MpDeviceLoader(train_loader, device)

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
    for epoch in range(1):
        progress_bar = tqdm(train_loader, disable=(rank != 0), desc=f"Epoch {epoch}")
        
        if rank == 0:
            print("--- [RANK 0] Начинаем итерации. ПЕРВЫЙ ШАГ вызовет компиляцию XLA графа (это может занять 5-15 минут, подождите!)... ---")
            
        for batch in progress_bar:
            if rank == 0 and global_step < 3: print(f"\n--- [RANK 0] Шаг {global_step+1}: Forward pass... ---")
            optimizer.zero_grad()
            student_states, teacher_targets, mu, logvar = distiller(batch['input_ids'], batch['attention_mask'])
            loss, _ = criterion(student_states, teacher_targets, batch['attention_mask'], mu, logvar, beta=0.0001)
            
            if rank == 0 and global_step < 3: print(f"--- [RANK 0] Шаг {global_step+1}: Backward pass... ---")
            loss.backward()
            
            if rank == 0 and global_step < 3: print(f"--- [RANK 0] Шаг {global_step+1}: Optimizer step (XLA Compile)... ---")
            xm.optimizer_step(optimizer)
            global_step += 1
            
            if rank == 0 and global_step <= 3: print(f"--- [RANK 0] Шаг {global_step} завершён! ---")
            
            if rank == 0:
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step})
                if global_step % 200 == 0:
                    torch.save({'model_state_dict': distiller.state_dict()}, "latest_checkpoint.pt")
                    subprocess.run(["gsutil", "cp", "latest_checkpoint.pt", "gs://bebladii-weigths/checkpoints/"])

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
            res = subprocess.run(["gsutil", "ls", "gs://bebladii-weigths/checkpoints/RESUME_TPU_STEP_*.pt"], capture_output=True, text=True)
            if res.returncode == 0 and res.stdout.strip():
                latest = sorted(res.stdout.strip().split('\n'))[-1]
                subprocess.run(["gsutil", "cp", latest, "latest_checkpoint.pt"], check=True)
        except: pass
        
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
