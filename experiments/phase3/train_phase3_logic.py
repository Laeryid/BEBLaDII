import os
import sys

# Настройка переменных окружения для TPU v6e
def setup_env():
    os.environ["PJRT_DEVICE"] = "TPU"
    os.environ["XLA_USE_BF16"] = "1"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "2,2,1"
    os.environ["TPU_NUM_DEVICES"] = "4"
    os.environ["XLA_USE_SPMD"] = "1"

setup_env()

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.experimental.xla_sharding as xs
import torch_xla.runtime as xr
import wandb
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel

from src.beb_la_dii.model.vae import LatentEncoder
from src.beb_la_dii.model.dus import DUSModel
from src.beb_la_dii.utils.loss import safe_normalize

xr.use_spmd()

# ---------------------------------------------------------------------------
# SPMD Mesh
# ---------------------------------------------------------------------------

def setup_spmd_mesh():
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (num_devices, 1)
    device_ids = np.array(range(num_devices))
    mesh = xs.Mesh(device_ids, mesh_shape, ("fsdp", "model"))
    xs.set_global_mesh(mesh)
    return mesh


# ---------------------------------------------------------------------------
# LogicAdapter: переводчик пространства учителя -> пространство студента
# teacher_dim ДОЛЖЕН совпадать с last_hidden_state учителя!
# Qwen2.5-1.5B -> 1536d, Qwen2.5-7B / DeepSeek-R1-7B -> 3584d
# ---------------------------------------------------------------------------

class LogicAdapter(nn.Module):
    def __init__(self, teacher_dim: int = 3584, student_dim: int = 1024):
        super().__init__()
        self.proj = nn.Linear(teacher_dim, student_dim)

    def forward(self, x):
        return self.proj(x)


# ---------------------------------------------------------------------------
# Основная модель Phase 3
# ---------------------------------------------------------------------------

class BEBLaDIIPhase3(nn.Module):
    def __init__(self, teacher_model_path: str, embedding_model_path: str,
                 dus_weights: str | None, encoder_weights: str | None,
                 teacher_dim: int = 3584):
        super().__init__()

        # 1. Замороженный учитель (DeepSeek-R1-7B, 3584d) — только для дистилляции скрытых состояний
        self.teacher = AutoModel.from_pretrained(teacher_model_path, torch_dtype=torch.bfloat16)
        for p in self.teacher.parameters():
            p.requires_grad = False

        # 1b. Замороженный Qwen2.5-1.5B — только embedding слой для получения 1536d входа энкодера
        # LatentEncoder обучен на эмбеддингах Qwen2.5-1.5B (1536d), поэтому нельзя использовать
        # эмбеддинги DeepSeek (тоже 3584d → иное пространство).
        _qwen_base = AutoModel.from_pretrained(embedding_model_path, torch_dtype=torch.bfloat16)
        self.qwen_embeddings = _qwen_base.get_input_embeddings()  # nn.Embedding (vocab, 1536)
        del _qwen_base
        for p in self.qwen_embeddings.parameters():
            p.requires_grad = False

        # 2. Замороженный LatentEncoder (input_dim=1536, output_dim=1024)
        self.encoder = LatentEncoder()
        if encoder_weights and os.path.exists(encoder_weights):
            state = torch.load(encoder_weights, map_location="cpu")
            # Веса могут быть вложены под ключом "encoder"
            if "encoder" in state:
                state = state["encoder"]
            self.encoder.load_state_dict(state, strict=False)
            print(f"[Init] LatentEncoder weights loaded from {encoder_weights}")
        else:
            print(f"[Init] WARN: encoder_weights not found ({encoder_weights}), using random init")
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.to(torch.bfloat16)

        # 3. DUS Backbone (обучаемый)
        self.dus = DUSModel.from_scratch(weights_path=dus_weights).model
        self.dus.to(torch.bfloat16)

        # 4. LogicAdapter (обучаемый): teacher_dim должен соответствовать реальному учителю
        self.logic_adapter = LogicAdapter(teacher_dim=teacher_dim, student_dim=1024)
        self.logic_adapter.to(torch.bfloat16)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                low_noise_amp: float = 0.5) -> dict:
        """
        Возвращает словарь тензоров для вычисления лосса.
        Все вычисления учителя и энкодера — без градиентов.
        """
        with torch.no_grad():
            # Эмбеддинги 1536d из Qwen2.5-1.5B (то же пространство, на котором обучен LatentEncoder)
            qwen_embeds = self.qwen_embeddings(input_ids)  # (B, T, 1536)

            # Скрытые состояния учителя (DeepSeek-R1-7B, 3584d) для дистилляции
            # Используем его собственный embedding слой: input_ids одинаковые (tokenizer Qwen-совместим)
            teacher_outputs = self.teacher(
                input_ids=input_ids, attention_mask=attention_mask
            )
            teacher_hidden = teacher_outputs.last_hidden_state  # (B, T, 3584)

            # Кодируем Qwen-эмбеддинги в латентное пространство
            z_clean, _, _ = self.encoder(qwen_embeds)  # (B, T, 1024)

            B, T, D = z_clean.shape

            # --- Hybrid Corruption ---
            # Генерируем бинарную маску зашумляемых позиций (XLA-совместимо через torch.where)
            rand_probs = torch.rand((B, T), device=z_clean.device, dtype=z_clean.dtype)
            noise_flag = torch.where(
                rand_probs < 0.15,
                torch.ones_like(rand_probs),
                torch.zeros_like(rand_probs)
            )
            # Исключаем паддинг
            noise_mask = noise_flag * attention_mask.to(z_clean.dtype)

            # Нормированный случайный шум заданной амплитуды
            noise = torch.randn_like(z_clean)
            noise = safe_normalize(noise, dim=-1) * low_noise_amp

            z_noisy = z_clean + noise * noise_mask.unsqueeze(-1)

            # Confidence map: косинус(чистый, зашумлённый) → [0, 1]
            z_clean_normed = safe_normalize(z_clean, dim=-1)
            z_noisy_normed = safe_normalize(z_noisy, dim=-1)
            c_true = torch.clamp((z_clean_normed * z_noisy_normed).sum(dim=-1), min=0.0)

        # Прогон зашумлённой последовательности через DUS (с градиентами)
        dus_outputs = self.dus(
            inputs_embeds=z_noisy.to(torch.bfloat16), attention_mask=attention_mask
        )
        dus_final = dus_outputs.last_hidden_state  # (B, T, 1024)

        # Адаптация пространства учителя -> пространство студента (с градиентами для адаптера)
        teacher_adapted = self.logic_adapter(teacher_hidden)  # (B, T, 1024)

        return {
            "z_clean": z_clean,
            "z_noisy": z_noisy,
            "c_true": c_true,
            "noise_mask": noise_mask,
            "dus_final": dus_final,
            "teacher_adapted": teacher_adapted,
        }


# ---------------------------------------------------------------------------
# Функция потерь
# ---------------------------------------------------------------------------

def compute_phase3_loss(
    outputs: dict,
    attention_mask: torch.Tensor,
    gamma: float = 20.0,
    w_denoise: float = 1.0,
    w_logic: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    z_clean       = outputs["z_clean"]
    z_noisy       = outputs["z_noisy"]
    c_true        = outputs["c_true"]
    noise_mask    = outputs["noise_mask"]
    dus_final     = outputs["dus_final"]
    teacher_adapted = outputs["teacher_adapted"]

    metrics: dict = {}

    D = z_clean.size(-1)
    attn_f = attention_mask.float()
    active_tokens  = attn_f.sum().clamp(min=1.0)
    noised_tokens  = noise_mask.sum().clamp(min=1.0)

    # 4.1 Геометрическое: Prior Loss (mean deviation from zero)
    z_flat   = dus_final.view(-1, D)
    mask_flat = attn_f.view(-1, 1)
    m_state  = (z_flat * mask_flat).sum(dim=0) / active_tokens
    prior_loss = m_state.pow(2).mean()
    metrics["prior_loss"] = prior_loss.detach()

    # Инициализируем total_loss тензором, чтобы не создавать граф-брейк на XLA
    total_loss = 0.1 * prior_loss

    # 4.2 Диффузное (Denoising): Huber Loss на зашумлённых позициях
    denoise_elementwise = F.huber_loss(dus_final, z_clean, reduction="none", delta=1.0).mean(dim=-1)
    denoise_loss = (denoise_elementwise * noise_mask).sum() / noised_tokens
    metrics["denoise_loss"] = denoise_loss.detach()
    total_loss = total_loss + w_denoise * denoise_loss

    # 4.3 Дистилляционное (Логика): Direction loss между студентом и адаптированным учителем
    # Считаем по всем активным токенам, а не только зашумлённым —
    # даём сигнал и на "чистые" позиции, чтобы DUS всегда смотрел в сторону логики учителя
    student_normed  = safe_normalize(dus_final, dim=-1)
    teacher_normed  = safe_normalize(teacher_adapted, dim=-1)
    delta_cos = 1.0 - (student_normed * teacher_normed).sum(dim=-1)
    logic_loss = (delta_cos * attn_f).sum() / active_tokens
    metrics["logic_loss"] = logic_loss.detach()
    total_loss = total_loss + w_logic * logic_loss

    # 4.4 Identity Penalty: наказываем DUS за изменение чистых токенов
    # W_c = c_true^gamma: при c_true=0.99, gamma=20 → W_c≈0.82; при c_true=0.9 → W_c≈0.12
    w_c = torch.pow(c_true, gamma)
    penalty_elementwise = F.huber_loss(dus_final, z_noisy, reduction="none", delta=1.0).mean(dim=-1)
    identity_penalty = (penalty_elementwise * w_c * attn_f).sum() / active_tokens
    metrics["identity_penalty"] = identity_penalty.detach()
    total_loss = total_loss + identity_penalty

    # Метрики мониторинга
    c_true_mean = (c_true * attn_f).sum() / active_tokens
    metrics["c_true_mean"] = c_true_mean.detach()

    return total_loss, metrics


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

def get_dataloader(data_path: str, batch_size: int = 8, max_length: int = 512,
                   tokenizer=None) -> DataLoader:
    from indexed_parquet_dataset import IndexedParquetDataset
    ds = IndexedParquetDataset.from_folder(data_path, pattern="*.parquet", auto_fill=True)
    ds = ds.shuffle(seed=42, rg_buffer=8)
    print(f"Dataset loaded: {len(ds)} samples")

    def collate_fn(batch):
        if tokenizer is not None and "text" in batch[0] and batch[0]["text"] is not None:
            texts = [item["text"] for item in batch]
            encoded = tokenizer(
                texts, padding="max_length", truncation=True,
                max_length=max_length, return_tensors="pt"
            )
            return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}

        input_ids_list, masks = [], []
        for item in batch:
            seq = list(item.get("input_ids", []))[:max_length]
            pad_len = max_length - len(seq)
            input_ids_list.append(seq + [151643] * pad_len)
            masks.append([1] * len(seq) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

    return DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        drop_last=True, num_workers=4, prefetch_factor=2
    )


# ---------------------------------------------------------------------------
# Обучение
# ---------------------------------------------------------------------------

def train(args):
    mesh = setup_spmd_mesh()
    device = xm.xla_device()

    model = BEBLaDIIPhase3(
        teacher_model_path=args.teacher_model_path,
        embedding_model_path=args.embedding_model_path,
        dus_weights=args.local_dus_weights,
        encoder_weights=args.local_encoder_weights,
        teacher_dim=args.teacher_dim,
    ).to(device)

    # Шардируем все тяжёлые компоненты через FSDP:
    # - учитель (7B): не обучается, огромный → шардируем обязательно
    # - encoder, qwen_embeddings: небольшие, не обучаются → шардируем для консистентности
    # - dus, logic_adapter: обучаются → шардируем
    model.teacher          = SpmdFullyShardedDataParallel(model.teacher, mesh=mesh)
    model.qwen_embeddings  = SpmdFullyShardedDataParallel(model.qwen_embeddings, mesh=mesh)
    model.encoder          = SpmdFullyShardedDataParallel(model.encoder, mesh=mesh)
    model.dus              = SpmdFullyShardedDataParallel(model.dus, mesh=mesh)
    model.logic_adapter    = SpmdFullyShardedDataParallel(model.logic_adapter, mesh=mesh)

    trainable_params = list(model.dus.parameters()) + list(model.logic_adapter.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-2)

    if xm.is_master_ordinal() and args.wandb_project:
        wandb.init(project=args.wandb_project, config=vars(args))

    local_data_path = "./data/train" if args.data_path.startswith("gs://") else args.data_path
    local_val_path = "./data/val" if args.val_data_path and args.val_data_path.startswith("gs://") else args.val_data_path

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 151643

    dataloader = get_dataloader(
        local_data_path, batch_size=args.batch_size,
        max_length=args.max_length, tokenizer=tokenizer
    )
    val_dataloader = None
    if local_val_path:
        val_dataloader = get_dataloader(
            local_val_path, batch_size=args.batch_size,
            max_length=args.max_length, tokenizer=tokenizer
        )

    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2000, T_mult=1, eta_min=1e-6)

    model.train()
    step = 0

    from tqdm.auto import tqdm
    pbar = tqdm(total=args.max_steps, desc="Training Phase 3") if xm.is_master_ordinal() else None

    for epoch in range(args.epochs):
        for batch in dataloader:
            if step >= args.max_steps:
                break

            input_ids     = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Шардируем батч по FSDP-оси (КРИТИЧНО для TPU скорости)
            xs.mark_sharding(input_ids,      mesh, ("fsdp", None))
            xs.mark_sharding(attention_mask, mesh, ("fsdp", None))

            optimizer.zero_grad()

            fwd_outputs = model(input_ids, attention_mask=attention_mask,
                                low_noise_amp=args.low_noise_amp)
            loss, metrics = compute_phase3_loss(
                fwd_outputs, attention_mask, gamma=args.gamma,
                w_denoise=args.w_denoise, w_logic=args.w_logic
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            xm.optimizer_step(optimizer, barrier=True)
            scheduler.step()

            # --- Цикличный прогрев LR (Initial & Restart Warmups) ---
            current_optim_step = step + 1
            warmup_steps = min(1000, int(args.max_steps * 0.1))
            
            if current_optim_step <= warmup_steps:
                # 1. Начальный прогрев
                lr_warmup_factor = max(0.01, current_optim_step / warmup_steps)
                for idx_p, param_group in enumerate(optimizer.param_groups):
                    param_group['lr'] = scheduler.base_lrs[idx_p] * lr_warmup_factor
            else:
                # 2. Прогрев при автоматических рестартах (каждые 2000 шагов)
                rel_step = current_optim_step % 2000
                restart_warmup_steps = 200
                if rel_step < restart_warmup_steps:
                    lr_warmup_factor = max(0.01, rel_step / restart_warmup_steps)
                    for idx_p, param_group in enumerate(optimizer.param_groups):
                        param_group['lr'] = param_group['lr'] * lr_warmup_factor
            # ---------------------------------------------------------

            # Логирование
            if step % args.log_steps == 0:
                metrics_dict = {k: v.item() for k, v in metrics.items()}
                metrics_dict["loss"] = loss.item()
                metrics_dict["lr"]   = scheduler.get_last_lr()[0]
                if xm.is_master_ordinal():
                    if args.wandb_project:
                        wandb.log(metrics_dict, step=step)
                    if pbar:
                        pbar.set_postfix({
                            "loss":    f"{metrics_dict['loss']:.4f}",
                            "logic":   f"{metrics_dict.get('logic_loss', 0):.4f}",
                            "c_true":  f"{metrics_dict.get('c_true_mean', 0):.4f}",
                        })

            # Валидация
            if val_dataloader is not None and step > 0 and step % args.val_steps == 0:
                model.eval()
                v_loss_sum = torch.tensor(0.0, device=device)
                max_val_batches = 50
                num_v_batches = 0
                
                with torch.no_grad():
                    for v_step, v_batch in enumerate(val_dataloader):
                        if v_step >= max_val_batches: break
                        
                        v_input_ids = v_batch["input_ids"].to(device)
                        v_attention_mask = v_batch["attention_mask"].to(device)
                        xs.mark_sharding(v_input_ids, mesh, ('fsdp', None))
                        xs.mark_sharding(v_attention_mask, mesh, ('fsdp', None))
                        
                        v_fwd = model(v_input_ids, attention_mask=v_attention_mask, low_noise_amp=args.low_noise_amp)
                        v_loss, _ = compute_phase3_loss(
                            v_fwd, v_attention_mask, gamma=args.gamma,
                            w_denoise=args.w_denoise, w_logic=args.w_logic
                        )
                        v_loss_sum += v_loss
                        num_v_batches += 1
                        xm.mark_step()
                
                if num_v_batches > 0:
                    v_loss_avg = (v_loss_sum / num_v_batches).item()
                    if xm.is_master_ordinal():
                        if args.wandb_project: wandb.log({"val/loss": v_loss_avg}, step=step)
                        print(f"\n[VAL] Step {step} | Loss: {v_loss_avg:.4f}")
                model.train()

            # Сохранение чекпоинта и GCS Sync
            if step > 0 and step % args.save_steps == 0:
                xm.mark_step()
                if xm.is_master_ordinal():
                    os.makedirs(args.output_dir, exist_ok=True)
                    ckpt_path = os.path.join(args.output_dir, f"phase3_step_{step}.pth")
                    dus_state     = {k: v.cpu() for k, v in model.dus.state_dict().items()}
                    adapter_state = {k: v.cpu() for k, v in model.logic_adapter.state_dict().items()}
                    torch.save({"dus": dus_state, "logic_adapter": adapter_state}, ckpt_path)
                    print(f"\n[SAVE] Checkpoint saved → {ckpt_path}")
                    
                    if args.gcs_checkpoint_dir:
                        import subprocess
                        print(f"[GCS] Syncing checkpoints to {args.gcs_checkpoint_dir}...")
                        subprocess.Popen(["gcloud", "storage", "rsync", "-r", args.output_dir, args.gcs_checkpoint_dir])

            step += 1
            if pbar:
                pbar.update(1)

        if step >= args.max_steps:
            break

    if pbar:
        pbar.close()

    # Финальное сохранение
    xm.mark_step()
    if xm.is_master_ordinal():
        os.makedirs(args.output_dir, exist_ok=True)
        final_path = os.path.join(args.output_dir, "phase3_final.pth")
        torch.save({
            "dus":          {k: v.cpu() for k, v in model.dus.state_dict().items()},
            "logic_adapter": {k: v.cpu() for k, v in model.logic_adapter.state_dict().items()},
        }, final_path)
        print(f"[SAVE] Final weights saved → {final_path}")
        if args.wandb_project:
            wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BEBLaDII Phase 3 — DUS Logic Training")

    # Модели
    parser.add_argument("--teacher_model_path", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="HuggingFace path или локальная папка модели-учителя (DeepSeek-R1-7B)")
    parser.add_argument("--embedding_model_path", type=str,
                        default="Qwen/Qwen2.5-1.5B",
                        help="Модель, из которой берём embedding слой для LatentEncoder (1536d)")
    parser.add_argument("--teacher_dim", type=int, default=3584,
                        help="Размерность last_hidden_state учителя (3584 для DeepSeek-R1-7B)")

    # GCS пути к весам
    parser.add_argument("--encoder_weights_gs", type=str,
                        default="gs://bebladii-weigths-us/planB/phase1/checkpoints/phase1_vae_step_10000.pth")
    parser.add_argument("--dus_weights_gs", type=str,
                        default="gs://bebladii-weigths-us/kaggle_upload_1_2/AWAKENED_WEIGHTS_FINAL.pt")
    parser.add_argument("--data_path", type=str,
                        default="gs://bebladii-datasets-us/phase 3/train_data/data")
    parser.add_argument("--val_data_path", type=str, default="",
                        help="Путь к данным валидации. Пусто = выключено.")
    parser.add_argument("--gcs_checkpoint_dir", type=str, default="",
                        help="Бакет для бэкапа весов (например: gs://bebladii-weigths-us/planB/phase3/checkpoints/)")

    # Параметры обучения
    parser.add_argument("--output_dir",    type=str,   default="checkpoints/phase3")
    parser.add_argument("--batch_size",    type=int,   default=8)
    parser.add_argument("--max_length",    type=int,   default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs",        type=int,   default=1)
    parser.add_argument("--max_steps",     type=int,   default=40000)
    parser.add_argument("--log_steps",     type=int,   default=10)
    parser.add_argument("--val_steps",     type=int,   default=1000)
    parser.add_argument("--save_steps",    type=int,   default=5000)

    # Гиперпараметры Phase 3
    parser.add_argument("--low_noise_amp", type=float, default=0.5,
                        help="Амплитуда шума LowNoiseAmp (норма случайного вектора)")
    parser.add_argument("--gamma",    type=float, default=20.0,
                        help="Степень нелинейности штрафа стабильности (W_c = c_true^gamma)")
    parser.add_argument("--w_denoise", type=float, default=1.0, help="Вес Denoising loss")
    parser.add_argument("--w_logic",   type=float, default=1.0, help="Вес Logic distillation loss")

    parser.add_argument("--wandb_project", type=str, default="BEBLaDII-Phase3")

    args = parser.parse_args()

    # Загрузка ресурсов с GCS (только master rank)
    if xm.is_master_ordinal():
        import subprocess
        os.makedirs("./weights_cache", exist_ok=True)

        args.local_encoder_weights = "./weights_cache/encoder.pth"
        if not os.path.exists(args.local_encoder_weights):
            print(f"[GCS] Downloading encoder: {args.encoder_weights_gs}")
            subprocess.run(
                ["gcloud", "storage", "cp", args.encoder_weights_gs, args.local_encoder_weights],
                check=True
            )

        args.local_dus_weights = "./weights_cache/dus.pth"
        if not os.path.exists(args.local_dus_weights):
            print(f"[GCS] Downloading DUS weights: {args.dus_weights_gs}")
            subprocess.run(
                ["gcloud", "storage", "cp", args.dus_weights_gs, args.local_dus_weights],
                check=True
            )

        if args.data_path.startswith("gs://"):
            os.makedirs("./data/train", exist_ok=True)
            try:
                subprocess.run(["gcloud", "storage", "rsync", "-r", args.data_path, "./data/train/"], check=True)
                # Удаляем старый индекс, если он прилетел с GCS, чтобы пути не сломались
                index_path = "./data/train/_dataset_index.json"
                if os.path.exists(index_path):
                    os.remove(index_path)
            except Exception as e:
                print(f"[GCS] ERROR syncing train data: {e}")
                
        if args.val_data_path and args.val_data_path.startswith("gs://"):
            os.makedirs("./data/val", exist_ok=True)
            try:
                subprocess.run(["gcloud", "storage", "rsync", "-r", args.val_data_path, "./data/val/"], check=True)
                index_path = "./data/val/_dataset_index.json"
                if os.path.exists(index_path):
                    os.remove(index_path)
            except Exception as e:
                print(f"[GCS] ERROR syncing val data: {e}")
    else:
        args.local_encoder_weights = "./weights_cache/encoder.pth"
        args.local_dus_weights     = "./weights_cache/dus.pth"

    # Ждём пока master закончит скачивание
    xm.rendezvous("resources_prepared")

    train(args)
