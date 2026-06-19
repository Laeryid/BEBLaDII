import os
import sys

# Настройка переменных окружения для TPU v4
def setup_env():
    os.environ["PJRT_DEVICE"] = "TPU"
    os.environ["XLA_USE_BF16"] = "1"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "2,2,1" 
    os.environ["TPU_NUM_DEVICES"] = "4"
    os.environ["XLA_USE_SPMD"] = "1"

setup_env()

# Добавляем корень проекта в пути поиска, чтобы Python видел папку src
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.experimental.xla_sharding as xs
import torch_xla.runtime as xr
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader

from src.beb_la_dii.model.vae import LatentDecoder, LatentEncoder
from src.beb_la_dii.utils.loss import safe_cosine_similarity, safe_normalize

# TPU single-process initialization
xr.use_spmd()

def setup_spmd_mesh():
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (num_devices, 1)
    device_ids = np.array(range(num_devices))
    mesh = xs.Mesh(device_ids, mesh_shape, ("fsdp", "model"))
    xs.set_global_mesh(mesh)
    return mesh


from transformers import AutoModelForCausalLM, AutoTokenizer

class BEBLaDIIVAE(nn.Module):
    def __init__(self, qwen_model_path, vae_encoder, vae_decoder):
        super().__init__()
        self.qwen = AutoModelForCausalLM.from_pretrained(
            qwen_model_path, torch_dtype=torch.bfloat16
        )
        # Замораживаем Qwen полностью
        for param in self.qwen.parameters():
            param.requires_grad = False

        self.encoder = vae_encoder
        self.decoder = vae_decoder

    def forward(self, input_ids, attention_mask=None):
        # 1. Получаем эмбеддинги из замороженного Qwen (teacher embedder)
        with torch.no_grad():
            inputs_embeds = self.qwen.get_input_embeddings()(input_ids)  # (B, T, 1536)

        # 2. VAE Encoder -> Latent Space
        z, mu, logvar = self.encoder(inputs_embeds)  # z: (B, T, 1024)

        # 3. VAE Decoder
        projected_embeds = self.decoder(z)  # (B, T, 1536)

        # 4. Прогоняем через Qwen Decoder для получения логитов (передаем кастомные эмбеддинги)
        outputs = self.qwen(
            inputs_embeds=projected_embeds, attention_mask=attention_mask
        )
        logits = outputs.logits

        return logits, z, mu, logvar, inputs_embeds


def compute_phase1_loss(
    logits, labels, z, mu, logvar, attention_mask, kl_beta=0.01, contrastive_lambda=0.01
):
    metrics = {}

    # 1. Reconstruction Loss (Cross-Entropy)
    # Сдвиг для авторегрессионного моделирования (Next-token prediction)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = (
        attention_mask[..., 1:].contiguous() if attention_mask is not None else None
    )

    loss_fct = nn.CrossEntropyLoss(reduction="none")
    ce_loss_raw = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    ce_loss_raw = ce_loss_raw.view(shift_labels.size())

    if shift_mask is not None:
        ce_loss = (ce_loss_raw * shift_mask).sum() / (shift_mask.sum() + 1e-6)
    else:
        ce_loss = ce_loss_raw.mean()

    metrics["ce_loss"] = ce_loss.detach()
    total_loss = ce_loss

    # 2. Variance-Only KL Divergence (logvar)
    if attention_mask is not None:
        mask_f = attention_mask.unsqueeze(-1).float()
        logvar_f = logvar * mask_f
    else:
        logvar_f = logvar

    # Оставляем штраф только за дисперсию (std -> 1), убираем mu
    kl_raw = -0.5 * torch.mean(1 + logvar_f - logvar_f.exp(), dim=-1)
    kl_clamped = torch.clamp(kl_raw, min=0.5)  # Free bits = 0.5

    if attention_mask is not None:
        kl_loss = (kl_clamped * attention_mask).sum() / (attention_mask.sum() + 1e-6)
    else:
        kl_loss = kl_clamped.mean()

    total_loss = total_loss + kl_beta * kl_loss
    metrics["kl_loss"] = kl_loss.detach()

    # 3. Covariance Loss & Prior (Spherical Repulsion & Isotropy)
    D = z.size(-1)
    if attention_mask is not None:
        mask_flat = attention_mask.view(-1, 1).float()
        z_flat = z.view(-1, D)
        N = mask_flat.sum().clamp(min=1.0)

        m_state = (z_flat * mask_flat).sum(dim=0) / N
        z_centered = z_flat - m_state.unsqueeze(0)
        z_masked = z_centered * mask_flat

        z_normed = safe_normalize(z_masked, dim=0)
        cov = z_normed.T @ z_normed
    else:
        m_state = z.mean(dim=(0, 1))
        z_centered = z - m_state.view(1, 1, -1)
        z_normed = safe_normalize(z_centered.view(-1, D), dim=0)
        cov = z_normed.T @ z_normed

    cov_off_diag = cov - torch.diag(torch.diag(cov))
    _cov_abs = cov_off_diag.abs()
    cov_loss = (
        2.0
        * torch.where(_cov_abs < 1.0, 0.5 * cov_off_diag.pow(2), _cov_abs - 0.5).sum()
        / D
    )
    metrics["cov_loss"] = cov_loss.detach()

    # Effective Dimensionality (TPU-friendly alternative to rank1_ratio)
    eff_dim = (cov.trace().pow(2)) / (cov.pow(2).sum() + 1e-8)
    metrics["effective_dim"] = eff_dim.detach()

    prior_loss = m_state.pow(2).mean() + 0.3 * cov_loss

    total_loss = total_loss + 0.1 * prior_loss
    metrics["prior_loss"] = prior_loss.detach()

    # 4. Uniformity Loss (Uniform Spherical Repulsion)
    # Расталкиваем все токены друг от друга для равномерного покрытия сферы.
    B, T, D = z.shape
    z_flat = z.view(-1, D)
    mask_flat = attention_mask.view(-1).float() if attention_mask is not None else torch.ones(B*T, device=z.device)

    # Subsample for performance
    max_tokens = min(4096, B * T)
    # Используем multinomial с весами по маске для обхода pad токенов
    weights = mask_flat + 1e-6
    indices = torch.multinomial(weights, num_samples=max_tokens, replacement=True)
    
    z_sampled = z_flat[indices]
    z_normed_active = safe_normalize(z_sampled, dim=-1)
    
    # Uniformity Loss (Wang & Isola)
    # sq_pdist = ||zi - zj||^2 = 2 - 2 * (zi @ zj)
    sq_pdist = 2.0 - 2.0 * (z_normed_active @ z_normed_active.T)
    
    # t = 2.0 (параметр концентрации)
    t = 2.0
    uniformity_raw = torch.logsumexp(-t * sq_pdist, dim=-1)
    contrastive_loss = uniformity_raw.mean()

    total_loss = total_loss + contrastive_lambda * contrastive_loss
    metrics["contrastive_loss"] = contrastive_loss.detach()

    # Сферичность (Norm CV)
    s_norms = z.norm(dim=-1)
    if attention_mask is not None:
        n_active = attention_mask.sum().clamp(min=1e-6)
        _norm_mean = (s_norms * attention_mask).sum() / n_active
        _norm_var = (((s_norms - _norm_mean) * attention_mask) ** 2).sum() / n_active
        _norm_std = torch.sqrt(_norm_var + 1e-8)
    else:
        _norm_mean = s_norms.mean().clamp(min=1e-6)
        _norm_std = s_norms.std(unbiased=False)
    metrics["norm_cv_l40_raw"] = (_norm_std / _norm_mean).detach()

    return total_loss, metrics


from transformers import AutoTokenizer

def get_dataloader(data_path, batch_size=8, max_length=1024, tokenizer=None):
    from indexed_parquet_dataset import IndexedParquetDataset

    print(f"Loading parquet dataset from {data_path} using IndexedParquetDataset...")
    ds = IndexedParquetDataset.from_folder(
        data_path, pattern="*.parquet", auto_fill=True
    )

    # Идеальный O(1) shuffle, чтение батчами
    ds = ds.shuffle(seed=42, rg_buffer=8)
    print(f"Total samples: {len(ds)}")

    def collate_fn(batch):
        # Если есть колонка "text", токенизируем на лету
        if tokenizer is not None and "text" in batch[0] and batch[0]["text"] is not None:
            texts = [item["text"] for item in batch]
            encoded = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
            return {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"]
            }

        # Иначе используем готовые input_ids (fallback)
        input_ids = []
        attention_mask = []
        for item in batch:
            seq = item.get("input_ids")
            if seq is None:
                seq = []
            elif not isinstance(seq, list):
                seq = list(seq)
            
            seq = seq[:max_length]
            pad_len = max_length - len(seq)
            # pad_token_id = 151643 for Qwen
            padded_seq = seq + [151643] * pad_len
            mask = [1] * len(seq) + [0] * pad_len
            input_ids.append(padded_seq)
            attention_mask.append(mask)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    return DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        drop_last=True,
        num_workers=4,
        prefetch_factor=2
    )


def train(args):
    mesh = setup_spmd_mesh()
    device = xm.xla_device()

    # Инициализация компонент
    encoder = LatentEncoder()
    decoder = LatentDecoder()

    model = BEBLaDIIVAE(
        qwen_model_path=args.model_path, vae_encoder=encoder, vae_decoder=decoder
    ).to(device)

    # Оборачиваем энкодер и декодер в FSDP/SPMD
    # Qwen заморожен, поэтому его оборачивать не нужно (или можно обернуть модель целиком)
    from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel
    model = SpmdFullyShardedDataParallel(model, mesh=mesh)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-2
    )

    if xm.is_master_ordinal() and args.wandb_project:
        wandb.init(project=args.wandb_project, config=vars(args))

    # Если мы передали gs://, к этому моменту данные уже скачаны в ./data в блоке __main__
    local_data_path = "./data/train" if args.data_path.startswith("gs://") else args.data_path
    local_val_path = "./data/val" if args.val_data_path and args.val_data_path.startswith("gs://") else args.val_data_path
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 151643
    
    dataloader = get_dataloader(
        local_data_path, batch_size=args.batch_size, max_length=args.max_length, tokenizer=tokenizer
    )
    
    val_dataloader = None
    if local_val_path:
        val_dataloader = get_dataloader(
            local_val_path, batch_size=args.batch_size, max_length=args.max_length, tokenizer=tokenizer
        )

    from transformers import get_cosine_schedule_with_warmup
    num_training_steps = args.max_steps
    num_warmup_steps = int(num_training_steps * 0.1) # 10% warmup
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    model.train()
    step = 0

    print("Starting training...")
    from tqdm.auto import tqdm
    
    pbar = None
    if xm.is_master_ordinal():
        pbar = tqdm(total=args.max_steps, desc="Training Phase 1")

    for epoch in range(args.epochs):
        for batch in dataloader:
            if step >= args.max_steps:
                break
                
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # SPMD Sharding батча по оси FSDP (КРИТИЧНО ДЛЯ СКОРОСТИ TPU)
            xs.mark_sharding(input_ids, mesh, ('fsdp', None))
            xs.mark_sharding(attention_mask, mesh, ('fsdp', None))

            optimizer.zero_grad()

            logits, z, mu, logvar, teacher_embeds = model(
                input_ids, attention_mask=attention_mask
            )

            loss, metrics = compute_phase1_loss(
                logits=logits,
                labels=input_ids,
                z=z,
                mu=mu,
                logvar=logvar,
                attention_mask=attention_mask,
                kl_beta=args.kl_beta,
                contrastive_lambda=args.contrastive_lambda,
            )

            loss.backward()
            xm.optimizer_step(optimizer, barrier=True)
            scheduler.step()

            if step % args.log_steps == 0:
                metrics_dict = {k: v.item() for k, v in metrics.items()}
                metrics_dict["loss"] = loss.item()
                metrics_dict["lr"] = scheduler.get_last_lr()[0]
                metrics_str = " | ".join(
                    [f"{k}: {v:.4f}" for k, v in metrics_dict.items() if k != "lr"]
                )
                
                if xm.is_master_ordinal():
                    if args.wandb_project:
                        wandb.log(metrics_dict, step=step)
                    if pbar is not None:
                        pbar.set_postfix({
                            "loss": f"{loss.item():.4f}",
                            "ce": f"{metrics_dict.get('ce_loss', 0):.4f}",
                            "lr": f"{metrics_dict['lr']:.2e}"
                        })

            # Validation Loop
            if val_dataloader is not None and step > 0 and step % args.val_steps == 0:
                model.eval()
                v_loss_sum = torch.tensor(0.0, device=device)
                v_ce_sum = torch.tensor(0.0, device=device)
                
                # Ограничиваем валидацию 50 батчами для скорости
                max_val_batches = 50 
                num_v_batches = 0
                
                with torch.no_grad():
                    for v_step, v_batch in enumerate(val_dataloader):
                        if v_step >= max_val_batches: break
                        
                        v_input_ids = v_batch["input_ids"].to(device)
                        v_attention_mask = v_batch["attention_mask"].to(device)
                        xs.mark_sharding(v_input_ids, mesh, ('fsdp', None))
                        xs.mark_sharding(v_attention_mask, mesh, ('fsdp', None))
                        
                        v_logits, v_z, v_mu, v_logvar, _ = model(v_input_ids, attention_mask=v_attention_mask)
                        v_loss, v_metrics = compute_phase1_loss(
                            logits=v_logits, labels=v_input_ids, z=v_z, mu=v_mu, logvar=v_logvar,
                            attention_mask=v_attention_mask, kl_beta=args.kl_beta, contrastive_lambda=args.contrastive_lambda
                        )
                        v_loss_sum += v_loss
                        v_ce_sum += v_metrics["ce_loss"]
                        num_v_batches += 1
                        
                        # Разрезаем граф, чтобы он не слипался в один гигантский
                        xm.mark_step()
                
                if num_v_batches > 0:
                    # Вызываем .item() ровно 1 раз в конце цикла!
                    v_loss_avg = (v_loss_sum / num_v_batches).item()
                    v_ce_avg = (v_ce_sum / num_v_batches).item()
                    
                    if xm.is_master_ordinal():
                        val_log = {
                            "val/loss": v_loss_avg,
                            "val/ce_loss": v_ce_avg
                        }
                        if args.wandb_project:
                            wandb.log(val_log, step=step)
                        print(f"\n--- [VAL] Step {step} | Loss: {v_loss_avg:.4f} | CE: {v_ce_avg:.4f} ---")
                
                # Синхронизация после валидации
                # xm.rendezvous("validation_done") - removed earlier
                model.train()

            # Промежуточное сохранение
            if step > 0 and step % args.save_steps == 0:
                if xm.is_master_ordinal():
                    ckpt_path = os.path.join(args.output_dir, f"phase1_vae_step_{step}.pth")
                    os.makedirs(args.output_dir, exist_ok=True)
                    # Извлекаем веса из FSDP-обёртки, отфильтровывая Qwen
                    full_state = model.state_dict()
                    cpu_encoder = {k.replace('encoder.', ''): v.cpu() for k, v in full_state.items() if k.startswith('encoder.')}
                    cpu_decoder = {k.replace('decoder.', ''): v.cpu() for k, v in full_state.items() if k.startswith('decoder.')}
                    torch.save({"encoder": cpu_encoder, "decoder": cpu_decoder}, ckpt_path)
                    print(f"\n[SAVE] Intermediate checkpoint saved to {ckpt_path}")

            step += 1
            if pbar is not None:
                pbar.update(1)
                
            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break

    if pbar is not None:
        pbar.close()

    # Save weights
    if xm.is_master_ordinal():
        save_path = os.path.join(args.output_dir, "phase1_vae_weights.pth")
        os.makedirs(args.output_dir, exist_ok=True)
        full_state = model.state_dict()
        cpu_encoder = {k.replace('encoder.', ''): v.cpu() for k, v in full_state.items() if k.startswith('encoder.')}
        cpu_decoder = {k.replace('decoder.', ''): v.cpu() for k, v in full_state.items() if k.startswith('decoder.')}
        torch.save(
            {"encoder": cpu_encoder, "decoder": cpu_decoder},
            save_path,
        )
        print(f"Saved weights to {save_path}")
        if args.wandb_project:
            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-1.5B",
        help="Path to frozen Qwen 1.5B",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="gs://bebladii-datasets-us/phase 3/train_data/data",
        help="Train dataset path",
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        default="",
        help="Validation dataset path (empty to disable)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="checkpoints/phase1", help="Output directory"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--kl_beta", type=float, default=0.01)
    parser.add_argument("--contrastive_lambda", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=40000)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--val_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="BEBLaDII-Phase1",
        help="wandb project name (empty to disable)",
    )
    args = parser.parse_args()

    # Синхронизация данных с GCS перед стартом TPU
    if xm.is_master_ordinal():
        print("--- [MASTER] Подготовка ресурсов ---")
        os.makedirs("./data/train", exist_ok=True)
        os.makedirs("./data/val", exist_ok=True)
        try:
            import subprocess
            if args.data_path.startswith("gs://"):
                subprocess.run(["gcloud", "storage", "rsync", "-r", args.data_path, "./data/train/"], check=True)
            if args.val_data_path and args.val_data_path.startswith("gs://"):
                subprocess.run(["gcloud", "storage", "rsync", "-r", args.val_data_path, "./data/val/"], check=True)
        except Exception as e:
            print(f"--- [MASTER] Ошибка GCS: {e} ---")
            
    xm.rendezvous("resources_prepared")

    train(args)
