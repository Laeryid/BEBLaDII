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

    # 2. KL Divergence (mu, logvar)
    if attention_mask is not None:
        mask_f = attention_mask.unsqueeze(-1).float()
        mu_f = mu * mask_f
        logvar_f = logvar * mask_f
    else:
        mu_f, logvar_f = mu, logvar

    kl_raw = -0.5 * torch.mean(1 + logvar_f - mu_f.pow(2) - logvar_f.exp(), dim=-1)
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

        v_state = (z_masked**2).sum(dim=0) / N.clamp(min=2.0)

        z_normed = safe_normalize(z_masked, dim=0)
        cov = z_normed.T @ z_normed
    else:
        m_state = z.mean(dim=(0, 1))
        v_state = z.var(dim=(0, 1), unbiased=False)
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

    _v_diff = v_state - 1.0
    _v_abs = _v_diff.abs()
    prior_loss = (
        m_state.pow(2).mean()
        + 2.0 * torch.where(_v_abs < 1.0, 0.5 * _v_diff.pow(2), _v_abs - 0.5).mean()
        + 0.3 * cov_loss
    )

    total_loss = total_loss + 0.1 * prior_loss
    metrics["prior_loss"] = prior_loss.detach()

    # 4. Contrastive Loss (Uniform Spherical Repulsion)
    # Расталкиваем все токены друг от друга, чтобы они равномерно распределились по сфере.
    # Мы НЕ используем teacher_embeds (Qwen), чтобы не переносить его анизотропную структуру.
    if attention_mask is not None:
        mask_flat = attention_mask.view(-1).bool()
        z_active = z.view(-1, D)[mask_flat]
    else:
        z_active = z.view(-1, D)

    # Subsample for performance if batch is too large
    max_tokens = 4096
    if z_active.size(0) > max_tokens:
        indices = torch.randperm(z_active.size(0), device=z_active.device)[:max_tokens]
        z_active = z_active[indices]

    z_normed_active = safe_normalize(z_active, dim=-1)
    sim_matrix = z_normed_active @ z_normed_active.T

    # Температура для Contrastive
    tau = 0.1
    sim_matrix.fill_diagonal_(-1e4)  # Исключаем self-similarity

    # Минимизируем сходство с другими токенами (Uniformity / InfoNCE-style repulsion)
    contrastive_loss = torch.logsumexp(sim_matrix / tau, dim=-1).mean()

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


from indexed_parquet_dataset import IndexedParquetDataset


def get_dataloader(data_path, batch_size=8, max_length=1024):
    print(f"Loading parquet dataset from {data_path} using IndexedParquetDataset...")
    ds = IndexedParquetDataset.from_folder(
        data_path, pattern="*.parquet", auto_fill=True
    )

    # Индексированный O(1) shuffle, без загрузки в память
    ds = ds.shuffle(seed=42, rg_buffer=8)
    print(f"Total samples: {len(ds)}")

    def collate_fn(batch):
        input_ids = []
        attention_mask = []
        for item in batch:
            seq = item["input_ids"][:max_length]
            pad_len = max_length - len(seq)
            padded_seq = seq + [0] * pad_len  # pad_token_id = 0 for Qwen
            mask = [1] * len(seq) + [0] * pad_len
            input_ids.append(padded_seq)
            attention_mask.append(mask)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    return DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True
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
    local_data_path = "./data" if args.data_path.startswith("gs://") else args.data_path
    
    dataloader = get_dataloader(
        local_data_path, batch_size=args.batch_size, max_length=args.max_length
    )

    model.train()
    step = 0

    print("Starting training...")
    for epoch in range(args.epochs):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

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
            xm.optimizer_step(optimizer)

            if step % args.log_steps == 0:
                metrics_dict = {k: v.item() for k, v in metrics.items()}
                metrics_dict["loss"] = loss.item()
                metrics_str = " | ".join(
                    [f"{k}: {v:.4f}" for k, v in metrics_dict.items()]
                )
                print(f"Step {step} | {metrics_str}")

                if xm.is_master_ordinal() and args.wandb_project:
                    wandb.log(metrics_dict, step=step)

            step += 1
            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break

    # Save weights
    if xm.is_master_ordinal():
        save_path = os.path.join(args.output_dir, "phase1_vae_weights.pth")
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(
            {"encoder": encoder.state_dict(), "decoder": decoder.state_dict()},
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
        help="Dataset path",
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
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="BEBLaDII-Phase1",
        help="wandb project name (empty to disable)",
    )
    args = parser.parse_args()

    # Синхронизация данных с GCS перед стартом TPU
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if rank == 0:
        if args.data_path.startswith("gs://"):
            print("--- [RANK 0] Подготовка ресурсов ---")
            os.makedirs("./data", exist_ok=True)
            try:
                import subprocess
                subprocess.run(["gcloud", "storage", "rsync", "-r", args.data_path, "./data/"], check=True)
            except Exception as e:
                print(f"--- [RANK 0] Ошибка GCS: {e} ---")
                
        with open("/tmp/resources_prepared.flag", "w") as f: f.write("ok")
    else:
        import time
        while not os.path.exists("/tmp/resources_prepared.flag"):
            time.sleep(1)

    train(args)
