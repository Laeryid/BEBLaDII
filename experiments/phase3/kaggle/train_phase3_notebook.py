# %% [markdown]
# # BEBLaDII Phase 3 Training (Kaggle T4 x2)
# *Автоматически сгенерировано через jupytext*

# %% [markdown]
# ## 1. Setup Environment
# Установка необходимых пакетов, которых может не быть в Kaggle по умолчанию.

# %%
# !pip install -q einops wandb

# %%
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

# Настройка путей для импорта src.
# Предполагается, что исходники проекта загружены как датасет (например, 'bebladii-src').
# Измените путь ниже на актуальный путь к вашему Kaggle Dataset с исходным кодом.
PROJECT_ROOT = "/kaggle/input/bebladii-src"
if PROJECT_ROOT not in sys.path and os.path.exists(PROJECT_ROOT):
    sys.path.insert(0, PROJECT_ROOT)

# Если исходников в Kaggle Dataset еще нет, загрузите их с помощью `sync_to_kaggle.ps1`

# Импорты проекта
try:
    from src.beb_la_dii.model.dus import DUSModel
    from src.beb_la_dii.model.vae import LatentEncoder
    from src.beb_la_dii.utils.data import get_dataloader
    from src.beb_la_dii.utils.loss import safe_normalize
except ImportError as e:
    print(
        f"Warning: Не удалось импортировать модули проекта. Убедитесь, что PROJECT_ROOT указан верно. Ошибка: {e}"
    )


# %% [markdown]
# ## 2. Configuration
# Использование класса Config вместо argparse для удобства работы в ячейках.


# %%
class Config:
    # Пути к базовым моделям (используйте Kaggle Models или Datasets)
    embedding_model_path = "/kaggle/input/qwen2.5-1.5b"
    modernbert_path = "/kaggle/input/modernbert-large"

    # Пути к данным
    dataset_path = "/kaggle/input/bebladii-planb-phase3-data/phase 3/train_data/data"

    # Пути к весам (из ваших предыдущих загрузок)
    local_encoder_weights = "/kaggle/input/bebladii-planb-phase3-data/planB_phase1_checkpoints_phase1_vae_step_20000.pth"
    local_dus_weights = "/kaggle/input/твое_старое_имя_датасета/AWAKENED_WEIGHTS_FINAL.pt"

    # Директории вывода (доступны для записи)
    output_dir = "/kaggle/working/checkpoints/phase3"

    # Гиперпараметры обучения
    batch_size = 16  # Можно увеличить для T4 x2 (по сравнению с 8 для TPU FSDP)
    max_length = 512
    learning_rate = 1e-4
    epochs = 1
    max_steps = 40000
    log_steps = 10
    val_steps = 1000
    save_steps = 1000

    # Специфика Phase 3
    low_noise_amp = 0.5
    gamma = 20.0
    w_denoise = 10.0
    w_identity = 5.0

    wandb_project = "BEBLaDII-Phase3-Kaggle"


args = Config()


# %% [markdown]
# ## 3. Utilities


# %%
class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Храним тени строго в float32
                self.shadow[name] = param.data.clone().detach().float()

    def step(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name].mul_(self.decay).add_(
                        param.data.float(), alpha=1.0 - self.decay
                    )

    def apply(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.backup[name] = param.data.clone().detach()
                    param.data.copy_(self.shadow[name].to(param.dtype))

    def restore(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(self.backup[name])
        self.backup = {}


# %% [markdown]
# ## 4. Model Definition


# %%
class BEBLaDIIPhase3(nn.Module):
    def __init__(
        self,
        embedding_model_path: str,
        modernbert_path: str,
        dus_weights: str | None,
        encoder_weights: str | None,
    ):
        super().__init__()

        # 1. Замороженный Qwen2.5-1.5B (только embedding)
        _qwen_base = AutoModel.from_pretrained(
            embedding_model_path, torch_dtype=torch.bfloat16
        )
        self.qwen_embeddings = _qwen_base.get_input_embeddings()
        del _qwen_base
        for p in self.qwen_embeddings.parameters():
            p.requires_grad = False

        # 2. Замороженный LatentEncoder
        self.encoder = LatentEncoder()
        if encoder_weights and os.path.exists(encoder_weights):
            state = torch.load(encoder_weights, map_location="cpu")
            self.encoder.load_state_dict(state, strict=False)
            print(
                f"[Init] LatentEncoder weights loaded from {encoder_weights}",
                flush=True,
            )
        else:
            print(
                f"[Init] WARN: encoder_weights not found ({encoder_weights}), using random init",
                flush=True,
            )

        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.to(torch.bfloat16)

        # 3. DUS Backbone (обучаемый)
        dus_wrapper = DUSModel.from_scratch(config={"base_model_id": modernbert_path}, weights_path=None)
        if dus_weights and os.path.exists(dus_weights):
            state = torch.load(dus_weights, map_location="cpu")
            if "latentBERT_state_dict" in state:
                state = state["latentBERT_state_dict"]
            elif "model_state_dict" in state:
                state = state["model_state_dict"]

            clean_state = {}
            for k, v in state.items():
                k_clean = k.replace("student.model.", "").replace("model.", "")
                clean_state[k_clean] = v

            model_sd = dus_wrapper.model.state_dict()
            matched = sum(1 for k in model_sd.keys() if k in clean_state)
            total_keys = len(model_sd.keys())
            dus_wrapper.model.load_state_dict(clean_state, strict=False)
            print(
                f"[Init] Awakened DUS weights loaded from {dus_weights} (Matched {matched}/{total_keys} params)",
                flush=True,
            )

        self.dus = dus_wrapper.model
        self.dus.to(torch.bfloat16)

        # 4. Confidence Embedding
        self.confidence_proj = nn.Sequential(
            nn.Linear(1, 256), nn.SiLU(), nn.Linear(256, 1024)
        )
        nn.init.zeros_(self.confidence_proj[-1].weight)
        nn.init.zeros_(self.confidence_proj[-1].bias)
        self.confidence_proj.to(torch.bfloat16)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "qwen_embeddings"):
            self.qwen_embeddings.eval()
        if hasattr(self, "encoder"):
            self.encoder.eval()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        low_noise_amp: float = 0.5,
    ) -> dict:
        with torch.no_grad():
            qwen_embeds = self.qwen_embeddings(input_ids)
            z_clean, _, _ = self.encoder(qwen_embeds)
            B, T, D = z_clean.shape

            # --- Windows generation ---
            noise_window_size = 5
            num_windows = 5
            block_size = T // num_windows

            starts = []
            for w in range(num_windows):
                min_start = 0
                max_start = max(min_start + 1, block_size - noise_window_size)
                start = torch.randint(
                    w * block_size + min_start,
                    w * block_size + max_start,
                    (B, 1),
                    device=z_clean.device,
                )
                starts.append(start)
            starts = torch.cat(starts, dim=1)

            noise_offsets = torch.arange(noise_window_size, device=z_clean.device).view(
                1, 1, noise_window_size
            )
            noise_window_indices = starts.unsqueeze(-1) + noise_offsets

            rand_noise = torch.rand(
                (B, num_windows, noise_window_size), device=z_clean.device
            )
            num_to_noise = torch.randint(
                1, noise_window_size + 1, (B, num_windows, 1), device=z_clean.device
            )
            _, noise_ranks = torch.sort(rand_noise, dim=-1)
            noise_subset_mask = (
                torch.arange(noise_window_size, device=z_clean.device).view(
                    1, 1, noise_window_size
                )
                < num_to_noise
            ).float()
            noise_subset_mask = torch.gather(
                noise_subset_mask, 2, noise_ranks.argsort(dim=-1)
            )

            flat_noise_indices = noise_window_indices.view(B, -1)
            full_noise_mask = torch.zeros(
                (B, T), device=z_clean.device, dtype=z_clean.dtype
            )
            full_noise_mask.scatter_(
                1, flat_noise_indices, noise_subset_mask.view(B, -1)
            )

            noise_mask = full_noise_mask * attention_mask.to(z_clean.dtype)

            noise = torch.randn_like(z_clean)
            noise = (
                safe_normalize(noise.float(), dim=-1).to(torch.bfloat16) * low_noise_amp
            )
            z_noisy = z_clean + noise * noise_mask.unsqueeze(-1)

            z_noisy = torch.where(
                noise_mask.unsqueeze(-1) > 0,
                safe_normalize(z_noisy.float(), dim=-1).to(torch.bfloat16),
                z_noisy,
            )

            z_clean_normed = safe_normalize(z_clean, dim=-1)
            z_noisy_normed = safe_normalize(z_noisy, dim=-1)
            c_true = torch.clamp((z_clean_normed * z_noisy_normed).sum(dim=-1), min=0.0)

        c_embed_raw = self.confidence_proj(c_true.unsqueeze(-1).to(torch.bfloat16))
        c_embed = safe_normalize(c_embed_raw.float(), dim=-1).to(torch.bfloat16) * 0.1
        dus_input = z_noisy.to(torch.bfloat16) + c_embed

        dus_outputs = self.dus(
            inputs_embeds=dus_input,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        pre_norm = dus_outputs.hidden_states[-1]
        clean_pre_norm = pre_norm - c_embed
        dus_final_raw = self.dus.final_norm(clean_pre_norm)

        dus_delta = dus_final_raw - z_noisy.to(torch.bfloat16)
        dus_final = safe_normalize(dus_final_raw.float(), dim=-1).to(torch.bfloat16)

        return {
            "z_clean": z_clean,
            "z_noisy": z_noisy,
            "c_true": c_true,
            "noise_mask": noise_mask,
            "dus_delta": dus_delta,
            "dus_final": dus_final,
            "attention_mask": attention_mask,
            "c_embed": c_embed,
            "dus_input": dus_input,
        }


# %% [markdown]
# ## 5. Loss Function


# %%
def compute_phase3_loss(
    outputs: dict,
    gamma: float = 20.0,
    w_denoise: float = 1.0,
    w_identity: float = 5.0,
) -> tuple[torch.Tensor, dict]:
    z_clean = outputs["z_clean"]
    z_noisy = outputs["z_noisy"]
    c_true = outputs["c_true"]
    noise_mask = outputs["noise_mask"]
    dus_delta = outputs["dus_delta"]
    dus_final = outputs["dus_final"]
    c_embed = outputs["c_embed"]
    dus_input = outputs["dus_input"]
    attn_f = outputs["attention_mask"].float()

    metrics = {}

    B, T, D = z_clean.size()
    active_tokens = attn_f.sum().clamp(min=1.0)
    noised_tokens = noise_mask.sum().clamp(min=1.0)

    # 4.1 Геометрическое: Prior Loss
    z_flat = dus_final.view(-1, D)
    mask_flat = attn_f.view(-1, 1)
    m_state = (z_flat * mask_flat).sum(dim=0) / active_tokens
    z_centered = z_flat - m_state

    v_state = (z_centered.pow(2) * mask_flat).sum(dim=0) / active_tokens
    var_floor = 1.0 / (D * 2)
    var_loss = F.relu(var_floor - v_state).mean()

    cov = (z_centered.T @ (z_centered * mask_flat)) / active_tokens
    cov_off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = cov_off_diag.pow(2).sum() / D

    prior_loss = m_state.pow(2).mean() + var_loss + 0.1 * cov_loss
    metrics["prior_loss"] = prior_loss.detach()
    metrics["cov_loss"] = cov_loss.detach()

    total_loss = 0.1 * prior_loss

    # 4.2 Диффузное: Denoising
    denoise_target = safe_normalize(z_clean.float(), dim=-1).to(torch.bfloat16)
    cos_denoise = (dus_final.float() * denoise_target.float()).sum(dim=-1)
    denoise_elementwise = 1.0 - cos_denoise
    denoise_loss = (denoise_elementwise * noise_mask).sum() / noised_tokens
    metrics["denoise_loss"] = denoise_loss.detach()

    total_loss = total_loss + w_denoise * denoise_loss

    # 4.4 Identity Penalty
    w_c = torch.pow(c_true.float(), gamma)
    identity_target = safe_normalize(z_noisy.float(), dim=-1).to(torch.bfloat16)
    cos_identity = (dus_final.float() * identity_target.float()).sum(dim=-1)
    penalty_elementwise = 1.0 - cos_identity
    identity_penalty = (penalty_elementwise * w_c * attn_f).sum() / active_tokens
    metrics["identity_penalty"] = identity_penalty.detach()

    total_loss = total_loss + w_identity * identity_penalty

    c_true_mean = (c_true * attn_f).sum() / active_tokens
    metrics["c_true_mean"] = c_true_mean.detach()

    # Для DataParallel лосс должен возвращать скаляр (мы уже сделали .sum() и усреднение)
    return total_loss, metrics


# %% [markdown]
# ## 6. Training Loop


# %%
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] Using device: {device}")
    num_gpus = torch.cuda.device_count()
    print(f"[Init] Available GPUs: {num_gpus}")

    print("[Init] Instantiating BEBLaDIIPhase3 model...", flush=True)

    model = BEBLaDIIPhase3(
        embedding_model_path=args.embedding_model_path,
        modernbert_path=args.modernbert_path,
        dus_weights=args.local_dus_weights,
        encoder_weights=args.local_encoder_weights,
    ).to(device)

    # Оборачиваем модель в DataParallel для распределения по T4 x2
    if num_gpus > 1:
        print(
            f"[Init] Wrapping model in DataParallel across {num_gpus} GPUs", flush=True
        )
        model = nn.DataParallel(model)

    # Оптимизатор
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.learning_rate, weight_decay=1e-2
    )

    # Из-за DataParallel доступ к модулям внутри EMA будет идти как model.module
    # но EMA обрабатывает named_parameters автоматически.
    ema = EMA(model, decay=0.998)

    if args.wandb_project:
        wandb.init(project=args.wandb_project, config=vars(args))

    # Даталоадеры
    try:
        dataloader = get_dataloader(
            stage="reasoning",
            batch_size=args.batch_size,
            max_length=args.max_length,
            split="train",
            val_ratio=0.0,
            data_dir=args.dataset_path,
        )
        val_dataloader = get_dataloader(
            stage="reasoning",
            batch_size=args.batch_size,
            max_length=args.max_length,
            split="val",
            val_ratio=0.0,
            data_dir=args.dataset_path,
        )
    except NameError:
        print(
            "WARN: get_dataloader is not defined. Please implement data loading logic or fix imports."
        )
        dataloader, val_dataloader = [], []  # Dummy

    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2000, T_mult=1, eta_min=1e-6)

    model.train()
    step = 0
    total_steps = (
        min(args.max_steps, len(dataloader) * args.epochs)
        if len(dataloader) > 0
        else args.max_steps
    )

    from tqdm.auto import tqdm

    pbar = tqdm(total=total_steps, desc="Training Phase 3")

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        for batch in dataloader:
            if step >= total_steps:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()

            fwd_outputs = model(
                input_ids,
                attention_mask=attention_mask,
                low_noise_amp=args.low_noise_amp,
            )

            # Если DataParallel - словари будут собраны (по первой размерности, B).
            # Функция потерь сможет обработать их корректно.
            loss, metrics = compute_phase3_loss(
                fwd_outputs,
                gamma=args.gamma,
                w_denoise=args.w_denoise,
                w_identity=args.w_identity,
            )

            # Для DataParallel если лосс возвращается как вектор (1 на GPU), нужно усреднить
            if loss.dim() > 0:
                loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

            optimizer.step()
            ema.step(model)
            scheduler.step()

            # Warmup логика
            current_optim_step = step + 1
            warmup_steps = min(1000, int(args.max_steps * 0.1))

            if current_optim_step <= warmup_steps:
                lr_warmup_factor = max(0.01, current_optim_step / warmup_steps)
                for idx_p, param_group in enumerate(optimizer.param_groups):
                    param_group["lr"] = scheduler.base_lrs[idx_p] * lr_warmup_factor
            else:
                rel_step = current_optim_step % 2000
                restart_warmup_steps = 200
                if rel_step < restart_warmup_steps:
                    lr_warmup_factor = max(0.01, rel_step / restart_warmup_steps)
                    for idx_p, param_group in enumerate(optimizer.param_groups):
                        param_group["lr"] = param_group["lr"] * lr_warmup_factor

            # Логирование
            if step % args.log_steps == 0:
                metrics_dict = {
                    k: (v.mean().item() if isinstance(v, torch.Tensor) else v)
                    for k, v in metrics.items()
                }
                metrics_dict["loss"] = loss.item()
                metrics_dict["lr"] = optimizer.param_groups[0]["lr"]

                if args.wandb_project:
                    wandb.log(metrics_dict, step=step)

                pbar.set_postfix(
                    {
                        "loss": f"{metrics_dict['loss']:.4f}",
                        "c_true": f"{metrics_dict.get('c_true_mean', 0):.4f}",
                    }
                )

            # Сохранение чекпоинта
            if step > 0 and step % args.save_steps == 0:
                ckpt_path = os.path.join(args.output_dir, f"phase3_step_{step}.pth")

                # Извлекаем state_dict с учетом DataParallel (module.)
                actual_model = (
                    model.module if isinstance(model, nn.DataParallel) else model
                )
                dus_state = {
                    k: v.cpu() for k, v in actual_model.dus.state_dict().items()
                }
                proj_state = {
                    k: v.cpu()
                    for k, v in actual_model.confidence_proj.state_dict().items()
                }

                ema.apply(model)
                dus_ema_state = {
                    k: v.cpu() for k, v in actual_model.dus.state_dict().items()
                }
                proj_ema_state = {
                    k: v.cpu()
                    for k, v in actual_model.confidence_proj.state_dict().items()
                }
                ema.restore(model)

                torch.save(
                    {
                        "dus": dus_state,
                        "confidence_proj": proj_state,
                        "dus_ema": dus_ema_state,
                        "confidence_proj_ema": proj_ema_state,
                    },
                    ckpt_path,
                )
                print(f"\n[SAVE] Checkpoint saved → {ckpt_path}")

            step += 1
            pbar.update(1)

        if step >= total_steps:
            break

    pbar.close()

    # Финальное сохранение
    final_path = os.path.join(args.output_dir, "phase3_final.pth")
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    dus_state = {k: v.cpu() for k, v in actual_model.dus.state_dict().items()}
    proj_state = {
        k: v.cpu() for k, v in actual_model.confidence_proj.state_dict().items()
    }

    ema.apply(model)
    dus_ema_state = {k: v.cpu() for k, v in actual_model.dus.state_dict().items()}
    proj_ema_state = {
        k: v.cpu() for k, v in actual_model.confidence_proj.state_dict().items()
    }
    ema.restore(model)

    torch.save(
        {
            "dus": dus_state,
            "confidence_proj": proj_state,
            "dus_ema": dus_ema_state,
            "confidence_proj_ema": proj_ema_state,
        },
        final_path,
    )
    print(f"[SAVE] Final weights saved → {final_path}")
    if args.wandb_project:
        wandb.finish()


# %%
if __name__ == "__main__":
    train()
