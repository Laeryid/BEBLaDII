# %% [markdown]
# # BEBLaDII Phase 2 Decoder Training (Kaggle T4 x2)
# *Автоматически сгенерировано через jupytext*

# %% [markdown]
# ## 1. Setup Environment

# %%
import os
import subprocess
import sys

# Установка необходимых пакетов для Kaggle
print("Установка зависимостей...")
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "einops",
        "wandb",
        "indexed_parquet_dataset",
    ],
    check=True,
)

PROJECT_ROOT = "/kaggle/working/BEBLaDII"
REPO_URL = "https://github.com/Laeryid/BEBLaDII.git"

if not os.path.exists(PROJECT_ROOT):
    print(f"Клонирование репозитория из {REPO_URL}...")
    subprocess.run(["git", "clone", REPO_URL, PROJECT_ROOT], check=True)
else:
    print("Репозиторий уже существует. Выполняю git pull...")
    subprocess.run(["git", "-C", PROJECT_ROOT, "pull"], check=True)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from src.beb_la_dii.model.modern_decoder import ModernLatentDecoder
    from src.beb_la_dii.model.vae import LatentEncoder
except ImportError as e:
    print(
        f"Warning: Не удалось импортировать модули проекта. Убедитесь, что PROJECT_ROOT указан верно. Ошибка: {e}"
    )


# %% [markdown]
# ## 2. Configuration


# %%
class Config:
    # Пути к моделям (Qwen2.5-1.5B используется для эмбеддингов и LM Head)
    qwen_model_path = "/kaggle/input/datasets/ragnar123/qwen2-5-1-5b"

    # Пути к данным
    dataset_path = "/kaggle/input/datasets/bogdanbuliakov/bebladii-planb-phase3-data/phase 3/train_data/data"

    # Пути к весам
    local_encoder_weights = "/kaggle/input/datasets/bogdanbuliakov/bebladii-planb-phase3-data/planB_phase1_checkpoints_phase1_vae_step_20000.pth"
    local_dus_weights = "/kaggle/input/datasets/bogdanbuliakov/bebladii-phase1-awakaned-weights/AWAKENED_WEIGHTS_FINAL.pt"

    # Директория вывода
    output_dir = "/kaggle/working/checkpoints/phase2"

    # Параметры архитектуры и обучения
    batch_size = 2
    grad_accum_steps = 4
    max_length = 1024
    learning_rate = 1e-4
    epochs = 1
    max_steps = 10000
    log_steps = 10
    save_steps = 1000

    wandb_project = "BEBLaDII-Phase2-Kaggle"


args = Config()


# %% [markdown]
# ## 3. Model Definition


# %%
class Phase2DecoderWrapper(nn.Module):
    def __init__(self, encoder, decoder, qwen_embed_weight, qwen_lm_head_weight):
        super().__init__()
        self.encoder = encoder

        # Строгая заморозка энкодера (он обучен в Фазе 1)
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.decoder = decoder
        # Матрицы Qwen передаем как тензоры
        self.qwen_embed_weight = nn.Parameter(qwen_embed_weight, requires_grad=False)
        self.qwen_lm_head_weight = nn.Parameter(
            qwen_lm_head_weight, requires_grad=False
        )

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "encoder"):
            self.encoder.eval()

    def forward(self, input_ids, attention_mask=None):
        # 1. Извлекаем эмбеддинги вручную из сохраненной матрицы Qwen
        inputs_embeds = F.embedding(input_ids, self.qwen_embed_weight)

        # 2. VAE Encoder -> Latent Space Z (заморожен)
        with torch.no_grad():
            z, _, _ = self.encoder(inputs_embeds)

        # 3. Наш новый ModernLatentDecoder
        projected_embeds = self.decoder(z, attention_mask)

        # 4. Проекция обратно в токены через сохраненную матрицу Qwen LM Head
        logits = F.linear(projected_embeds, self.qwen_lm_head_weight)

        return logits


# %% [markdown]
# ## 4. Utilities and Data Loading


# %%
def get_dataloader(data_path, batch_size=4, max_length=1024, tokenizer=None):
    from indexed_parquet_dataset import IndexedParquetDataset

    print(f"Loading parquet dataset from {data_path} using IndexedParquetDataset...")
    ds = IndexedParquetDataset.from_folder(
        data_path, pattern="*.parquet", auto_fill=True
    )
    ds = ds.shuffle(seed=42, rg_buffer=8)

    def collate_fn(batch):
        if (
            tokenizer is not None
            and "text" in batch[0]
            and batch[0]["text"] is not None
        ):
            texts = [item["text"] for item in batch]
            encoded = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }

        input_ids, attention_mask = [], []
        for item in batch:
            seq = item.get("input_ids", [])
            if not isinstance(seq, list):
                seq = list(seq)
            seq = seq[:max_length]
            pad_len = max_length - len(seq)
            # 151643 - padding для Qwen
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
        prefetch_factor=2,
    )


# %% [markdown]
# ## 5. Training Loop


# %%
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] Using device: {device}")
    num_gpus = torch.cuda.device_count()
    print(f"[Init] Available GPUs: {num_gpus}")

    if args.wandb_project:
        try:
            from kaggle_secrets import UserSecretsClient

            user_secrets = UserSecretsClient()
            wandb_api = user_secrets.get_secret("WANDB_API_KEY")
            wandb.login(key=wandb_api)
            print("[Init] W&B login successful via Kaggle Secrets.")

            try:
                gcp_sa = user_secrets.get_secret("GCP_SA_JSON")
                with open("gcp_sa.json", "w") as f:
                    f.write(gcp_sa)
                subprocess.run(
                    [
                        "gcloud",
                        "auth",
                        "activate-service-account",
                        "--key-file",
                        "gcp_sa.json",
                    ],
                    check=True,
                )
                print("[Init] GCP Authentication successful via Kaggle Secrets.")
            except Exception as e_gcp:
                print(f"[Init] WARN: Could not authenticate GCP automatically: {e_gcp}")

        except Exception as e:
            print(f"[Init] WARN: Could not login to W&B automatically: {e}")

        wandb.init(project=args.wandb_project, config=vars(args))

    # 1. Загрузка Qwen и извлечение матриц
    print(f"Loading weights from {args.qwen_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 151643

    # Загружаем модель только для того, чтобы вытащить 2 матрицы
    qwen = AutoModelForCausalLM.from_pretrained(
        args.qwen_model_path, torch_dtype=torch.bfloat16
    )
    embed_weight = qwen.model.embed_tokens.weight.detach().clone().to(torch.bfloat16)
    lm_head_weight = qwen.lm_head.weight.detach().clone().to(torch.bfloat16)

    # СУПЕР-ОПТИМИЗАЦИЯ: Удаляем тушу Qwen из памяти!
    del qwen
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        "Qwen deleted from memory. Kept only Embeddings and LM Head. Massive VRAM saving!"
    )

    # 2. Загрузка замороженного Encoder из Фазы 1
    encoder = LatentEncoder().to(torch.bfloat16)
    if os.path.exists(args.local_encoder_weights):
        print(f"Loading Phase 1 Encoder from {args.local_encoder_weights}")
        ckpt = torch.load(args.local_encoder_weights, map_location="cpu")
        encoder.load_state_dict(ckpt["encoder"] if "encoder" in ckpt else ckpt)
    else:
        print(
            f"Warning: {args.local_encoder_weights} not found. Using random LatentEncoder."
        )

    # 3. Инициализация Декодера
    decoder = ModernLatentDecoder(
        latent_dim=1024,
        qwen_dim=1536,
        num_layers=3,
    ).to(torch.bfloat16)

    # 4. Сборка финальной модели
    model = Phase2DecoderWrapper(encoder, decoder, embed_weight, lm_head_weight)
    model = model.to(device)

    # Оборачиваем модель в DataParallel
    if num_gpus > 1:
        print(
            f"[Init] Wrapping model in DataParallel across {num_gpus} GPUs", flush=True
        )
        model = nn.DataParallel(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    loss_fct = nn.CrossEntropyLoss(reduction="none")

    model.train()
    print("Model initialized. Ready for training loop.")

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        dataloader = get_dataloader(
            args.dataset_path,
            batch_size=args.batch_size,
            max_length=args.max_length,
            tokenizer=tokenizer,
        )
    except Exception as e:
        print(f"Dataset warning: {e}. Dataloader is empty.")
        dataloader = []

    from tqdm.auto import tqdm

    print("Starting training loop...")

    step = 0
    total_steps = (
        min(args.max_steps, ((len(dataloader) + args.grad_accum_steps - 1) // args.grad_accum_steps) * args.epochs)
        if len(dataloader) > 0
        else args.max_steps
    )
    pbar = tqdm(total=total_steps, desc="Phase 2 Decoder")

    optimizer.zero_grad()
    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(dataloader):
            if step >= total_steps:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(input_ids, attention_mask)

            # Вычисляем Loss
            ce_loss_raw = loss_fct(logits.view(-1, logits.size(-1)), input_ids.view(-1))
            ce_loss_raw = ce_loss_raw.view(input_ids.size())

            attention_mask_bf = attention_mask.to(torch.bfloat16)
            ce_loss = (ce_loss_raw * attention_mask_bf).sum() / (
                attention_mask_bf.sum() + 1e-6
            )

            # Если DataParallel - усредняем loss
            if ce_loss.dim() > 0:
                ce_loss = ce_loss.mean()

            ce_loss = ce_loss / args.grad_accum_steps
            ce_loss.backward()

            if (batch_idx + 1) % args.grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()
                step += 1

                if step % args.log_steps == 0:
                    if args.wandb_project:
                        wandb.log(
                            {"train/ce_loss": ce_loss.item() * args.grad_accum_steps, "step": step}, step=step
                        )
                    pbar.set_postfix({"Loss": f"{ce_loss.item() * args.grad_accum_steps:.4f}"})

                if step > 0 and step % args.save_steps == 0:
                    ckpt_path = os.path.join(args.output_dir, f"decoder_step_{step}.pth")
                    actual_model = (
                        model.module if isinstance(model, nn.DataParallel) else model
                    )
                    torch.save({"decoder": actual_model.decoder.state_dict()}, ckpt_path)
                    print(f"\n[SAVE] Saved checkpoint to {ckpt_path}")
                    try:
                        subprocess.Popen(
                            [
                                "gsutil",
                                "-q",
                                "cp",
                                ckpt_path,
                                "gs://bebladii-weigths-us/planB/phase2/checkpoints/",
                            ]
                        )
                    except Exception:
                        pass

                pbar.update(1)

        if step >= total_steps:
            break

    pbar.close()

    # Final save
    final_path = os.path.join(args.output_dir, "decoder_final.pth")
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    torch.save({"decoder": actual_model.decoder.state_dict()}, final_path)
    print(f"[SAVE] Final weights saved → {final_path}")
    try:
        subprocess.run(
            [
                "gsutil",
                "-q",
                "cp",
                final_path,
                "gs://bebladii-weigths-us/planB/phase2/checkpoints/",
            ]
        )
    except Exception:
        pass

    if args.wandb_project:
        wandb.finish()


# %%
if __name__ == "__main__":
    train()
