# %% [markdown]
# # BEBLaDII Phase 3 Training (Kaggle T4 x2)
# *Архитектура: ADR 058 (sep_token) + ADR 059 (c_embed hooks, unified_loss)*
# *LR: ConstantLR с однократным warmup (без CosineAnnealingWarmRestarts)*

# %% [markdown]
# ## 1. Setup Environment

# %%
# !pip install -q einops wandb indexed_parquet_dataset

# %%
import math
import os
import re
import subprocess
import sys

# --- Отладочный вывод структуры Kaggle ---
if os.path.exists("/kaggle/input"):
    print("=== Kaggle Input Structure ===")
    for root, dirs, files in os.walk("/kaggle/input"):
        level = root.replace("/kaggle/input", "").count(os.sep)
        if level < 3: # Ограничиваем глубину вывода
            indent = " " * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 4 * (level + 1)
            for f in files:
                if f.endswith(".json") or f.endswith(".pt") or f.endswith(".pth"):
                    print(f"{subindent}{f}")
    print("==============================")
# -----------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModel, AutoTokenizer

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

try:
    from src.beb_la_dii.model.dus import DUSModel
    from src.beb_la_dii.model.vae import LatentEncoder
    from src.beb_la_dii.utils.data import get_dataloader
    from src.beb_la_dii.utils.loss import safe_normalize
except ImportError as e:
    print(f"Warning: Не удалось импортировать модули проекта. Ошибка: {e}")


def resolve_model_path(base_path: str) -> str:
    """
    Находит директорию с config.json начиная с base_path.
    Если не находит — делает глобальный fallback-поиск по /kaggle/input/.
    """
    import pathlib
    p = pathlib.Path(base_path)

    def check_dir(dir_path):
        return (dir_path / "config.json").exists()

    # 1. Точный путь
    if check_dir(p):
        print(f"[resolve_model_path] Found config.json at: {p}")
        return str(p)

    # 2. Вверх по родителям
    for parent in list(p.parents)[:4]:
        if check_dir(parent):
            print(f"[resolve_model_path] Found config.json in parent: {parent}")
            return str(parent)

    # 3. Вниз рекурсивно
    if p.exists():
        for config_file in sorted(p.rglob("config.json")):
            print(f"[resolve_model_path] Found config.json recursively: {config_file.parent}")
            return str(config_file.parent)

    # 4. Глобальный fallback по ключевому слову
    print(f"[resolve_model_path] WARNING: config.json not found under {base_path}. Searching globally...")
    keyword = ""
    if "qwen" in base_path.lower(): keyword = "qwen"
    elif "modernbert" in base_path.lower(): keyword = "modernbert"

    if keyword:
        kaggle_input = pathlib.Path("/kaggle/input")
        if kaggle_input.exists():
            for config_file in kaggle_input.rglob("config.json"):
                if keyword in str(config_file).lower():
                    print(f"[resolve_model_path] Found fallback config for '{keyword}': {config_file.parent}")
                    return str(config_file.parent)

    print(f"[resolve_model_path] FAILED to resolve {base_path}, using as-is")
    return base_path


# %% [markdown]
# ## 2. Configuration

# %%
class Config:
    # Пути к базовым моделям
    embedding_model_path = resolve_model_path("/kaggle/input/datasets/ragnar123/qwen2-5-1-5b")
    modernbert_path      = resolve_model_path("/kaggle/input/models/answer-ai/modernbert/transformers/large/2")

    # Пути к данным
    dataset_path = "/kaggle/input/datasets/bogdanbuliakov/bebladii-planb-phase3-data/phase 3/train_data/data"

    # Пути к весам
    local_encoder_weights = "/kaggle/input/datasets/bogdanbuliakov/bebladii-planb-phase3-data/planB_phase1_checkpoints_phase1_vae_step_20000.pth"
    local_dus_weights     = "/kaggle/input/datasets/bogdanbuliakov/bebladii-phase1-awakaned-weights/AWAKENED_WEIGHTS_FINAL.pt"
    # sep_token из датасета (загруженного в Kaggle)
    local_sep_token       = "/kaggle/working/BEBLaDII/storage/components/sep_token.pt"

    # GCS (для resume и сохранения чекпоинтов)
    resume_from_checkpoint = True
    gcs_checkpoint_dir = "gs://bebladii-weigths-us/planB/phase3/checkpoints/"

    # Директория вывода
    output_dir = "/kaggle/working/checkpoints/phase3"

    # Гиперпараметры
    batch_size   = 8
    max_length   = 512
    learning_rate = 2e-4   # ConstantLR после warmup
    epochs       = 1
    max_steps    = 40000
    log_steps    = 10
    val_steps    = 200
    save_steps   = 1000

    # Phase 3
    low_noise_amp = 0.5

    wandb_project = "BEBLaDII-Phase3-Kaggle"


args = Config()


# %% [markdown]
# ## 3. Utilities

# %%
class EMA:
    """
    Тени хранятся на CPU (float32) для экономии VRAM.
    Копирование CPU<->GPU только при apply/restore (раз в save_steps).
    """
    def __init__(self, model, decay=0.998):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach().float().cpu()

    def step(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param_cpu = param.data.float().cpu()
                    self.shadow[name].mul_(self.decay).add_(param_cpu, alpha=1.0 - self.decay)

    def apply(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.backup[name] = param.data.clone().detach().cpu()
                    param.data.copy_(self.shadow[name].to(param.device, dtype=param.dtype))

    def restore(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(self.backup[name].to(param.device, dtype=param.dtype))
        self.backup = {}


def get_latest_gcs_checkpoint(gcs_dir):
    try:
        result = subprocess.run(["gsutil", "ls", gcs_dir], capture_output=True, text=True, check=True)
        files = result.stdout.splitlines()
        ckpt_files = [f for f in files if "phase3_step_" in f and f.endswith(".pth")]
        if not ckpt_files:
            return None
        def extract_step(filename):
            try:
                return int(filename.split("_step_")[-1].split(".pth")[0])
            except ValueError:
                return -1
        ckpt_files.sort(key=extract_step)
        return ckpt_files[-1]
    except Exception as e:
        print(f"Failed to list GCS checkpoints: {e}")
        return None


# %% [markdown]
# ## 4. Model Definition
# Архитектура: ADR 058 (sep_token) + ADR 059 (c_embed per-layer hooks, float32 DUS)

# %%
class BEBLaDIIPhase3(nn.Module):
    def __init__(
        self,
        embedding_model_path: str,
        modernbert_path: str,
        dus_weights: str | None,
        encoder_weights: str | None,
        sep_token_path: str | None,
    ):
        super().__init__()

        # 1. Qwen Embeddings (заморожены)
        _qwen_base = AutoModel.from_pretrained(
            embedding_model_path, torch_dtype=torch.bfloat16, local_files_only=True
        )
        self.qwen_embeddings = _qwen_base.get_input_embeddings()
        del _qwen_base
        for p in self.qwen_embeddings.parameters():
            p.requires_grad = False

        # 2. LatentEncoder (заморожен)
        self.encoder = LatentEncoder()
        if encoder_weights and os.path.exists(encoder_weights):
            state = torch.load(encoder_weights, map_location="cpu")
            if "encoder" in state:
                state = state["encoder"]
            self.encoder.load_state_dict(state, strict=False)
            print(f"[Init] LatentEncoder weights loaded from {encoder_weights}", flush=True)
        else:
            raise FileNotFoundError(f"CRITICAL: encoder_weights not found at '{encoder_weights}'.")
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.to(torch.bfloat16)

        # 3. DUS Backbone (обучаемый, в float32 — ADR 059)
        dus_wrapper = DUSModel.from_scratch(
            config={"base_model_id": modernbert_path}, weights_path=None, local_files_only=True
        )
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
            dus_wrapper.model.load_state_dict(clean_state, strict=False)
            print(f"[Init] Awakened DUS weights loaded from {dus_weights}", flush=True)
        else:
            raise FileNotFoundError(f"CRITICAL: dus_weights not found at '{dus_weights}'.")

        self.dus = dus_wrapper.model
        # DUS в float32 для корректного обновления оптимайзером (ADR 057)
        # gradient_checkpointing с use_reentrant=False — обязателен при хуках (ADR 059)
        if hasattr(self.dus, "gradient_checkpointing_enable"):
            self.dus.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if hasattr(self.dus, "_maybe_set_compile"):
            self.dus._maybe_set_compile = lambda *a, **kw: None
        type(self.dus).device = property(lambda self: torch.device("cuda"))
        type(self.dus).dtype = property(lambda self: torch.float32)

        # 4. Confidence Projector (обучаемый)
        self.confidence_proj = nn.Sequential(
            nn.Linear(1, 256),
            nn.GELU(),
            nn.Linear(256, 1024)
        )

        # 5. C_Embed Layer Hooks (ADR 059)
        # Инициализируем 10.0 — нужно для преодоления начальной нормы residual stream (~17-30)
        self.c_embed_alphas = nn.Parameter(torch.ones(len(self.dus.layers)) * 10.0)
        for i, layer in enumerate(self.dus.layers):
            layer.register_forward_pre_hook(self._make_c_embed_hook(i))

        # 6. Токен-разделитель (ADR 058)
        if sep_token_path and os.path.exists(sep_token_path):
            sep_tensor = torch.load(sep_token_path, map_location="cpu").float()
            self.register_buffer("sep_embed", sep_tensor)
            print(f"[Init] Separator token loaded from {sep_token_path}", flush=True)
        else:
            self.register_buffer("sep_embed", torch.zeros(1024, dtype=torch.float32))
            print(f"[WARN] Separator token NOT found at {sep_token_path}, initialized to zeros.", flush=True)

    def _make_c_embed_hook(self, layer_idx):
        def hook(module, args):
            hidden_states = args[0]
            if hasattr(self, "_current_c_embed") and self._current_c_embed is not None:
                current_alpha = self.c_embed_alphas[layer_idx].to(hidden_states.device)
                current_c = self._current_c_embed.to(hidden_states.device)
                hidden_states = hidden_states + current_alpha * current_c
            return (hidden_states,) + args[1:]
        return hook

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

            # Генерация маски шума (5 окон × 5 токенов)
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

            noise_offsets = torch.arange(noise_window_size, device=z_clean.device).view(1, 1, noise_window_size)
            noise_window_indices = starts.unsqueeze(-1) + noise_offsets

            rand_noise = torch.rand((B, num_windows, noise_window_size), device=z_clean.device)
            num_to_noise = torch.randint(1, noise_window_size + 1, (B, num_windows, 1), device=z_clean.device)
            _, noise_ranks = torch.sort(rand_noise, dim=-1)
            noise_subset_mask = (
                torch.arange(noise_window_size, device=z_clean.device).view(1, 1, noise_window_size)
                < num_to_noise
            ).to(z_clean.dtype)
            noise_subset_mask = torch.gather(noise_subset_mask, 2, noise_ranks.argsort(dim=-1))

            flat_noise_indices = noise_window_indices.view(B, -1)
            full_noise_mask = torch.zeros((B, T), device=z_clean.device, dtype=z_clean.dtype)
            full_noise_mask.scatter_(1, flat_noise_indices, noise_subset_mask.view(B, -1))
            noise_mask = full_noise_mask * attention_mask.to(z_clean.dtype)

            # Применение шума
            z_clean_norm = torch.linalg.vector_norm(z_clean.float(), dim=-1, keepdim=True)
            noise = torch.randn_like(z_clean)
            noise = (
                safe_normalize(noise.float(), dim=-1).to(z_clean.dtype)
                * low_noise_amp
                * z_clean_norm.to(z_clean.dtype)
            )
            z_noisy = z_clean + noise * noise_mask.unsqueeze(-1)
            z_noisy = torch.where(
                noise_mask.unsqueeze(-1) > 0,
                safe_normalize(z_noisy.float(), dim=-1).to(z_clean.dtype),
                z_noisy,
            )

            # Confidence signal
            z_clean_normed = safe_normalize(z_clean, dim=-1)
            z_noisy_normed = safe_normalize(z_noisy, dim=-1)
            c_true = torch.clamp((z_clean_normed * z_noisy_normed).sum(dim=-1), min=0.0)

        # Confidence embedding (float32, норма 0.1 — ADR 047, 051)
        c_embed_raw = self.confidence_proj(c_true.unsqueeze(-1).float())
        c_embed = safe_normalize(c_embed_raw, dim=-1) * 0.1  # float32

        # Сохраняем c_embed для хуков (с нулевым вектором на позицию sep_prefix)
        self._current_c_embed = F.pad(c_embed, (0, 0, 1, 0), value=0.0)  # [B, T+1, D]

        # Конкатенируем sep_prefix (ADR 058)
        x_in = z_noisy.float()
        sep_prefix = self.sep_embed.unsqueeze(0).unsqueeze(0).expand(B, 1, -1).to(x_in.dtype)
        dus_input_extended = torch.cat([sep_prefix, x_in], dim=1)
        attention_mask_extended = F.pad(attention_mask, (1, 0), value=1)

        # DUS forward в float32 (ADR 059)
        dus_outputs = self.dus(
            inputs_embeds=dus_input_extended,
            attention_mask=attention_mask_extended,
            output_hidden_states=True,
        )

        # Отрезаем sep_prefix, нормализуем
        pre_norm = dus_outputs.hidden_states[-1][:, 1:, :].float()
        dus_final_raw = self.dus.final_norm(pre_norm.to(self.dus.dtype)).float()
        dus_delta = dus_final_raw - z_noisy.float()
        dus_final = safe_normalize(dus_final_raw, dim=-1)  # float32

        return {
            "z_clean": z_clean,
            "z_noisy": z_noisy,
            "c_true": c_true,
            "noise_mask": noise_mask,
            "dus_delta": dus_delta,
            "dus_final": dus_final,
            "dus_final_raw": dus_final_raw,
            "attention_mask": attention_mask,
            "c_embed": c_embed,
            "c_embed_raw": c_embed_raw,
            "hidden_states": dus_outputs.hidden_states,
            "c_embed_alphas": self.c_embed_alphas,
        }


# %% [markdown]
# ## 5. Loss Function
# Unified Loss: 1 - cos(dus_final, normalize(z_clean)) для всех активных токенов (ADR 059)

# %%
def compute_phase3_loss(outputs: dict, w_prior: float = 0.1):
    z_clean   = outputs["z_clean"].float()
    dus_final = outputs["dus_final"].float()
    attn_f    = outputs["attention_mask"].float()
    hidden_states = outputs.get("hidden_states", [])

    metrics = {}
    B, T, D = z_clean.size()
    active_tokens = attn_f.sum().clamp(min=1.0)

    # Unified Loss (ADR 059)
    target = safe_normalize(z_clean, dim=-1)
    cos_sim = (dus_final * target).sum(dim=-1)
    loss_elementwise = 1.0 - cos_sim
    main_loss = (loss_elementwise * attn_f).sum() / active_tokens
    metrics["unified_loss"] = main_loss.detach()
    metrics["cos_sim_all"] = (cos_sim * attn_f).sum().detach() / active_tokens

    # Prior Loss
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
    metrics["var_loss"]   = var_loss.detach()
    metrics["cov_loss"]   = cov_loss.detach()

    total_loss = main_loss + w_prior * prior_loss

    # Diagnostic Norms
    if hidden_states and len(hidden_states) > 1:
        l0_norm   = hidden_states[0][:, 1:, :].float().norm(dim=-1).mean()
        l40_norm  = hidden_states[-1][:, 1:, :].float().norm(dim=-1).mean()
        metrics["norm_L0"]    = l0_norm.detach()
        metrics["norm_Llast"] = l40_norm.detach()
        total_delta = 0.0
        for i in range(len(hidden_states) - 1):
            delta = hidden_states[i+1][:, 1:, :].float() - hidden_states[i][:, 1:, :].float()
            total_delta += delta.norm(dim=-1).mean()
        metrics["delta_norm_avg"] = (total_delta / (len(hidden_states) - 1)).detach()

    if "c_embed_alphas" in outputs:
        alphas = outputs["c_embed_alphas"]
        metrics["c_embed_alphas_mean"]     = alphas.mean().detach()
        metrics["c_embed_alphas_abs_mean"] = alphas.abs().mean().detach()
        metrics["c_embed_alphas_max"]      = alphas.abs().max().detach()

    if "c_embed_raw" in outputs:
        metrics["c_embed_raw_norm"] = outputs["c_embed_raw"].float().norm(dim=-1).mean().detach()

    metrics["c_true_mean"] = (outputs["c_true"].float() * attn_f).sum().detach() / active_tokens

    return total_loss, metrics


# %% [markdown]
# ## 6. Training Loop

# %%
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] Using device: {device}")
    num_gpus = torch.cuda.device_count()
    print(f"[Init] Available GPUs: {num_gpus}")

    # WandB + GCP Auth
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
                subprocess.run(["gcloud", "auth", "activate-service-account", "--key-file", "gcp_sa.json"], check=True)
                print("[Init] GCP Authentication successful.")
                with open("gcs_test.txt", "w") as f:
                    f.write("GCS Auth Test Successful")
                subprocess.run(["gsutil", "cp", "gcs_test.txt",
                                "gs://bebladii-weigths-us/planB/phase3/checkpoints/gcs_test.txt"], check=True)
                print("[Init] GCS write access verified.")
            except Exception as e_gcp:
                print(f"[Init] WARN: GCP auth failed: {e_gcp}")
        except Exception as e:
            print(f"[Init] WARN: Could not login to W&B: {e}")
        wandb.init(project=args.wandb_project, config=vars(args), resume="allow")

    print("[Init] Instantiating BEBLaDIIPhase3 model...", flush=True)
    model = BEBLaDIIPhase3(
        embedding_model_path=args.embedding_model_path,
        modernbert_path=args.modernbert_path,
        dus_weights=args.local_dus_weights,
        encoder_weights=args.local_encoder_weights,
        sep_token_path=args.local_sep_token,
    ).to(device)

    if num_gpus > 1:
        print(f"[Init] Wrapping model in DataParallel across {num_gpus} GPUs", flush=True)
        model = nn.DataParallel(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-2)

    # ConstantLR с однократным warmup (ADR: замена CosineAnnealingWarmRestarts)
    warmup_steps = min(1000, int(args.max_steps * 0.1))
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return max(0.01, current_step / warmup_steps)
        return 1.0
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    ema = EMA(model, decay=0.998)

    # Dataloaders
    try:
        dataloader = get_dataloader(
            stage="reasoning", batch_size=args.batch_size, max_length=args.max_length,
            split="train", val_ratio=0.05, data_dir=args.dataset_path,
        )
        val_dataloader = get_dataloader(
            stage="reasoning", batch_size=args.batch_size, max_length=args.max_length,
            split="val", val_ratio=0.05, data_dir=args.dataset_path,
        )
    except Exception as e:
        print(f"WARN: Failed to load dataset: {e}. Using dummy dataloaders.")
        dataloader, val_dataloader = [], []

    if len(dataloader) == 0:
        raise RuntimeError(f"[FATAL] Dataset is empty or not found at: '{args.dataset_path}'. "
                           f"Training cannot start.")

    total_steps = (
        min(args.max_steps, len(dataloader) * args.epochs)
        if len(dataloader) > 0 else args.max_steps
    )

    # Resume from GCS checkpoint
    start_step = 0
    metrics_history = []
    if getattr(args, "resume_from_checkpoint", False):
        latest_gs_ckpt = get_latest_gcs_checkpoint(args.gcs_checkpoint_dir)
        if latest_gs_ckpt:
            print(f"Found latest checkpoint on GCS: {latest_gs_ckpt}")
            os.makedirs(args.output_dir, exist_ok=True)
            local_ckpt_path = os.path.join(args.output_dir, "resume_checkpoint.pth")
            try:
                subprocess.run(["gsutil", "-q", "cp", latest_gs_ckpt, local_ckpt_path], check=True)
                if os.path.exists(local_ckpt_path):
                    print(f"Loading checkpoint {local_ckpt_path}...")
                    ckpt = torch.load(local_ckpt_path, map_location="cpu")
                    actual_model = model.module if isinstance(model, nn.DataParallel) else model

                    if "dus" in ckpt:
                        clean_dus = {k.replace("_orig_module.", ""): v for k, v in ckpt["dus"].items()}
                        actual_model.dus.load_state_dict(clean_dus)
                    if "confidence_proj" in ckpt:
                        clean_proj = {k.replace("_orig_module.", ""): v for k, v in ckpt["confidence_proj"].items()}
                        actual_model.confidence_proj.load_state_dict(clean_proj)
                    if "c_embed_alphas" in ckpt:
                        actual_model.c_embed_alphas.data.copy_(ckpt["c_embed_alphas"].to(device))
                        print(f"[Resume] c_embed_alphas loaded (mean={ckpt['c_embed_alphas'].mean():.4f})")

                    if "dus_ema" in ckpt and "confidence_proj_ema" in ckpt:
                        shadow_update = {
                            **{f"dus.{k}": v for k, v in ckpt["dus_ema"].items()},
                            **{f"confidence_proj.{k}": v for k, v in ckpt["confidence_proj_ema"].items()},
                        }
                        if "c_embed_alphas_ema" in ckpt:
                            shadow_update["c_embed_alphas"] = ckpt["c_embed_alphas_ema"]
                            print(f"[Resume] c_embed_alphas_ema loaded (mean={ckpt['c_embed_alphas_ema'].mean():.4f})")
                        ema.shadow.update(shadow_update)

                    if "optimizer" in ckpt:
                        try:
                            optimizer.load_state_dict(ckpt["optimizer"])
                        except Exception as e:
                            print(f"[Resume] WARN: Skipping optimizer load ({e}).")
                    if "scheduler" in ckpt:
                        try:
                            scheduler.load_state_dict(ckpt["scheduler"])
                        except Exception as e:
                            print(f"[Resume] INFO: Skipping scheduler state load ({e}).")
                    if "step" in ckpt:
                        start_step = ckpt["step"]
                        if start_step > 0:
                            for _ in range(start_step):
                                scheduler.step()
                    if "metrics_history" in ckpt: metrics_history = ckpt["metrics_history"]

                    print(f"Successfully resumed from step {start_step}!")
            except Exception as e:
                print(f"Warning: Failed to resume from GCS checkpoint! Error: {e}")

    model.train()
    step = start_step

    from tqdm.auto import tqdm
    pbar = tqdm(total=total_steps, initial=start_step, desc="Training Phase 3")
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        for batch in dataloader:
            if step >= total_steps:
                break

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            fwd_outputs = model(input_ids, attention_mask=attention_mask, low_noise_amp=args.low_noise_amp)
            loss, metrics = compute_phase3_loss(fwd_outputs)

            if loss.dim() > 0:
                loss = loss.mean()

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0).item()
            optimizer.step()
            ema.step(model)
            scheduler.step()

            # Логирование
            if step % args.log_steps == 0:
                metrics_dict = {k: (v.mean().item() if isinstance(v, torch.Tensor) else v) for k, v in metrics.items()}
                metrics_dict["loss"]      = loss.item()
                metrics_dict["lr"]        = optimizer.param_groups[0]["lr"]
                metrics_dict["step"]      = step
                metrics_dict["grad_norm"] = grad_norm
                metrics_history.append(metrics_dict)
                if args.wandb_project:
                    wandb.log(metrics_dict, step=step)
                pbar.set_postfix({
                    "loss": f"{metrics_dict['loss']:.4f}",
                    "grad": f"{grad_norm:.3f}",
                    "c_true": f"{metrics_dict.get('c_true_mean', 0):.4f}",
                })

            # Validation
            if step > start_step and (step + 5) % args.val_steps == 0:
                model.eval()
                val_metrics_sum = {}
                val_batches = 0
                with torch.no_grad():
                    for val_batch in val_dataloader:
                        v_input_ids      = val_batch["input_ids"].to(device)
                        v_attention_mask = val_batch["attention_mask"].to(device)
                        v_outputs = model(v_input_ids, attention_mask=v_attention_mask, low_noise_amp=args.low_noise_amp)
                        v_loss, v_metrics = compute_phase3_loss(v_outputs)
                        for k, v in v_metrics.items():
                            val_metrics_sum[f"val_{k}"] = val_metrics_sum.get(f"val_{k}", 0) + (v.mean().item() if isinstance(v, torch.Tensor) else v)
                        val_metrics_sum["val_loss"] = val_metrics_sum.get("val_loss", 0) + (v_loss.mean().item() if isinstance(v_loss, torch.Tensor) else v_loss)
                        val_batches += 1
                        if val_batches >= 50:
                            break
                if val_batches > 0:
                    val_metrics_avg = {k: v / val_batches for k, v in val_metrics_sum.items()}
                    if args.wandb_project:
                        wandb.log(val_metrics_avg, step=step)
                    print(f"\n[VAL] Step {step} | val_loss: {val_metrics_avg.get('val_loss', 0):.4f} | val_cos_sim_all: {val_metrics_avg.get('val_cos_sim_all', 0):.4f}")
                model.train()

            # Сохранение чекпоинта
            if step > start_step and (step + 5) % args.save_steps == 0:
                ckpt_path = os.path.join(args.output_dir, f"phase3_step_{step}.pth")
                actual_model = model.module if isinstance(model, nn.DataParallel) else model

                dus_state  = {k: v.cpu() for k, v in actual_model.dus.state_dict().items()}
                proj_state = {k: v.cpu() for k, v in actual_model.confidence_proj.state_dict().items()}
                alphas_state = actual_model.c_embed_alphas.detach().cpu()

                ema.apply(model)
                dus_ema_state  = {k: v.cpu() for k, v in actual_model.dus.state_dict().items()}
                proj_ema_state = {k: v.cpu() for k, v in actual_model.confidence_proj.state_dict().items()}
                alphas_ema_state = actual_model.c_embed_alphas.detach().cpu()
                ema.restore(model)

                torch.save({
                    "dus": dus_state,
                    "confidence_proj": proj_state,
                    "c_embed_alphas": alphas_state,
                    "dus_ema": dus_ema_state,
                    "confidence_proj_ema": proj_ema_state,
                    "c_embed_alphas_ema": alphas_ema_state,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "metrics_history": metrics_history,
                }, ckpt_path)
                print(f"\n[SAVE] Checkpoint saved → {ckpt_path}")
                try:
                    subprocess.Popen(["gsutil", "-q", "cp", ckpt_path, args.gcs_checkpoint_dir])
                except Exception as e_sync:
                    print(f"[SYNC] Error syncing to GCS: {e_sync}")

            step += 1
            pbar.update(1)

        if step >= total_steps:
            break

    pbar.close()

    # Финальное сохранение
    final_path = os.path.join(args.output_dir, "phase3_final.pth")
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    dus_state  = {k: v.cpu() for k, v in actual_model.dus.state_dict().items()}
    proj_state = {k: v.cpu() for k, v in actual_model.confidence_proj.state_dict().items()}
    alphas_state = actual_model.c_embed_alphas.detach().cpu()

    ema.apply(model)
    dus_ema_state  = {k: v.cpu() for k, v in actual_model.dus.state_dict().items()}
    proj_ema_state = {k: v.cpu() for k, v in actual_model.confidence_proj.state_dict().items()}
    alphas_ema_state = actual_model.c_embed_alphas.detach().cpu()
    ema.restore(model)

    torch.save({
        "dus": dus_state,
        "confidence_proj": proj_state,
        "c_embed_alphas": alphas_state,
        "dus_ema": dus_ema_state,
        "confidence_proj_ema": proj_ema_state,
        "c_embed_alphas_ema": alphas_ema_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "metrics_history": metrics_history,
    }, final_path)
    print(f"[SAVE] Final weights saved → {final_path}")
    try:
        subprocess.run(["gsutil", "-q", "cp", final_path, args.gcs_checkpoint_dir])
        print(f"[SYNC] Final sync to GCS complete.")
    except Exception as e:
        print(f"[SYNC] Error syncing final: {e}")

    if args.wandb_project:
        wandb.finish()


# %%
train()
