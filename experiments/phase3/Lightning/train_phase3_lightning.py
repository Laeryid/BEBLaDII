import os
import subprocess
import sys
import gc
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Ensure the root of the project is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.beb_la_dii.model.dus import DUSModel
    from src.beb_la_dii.model.vae import LatentEncoder
    from src.beb_la_dii.utils.data import get_dataloader
    from src.beb_la_dii.utils.loss import safe_normalize
except ImportError as e:
    print(f"Error: Failed to import project modules. Make sure you run this from the project root. Error: {e}")
    sys.exit(1)


# ==========================================
# 1. Configuration
# ==========================================
class Config:
    # Model and Data Paths
    embedding_model_path = "Qwen/Qwen2.5-1.5B"
    modernbert_path = "answer-ai/ModernBERT-large"

    dataset_path = "./data/train_data/data"

    local_encoder_weights = "gs://bebladii-weigths-us/planB/phase1/checkpoints/phase1_vae_step_20000.pth"
    local_dus_weights = "gs://bebladii-weigths-us/kaggle_upload_1_2/AWAKENED_WEIGHTS_FINAL.pt"
    local_sep_token = "storage/components/sep_token.pt"

    # Checkpointing and GCS
    resume_from_checkpoint = False
    gcs_checkpoint_dir = "gs://bebladii-weigths-us/planB/phase3/checkpoints/"
    output_dir = "./checkpoints/phase3"

    # Training Hyperparameters
    batch_size = 8
    max_length = 512
    learning_rate = 3e-4
    epochs = 1
    max_steps = 40000
    log_steps = 10
    val_steps = 200
    save_steps = 1000

    # Phase 3 Specifics
    low_noise_amp = 0.5
    gamma = 20.0
    w_denoise = 10.0
    w_identity = 5.0

    wandb_project = "BEBLaDII-Phase3-Lightning"


args = Config()


# ==========================================
# 2. Utilities
# ==========================================
class EMA:
    """
    Тени хранятся на CPU (float32) для экономии ~1.6 GB VRAM на T4.
    Оверхед: CPU<->GPU копирование только при apply/restore (раз в save_steps)
    и маленькая передача GPU->CPU при step().
    """
    def __init__(self, model, decay=0.998):
        self.decay = decay
        self.shadow = {}  # CPU float32
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                # .cpu() — принципиально важно: тени на CPU
                self.shadow[name] = param.data.clone().detach().float().cpu()

    def step(self, model):
        """Обновляет CPU-тени. param.data переносится на CPU per-step."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # Передаём GPU->CPU только нужный тензор
                    param_cpu = param.data.float().cpu()
                    self.shadow[name].mul_(self.decay).add_(param_cpu, alpha=1.0 - self.decay)

    def apply(self, model):
        """Копирует CPU-тени в GPU-параметры (для save/eval)."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.backup[name] = param.data.clone().detach().cpu()
                    param.data.copy_(self.shadow[name].to(param.device, dtype=param.dtype))

    def restore(self, model):
        """Восстанавливает оригинальные GPU-параметры из CPU-бэкапа."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(self.backup[name].to(param.device, dtype=param.dtype))
        self.backup = {}


def compute_layer_divergence(model_module):
    """
    Вычисляет среднюю нормализованную L2-дистанцию между "парными" слоями DUS.

    DUS-схема (из create_latentbert):
      Block 1: слои 0-19  ← оригинальные BERT слои [0..19]
      Block 2: слои 20-39 ← оригинальные BERT слои [8..27]

    Слои 8-19 оригинала продублированы в обоих блоках:
      Block1[k] == Block2[k-8]  для k=8..19
      т.е. layer[k] и layer[k+12]  для k=8..19

    Итого 12 пар: (8,20), (9,21), ..., (19,31).

    Важно: в AWAKENED-весах эти слои уже разошлись и не равны, поэтому
    стартовое значение — ненулевое базелиню.
    Смысл метрики — наблюдать её ДИНАМИКУ в ходе Phase 3:
    - рост → слои специализируются, обучение идёт
    - плато или падение → модель сходится к соллапсу или учится на идентичных преобразованиях
    """
    import re
    # Собираем параметры по индексу слоя: {layer_idx: {suffix: tensor}}
    layer_params: dict[int, dict[str, torch.Tensor]] = {}
    for name, param in model_module.named_parameters():
        m = re.search(r'layers\.([0-9]+)\.', name)
        if m:
            idx = int(m.group(1))
            suffix = name[m.end():]  # всё после "layers.N."
            if idx not in layer_params:
                layer_params[idx] = {}
            layer_params[idx][suffix] = param.data.float().cpu()

    if not layer_params:
        return None

    # 12 пар дублированных слоёв
    pairs = [(8 + k, 20 + k) for k in range(12)]  # (8,20)..(19,31)
    diffs = []
    for l1, l2 in pairs:
        if l1 not in layer_params or l2 not in layer_params:
            continue
        for suffix, p1 in layer_params[l1].items():
            if suffix in layer_params[l2]:
                p2 = layer_params[l2][suffix]
                if p1.shape == p2.shape and p1.numel() > 0:
                    # Нормируем на sqrt(numel) → сопоставимо между слоями разных размеров
                    diff = (p1 - p2).norm().item() / (p1.numel() ** 0.5)
                    diffs.append(diff)

    return float(sum(diffs) / len(diffs)) if diffs else None


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


# ==========================================
# 3. Model Definition
# ==========================================
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

        # 1. Qwen Embeddings
        _qwen_base = AutoModel.from_pretrained(
            embedding_model_path, torch_dtype=torch.bfloat16
        )
        self.qwen_embeddings = _qwen_base.get_input_embeddings()
        del _qwen_base
        for p in self.qwen_embeddings.parameters():
            p.requires_grad = False

        # 2. LatentEncoder
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

        # 3. DUS Backbone
        dus_wrapper = DUSModel.from_scratch(
            config={"base_model_id": modernbert_path}, weights_path=None
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
        # ADR 057: DUS остаётся в float32 — мастер-копия весов для корректного обновления оптимайзером.
        # Forward pass будет идти в bfloat16 через torch.autocast.

        if hasattr(self.dus, 'gradient_checkpointing_enable'):
            self.dus.gradient_checkpointing_enable()

        if hasattr(self.dus, '_maybe_set_compile'):
            self.dus._maybe_set_compile = lambda *args, **kwargs: None

        # Не патчим dtype-свойство: модель теперь float32.
        type(self.dus).device = property(lambda self: torch.device("cuda"))

        # 4. Confidence Projector
        self.confidence_proj = nn.Sequential(
            nn.Linear(1, 256),
            nn.GELU(),
            nn.Linear(256, 1024)
        )

        # 5. C_Embed Layer Hooks
        # Инициализируем 10.0, чтобы разорвать градиентный дедлок и сделать сигнал
        # сопоставимым с начальной нормой остаточного потока (которая около 17-30)
        self.c_embed_alphas = nn.Parameter(torch.ones(len(self.dus.layers)) * 10.0)
        for i, layer in enumerate(self.dus.layers):
            layer.register_forward_pre_hook(self._make_c_embed_hook(i))

        # 6. Токен-разделитель (ADR 058)
        if sep_token_path and os.path.exists(sep_token_path):
            sep_tensor = torch.load(sep_token_path, map_location="cpu").float()
            self.register_buffer("sep_embed", sep_tensor)
            print(f"[Init] Separator token loaded from {sep_token_path}", flush=True)
        else:
            # Fallback для тестов
            self.register_buffer("sep_embed", torch.zeros(1024, dtype=torch.float32))
            print(f"[WARN] Separator token NOT found at {sep_token_path}, initialized to zeros.", flush=True)

    def _make_c_embed_hook(self, layer_idx):
        def hook(module, args):
            hidden_states = args[0]
            if hasattr(self, '_current_c_embed') and self._current_c_embed is not None:
                hidden_states = hidden_states + self.c_embed_alphas[layer_idx] * self._current_c_embed
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

            z_clean_norm = torch.linalg.vector_norm(z_clean.float(), dim=-1, keepdim=True)
            noise = torch.randn_like(z_clean)
            noise = (
                safe_normalize(noise.float(), dim=-1).to(torch.bfloat16)
                * low_noise_amp
                * z_clean_norm.to(torch.bfloat16)
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

        c_embed_raw = self.confidence_proj(c_true.unsqueeze(-1).float())
        c_embed = safe_normalize(c_embed_raw, dim=-1) * 0.1  # float32

        # Убираем жесткое прибавление к входу, отдаем контроль хукам
        x_in = z_noisy.float()

        # Конкатенируем префикс разделителя (ADR 058)
        B = x_in.size(0)
        sep_prefix = self.sep_embed.unsqueeze(0).unsqueeze(0).expand(B, 1, -1).to(x_in.dtype)
        dus_input_extended = torch.cat([sep_prefix, x_in], dim=1)
        attention_mask_extended = F.pad(attention_mask, (1, 0), value=1)

        # Вычисляем DUS в float32 (без autocast)
        dus_outputs = self.dus(
            inputs_embeds=dus_input_extended,
            attention_mask=attention_mask_extended,
            output_hidden_states=True,
        )

        # Отрезаем первый токен (sep_prefix)
        pre_norm = dus_outputs.hidden_states[-1][:, 1:, :].float()
        dus_final_raw = self.dus.final_norm(pre_norm.to(next(self.dus.final_norm.parameters()).dtype)).float()

        dus_delta = dus_final_raw - z_noisy.float()
        dus_final = safe_normalize(dus_final_raw, dim=-1)  # float32

        return {
            "z_clean": z_clean,
            "z_noisy": z_noisy,
            "c_true": c_true,
            "noise_mask": noise_mask,
            "dus_delta": dus_delta,
            "dus_final": dus_final,
            "attention_mask": attention_mask,
            "c_embed": c_embed,
            "dus_input": x_in,
            "hidden_states": dus_outputs.hidden_states,
            "c_embed_alphas": self.c_embed_alphas,
        }


# ==========================================
# 4. Loss Function
# ==========================================
def compute_phase3_loss(outputs: dict, w_prior: float = 0.1):
    z_clean = outputs["z_clean"].float()
    dus_final = outputs["dus_final"].float()
    attn_f = outputs["attention_mask"].float()
    hidden_states = outputs.get("hidden_states", [])

    metrics = {}
    B, T, D = z_clean.size()
    active_tokens = attn_f.sum().clamp(min=1.0)

    # Denoise/Identity Loss (Unified Target)
    target = safe_normalize(z_clean, dim=-1)
    cos_sim = (dus_final * target).sum(dim=-1)

    # 1.0 - cos_sim for all active tokens
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
    metrics["var_loss"] = var_loss.detach()
    metrics["cov_loss"] = cov_loss.detach()

    total_loss = main_loss + w_prior * prior_loss

    # Diagnostic Norms
    if hidden_states and len(hidden_states) > 1:
        l0_norm = hidden_states[0][:, 1:, :].float().norm(dim=-1).mean()
        metrics["norm_L0"] = l0_norm.detach()
        l40_norm = hidden_states[-1][:, 1:, :].float().norm(dim=-1).mean()
        metrics["norm_Llast"] = l40_norm.detach()

        total_delta_norm = 0.0
        for i in range(len(hidden_states) - 1):
            delta = hidden_states[i+1][:, 1:, :].float() - hidden_states[i][:, 1:, :].float()
            total_delta_norm += delta.norm(dim=-1).mean()
        metrics["delta_norm_avg"] = (total_delta_norm / (len(hidden_states) - 1)).detach()

    if "c_embed_alphas" in outputs:
        alphas = outputs["c_embed_alphas"]
        metrics["c_embed_alphas_mean"] = alphas.mean().detach()
        metrics["c_embed_alphas_abs_mean"] = alphas.abs().mean().detach()
        metrics["c_embed_alphas_max"] = alphas.abs().max().detach()

    # Log old diagnostics so wandb doesn't break
    metrics["c_true_mean"] = (outputs["c_true"].float() * attn_f).sum().detach() / active_tokens

    return total_loss, metrics


# ==========================================
# 5. Training Loop
# ==========================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] Using device: {device}")
    num_gpus = torch.cuda.device_count()
    print(f"[Init] Available GPUs: {num_gpus}")

    # GCP Auth
    if os.path.exists("gcp_sa.json"):
        try:
            subprocess.run(["gcloud", "auth", "activate-service-account", "--key-file", "gcp_sa.json"], check=True)
            print("[Init] GCP Authentication successful.")
        except Exception as e_gcp:
            print(f"[Init] WARN: Could not authenticate GCP: {e_gcp}")

    # WandB
    if args.wandb_project:
        wandb_api = os.environ.get("WANDB")
        if wandb_api:
            wandb.login(key=wandb_api)
        wandb.init(project=args.wandb_project, config=vars(args), resume="allow")

    # Download Weights if necessary
    os.makedirs("./weights", exist_ok=True)

    enc_path = args.local_encoder_weights
    if enc_path.startswith("gs://"):
        local_enc = os.path.join("./weights", os.path.basename(enc_path))
        if not os.path.exists(local_enc):
            print(f"Downloading Encoder from {enc_path}...")
            subprocess.run(["gsutil", "-q", "cp", enc_path, local_enc], check=True)
        enc_path = local_enc

    dus_path = args.local_dus_weights
    if dus_path.startswith("gs://"):
        local_dus = os.path.join("./weights", os.path.basename(dus_path))
        if not os.path.exists(local_dus):
            print(f"Downloading DUS from {dus_path}...")
            subprocess.run(["gsutil", "-q", "cp", dus_path, local_dus], check=True)
        dus_path = local_dus

    print("[Init] Instantiating BEBLaDIIPhase3 model...", flush=True)
    model = BEBLaDIIPhase3(
        embedding_model_path=args.embedding_model_path,
        modernbert_path="answerdotai/ModernBERT-large",
        dus_weights=dus_path,
        encoder_weights=enc_path,
        sep_token_path=args.local_sep_token
    ).to(device)

    if num_gpus > 1:
        print(f"[Init] Wrapping model in DataParallel across {num_gpus} GPUs", flush=True)
        model = nn.DataParallel(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-2)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2000, T_mult=1, eta_min=1e-6)
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

    total_steps = (min(args.max_steps, len(dataloader) * args.epochs) if len(dataloader) > 0 else args.max_steps)

    # Resume from checkpoint
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

                    if "dus_ema" in ckpt and "confidence_proj_ema" in ckpt:
                        ema.shadow = {
                            **{f"dus.{k}": v for k, v in ckpt["dus_ema"].items()},
                            **{f"confidence_proj.{k}": v for k, v in ckpt["confidence_proj_ema"].items()}
                        }

                    if "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
                    if "scheduler" in ckpt: scheduler.load_state_dict(ckpt["scheduler"])
                    if "step" in ckpt: start_step = ckpt["step"]
                    if "metrics_history" in ckpt: metrics_history = ckpt["metrics_history"]

                    print(f"Successfully resumed from step {start_step}!")
            except Exception as e:
                print(f"Warning: Failed to resume from GS checkpoint! Error: {e}")

    model.train()
    step = start_step

    from tqdm.auto import tqdm
    pbar = tqdm(total=total_steps, initial=start_step, desc="Training Phase 3")
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        for batch in dataloader:
            if step >= total_steps:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            # ADR 057: forward без внешнего autocast — autocast применяется внутри forward()
            # точечно только для DUS. Параметры — float32, обновления точные.
            fwd_outputs = model(input_ids, attention_mask=attention_mask, low_noise_amp=args.low_noise_amp)
            # compute_phase3_loss работает полностью в float32
            loss, metrics = compute_phase3_loss(fwd_outputs)

            if loss.dim() > 0:
                loss = loss.mean()

            loss.backward()
            # clip_grad_norm_ возвращает суммарную норму ДО клиппинга — ценная диагностика
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0).item()

            optimizer.step()
            ema.step(model)
            scheduler.step()

            # Warmup
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

            if step % args.log_steps == 0:
                metrics_dict = {k: (v.mean().item() if isinstance(v, torch.Tensor) else v) for k, v in metrics.items()}
                metrics_dict["loss"] = loss.item()
                metrics_dict["lr"] = optimizer.param_groups[0]["lr"]
                metrics_dict["step"] = step
                # Норма градиентов — главный индикатор "живости" обучения (ADR 057)
                metrics_dict["grad_norm"] = grad_norm
                metrics_history.append(metrics_dict)

                if args.wandb_project:
                    wandb.log(metrics_dict, step=step)

                pbar.set_postfix({
                    "loss": f"{metrics_dict['loss']:.4f}",
                    "grad": f"{grad_norm:.3f}",
                    "c_true": f"{metrics_dict.get('c_true_mean', 0):.4f}",
                })

            if step > start_step and step % args.save_steps == 0:
                # --- Layer Divergence (до eval, пока модель в train-параметрах) ---
                actual_model_ref = model.module if isinstance(model, nn.DataParallel) else model
                layer_div = compute_layer_divergence(actual_model_ref.dus)
                if layer_div is not None:
                    print(f"[DIAG] Step {step} | layer_divergence (DUS pairs 8-19 vs 20-31): {layer_div:.6f} (baseline from AWAKENED ≠ 0)")
                    if args.wandb_project:
                        wandb.log({"diag/layer_divergence": layer_div}, step=step)

                # --- Validation ---
                model.eval()
                val_metrics_sum = {}
                val_batches = 0
                with torch.no_grad():
                    for val_batch in val_dataloader:
                        v_input_ids = val_batch["input_ids"].to(device)
                        v_attention_mask = val_batch["attention_mask"].to(device)
                        # autocast применяется внутри model.forward(), здесь не нужен
                        v_outputs = model(v_input_ids, attention_mask=v_attention_mask, low_noise_amp=args.low_noise_amp)
                        v_loss, v_metrics = compute_phase3_loss(v_outputs)
                        for k, v in v_metrics.items():
                            val_metrics_sum[f"val_{k}"] = val_metrics_sum.get(f"val_{k}", 0) + (v.mean().item() if isinstance(v, torch.Tensor) else v)
                        val_metrics_sum["val_loss"] = val_metrics_sum.get("val_loss", 0) + (v_loss.mean().item() if isinstance(v_loss, torch.Tensor) else v_loss)
                        val_batches += 1
                        if val_batches >= 50: break

                if val_batches > 0:
                    val_metrics_avg = {k: v / val_batches for k, v in val_metrics_sum.items()}
                    if args.wandb_project: wandb.log(val_metrics_avg, step=step)
                    print(f"\n[VAL] Step {step} | val_loss: {val_metrics_avg.get('val_loss', 0):.4f} | val_cos_sim_all: {val_metrics_avg.get('val_cos_sim_all', 0):.4f}")
                model.train()

                ckpt_path = os.path.join(args.output_dir, f"phase3_step_{step}.pth")
                actual_model = model.module if isinstance(model, nn.DataParallel) else model

                dus_state = {k: v.cpu() for k, v in actual_model.dus.state_dict().items()}
                proj_state = {k: v.cpu() for k, v in actual_model.confidence_proj.state_dict().items()}

                ema.apply(model)
                dus_ema_state = {k: v.cpu() for k, v in actual_model.dus.state_dict().items()}
                proj_ema_state = {k: v.cpu() for k, v in actual_model.confidence_proj.state_dict().items()}
                ema.restore(model)

                torch.save({
                    "dus": dus_state,
                    "confidence_proj": proj_state,
                    "dus_ema": dus_ema_state,
                    "confidence_proj_ema": proj_ema_state,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "metrics_history": metrics_history
                }, ckpt_path)
                print(f"\\n[SAVE] Checkpoint saved → {ckpt_path}")
                try:
                    subprocess.Popen(["gsutil", "-q", "cp", ckpt_path, args.gcs_checkpoint_dir])
                except Exception: pass

            step += 1
            pbar.update(1)

        if step >= total_steps:
            break

    pbar.close()

    # Final Save
    final_path = os.path.join(args.output_dir, "phase3_final.pth")
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    dus_state = {k: v.cpu() for k, v in actual_model.dus.state_dict().items()}
    proj_state = {k: v.cpu() for k, v in actual_model.confidence_proj.state_dict().items()}

    ema.apply(model)
    dus_ema_state = {k: v.cpu() for k, v in actual_model.dus.state_dict().items()}
    proj_ema_state = {k: v.cpu() for k, v in actual_model.confidence_proj.state_dict().items()}
    ema.restore(model)

    torch.save({
        "dus": dus_state,
        "confidence_proj": proj_state,
        "dus_ema": dus_ema_state,
        "confidence_proj_ema": proj_ema_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "metrics_history": metrics_history
    }, final_path)

    print(f"[SAVE] Final weights saved → {final_path}")
    try:
        subprocess.run(["gsutil", "-q", "cp", final_path, args.gcs_checkpoint_dir])
    except Exception: pass

    if args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    train()
