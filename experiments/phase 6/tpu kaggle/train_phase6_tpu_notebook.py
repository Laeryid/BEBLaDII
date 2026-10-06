import math
import os
import subprocess
import sys
import time

# Install Kaggle TPU dependencies (equivalent to: !pip install -q -U --no-cache-dir einops wandb indexed_parquet_dataset google-cloud-storage)
try:
    import wandb
    import einops
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "--no-cache-dir",
                    "einops", "wandb", "indexed_parquet_dataset", "google-cloud-storage"])

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
import numpy as np

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

# Handle torch_xla versions
try:
    import torch_xla.experimental.xla_sharding as xs
except ImportError:
    import torch_xla.distributed.spmd as xs
    if not hasattr(xs, 'mark_sharding'):
        import torch_xla.distributed.spmd.xla_sharding as xs

try:
    from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel
except ImportError:
    try:
        from torch_xla.distributed.spmd import SpmdFullyShardedDataParallel
    except ImportError:
        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as SpmdFullyShardedDataParallel

xr.use_spmd()

def setup_spmd_mesh():
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (num_devices, 1)
    device_ids = np.array(range(num_devices))
    mesh = xs.Mesh(device_ids, mesh_shape, ("fsdp", "model"))
    xs.set_global_mesh(mesh)
    return mesh

from pathlib import Path
if Path("/kaggle/input").exists():
    PROJECT_ROOT = "/kaggle/working/BEBLaDII"
    REPO_URL = "https://github.com/Laeryid/BEBLaDII.git"
    if not os.path.exists(PROJECT_ROOT):
        subprocess.run(["git", "clone", REPO_URL, PROJECT_ROOT], check=True)
    else:
        subprocess.run(["git", "-C", PROJECT_ROOT, "pull"], check=True)
else:
    PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "C:/Experiments/BEBLaDII")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.beb_la_dii.model.dus import DUSModel
from src.beb_la_dii.model.vae import LatentEncoder
from src.beb_la_dii.utils.loss import safe_normalize

def resolve_model_path(base_path: str, fallback: str = "") -> str:
    import pathlib
    p = pathlib.Path(base_path)
    if not p.exists(): return fallback
    def check_dir(dir_path):
        return (dir_path / "config.json").exists()
    if check_dir(p): return str(p)
    for parent in list(p.parents)[:4]:
        if check_dir(parent): return str(parent)
    for config_file in sorted(p.rglob("config.json")):
        return str(config_file.parent)
    return fallback

def resolve_file_path(filename: str, fallback_dir="/kaggle/input") -> str:
    import pathlib
    p = pathlib.Path(fallback_dir)
    if p.exists():
        for f in p.rglob(filename):
            return str(f)
    return filename

def sync_to_gcs_and_delete(local_path: str, gcs_dir: str):
    if not gcs_dir.endswith("/"): gcs_dir += "/"
    gcs_path = gcs_dir + os.path.basename(local_path)
    try:
        subprocess.run(["gsutil", "-q", "cp", local_path, gcs_path], check=True)
        os.remove(local_path)
    except Exception as e:
        print(f"[GCS] Error syncing {local_path}: {e}")

def get_latest_gcs_checkpoint(gcs_dir: str, prefix: str = "phase6_ca_layers_step_"):
    try:
        if not gcs_dir.startswith("gs://"):
            return None, 0
        result = subprocess.run(["gsutil", "ls", gcs_dir], capture_output=True, text=True)
        if result.returncode != 0: return None, 0

        files = result.stdout.strip().split("\n")
        ckpt_files = [f for f in files if prefix in f and f.endswith(".pth")]
        if not ckpt_files: return None, 0

        def extract_step(filename):
            try: return int(filename.split("_step_")[-1].replace(".pth", ""))
            except ValueError: return -1

        ckpt_files.sort(key=extract_step)
        latest_file = ckpt_files[-1]
        return latest_file, extract_step(latest_file)
    except Exception:
        return None, 0

def sample_token_noise_levels(t_global: torch.Tensor, T: int, kappa_min: float = 1.0, kappa_max: float = 8.0) -> torch.Tensor:
    # Вычисляется на CPU, чтобы XLA не падал на torch.distributions.Beta
    dev = t_global.device
    B = t_global.shape[0]
    u = torch.rand(B, device=dev)
    kappa = torch.exp(math.log(kappa_min) + u * (math.log(kappa_max) - math.log(kappa_min)))
    a = (t_global * kappa).unsqueeze(1).expand(B, T)
    b = ((1.0 - t_global) * kappa).unsqueeze(1).expand(B, T)

    try:
        from torch.distributions import Beta
        t_actual = Beta(a, b).sample()
    except Exception:
        # XLA fallback safe approximation (хотя мы вызываем на CPU, это перестраховка)
        t_actual = torch.rand((B, T), device=dev)
        # linear blend between uniform and extreme as a fallback
        pass

    t_actual = t_actual.clamp(1e-4, 1.0)
    return t_actual

def content_cos_stats(outputs):
    target = safe_normalize(outputs["z_clean"], dim=-1)
    cp = (outputs["dus_final"] * target).sum(dim=-1)
    cb = (outputs["z_noisy"] * target).sum(dim=-1)
    cm = (~outputs["void_mask"].bool()).float()
    return cp, cb, cm

class Config:
    embedding_model_path = resolve_model_path("/kaggle/input/datasets/ragnar123/qwen2-5-1-5b", "Qwen/Qwen2.5-1.5B")
    modernbert_path      = resolve_model_path("/kaggle/input/models/answer-ai/modernbert/transformers/large/2", "answerdotai/ModernBERT-large")
    local_files_only     = Path("/kaggle/input").exists()
    dataset_path = resolve_file_path("train_phase6.parquet")
    val_dataset_path = resolve_file_path("val_phase6.parquet")
    encoder_weights = resolve_file_path("planB_phase1_checkpoints_phase1_vae_step_20000.pth")
    dus_weights     = resolve_file_path("phase4_step_85995.pth")
    sep_token       = os.path.join(PROJECT_ROOT, "storage/components/sep_token.pt")
    void_token      = os.path.join(PROJECT_ROOT, "storage/components/void_token.pt")
    latent_dict     = resolve_file_path("latent_dict.pt")
    output_dir      = "/kaggle/working/checkpoints/phase6_v2" if Path("/kaggle/input").exists() else os.path.join(PROJECT_ROOT, "checkpoints/phase6_v2")
    resume_from_checkpoint = False
    gcs_checkpoint_dir = "gs://bebladii-weigths-us/planB/phase6_v2/checkpoints/"
    batch_size    = 64 * 2
    max_length_q  = 512
    max_length_a  = 512
    learning_rate = 1e-4
    epochs        = 50
    max_steps     = 200000
    log_steps     = 10
    val_steps     = 200
    save_steps    = 1000
    warmup_steps  = 1000
    ema_decay     = 0.998
    pace_alpha    = 0.001
    unfreeze_k_after_ca = 4
    use_gradient_checkpointing = False
    loss_t_power  = 0.0   # 0.0 = равномерный вес лосса по токенам (ADR 098)
    use_prompt_sa = True  # Prompt-SA + RoPE перед CA
    val_num_samples = 200 # Размер валидационной выборки
    wandb_project = "BEBLaDII-Phase6-Kaggle"

args = Config()

class EMATracker:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone().float()

    def update(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name].copy_(self.decay * self.shadow[name] + (1.0 - self.decay) * param.float())

    def pace_pullback(self, model, alpha):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    ema_casted = self.shadow[name].to(param.dtype)
                    param.sub_(alpha * (param - ema_casted))

    def apply_shadow(self, model):
        self.backup = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.backup[name] = param.detach().clone()
                    param.copy_(self.shadow[name].to(param.dtype))
        xm.mark_step()

    def restore(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.copy_(self.backup[name])
        self.backup = {}
        xm.mark_step()

class QADataset(Dataset):
    def __init__(self, parquet_path, tokenizer, max_length_q=512, max_length_a=512):
        self.df = pd.read_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.max_length_q = max_length_q
        self.max_length_a = max_length_a

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        q_text, a_text = str(row['Q']), str(row['A'])
        q_enc = self.tokenizer(q_text, truncation=True, max_length=self.max_length_q, padding='max_length', return_tensors='pt')
        a_enc = self.tokenizer(a_text, truncation=True, max_length=self.max_length_a, padding='max_length', return_tensors='pt')
        return {
            'input_ids_q': q_enc.input_ids.squeeze(0),
            'attention_mask_q': q_enc.attention_mask.squeeze(0),
            'input_ids_a': a_enc.input_ids.squeeze(0),
            'attention_mask_a': a_enc.attention_mask.squeeze(0),
        }

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / (half - 1))
        args = t.unsqueeze(-1) * freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

def cosine_noise_schedule(t: torch.Tensor) -> torch.Tensor:
    return torch.cos(t * (math.pi / 2))

def spherical_noise(x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    B, T, D = x0.shape
    eps = safe_normalize(torch.randn_like(x0), dim=-1)
    if t.dim() == 1: t = t.view(B, 1, 1)
    elif t.dim() == 2: t = t.unsqueeze(-1)
    mu = cosine_noise_schedule(t)
    sigma = torch.sin(t * (math.pi / 2))
    return safe_normalize(mu * x0 + sigma * eps, dim=-1)

class AdaLNModulation(nn.Module):
    def __init__(self, t_emb_dim: int, hidden_dim: int):
        super().__init__()
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, 2 * hidden_dim))
        nn.init.zeros_(self.modulation[-1].weight)
        bias = torch.zeros(2 * hidden_dim)
        bias[hidden_dim:] = 1.0
        self.modulation[-1].bias = nn.Parameter(bias)
    def forward(self, t_emb: torch.Tensor) -> tuple:
        out = self.modulation(t_emb)
        shift, scale = out.chunk(2, dim=-1)
        if shift.dim() == 2: return shift.unsqueeze(1), scale.unsqueeze(1)
        return shift, scale

class AdaLNWrappedLayerNorm(nn.Module):
    def __init__(self, original_norm, adaln_module):
        super().__init__()
        self.original_norm = original_norm
        self.adaln = adaln_module
        self._current_t_emb = None
    def forward(self, x):
        out = self.original_norm(x)
        if self._current_t_emb is None: return out
        shift, scale = self.adaln(self._current_t_emb)
        return out * scale.to(out.dtype) + shift.to(out.dtype)

class CAPromptLayer(nn.Module):
    def __init__(self, dim=1024, t_emb_dim=256):
        super().__init__()
        self._disable = False
        self.last_ca_ratio = 0.0

        self.norm1 = nn.RMSNorm(dim)
        self.norm_q = nn.RMSNorm(dim)  # отдельная норма для Q промпта
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.adaln_ca = AdaLNModulation(t_emb_dim, dim)  # AdaLN для CA (модулирует A перед Q-проекцией)

        self.norm2 = nn.RMSNorm(dim)
        self.qkv_proj_sa = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj_sa = nn.Linear(dim, dim, bias=False)
        self.adaln_sa = AdaLNModulation(t_emb_dim, dim)  # AdaLN для SA (модулирует холст перед Self-Attention)

        self.gate = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.qkv_proj_sa.weight)
        nn.init.zeros_(self.out_proj_sa.weight)

    def forward(self, A, Q, mask_Q=None, warmup_factor=1.0, t_emb=None):
        if getattr(self, '_disable', False):
            return A

        # --- Cross-Attention ---
        # Q из промпта: статическая нормализация (вопрос не зависит от t)
        Q_norm = self.norm_q(Q)
        # A (холст): AdaLN — модулируем, что именно ищем в вопросе, в зависимости от t
        A_norm = self.norm1(A)
        if t_emb is not None:
            shift_ca, scale_ca = self.adaln_ca(t_emb)
            A_norm = A_norm * scale_ca + shift_ca

        q = self.q_proj(A_norm)
        kv = self.kv_proj(Q_norm)
        k, v = kv.chunk(2, dim=-1)

        attn_mask = None
        if mask_Q is not None:
            attn_mask = mask_Q.view(A.shape[0], 1, 1, -1).expand(-1, 1, A.shape[1], -1).bool()

        B, T_a, D = A.shape
        heads = 16
        head_dim = D // heads

        q = q.view(B, T_a, heads, head_dim).transpose(1, 2)
        k = k.view(B, -1, heads, head_dim).transpose(1, 2)
        v = v.view(B, -1, heads, head_dim).transpose(1, 2)

        ca_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        ca_out = ca_out.transpose(1, 2).reshape(B, T_a, D)
        ca_out = self.out_proj(ca_out)
        with torch.no_grad():
            self.last_ca_ratio = ca_out.norm(dim=-1).mean() / (A.norm(dim=-1).mean() + 1e-8)
        A = A + ca_out

        # --- Self-Attention ---
        # AdaLN — модулируем пропорции смешивания токенов холста в зависимости от t
        A_norm2 = self.norm2(A)
        if t_emb is not None:
            shift_sa, scale_sa = self.adaln_sa(t_emb)
            A_norm2 = A_norm2 * scale_sa + shift_sa

        qkv = self.qkv_proj_sa(A_norm2)
        q_sa, k_sa, v_sa = qkv.chunk(3, dim=-1)

        q_sa = q_sa.view(B, T_a, heads, head_dim).transpose(1, 2)
        k_sa = k_sa.view(B, T_a, heads, head_dim).transpose(1, 2)
        v_sa = v_sa.view(B, T_a, heads, head_dim).transpose(1, 2)

        sa_out = F.scaled_dot_product_attention(q_sa, k_sa, v_sa)
        sa_out = sa_out.transpose(1, 2).reshape(B, T_a, D)
        sa_out = self.out_proj_sa(sa_out)

        A = A + sa_out
        return A

class Phase6BlockWrapper(nn.Module):
    def __init__(self, original_layer, ca_layer=None):
        super().__init__()
        self.original_layer = original_layer
        self.ca_layer = ca_layer
    @property
    def attention_type(self): return self.original_layer.attention_type
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        out = self.original_layer(hidden_states, attention_mask=attention_mask, **kwargs)
        is_tuple = isinstance(out, (tuple, list))
        hidden = out[0] if is_tuple else out

        if self.ca_layer is not None:
            Z_prompt = getattr(self.ca_layer, '_current_Z_prompt', None)
            mask_Q = getattr(self.ca_layer, '_current_mask_Q', None)
            warmup_factor = getattr(self.ca_layer, '_current_warmup_factor', 1.0)
            t_emb = getattr(self.ca_layer, '_current_t_emb', None)
            if Z_prompt is not None:
                sep = hidden[:, 0:1, :]
                ans = hidden[:, 1:, :]
                ans = self.ca_layer(ans, Z_prompt, mask_Q, warmup_factor, t_emb=t_emb)
                hidden = torch.cat([sep, ans], dim=1)

        if is_tuple:
            return (hidden,) + tuple(out[1:])
        return hidden

class BEBLaDIIPhase6(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        _qwen = AutoModel.from_pretrained(config.embedding_model_path, torch_dtype=torch.bfloat16, local_files_only=config.local_files_only)
        self.qwen_embeddings = _qwen.get_input_embeddings()
        del _qwen

        self.encoder = LatentEncoder()
        if os.path.exists(config.encoder_weights):
            state = torch.load(config.encoder_weights, map_location="cpu", weights_only=False)
            if "encoder" in state: state = state["encoder"]
            self.encoder.load_state_dict(state, strict=False)
        self.encoder.to(torch.bfloat16)

        dus_wrapper = DUSModel.from_scratch(config={"base_model_id": config.modernbert_path}, weights_path=None, local_files_only=config.local_files_only)
        self.dus = dus_wrapper.model
        if config.use_gradient_checkpointing and hasattr(self.dus, "gradient_checkpointing_enable"):
            self.dus.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True, "preserve_rng_state": False})

        t_emb_dim = 256
        hidden_dim = 1024
        self.t_sin_embed = SinusoidalEmbedding(t_emb_dim)
        self.t_proj_global = nn.Sequential(nn.Linear(t_emb_dim, t_emb_dim * 4), nn.SiLU(), nn.Linear(t_emb_dim * 4, t_emb_dim))
        self.t_proj_token = nn.Sequential(nn.Linear(t_emb_dim, t_emb_dim * 4), nn.SiLU(), nn.Linear(t_emb_dim * 4, t_emb_dim))
        self.t_joint_proj = nn.Linear(t_emb_dim * 2, t_emb_dim)

        self.adaLN_attn = nn.ModuleList([AdaLNModulation(t_emb_dim, hidden_dim) for _ in range(40)])
        self.adaLN_mlp = nn.ModuleList([AdaLNModulation(t_emb_dim, hidden_dim) for _ in range(40)])
        for i, layer in enumerate(self.dus.layers):
            layer.attn_norm = AdaLNWrappedLayerNorm(layer.attn_norm, self.adaLN_attn[i])
            layer.mlp_norm = AdaLNWrappedLayerNorm(layer.mlp_norm, self.adaLN_mlp[i])

        self.register_buffer("sep_embed", torch.load(config.sep_token).float())
        self.register_buffer("void_embed", torch.load(config.void_token).float())
        self.register_buffer("latent_dict", torch.load(config.latent_dict).float())

        if os.path.exists(config.dus_weights):
            state = torch.load(config.dus_weights, map_location="cpu", weights_only=False)
            if "dus_ema" in state: state = state["dus_ema"]
            elif "dus" in state: state = state["dus"]
            elif "model_state_dict" in state: state = state["model_state_dict"]
            elif "model" in state: state = state["model"]
            clean_state = {k.replace("student.model.", "").replace("model.", "").replace("_orig_module.", ""): v for k, v in state.items()}
            self.load_state_dict(clean_state, strict=False)

        for p in self.parameters(): p.requires_grad = False

        self.ca_layers = nn.ModuleDict({
            "12": CAPromptLayer(1024, t_emb_dim=256),
            "24": CAPromptLayer(1024, t_emb_dim=256),
            "36": CAPromptLayer(1024, t_emb_dim=256),
        })
        for i in [11, 23, 35]:
            self.dus.layers[i] = Phase6BlockWrapper(self.dus.layers[i], self.ca_layers[str(i+1)])

        for p in self.ca_layers.parameters(): p.requires_grad = True
        unfreeze_k = getattr(config, "unfreeze_k_after_ca", 0)
        if unfreeze_k > 0:
            ca_indices = [11, 23, 35]
            unfreeze_indices = []
            for idx in ca_indices:
                for k in range(1, unfreeze_k + 1):
                    if idx + k < len(self.dus.layers):
                        unfreeze_indices.append(idx + k)
            for i, layer in enumerate(self.dus.layers):
                if i in unfreeze_indices:
                    for p in layer.parameters():
                        p.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "qwen_embeddings"): self.qwen_embeddings.eval()
        if hasattr(self, "encoder"): self.encoder.eval()

    def forward(self, input_ids_q, attention_mask_q, input_ids_a, attention_mask_a, warmup_factor=1.0, t_global=None):
        B, T_a = input_ids_a.shape
        with torch.no_grad():
            qwen_embeds_a = self.qwen_embeddings(input_ids_a)
            Z_A_clean, _, _ = self.encoder(qwen_embeds_a)
            Z_A_clean = safe_normalize(Z_A_clean.float(), dim=-1)

            qwen_embeds_q = self.qwen_embeddings(input_ids_q)
            Z_prompt, _, _ = self.encoder(qwen_embeds_q)
            Z_prompt = safe_normalize(Z_prompt.float(), dim=-1)

            # --- VOID TOKEN INJECTION ---
            void_embed = self.void_embed.to(Z_A_clean.dtype)

            # 1. Генерация маски слотов для void (5% шанс для mid-voids)
            is_void = torch.rand((B, T_a), device=Z_A_clean.device) < 0.05

            # 2. Префиксный сдвиг (0-10 void в начале)
            shift_k = torch.randint(0, 11, (B,), device=Z_A_clean.device)
            seq_indices = torch.arange(T_a, device=Z_A_clean.device).unsqueeze(0).expand(B, T_a)
            prefix_mask = seq_indices < shift_k.unsqueeze(1)
            is_void = is_void | prefix_mask

            # 3. Проверка на переполнение холста
            content_indices = torch.cumsum((~is_void).long(), dim=1) - 1
            orig_len = attention_mask_a.sum(dim=1)
            overflow_mask = content_indices[:, -1] < (orig_len - 1)

            # Отменяем инъекцию для тех фраз, где контент не помещается (оставляем только PAD)
            is_void = is_void & ~overflow_mask.unsqueeze(1)

            # Пересчитываем индексы после отмены
            content_indices = torch.cumsum((~is_void).long(), dim=1) - 1
            content_indices = content_indices.clamp(min=0, max=T_a - 1)

            # 4. Векторизованный сдвиг
            batch_indices = torch.arange(B, device=Z_A_clean.device).unsqueeze(1).expand(B, T_a)
            shifted_Z_A_clean = Z_A_clean[batch_indices, content_indices, :]
            shifted_attention_mask_a = attention_mask_a[batch_indices, content_indices]

            # 5. Итоговая маска void: инжектированные + оригинальные PAD-позиции
            void_mask = is_void | (shifted_attention_mask_a == 0)

            # 6. Применение void_embed
            void_mask_expanded = void_mask.unsqueeze(-1).expand(-1, -1, Z_A_clean.shape[-1])
            void_embed_expanded = void_embed.view(1, 1, -1).expand(B, T_a, -1)
            Z_A_clean = torch.where(void_mask_expanded, void_embed_expanded, shifted_Z_A_clean)
            Z_A_clean = safe_normalize(Z_A_clean.float(), dim=-1).to(void_embed.dtype)

            attention_mask_a = shifted_attention_mask_a
            # ---------------------------

            # === XLA SAFE SAMPLING (CPU) ===
            cpu_dev = torch.device('cpu')
            if t_global is None:
                t_global_cpu = torch.randint(1, 26, (B,), device=cpu_dev).float() / 25.0
            else:
                t_global_cpu = t_global.detach().to(cpu_dev).float()
            t_actual_cpu = sample_token_noise_levels(t_global_cpu, T_a)
            t_global = t_global_cpu.to(Z_A_clean.device)
            t_actual = t_actual_cpu.to(Z_A_clean.device)
            # ===============================

            z_noisy = spherical_noise(Z_A_clean, t_actual)

            sims = torch.matmul(z_noisy, self.latent_dict.T)
            RawDProx, _ = sims.max(dim=-1)
            t_reported = (1.0 - RawDProx).clamp(0.0, 1.0)

        # ==========================================
        # ПРОГОН ПРОМПТА (Слои 0-11, t=0)
        # ==========================================
        t_global_prompt = torch.zeros((B,), device=Z_A_clean.device, dtype=torch.float32)
        t_sin_prompt_global = self.t_sin_embed(t_global_prompt)
        t_emb_prompt_global = self.t_proj_global(t_sin_prompt_global)
        
        t_sin_prompt_token = self.t_sin_embed(t_global_prompt) # t=0
        t_emb_prompt_token = self.t_proj_token(t_sin_prompt_token)
        
        cond_prompt = torch.cat([t_emb_prompt_token, t_emb_prompt_global.unsqueeze(1).expand(-1, Z_prompt.shape[1], -1)], dim=-1)
        t_emb_prompt = self.t_joint_proj(cond_prompt)
        
        sep_t_emb_prompt = torch.zeros(B, 1, t_emb_prompt.shape[-1], device=t_emb_prompt.device, dtype=t_emb_prompt.dtype)
        t_emb_prompt_extended = torch.cat([sep_t_emb_prompt, t_emb_prompt], dim=1)

        for layer in self.dus.layers:
            layer_to_check = layer.original_layer if isinstance(layer, Phase6BlockWrapper) else layer
            if hasattr(layer_to_check, "attn_norm"): layer_to_check.attn_norm._current_t_emb = t_emb_prompt_extended
            if hasattr(layer_to_check, "mlp_norm"): layer_to_check.mlp_norm._current_t_emb = t_emb_prompt_extended

        # Подготовка входа для промпта
        prompt_in = Z_prompt.float()
        sep_prefix_prompt = self.sep_embed.unsqueeze(0).unsqueeze(0).expand(B, 1, -1).to(prompt_in.dtype)
        dus_prompt_extended = torch.cat([sep_prefix_prompt, prompt_in], dim=1)
        
        ones_sep_prompt = torch.ones((B, 1), device=attention_mask_q.device, dtype=attention_mask_q.dtype)
        prompt_mask_extended = torch.cat([ones_sep_prompt, attention_mask_q], dim=1)

        # Отключаем CA и прогоняем промпт
        with torch.no_grad():
            for ca in self.ca_layers.values():
                ca._disable = True
            
            prompt_outputs = self.dus(
                inputs_embeds=dus_prompt_extended,
                attention_mask=prompt_mask_extended,
                output_hidden_states=True,
            )
            
            # Индекс 12 соответствует выходу 12-го слоя (11-й по индексу 0-11)
            # Нам нужен выход слоя 11 (12-го по счету). Это hidden_states[12]
            Z_prompt_12 = prompt_outputs.hidden_states[12][:, 1:, :].to(Z_prompt.dtype)
            
            for ca in self.ca_layers.values():
                ca._disable = False

        # Обновляем Q для CA-слоев
        for ca in self.ca_layers.values():
            ca._current_Z_prompt = Z_prompt_12

        # ==========================================
        # ПРОГОН ХОЛСТА (Слои 0-39, реальный t)
        # ==========================================
        t_sin_global = self.t_sin_embed(t_global)
        t_emb_global = self.t_proj_global(t_sin_global)

        t_sin_token = self.t_sin_embed(t_reported)
        t_emb_token = self.t_proj_token(t_sin_token)

        cond = torch.cat([t_emb_token, t_emb_global.unsqueeze(1).expand(-1, T_a, -1)], dim=-1)
        t_emb = self.t_joint_proj(cond)

        sep_t_emb = torch.zeros(B, 1, t_emb.shape[-1], device=t_emb.device, dtype=t_emb.dtype)
        t_emb_extended = torch.cat([sep_t_emb, t_emb], dim=1)

        # Инъекция t_emb и контекста CA
        for layer in self.dus.layers:
            layer_to_check = layer.original_layer if isinstance(layer, Phase6BlockWrapper) else layer
            if hasattr(layer_to_check, "attn_norm"): layer_to_check.attn_norm._current_t_emb = t_emb_extended
            if hasattr(layer_to_check, "mlp_norm"): layer_to_check.mlp_norm._current_t_emb = t_emb_extended

        for ca in self.ca_layers.values():
            ca._current_mask_Q = attention_mask_q
            ca._current_warmup_factor = warmup_factor
            ca._current_t_emb = t_emb  # [B, T_a, 256] — per-token time embedding для AdaLN в CA/SA

        x_in = z_noisy.float()
        sep_prefix = self.sep_embed.unsqueeze(0).unsqueeze(0).expand(B, 1, -1).to(x_in.dtype)
        dus_input_extended = torch.cat([sep_prefix, x_in], dim=1)

        # Полностью снимаем маску с холста для DUS, так как void-позиции тоже обучаются
        attention_mask_extended = torch.ones((B, T_a + 1), device=x_in.device, dtype=torch.long)

        # FIX: Принудительно устанавливаем requires_grad для запуска Gradient Checkpointing внутри DUS
        # (На TPU use_reentrant=True используется в config, поэтому requires_grad_(True) не обязателен, но не мешает)
        # dus_input_extended.requires_grad_(True)

        dus_outputs = self.dus(
            inputs_embeds=dus_input_extended,
            attention_mask=attention_mask_extended,
            output_hidden_states=False,
        )

        pre_norm = dus_outputs.last_hidden_state[:, 1:, :].float()
        dus_final_raw = self.dus.final_norm(pre_norm.to(self.dus.dtype)).float()
        h_39 = safe_normalize(dus_final_raw, dim=-1)

        gate = torch.sin(t_global * (math.pi / 2)).view(B, 1, 1).to(h_39.dtype)
        dus_gated = gate * h_39 + (1.0 - gate) * x_in
        dus_final = safe_normalize(dus_gated, dim=-1)

        return {
            "z_clean": Z_A_clean,
            "z_noisy": z_noisy,
            "dus_final": dus_final,
            "t_actual": t_actual,
            "t_global": t_global,
            "void_mask": void_mask,
            "void_embed": void_embed
        }

def compute_phase6_loss(outputs, void_margin=0.0, loss_t_power=0.0):
    z_clean = outputs["z_clean"].float()
    dus_final = outputs["dus_final"].float()
    t_actual = outputs["t_actual"].float()
    void_mask = outputs["void_mask"].float()
    void_embed = outputs["void_embed"].float()

    target = safe_normalize(z_clean, dim=-1)
    cos_sim = (dus_final * target).sum(dim=-1)
    loss_el = 1.0 - cos_sim

    # --- NEW: Dynamic Void Penalty ---
    cos_sim_to_void = (dus_final * void_embed.view(1, 1, -1)).sum(dim=-1)
    cos_sim_target_to_void = (target * void_embed.view(1, 1, -1)).sum(dim=-1)
    content_mask = 1.0 - void_mask

    # 80% от расстояния между void и z_clean (в терминах косинусного сходства)
    dynamic_margin = 1.0 - 0.8 * (1.0 - cos_sim_target_to_void)

    import torch.nn.functional as F
    void_penalty = F.relu(cos_sim_to_void - dynamic_margin) * content_mask
    loss_el = loss_el + 0.5 * void_penalty
    # -------------------------

    # loss_t_power > 0 -> (1 - t_actual)^power, иначе равномерный вес 1.0
    if loss_t_power > 0.0:
        w_weighted = (1.0 - t_actual).pow(loss_t_power)
    else:
        w_weighted = torch.ones_like(t_actual)
    loss = (w_weighted * loss_el).sum() / w_weighted.sum().clamp(min=1e-8)

    avg_cos_sim = cos_sim.mean()
    cos_sim_to_void = (dus_final * void_embed.view(1, 1, -1)).sum(dim=-1)
    content_mask = 1.0 - void_mask

    true_void_sim = (cos_sim_to_void * void_mask).sum() / void_mask.sum().clamp(min=1e-8)
    content_to_void_sim = (cos_sim_to_void * content_mask).sum() / content_mask.sum().clamp(min=1e-8)

    return loss, avg_cos_sim, true_void_sim, content_to_void_sim

def _ca_inner(ca):
    """Возвращает исходный CAPromptLayer (снимает FSDP-обёртку, если есть)."""
    return getattr(ca, "module", ca)

def _set_seed(seed: int):
    torch.manual_seed(seed)
    xm.set_rng_state(seed)

def run_validation(model, ema_tracker, val_dataloader, device, mesh, global_step):
    ema_tracker.apply_shadow(model)
    model.eval()
    val_limit_batches = max(1, math.ceil(getattr(args, "val_num_samples", 200) / args.batch_size))

    def to_dev(v_batch):
        ids_q = v_batch['input_ids_q'].to(device)
        m_q = v_batch['attention_mask_q'].to(device)
        ids_a = v_batch['input_ids_a'].to(device)
        m_a = v_batch['attention_mask_a'].to(device)
        for t in (ids_q, m_q, ids_a, m_a):
            xs.mark_sharding(t, mesh, ("fsdp", None))
        return ids_q, m_q, ids_a, m_a

    val_loss = val_cos = val_void_cos = val_content_cos = 0.0
    val_batches = 0
    with torch.no_grad():
        for v_batch in val_dataloader:
            ids_q, m_q, ids_a, m_a = to_dev(v_batch)
            v_out = model(ids_q, m_q, ids_a, m_a, warmup_factor=1.0)
            v_loss, v_cos, v_v_cos, v_c_cos = compute_phase6_loss(v_out, void_margin=0.5, loss_t_power=args.loss_t_power)
            if val_batches == 0:
                val_loss, val_cos, val_void_cos, val_content_cos = v_loss, v_cos, v_v_cos, v_c_cos
            else:
                val_loss += v_loss; val_cos += v_cos
                val_void_cos += v_v_cos; val_content_cos += v_c_cos
            val_batches += 1
            xm.mark_step()
            if val_batches >= val_limit_batches: break
    n = max(1, val_batches)
    vals = torch.stack([val_loss, val_cos, val_void_cos, val_content_cos]).cpu().tolist() if val_batches > 0 else [0.0]*4
    log = {
        "val_loss_ema": vals[0] / n,
        "val_cos_ema": vals[1] / n,
        "val_void_cos_ema": vals[2] / n,
        "val_content_to_void_ema": vals[3] / n,
    }

    # --- fixed-t валидация: baseline (копия входа), с CA, без CA (абляция), чистый вклад CA ---
    with torch.no_grad():
        for tg_val in (0.2, 0.5, 0.8, 1.0):
            cp_sum = cp_no_ca_sum = cb_sum = n_tok = 0.0
            vb = 0
            for v_batch in val_dataloader:
                ids_q, m_q, ids_a, m_a = to_dev(v_batch)
                tg_t = torch.full((ids_a.shape[0],), tg_val)  # CPU: шум сэмплируется на CPU, sync не нужен
                seed = 42 + vb * 100 + int(tg_val * 10)

                _set_seed(seed)
                v_o = model(ids_q, m_q, ids_a, m_a, warmup_factor=1.0, t_global=tg_t)
                cp, cb, cm = content_cos_stats(v_o)
                cp_s = (cp * cm).sum(); cb_s = (cb * cm).sum(); cm_s = cm.sum()

                for ca in model.ca_layers.values(): _ca_inner(ca)._disable = True
                _set_seed(seed)
                v_o_no = model(ids_q, m_q, ids_a, m_a, warmup_factor=1.0, t_global=tg_t)
                cp_no, _, _ = content_cos_stats(v_o_no)
                cp_no_s = (cp_no * cm).sum()
                for ca in model.ca_layers.values(): _ca_inner(ca)._disable = False

                if vb == 0:
                    cp_sum, cp_no_ca_sum, cb_sum, n_tok = cp_s, cp_no_s, cb_s, cm_s
                else:
                    cp_sum += cp_s; cp_no_ca_sum += cp_no_s; cb_sum += cb_s; n_tok += cm_s
                vb += 1
                xm.mark_step()
                if vb >= val_limit_batches: break
            
            s_vals = torch.stack([cp_sum, cp_no_ca_sum, cb_sum, n_tok]).cpu().tolist() if vb > 0 else [0.0]*4
            cp_sum_v, cp_no_ca_sum_v, cb_sum_v, n_tok_v = s_vals
            
            n_tok_v = max(1.0, n_tok_v)
            with_ca, without_ca, base = cp_sum_v / n_tok_v, cp_no_ca_sum_v / n_tok_v, cb_sum_v / n_tok_v
            log[f"val_content_cos_tg{tg_val}"] = with_ca
            log[f"val_content_cos_noCA_tg{tg_val}"] = without_ca
            log[f"val_content_baseline_tg{tg_val}"] = base
            log[f"val_content_gain_tg{tg_val}"] = with_ca - base
            log[f"val_content_gain_CA_tg{tg_val}"] = with_ca - without_ca

    wandb.log(log, step=global_step)
    print(f"Step {global_step} | Val Loss (EMA): {log['val_loss_ema']:.4f} | Val Cos (EMA): {log['val_cos_ema']:.4f}")
    print(f"Step {global_step} | Val gain (Total / CA only): " + ", ".join(
        f"tg{tg}: {log[f'val_content_gain_tg{tg}']:+.3f} (CA: {log[f'val_content_gain_CA_tg{tg}']:+.3f})"
        for tg in (0.2, 0.5, 0.8, 1.0)))
    model.train()
    ema_tracker.restore(model)

def main():
    mesh = setup_spmd_mesh()
    device = xm.xla_device()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.wandb_project:
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            wandb.login(key=user_secrets.get_secret("WANDB_API_KEY"))
            gcp_sa = user_secrets.get_secret("GCP_SA_JSON")
            with open("gcp_sa.json", "w") as f: f.write(gcp_sa)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath("gcp_sa.json")
        except Exception: pass
        wandb.init(project=args.wandb_project, config=vars(args))

    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_path, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    val_dataloader = None
    try:
        dataset = QADataset(args.dataset_path, tokenizer, args.max_length_q, args.max_length_a)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
        if os.path.exists(args.val_dataset_path):
            val_dataset = QADataset(args.val_dataset_path, tokenizer, args.max_length_q, args.max_length_a)
            val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=2)
        else:
            print(f"Validation dataset not found at {args.val_dataset_path}. Validation will be skipped.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    model = BEBLaDIIPhase6(args).to(device)

    def shard_output(output, mesh): return None
    for key in list(model.ca_layers.keys()):
        wrapped = SpmdFullyShardedDataParallel(model.ca_layers[key], mesh=mesh, shard_output=shard_output)
        model.ca_layers[key] = wrapped
        layer_idx = int(key) - 1
        model.dus.layers[layer_idx].ca_layer = wrapped

    unfreeze_k = getattr(args, "unfreeze_k_after_ca", 0)
    if unfreeze_k > 0:
        ca_indices = [11, 23, 35]
        unfreeze_indices = []
        for idx in ca_indices:
            for k in range(1, unfreeze_k + 1):
                if idx + k < len(model.dus.layers):
                    unfreeze_indices.append(idx + k)
        for i in unfreeze_indices:
            wrapped = SpmdFullyShardedDataParallel(model.dus.layers[i], mesh=mesh, shard_output=shard_output)
            model.dus.layers[i] = wrapped

    ema_tracker = EMATracker(model, decay=args.ema_decay)
    ca_out_params, ca_qkv_params, ca_other_params = [], [], []
    for name, p in model.ca_layers.named_parameters():
        if p.requires_grad:
            if "out_proj" in name:
                ca_out_params.append(p)
            elif "q_proj" in name or "k_proj" in name or "v_proj" in name or "qkv_proj" in name:
                ca_qkv_params.append(p)
            else:
                ca_other_params.append(p)
    dus_params = []
    for name, p in model.dus.named_parameters():
        if p.requires_grad and "ca_layer" not in name:
            dus_params.append(p)

    optimizer = torch.optim.AdamW([
        {'params': ca_out_params, 'lr': 5e-4},     # x10 boost для out_proj (CA + Prompt-SA)
        {'params': ca_qkv_params, 'lr': 2e-4},     # q_proj / kv_proj / qkv_proj
        {'params': ca_other_params, 'lr': 5e-5},   # AdaLN, нормы
        {'params': dus_params, 'lr': 5e-5}
    ])

    model.train()
    global_step = 0
    start_time = time.time()
    checkpoints_saved = 0
    time_limit_seconds = 9.0 * 3600  # 9 часов для Kaggle TPU

    if args.resume_from_checkpoint and args.gcs_checkpoint_dir:
        latest_ckpt, step = get_latest_gcs_checkpoint(args.gcs_checkpoint_dir)
        if latest_ckpt:
            local_ckpt = os.path.join(args.output_dir, "resume_ca_layers.pth")
            try:
                subprocess.run(["gsutil", "-q", "cp", latest_ckpt, local_ckpt], check=True)
                ckpt_state = torch.load(local_ckpt, map_location="cpu", weights_only=False)
                model.ca_layers.load_state_dict(ckpt_state)
                ema_tracker = EMATracker(model, decay=args.ema_decay)
                global_step = step
                os.remove(local_ckpt)
            except Exception as e: print(f"Resume failed: {e}")

    for epoch in range(args.epochs):
        for batch in dataloader:
            if global_step >= args.max_steps: return

            warmup_factor_val = min(1.0, global_step / args.warmup_steps)
            warmup_factor = torch.tensor(warmup_factor_val, dtype=torch.float32, device=device)

            input_ids_q = batch['input_ids_q'].to(device)
            mask_q = batch['attention_mask_q'].to(device)
            input_ids_a = batch['input_ids_a'].to(device)
            mask_a = batch['attention_mask_a'].to(device)

            xs.mark_sharding(input_ids_q, mesh, ("fsdp", None))
            xs.mark_sharding(mask_q, mesh, ("fsdp", None))
            xs.mark_sharding(input_ids_a, mesh, ("fsdp", None))
            xs.mark_sharding(mask_a, mesh, ("fsdp", None))

            optimizer.zero_grad()
            outputs = model(input_ids_q, mask_q, input_ids_a, mask_a, warmup_factor=warmup_factor)
            loss, avg_cos_sim, true_void_sim, content_to_void_sim = compute_phase6_loss(outputs, void_margin=0.5, loss_t_power=args.loss_t_power)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            xm.optimizer_step(optimizer)

            ema_tracker.update(model)
            ema_tracker.pace_pullback(model, alpha=args.pace_alpha)

            xm.mark_step()

            cp, cb, cm = content_cos_stats(outputs)

            try:
                op12 = model.ca_layers["12"].out_proj.weight.detach().abs().mean()
                op36 = model.ca_layers["36"].out_proj.weight.detach().abs().mean()
                qp36 = model.ca_layers["36"].q_proj.weight.detach().abs().mean()
            except AttributeError:
                op12 = model.ca_layers["12"].module.out_proj.weight.detach().abs().mean()
                op36 = model.ca_layers["36"].module.out_proj.weight.detach().abs().mean()
                qp36 = model.ca_layers["36"].module.q_proj.weight.detach().abs().mean()

            # ca_ratio_* = ||ca_out|| / ||A|| (тензоры, без .item() в графе XLA)
            ratio12 = _ca_inner(model.ca_layers["12"]).last_ca_ratio
            ratio24 = _ca_inner(model.ca_layers["24"]).last_ca_ratio
            ratio36 = _ca_inner(model.ca_layers["36"]).last_ca_ratio

            if global_step % args.log_steps == 0:
                def log_step(s, l, c, vc, cc, w, o12, o36, q36, cp_t, cm_t, tg_t, r12, r24, r36):
                    log_dict = {
                        "loss": l.item(),
                        "cos_sim": c.item(),
                        "void_cos_sim": vc.item(),
                        "content_to_void_sim": cc.item(),
                        "warmup_factor": w.item(),
                        "out_proj_12_amp": o12.item(),
                        "out_proj_36_amp": o36.item(),
                        "q_proj_36_amp": q36.item(),
                        "ca_ratio_12": float(r12.item()) if torch.is_tensor(r12) else float(r12),
                        "ca_ratio_24": float(r24.item()) if torch.is_tensor(r24) else float(r24),
                        "ca_ratio_36": float(r36.item()) if torch.is_tensor(r36) else float(r36),
                    }
                    cp_cpu = cp_t.cpu()
                    cm_cpu = cm_t.cpu()
                    tg_cpu = tg_t.cpu()
                    B, T = cp_cpu.shape
                    tg_expanded = tg_cpu.unsqueeze(1).expand(B, T)

                    bins = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
                    for lo, hi in bins:
                        sel = ((tg_expanded > lo) & (tg_expanded <= hi)).float() * cm_cpu
                        sel_sum = sel.sum()
                        if sel_sum > 0:
                            log_dict[f"train_content_cos_tg_{lo}_{min(hi, 1.0)}"] = float(((cp_cpu * sel).sum() / sel_sum))

                    wandb.log(log_dict, step=s)

                xm.add_step_closure(log_step, args=(global_step, loss, avg_cos_sim, true_void_sim, content_to_void_sim, warmup_factor, op12, op36, qp36, cp, cm, outputs["t_global"], ratio12, ratio24, ratio36))

            if val_dataloader is not None and global_step % args.val_steps == 0 and global_step > 0:
                run_validation(model, ema_tracker, val_dataloader, device, mesh, global_step)

            if global_step % args.save_steps == 0 and global_step > 0:
                ema_tracker.apply_shadow(model)
                ckpt_path = os.path.join(args.output_dir, f"phase6_ca_layers_step_{global_step}.pth")
                state_dict = model.state_dict()
                named_params = dict(model.named_parameters())
                trainable_state = {k: v for k, v in state_dict.items() if k in named_params and named_params[k].requires_grad}
                xm.save(trainable_state, ckpt_path)
                ema_tracker.restore(model)
                if args.gcs_checkpoint_dir:
                    xm.add_step_closure(lambda: sync_to_gcs_and_delete(ckpt_path, args.gcs_checkpoint_dir))

                checkpoints_saved += 1
                elapsed = time.time() - start_time
                time_per_ckpt = elapsed / checkpoints_saved

                if elapsed + time_per_ckpt > time_limit_seconds:
                    print(f"Внимание: Оставшегося времени Kaggle ({time_limit_seconds - elapsed:.0f}s) "
                          f"может не хватить на следующий чекпоинт (нужно ~{time_per_ckpt:.0f}s). Прерывание.")
                    return

            global_step += 1

if __name__ == "__main__":
    main()
