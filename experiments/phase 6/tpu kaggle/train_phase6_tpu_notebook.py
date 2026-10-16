import math
import os
import subprocess
import sys
import time

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"): sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

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

GLOBAL_SPMD_MESH = None

def setup_spmd_mesh():
    global GLOBAL_SPMD_MESH
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (num_devices, 1)
    device_ids = np.array(range(num_devices))
    mesh = xs.Mesh(device_ids, mesh_shape, ("fsdp", "model"))
    xs.set_global_mesh(mesh)
    GLOBAL_SPMD_MESH = mesh
    return mesh

def mark_fsdp_sharding(tensor):
    if tensor is None or not torch.is_tensor(tensor):
        return tensor
    mesh = GLOBAL_SPMD_MESH
    if mesh is None and hasattr(xs, "get_global_mesh"):
        try:
            mesh = xs.get_global_mesh()
        except Exception:
            mesh = None
    if mesh is not None and hasattr(xs, "mark_sharding"):
        try:
            spec = ("fsdp",) + (None,) * (tensor.dim() - 1)
            xs.mark_sharding(tensor, mesh, spec)
        except Exception:
            pass
    return tensor

GLOBAL_NAN_NAMES = []
GLOBAL_NAN_CHECKS = []

def check_tensor_nan(name: str, tensor: torch.Tensor, step: int = 0, step_limit: int = 10):
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return
    if step > step_limit:
        return
    is_bad = torch.isnan(tensor).any() | torch.isinf(tensor).any()
    GLOBAL_NAN_NAMES.append(name)
    GLOBAL_NAN_CHECKS.append(is_bad)

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
    if os.path.exists(filename):
        return filename
    p_root = pathlib.Path(PROJECT_ROOT)
    for sub in ["BEBLaDII-planB-Phase6-Data", "BEBLaDII-planB-Phase5-Data", "BEBLaDII-planB-Phase4-Data", "BEBLaDII-planB-Phase3-Data", "experiments", "storage", "checkpoints"]:
        s = p_root / sub
        if s.exists():
            for m in s.rglob(filename):
                return str(m)
    return filename

def get_gcs_client():
    try:
        from google.cloud import storage
        return storage.Client()
    except Exception:
        return None

def is_state_dict_finite(state_dict):
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            if not torch.isfinite(v).all():
                return False
    return True

def sync_to_gcs_and_delete(local_path: str, gcs_dir: str, keep_local: bool = False):
    if not gcs_dir.endswith("/"): gcs_dir += "/"
    gcs_path = gcs_dir + os.path.basename(local_path)
    synced = False
    try:
        client = get_gcs_client()
        if client is not None:
            bucket_name = gcs_path.replace("gs://", "").split("/")[0]
            blob_name = gcs_path.replace(f"gs://{bucket_name}/", "")
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_path)
            synced = True
            print(f"[GCS] Synced: {local_path} → {gcs_path}")
        else:
            raise RuntimeError("GCS client unavailable")
    except Exception as e:
        import shutil
        if shutil.which("gsutil"):
            try:
                subprocess.run(["gsutil", "-q", "cp", local_path, gcs_path], check=True)
                synced = True
                print(f"[GCS-gsutil] Synced: {local_path} → {gcs_path}")
            except Exception as e2:
                print(f"[GCS] Error syncing {local_path}: {e} / gsutil: {e2}")
        else:
            print(f"[GCS] Error syncing {local_path}: {e} (gsutil unavailable)")

    if synced and not keep_local and os.path.exists(local_path):
        try:
            os.remove(local_path)
        except OSError:
            pass

def get_gcs_checkpoints(gcs_dir: str, prefix: str = "phase6_ca_layers_step_"):
    try:
        if not gcs_dir.startswith("gs://"):
            return []
        bucket_name = gcs_dir.replace("gs://", "").split("/")[0]
        pfx = gcs_dir.replace(f"gs://{bucket_name}/", "")
        client = get_gcs_client()
        ckpt_files = []
        if client is not None:
            try:
                bucket = client.bucket(bucket_name)
                blobs = bucket.list_blobs(prefix=pfx)
                ckpt_files = [f"gs://{bucket_name}/{b.name}" for b in blobs if prefix in b.name and b.name.endswith(".pth")]
            except Exception as e_client:
                print(f"[GCS Checkpoints] Client list error: {e_client}, falling back to gsutil...")
                client = None
        if not ckpt_files:
            import shutil
            if shutil.which("gsutil"):
                result = subprocess.run(["gsutil", "ls", gcs_dir], capture_output=True, text=True)
                if result.returncode == 0:
                    files = result.stdout.strip().split("\n")
                    ckpt_files = [f for f in files if prefix in f and f.endswith(".pth")]

        def extract_step(filename):
            try: return int(filename.split("_step_")[-1].replace(".pth", ""))
            except ValueError: return -1

        valid_ckpts = []
        for f in ckpt_files:
            s = extract_step(f)
            if s >= 0:
                valid_ckpts.append((f, s))
        valid_ckpts.sort(key=lambda x: x[1], reverse=True)
        return valid_ckpts
    except Exception as e:
        print(f"[GCS Checkpoints] Error listing checkpoints: {e}")
        return []

def get_latest_gcs_checkpoint(gcs_dir: str, prefix: str = "phase6_ca_layers_step_"):
    ckpts = get_gcs_checkpoints(gcs_dir, prefix=prefix)
    if ckpts:
        return ckpts[0][0], ckpts[0][1]
    return None, 0

def sample_token_noise_levels(t_global: torch.Tensor, T: int, kappa_min: float = 1.0,
                              kappa_max: float = 8.0, steps: int = 25) -> torch.Tensor:
    """
    Per-token t_i ~ Beta(t_g*kappa, (1-t_g)*kappa): E[t_i] = t_g.
    Малое kappa -> бимодально (чистые vs сильно зашумлённые): при низком t_g в основном чистые токены,
    при высоком - в основном зашумлённые. kappa выбирается на сэмпл (log-uniform [kappa_min, kappa_max]).
    t_global >= 1.0 -> чистый шум (все t_i = 1). Результат квантуется до сетки k/steps.
    Вычисляется на CPU, чтобы XLA не падал на torch.distributions.Beta.
    """
    B = t_global.shape[0]
    dev = t_global.device
    tg = t_global.float().clamp(0.02, 0.98)
    u = torch.rand(B, device=dev)
    kappa = torch.exp(math.log(kappa_min) + u * (math.log(kappa_max) - math.log(kappa_min)))
    a = (tg * kappa).view(B, 1).expand(B, T).contiguous()
    b = ((1.0 - tg) * kappa).view(B, 1).expand(B, T).contiguous()

    try:
        from torch.distributions import Beta
        t = Beta(a, b).sample()
    except Exception:
        t = tg.view(B, 1).expand(B, T).contiguous()

    # Защита от редких NaN при марсалья-цикуда сэмплировании малых концентраций гамма-распределения на CPU
    fallback_t = tg.view(B, 1).expand(B, T).contiguous()
    t = torch.where(torch.isfinite(t), t, fallback_t)

    t = torch.round(t * steps) / steps
    t = torch.where(t_global.view(B, 1) >= 1.0, torch.ones_like(t), t)
    return t.clamp(0.0, 1.0)

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
    resume_from_checkpoint = False if os.environ.get("SMOKE_TEST") else True
    gcs_checkpoint_dir = "gs://bebladii-weigths-us/planB/phase6/checkpoints/"
    batch_size    = 2 if os.environ.get("SMOKE_TEST") else 64*2
    max_length_q  = 64 if os.environ.get("SMOKE_TEST") else 512
    max_length_a  = 64 if os.environ.get("SMOKE_TEST") else 512
    learning_rate = 1e-4
    epochs        = 50
    max_steps     = int(os.environ.get("MAX_STEPS", 200000))
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
    val_num_samples = 80 # Размер валидационной выборки
    wandb_project = "BEBLaDII-Phase6-Kaggle"
    run_diagnosis_first  = True  # Запуск 1-шаговой изоляционной диагностики NaN перед обучением
    stop_after_diagnosis = False # Не прерывать после успешной диагностики, продолжать обучение

args = Config()

class EMATracker:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone().float()

    def update(self, model, is_safe=None):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if is_safe is not None:
                        new_shadow = self.decay * self.shadow[name] + (1.0 - self.decay) * param.float()
                        self.shadow[name].copy_(torch.where(is_safe, new_shadow, self.shadow[name]))
                    else:
                        self.shadow[name].copy_(self.decay * self.shadow[name] + (1.0 - self.decay) * param.float())

    def pace_pullback(self, model, alpha, is_safe=None):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    ema_casted = self.shadow[name].to(param.dtype)
                    if is_safe is not None:
                        new_param = param - alpha * (param - ema_casted)
                        param.copy_(torch.where(is_safe, new_param, param))
                    else:
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
    eps = torch.randn_like(x0)
    # Ортогонализация eps к x0 в касательном пространстве сферы T_{x0}S^{D-1}
    proj = (eps * x0).sum(dim=-1, keepdim=True) * x0
    eps_ortho = safe_normalize(eps - proj, dim=-1)
    if t.dim() == 1: t = t.view(B, 1, 1)
    elif t.dim() == 2: t = t.unsqueeze(-1)
    mu = cosine_noise_schedule(t)
    sigma = torch.sin(t * (math.pi / 2))
    return safe_normalize(mu * x0 + sigma * eps_ortho, dim=-1)

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

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    """
    XLA-friendly rotary position embedding (RoPE) without learnable parameters.
    """
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().view(1, 1, seq_len, self.dim).to(dtype=x.dtype)
        sin = emb.sin().view(1, 1, seq_len, self.dim).to(dtype=x.dtype)
        return cos, sin

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

        heads = 16
        head_dim = dim // heads
        self.rope_sa = RotaryEmbedding(dim=head_dim)

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

        B, T_a, D = A.shape
        heads = 16
        head_dim = D // heads

        q = q.view(B, T_a, heads, head_dim).transpose(1, 2)
        k = k.view(B, -1, heads, head_dim).transpose(1, 2)
        v = v.view(B, -1, heads, head_dim).transpose(1, 2)

        attn_mask = None
        if mask_Q is not None:
            # Безопасная аддитивная float-маска для TPU XLA (ADR 103):
            # 0.0 для валидных токенов, -10000.0 для паддинга (устраняет underflow/NaN в backward)
            mask_q_4d = mask_Q.view(B, 1, 1, -1).bool()
            attn_mask = torch.where(
                mask_q_4d,
                torch.zeros(1, dtype=q.dtype, device=q.device),
                torch.tensor(-10000.0, dtype=q.dtype, device=q.device)
            )

        ca_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        ca_out = ca_out.transpose(1, 2).reshape(B, T_a, D)
        ca_out = self.out_proj(ca_out)
        with torch.no_grad():
            self.last_ca_ratio = ca_out.norm(dim=-1).mean() / (A.norm(dim=-1).mean() + 1e-8)
        A = A + warmup_factor * ca_out

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

        cos_sa, sin_sa = self.rope_sa(q_sa, T_a)
        q_sa, k_sa = apply_rotary_pos_emb(q_sa, k_sa, cos_sa, sin_sa)

        sa_out = F.scaled_dot_product_attention(q_sa, k_sa, v_sa)
        sa_out = sa_out.transpose(1, 2).reshape(B, T_a, D)
        sa_out = self.out_proj_sa(sa_out)

        A = A + warmup_factor * sa_out
        return A

def _unwrap_module(m):
    """Снимает обёртки SpmdFullyShardedDataParallel / DDP / DataParallel / torch._dynamo."""
    while hasattr(m, "module") or hasattr(m, "_orig_module"):
        m = getattr(m, "module", None) or getattr(m, "_orig_module", None)
    return m

def _ca_inner(ca):
    """Возвращает исходный CAPromptLayer (снимает FSDP/SPMD обёртку, если есть)."""
    return _unwrap_module(ca)

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
            ca_inner = _ca_inner(self.ca_layer)
            if not getattr(ca_inner, '_disable', False):
                Z_prompt = getattr(self.ca_layer, '_current_Z_prompt', None)
                if Z_prompt is None:
                    Z_prompt = getattr(ca_inner, '_current_Z_prompt', None)
                mask_Q = getattr(self.ca_layer, '_current_mask_Q', None)
                if mask_Q is None:
                    mask_Q = getattr(ca_inner, '_current_mask_Q', None)
                warmup_factor = getattr(self.ca_layer, '_current_warmup_factor', 1.0)
                t_emb = getattr(self.ca_layer, '_current_t_emb', None)
                if t_emb is None:
                    t_emb = getattr(ca_inner, '_current_t_emb', None)
                if Z_prompt is not None:
                    sep = hidden[:, 0:1, :]
                    ans = hidden[:, 1:, :]
                    ans = self.ca_layer(ans, Z_prompt, mask_Q, warmup_factor, t_emb=t_emb)
                    hidden = torch.cat([sep, ans], dim=1)

        if is_tuple:
            return (hidden,) + tuple(out[1:])
        return hidden

def patch_modernbert_xla_safe_mask(modernbert_model):
    """
    Патчит генерацию и применение масок в ModernBertModel для стабильности на TPU XLA (ADR 103, ADR 104).
    1. Заменяет булевы маски (True=attend, False=mask) на аддитивные float-маски (0.0 и -10000.0).
    2. Заменяет экстремальные отрицательные числа (< -10000.0, -inf, -3.4e38) на безопасный сентинел -10000.0.
    3. Перехватывает как attention_mask, так и sliding_window_mask на всех уровнях:
       - legacy _update_attention_mask (transformers <= 4.49)
       - create_bidirectional_mask, create_bidirectional_sliding_window_mask, sdpa_mask
       - MODERNBERT_ATTENTION_FUNCTION["sdpa"]
       - layer.attn.forward
    """
    def _clamp_mask_tensor(mask, target_dtype=torch.float32):
        if mask is None or not torch.is_tensor(mask):
            return mask
        if mask.dtype == torch.bool:
            zero = torch.tensor(0.0, dtype=target_dtype, device=mask.device)
            neg = torch.tensor(-10000.0, dtype=target_dtype, device=mask.device)
            return torch.where(mask, zero, neg)
        if mask.is_floating_point():
            safe_min = torch.tensor(-10000.0, dtype=mask.dtype, device=mask.device)
            return torch.where((mask < -10000.0) | torch.isneginf(mask), safe_min, mask)
        return mask

    def _wrap_mask_fn(orig_fn):
        def safe_mask_fn(*args, **kwargs):
            mask = orig_fn(*args, **kwargs)
            return _clamp_mask_tensor(mask)
        return safe_mask_fn

    # 1. Если есть legacy-метод _update_attention_mask (transformers <= 4.49)
    if hasattr(modernbert_model, "_update_attention_mask"):
        orig_update_mask = modernbert_model._update_attention_mask
        def safe_update_attention_mask(attention_mask: torch.Tensor, output_attentions: bool = False):
            global_mask, sliding_mask = orig_update_mask(attention_mask, output_attentions=output_attentions)
            return _clamp_mask_tensor(global_mask), _clamp_mask_tensor(sliding_mask)
        modernbert_model._update_attention_mask = safe_update_attention_mask

    # 2. Модуль modeling_modernbert (transformers >= 4.50)
    mod = sys.modules.get(modernbert_model.__class__.__module__)
    if mod is not None:
        for fn_name in ["create_bidirectional_mask", "create_bidirectional_sliding_window_mask"]:
            if hasattr(mod, fn_name):
                setattr(mod, fn_name, _wrap_mask_fn(getattr(mod, fn_name)))
        if hasattr(mod, "MODERNBERT_ATTENTION_FUNCTION") and "sdpa" in mod.MODERNBERT_ATTENTION_FUNCTION:
            orig_sdpa = mod.MODERNBERT_ATTENTION_FUNCTION["sdpa"]
            def safe_sdpa(module, qkv, attention_mask=None, sliding_window_mask=None, *args, **kwargs):
                td = qkv.dtype if torch.is_tensor(qkv) else torch.float32
                if attention_mask is not None:
                    attention_mask = _clamp_mask_tensor(attention_mask, target_dtype=td)
                if sliding_window_mask is not None:
                    sliding_window_mask = _clamp_mask_tensor(sliding_window_mask, target_dtype=td)
                return orig_sdpa(module, qkv, attention_mask=attention_mask, sliding_window_mask=sliding_window_mask, *args, **kwargs)
            mod.MODERNBERT_ATTENTION_FUNCTION["sdpa"] = safe_sdpa

    # 3. Общий модуль transformers.masking_utils
    try:
        import transformers.masking_utils as mu
        for fn_name in ["create_bidirectional_mask", "create_bidirectional_sliding_window_mask", "sdpa_mask"]:
            if hasattr(mu, fn_name):
                setattr(mu, fn_name, _wrap_mask_fn(getattr(mu, fn_name)))
    except (ImportError, AttributeError):
        pass

    # 4. Непосредственный перехват на уровне слоев внимания (layer.attn.forward)
    if hasattr(modernbert_model, "layers"):
        for layer in modernbert_model.layers:
            attn_mod = getattr(layer, "attn", None)
            if attn_mod is not None and hasattr(attn_mod, "forward"):
                orig_attn_forward = attn_mod.forward
                def _make_safe_attn_forward(orig_fn):
                    def safe_attn_forward(*args, **kwargs):
                        td = args[0].dtype if (len(args) > 0 and torch.is_tensor(args[0])) else torch.float32
                        if "hidden_states" in kwargs and torch.is_tensor(kwargs["hidden_states"]):
                            td = kwargs["hidden_states"].dtype
                        if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                            kwargs["attention_mask"] = _clamp_mask_tensor(kwargs["attention_mask"], target_dtype=td)
                        if "sliding_window_mask" in kwargs and kwargs["sliding_window_mask"] is not None:
                            kwargs["sliding_window_mask"] = _clamp_mask_tensor(kwargs["sliding_window_mask"], target_dtype=td)
                        if len(args) >= 3 and args[2] is not None:
                            args_list = list(args)
                            args_list[2] = _clamp_mask_tensor(args_list[2], target_dtype=td)
                            args = tuple(args_list)
                        return orig_fn(*args, **kwargs)
                    return safe_attn_forward
                attn_mod.forward = _make_safe_attn_forward(orig_attn_forward)

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
        self.dus.to(torch.float32)
        type(self.dus).dtype = property(lambda self: torch.float32)
        patch_modernbert_xla_safe_mask(self.dus)
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
        self.register_buffer("latent_dict", torch.load(config.latent_dict).to(torch.bfloat16))

        if os.path.exists(config.dus_weights):
            state = torch.load(config.dus_weights, map_location="cpu", weights_only=False)
            dus_state = state.get("dus_ema", state.get("dus", state.get("model_state_dict", state.get("model", state))))
            clean_dus_state = {k.replace("student.model.", "").replace("model.", "").replace("_orig_module.", ""): v for k, v in dus_state.items()}
            self.dus.load_state_dict(clean_dus_state, strict=False)

            if "adaLN_attn_ema" in state or "adaLN_attn" in state:
                ada_attn_st = state.get("adaLN_attn_ema", state.get("adaLN_attn"))
                clean_ada_attn = {k.replace("_orig_module.", ""): v for k, v in ada_attn_st.items()}
                self.adaLN_attn.load_state_dict(clean_ada_attn, strict=False)

            if "adaLN_mlp_ema" in state or "adaLN_mlp" in state:
                ada_mlp_st = state.get("adaLN_mlp_ema", state.get("adaLN_mlp"))
                clean_ada_mlp = {k.replace("_orig_module.", ""): v for k, v in ada_mlp_st.items()}
                self.adaLN_mlp.load_state_dict(clean_ada_mlp, strict=False)

            if "t_proj_global_ema" in state or "t_proj_global" in state:
                st_glob = state.get("t_proj_global_ema", state.get("t_proj_global"))
                self.t_proj_global.load_state_dict({k.replace("_orig_module.", ""): v for k, v in st_glob.items()}, strict=False)

            if "t_proj_token_ema" in state or "t_proj_token" in state:
                st_tok = state.get("t_proj_token_ema", state.get("t_proj_token"))
                self.t_proj_token.load_state_dict({k.replace("_orig_module.", ""): v for k, v in st_tok.items()}, strict=False)

            if "t_joint_proj_ema" in state or "t_joint_proj" in state:
                st_joint = state.get("t_joint_proj_ema", state.get("t_joint_proj"))
                self.t_joint_proj.load_state_dict({k.replace("_orig_module.", ""): v for k, v in st_joint.items()}, strict=False)

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

    def forward(self, input_ids_q, attention_mask_q, input_ids_a, attention_mask_a, warmup_factor=1.0, t_global=None, disable_ca: bool = False):
        B, T_a = input_ids_a.shape
        with torch.no_grad():
            qwen_embeds_a = self.qwen_embeddings(input_ids_a)
            Z_A_clean, _, _ = self.encoder(qwen_embeds_a)
            Z_A_clean = safe_normalize(Z_A_clean.float(), dim=-1)

            if not disable_ca:
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

            # 4. Векторизованный сдвиг вдоль dim=1 (сохраняет SPMD batch sharding по dim=0)
            gather_idx_3d = content_indices.unsqueeze(-1).expand(-1, -1, Z_A_clean.shape[-1])
            shifted_Z_A_clean = torch.gather(Z_A_clean, 1, gather_idx_3d)
            shifted_attention_mask_a = torch.gather(attention_mask_a, 1, content_indices)

            # 5. Итоговая маска void: инжектированные + оригинальные PAD-позиции
            void_mask = is_void | (shifted_attention_mask_a == 0)

            # 6. Применение void_embed
            void_mask_expanded = void_mask.unsqueeze(-1).expand(-1, -1, Z_A_clean.shape[-1])
            void_embed_expanded = void_embed.view(1, 1, -1).expand(B, T_a, -1)
            Z_A_clean = torch.where(void_mask_expanded, void_embed_expanded, shifted_Z_A_clean)
            Z_A_clean = safe_normalize(Z_A_clean.float(), dim=-1).to(void_embed.dtype)
            Z_A_clean = mark_fsdp_sharding(Z_A_clean)

            attention_mask_a = shifted_attention_mask_a
            # ---------------------------

            # === XLA SAFE SAMPLING (CPU) ===
            cpu_dev = torch.device('cpu')
            if t_global is None:
                t_global_cpu = torch.randint(1, 26, (B,), device=cpu_dev).float() / 25.0
            else:
                t_global_cpu = t_global.detach().to(cpu_dev).float()
            t_actual_cpu = sample_token_noise_levels(t_global_cpu, T_a)
            t_global = mark_fsdp_sharding(t_global_cpu.to(Z_A_clean.device))
            t_actual = mark_fsdp_sharding(t_actual_cpu.to(Z_A_clean.device))
            # ===============================

            z_noisy = mark_fsdp_sharding(spherical_noise(Z_A_clean, t_actual))

            # Чанкованное вычисление RawDProx: делим latent_dict на 16 чанков по ~9.5K векторов
            # и считаем матричное умножение в bfloat16 для предотвращения OOM в HBM
            z_noisy_bf16 = z_noisy.to(torch.bfloat16)
            RawDProx = torch.full((B, T_a), -1.0, device=z_noisy.device, dtype=torch.bfloat16)
            for dict_chunk in self.latent_dict.chunk(16, dim=0):
                sims_chunk = torch.matmul(z_noisy_bf16, dict_chunk.T)
                RawDProx = torch.maximum(RawDProx, sims_chunk.max(dim=-1).values)
            t_reported = (1.0 - RawDProx.float()).clamp(0.0, 1.0)

        # ==========================================
        # ПРОГОН ПРОМПТА (Слои 0-11, t=0, только если CA включен)
        # ==========================================
        if not disable_ca:
            t_global_prompt = torch.zeros((B,), device=Z_A_clean.device, dtype=torch.float32)
            t_sin_prompt_global = self.t_sin_embed(t_global_prompt)
            t_emb_prompt_global = self.t_proj_global(t_sin_prompt_global)

            t_sin_prompt_token = self.t_sin_embed(t_global_prompt) # t=0
            t_emb_prompt_token = self.t_proj_token(t_sin_prompt_token)

            cond_prompt = torch.cat([t_emb_prompt_token.unsqueeze(1).expand(-1, Z_prompt.shape[1], -1), t_emb_prompt_global.unsqueeze(1).expand(-1, Z_prompt.shape[1], -1)], dim=-1)
            t_emb_prompt = self.t_joint_proj(cond_prompt)

            sep_t_emb_prompt = torch.zeros(B, 1, t_emb_prompt.shape[-1], device=t_emb_prompt.device, dtype=t_emb_prompt.dtype)
            t_emb_prompt_extended = mark_fsdp_sharding(torch.cat([sep_t_emb_prompt, t_emb_prompt], dim=1))

            for layer in self.dus.layers:
                layer_to_check = layer.original_layer if isinstance(layer, Phase6BlockWrapper) else layer
                layer_to_check = _unwrap_module(layer_to_check)
                if hasattr(layer_to_check, "attn_norm"): layer_to_check.attn_norm._current_t_emb = t_emb_prompt_extended
                if hasattr(layer_to_check, "mlp_norm"): layer_to_check.mlp_norm._current_t_emb = t_emb_prompt_extended

            # Подготовка входа для промпта
            prompt_in = Z_prompt.float()
            sep_prefix_prompt = self.sep_embed.unsqueeze(0).unsqueeze(0).expand(B, 1, -1).to(prompt_in.dtype)
            dus_prompt_extended = mark_fsdp_sharding(torch.cat([sep_prefix_prompt, prompt_in], dim=1))

            ones_sep_prompt = torch.ones((B, 1), device=attention_mask_q.device, dtype=attention_mask_q.dtype)
            prompt_mask_extended = mark_fsdp_sharding(torch.cat([ones_sep_prompt, attention_mask_q], dim=1))

            # Отключаем CA и прогоняем промпт через первые 12 слоев (0-11) DUS (ADR 099)
            with torch.no_grad():
                for ca in self.ca_layers.values():
                    _ca_inner(ca)._disable = True
                    ca._disable = True

                seq_len_p = dus_prompt_extended.shape[1]
                position_ids_p = torch.arange(seq_len_p, device=dus_prompt_extended.device).unsqueeze(0)
                if hasattr(self.dus, "_update_attention_mask"):
                    attention_mask_p, sliding_window_mask_p = self.dus._update_attention_mask(
                        prompt_mask_extended, output_attentions=False
                    )
                else:
                    attention_mask_p = prompt_mask_extended
                    sliding_window_mask_p = None

                hidden_p = dus_prompt_extended
                for layer in self.dus.layers[:12]:
                    kwargs_p = {
                        "attention_mask": attention_mask_p,
                        "position_ids": position_ids_p,
                    }
                    if sliding_window_mask_p is not None:
                        kwargs_p["sliding_window_mask"] = sliding_window_mask_p
                    layer_out = layer(hidden_p, **kwargs_p)
                    hidden_p = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out

                Z_prompt_12 = hidden_p[:, 1:, :].to(Z_prompt.dtype)

                for ca in self.ca_layers.values():
                    _ca_inner(ca)._disable = False
                    ca._disable = False

            # Обновляем Q для CA-слоев
            for ca in self.ca_layers.values():
                ca._current_Z_prompt = Z_prompt_12
                _ca_inner(ca)._current_Z_prompt = Z_prompt_12
        else:
            for ca in self.ca_layers.values():
                _ca_inner(ca)._disable = True
                ca._disable = True

        # ==========================================
        # ПРОГОН ХОЛСТА (Слои 0-39, реальный t)
        # ==========================================
        t_sin_global = self.t_sin_embed(t_global)
        t_emb_global = self.t_proj_global(t_sin_global)

        t_sin_token = self.t_sin_embed(t_reported)
        t_emb_token = self.t_proj_token(t_sin_token)

        cond = torch.cat([t_emb_token, t_emb_global.unsqueeze(1).expand(-1, T_a, -1)], dim=-1)
        t_emb = mark_fsdp_sharding(self.t_joint_proj(cond))

        sep_t_emb = torch.zeros(B, 1, t_emb.shape[-1], device=t_emb.device, dtype=t_emb.dtype)
        t_emb_extended = mark_fsdp_sharding(torch.cat([sep_t_emb, t_emb], dim=1))

        # Инъекция t_emb и контекста CA
        for layer in self.dus.layers:
            layer_to_check = layer.original_layer if isinstance(layer, Phase6BlockWrapper) else layer
            layer_to_check = _unwrap_module(layer_to_check)
            if hasattr(layer_to_check, "attn_norm"): layer_to_check.attn_norm._current_t_emb = t_emb_extended
            if hasattr(layer_to_check, "mlp_norm"): layer_to_check.mlp_norm._current_t_emb = t_emb_extended

        for ca in self.ca_layers.values():
            ca._current_mask_Q = attention_mask_q
            ca._current_warmup_factor = warmup_factor
            ca._current_t_emb = t_emb  # [B, T_a, 256] — per-token time embedding для AdaLN в CA/SA
            _ca_inner(ca)._current_mask_Q = attention_mask_q
            _ca_inner(ca)._current_warmup_factor = warmup_factor
            _ca_inner(ca)._current_t_emb = t_emb

        x_in = z_noisy.float()
        sep_prefix = self.sep_embed.unsqueeze(0).unsqueeze(0).expand(B, 1, -1).to(x_in.dtype)
        dus_input_extended = mark_fsdp_sharding(torch.cat([sep_prefix, x_in], dim=1))

        # Полностью снимаем маску с холста для DUS, так как void-позиции тоже обучаются
        attention_mask_extended = mark_fsdp_sharding(torch.ones((B, T_a + 1), device=x_in.device, dtype=torch.long))

        # FIX: Принудительно устанавливаем requires_grad для корректной работы backward-хуков FSDP
        dus_input_extended = dus_input_extended.requires_grad_(True)

        dus_outputs = self.dus(
            inputs_embeds=dus_input_extended,
            attention_mask=attention_mask_extended,
            output_hidden_states=False,
        )

        pre_norm = dus_outputs.last_hidden_state[:, 1:, :].float()
        dus_final_raw = self.dus.final_norm(pre_norm).float()
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

def _set_seed(seed: int):
    torch.manual_seed(seed)
    xm.set_rng_state(seed)

def run_validation(model, ema_tracker, val_dataloader, device, mesh, global_step, metrics_history=None):
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

                # 1. Прогон с CA
                _set_seed(seed)
                v_o = model(ids_q, m_q, ids_a, m_a, warmup_factor=1.0, t_global=tg_t, disable_ca=False)
                cp, cb, cm = content_cos_stats(v_o)
                cp_s = (cp * cm).sum(); cb_s = (cb * cm).sum(); cm_s = cm.sum()
                del v_o, cp, cb
                xm.mark_step()

                # 2. Прогон без CA (абляция)
                _set_seed(seed)
                v_o_no = model(ids_q, m_q, ids_a, m_a, warmup_factor=1.0, t_global=tg_t, disable_ca=True)
                cp_no, _, _ = content_cos_stats(v_o_no)
                cp_no_s = (cp_no * cm).sum()
                del v_o_no, cp_no
                xm.mark_step()

                if vb == 0:
                    cp_sum, cp_no_ca_sum, cb_sum, n_tok = cp_s, cp_no_s, cb_s, cm_s
                else:
                    cp_sum += cp_s; cp_no_ca_sum += cp_no_s; cb_sum += cb_s; n_tok += cm_s
                vb += 1
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
    if metrics_history is not None:
        metrics_history.append({"step": global_step, "is_val": True, **log})
    print(f"Step {global_step} | Val Loss (EMA): {log['val_loss_ema']:.4f} | Val Cos (EMA): {log['val_cos_ema']:.4f}")
    print(f"Step {global_step} | Val gain (Total / CA only): " + ", ".join(
        f"tg{tg}: {log[f'val_content_gain_tg{tg}']:+.3f} (CA: {log[f'val_content_gain_CA_tg{tg}']:+.3f})"
        for tg in (0.2, 0.5, 0.8, 1.0)))
    model.train()
    ema_tracker.restore(model)

def run_nan_diagnosis(model, batch, device, mesh, ca_params=None, dus_params=None):
    print("\n" + "="*70)
    print("           [DIAGNOSIS] FULL NaN GRADIENT AUDIT (TPU/CPU)          ")
    print("="*70)

    def clear_tpu_memory():
        import gc
        gc.collect()
        try:
            torch_xla.sync(wait=True, reset_scope=True)
        except Exception:
            try:
                xm.mark_step()
            except Exception:
                pass

    input_ids_q = batch['input_ids_q'].to(device)
    mask_q = batch['attention_mask_q'].to(device)
    input_ids_a = batch['input_ids_a'].to(device)
    mask_a = batch['attention_mask_a'].to(device)

    xs.mark_sharding(input_ids_q, mesh, ("fsdp", None))
    xs.mark_sharding(mask_q, mesh, ("fsdp", None))
    xs.mark_sharding(input_ids_a, mesh, ("fsdp", None))
    xs.mark_sharding(mask_a, mesh, ("fsdp", None))

    named_params = dict(model.named_parameters())
    trainable_named = {n: p for n, p in named_params.items() if p.requires_grad}
    print(f"Total trainable parameters: {len(trainable_named)}")

    # -------------------------------------------------------------
    # TEST 0: Input Attention Mask Audit
    # -------------------------------------------------------------
    print("\n[TEST 0] Input Attention Mask Audit:")
    q_act = mask_q.sum(dim=-1).tolist()
    a_act = mask_a.sum(dim=-1).tolist()
    print(f"  Active tokens Q (Question): min={min(q_act)}, max={max(q_act)}, zero_count={(mask_q.sum(dim=-1) == 0).sum().item()}")
    print(f"  Active tokens A (Answer):   min={min(a_act)}, max={max(a_act)}, zero_count={(mask_a.sum(dim=-1) == 0).sum().item()}")
    if any(c == 0 for c in q_act):
        print("  [ALERT] Samples with all-zero attention_mask_q detected! In SDPA this causes NaN in backward!")

    # -------------------------------------------------------------
    # TEST 1: Raw Backward (current architecture, unsanitized)
    # -------------------------------------------------------------
    print("\n[TEST 1] Raw Backward (Current architecture, unsanitized)...")
    model.zero_grad(set_to_none=True)
    warmup_factor = torch.tensor(0.05, dtype=torch.float32, device=device)
    outputs = model(input_ids_q, mask_q, input_ids_a, mask_a, warmup_factor=warmup_factor)
    loss, avg_cos, _, _ = compute_phase6_loss(outputs, void_margin=0.5, loss_t_power=args.loss_t_power)

    loss_is_finite = torch.isfinite(loss).item()
    print(f"  Forward: loss={loss.item():.4f} (finite: {loss_is_finite}), avg_cos={avg_cos.item():.4f}")
    for k in ["z_clean", "z_noisy", "dus_final"]:
        v = outputs[k]
        print(f"    outputs['{k}']: finite={torch.isfinite(v).all().item()}, min={v.min().item():.3f}, max={v.max().item():.3f}")

    loss.backward()
    xm.mark_step()

    nan_before = []
    finite_before = []
    for name, p in trainable_named.items():
        if p.grad is None: continue
        if not torch.isfinite(p.grad).all():
            nan_before.append(name)
        else:
            finite_before.append(name)

    print(f"\n  [Results BEFORE xm.reduce_gradients]:")
    print(f"    Finite gradients:  {len(finite_before)}")
    print(f"    NaN/Inf gradients: {len(nan_before)}")
    if finite_before:
        print("    Parameters with FINITE gradients BEFORE reduction:")
        for n in finite_before:
            print(f"      + {n}")
    if nan_before:
        print("    Examples of parameters with NaN BEFORE reduction:")
        for n in nan_before[:10]:
            print(f"      - {n}")

    dummy_opt = torch.optim.AdamW(list(trainable_named.values()), lr=1e-4)
    xm.reduce_gradients(dummy_opt)
    xm.mark_step()

    nan_after = []
    for name, p in trainable_named.items():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            nan_after.append(name)

    print(f"\n  [Results AFTER xm.reduce_gradients]:")
    print(f"    NaN/Inf gradients: {len(nan_after)}")
    if not nan_before and nan_after:
        print("  [ALERT] CONCLUSION: NaNs generated INSIDE xm.reduce_gradients!")
        print("     (Reduction conflict between SPMD-FSDP sharded and unsharded weights).")
    elif nan_before:
        print("  [ALERT] CONCLUSION: NaNs generated INSIDE backward graph BEFORE all-reduce!")

    # Очистка памяти TPU после TEST 1
    dummy_opt.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    del dummy_opt, outputs, loss
    clear_tpu_memory()

    # -------------------------------------------------------------
    # TEST 2: CA Isolation (disable_ca=True)
    # -------------------------------------------------------------
    print("\n[TEST 2] Bisection: Forward + Backward with disable_ca=True (DUS backbone isolated)...")
    model.zero_grad(set_to_none=True)
    outputs_no_ca = model(input_ids_q, mask_q, input_ids_a, mask_a, warmup_factor=warmup_factor, disable_ca=True)
    loss_no_ca, _, _, _ = compute_phase6_loss(outputs_no_ca, void_margin=0.5, loss_t_power=args.loss_t_power)
    loss_no_ca.backward()
    xm.mark_step()

    dummy_opt_no_ca = torch.optim.AdamW([p for p in trainable_named.values() if p.requires_grad], lr=1e-4)
    xm.reduce_gradients(dummy_opt_no_ca)
    xm.mark_step()

    nan_no_ca = [n for n, p in trainable_named.items() if p.grad is not None and not torch.isfinite(p.grad).all()]
    print(f"  NaN gradients without CA (after reduction): {len(nan_no_ca)} of {len(trainable_named)}")
    if len(nan_before) > 0 and len(nan_no_ca) == 0:
        print("  [CONCLUSION] Without CA, gradients are 100% FINITE! Root cause is strictly in CAPromptLayer or Prompt-Pass!")
    elif len(nan_no_ca) > 0:
        print("  [INFO] Parameter gradients without CA:")
        for n in nan_no_ca[:5]:
            print(f"      - {n}")

    # Очистка памяти TPU после TEST 2
    dummy_opt_no_ca.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    del dummy_opt_no_ca, outputs_no_ca, loss_no_ca
    clear_tpu_memory()

    # -------------------------------------------------------------
    # TEST 3: DUS Isolation (Train ONLY CA, freeze DUS)
    # -------------------------------------------------------------
    if dus_params and ca_params:
        print("\n[TEST 3] Bisection: Freeze DUS, backward only for CA layers...")
        model.zero_grad(set_to_none=True)
        try:
            for p in dus_params: p.requires_grad = False
            outputs_ca_only = model(input_ids_q, mask_q, input_ids_a, mask_a, warmup_factor=warmup_factor)
            loss_ca_only, _, _, _ = compute_phase6_loss(outputs_ca_only, void_margin=0.5, loss_t_power=args.loss_t_power)
            loss_ca_only.backward()
            xm.mark_step()

            ca_named = {n: p for n, p in model.ca_layers.named_parameters() if p.requires_grad}
            nan_ca_only = [n for n, p in ca_named.items() if p.grad is not None and not torch.isfinite(p.grad).all()]
            print(f"  NaN gradients in CA (with frozen DUS): {len(nan_ca_only)} of {len(ca_named)}")
            if len(nan_ca_only) == 0:
                print("  [CONCLUSION] With frozen DUS, CA gradients are 100% FINITE!")
            else:
                print("  [INFO] CA gradients with frozen DUS (note: freezing FSDP modules on XLA may affect reduction):")
                for n in nan_ca_only[:5]:
                    print(f"      - {n}")
        finally:
            for p in dus_params: p.requires_grad = True

        # Очистка памяти TPU после TEST 3
        model.zero_grad(set_to_none=True)
        del outputs_ca_only, loss_ca_only
        clear_tpu_memory()

    model.zero_grad(set_to_none=True)
    clear_tpu_memory()
    print("\n" + "="*70)
    print("                    [FINISH] DIAGNOSIS COMPLETED                    ")
    print("="*70 + "\n")

def main():
    mesh = setup_spmd_mesh()
    device = xm.xla_device()
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        gcp_sa = user_secrets.get_secret("GCP_SA_JSON")
        if gcp_sa:
            with open("gcp_sa.json", "w") as f: f.write(gcp_sa)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath("gcp_sa.json")
    except Exception: pass

    if args.wandb_project:
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            wandb.login(key=user_secrets.get_secret("WANDB_API_KEY"))
        except Exception: pass
        wandb.init(project=args.wandb_project, config=vars(args))

    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_path, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    val_dataloader = None
    try:
        dataset = QADataset(args.dataset_path, tokenizer, args.max_length_q, args.max_length_a)
        num_workers = 0 if os.name == "nt" or os.environ.get("SMOKE_TEST") else 2
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        if os.path.exists(args.val_dataset_path):
            val_dataset = QADataset(args.val_dataset_path, tokenizer, args.max_length_q, args.max_length_a)
            val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=num_workers)
        else:
            print(f"Validation dataset not found at {args.val_dataset_path}. Validation will be skipped.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        sys.exit(1)

    model = BEBLaDIIPhase6(args).to(device)

    def shard_output(output, mesh): return None
    for key in list(model.ca_layers.keys()):
        layer_idx = int(key) - 1
        model.dus.layers[layer_idx].ca_layer = model.ca_layers[key]

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

    trainable_params = ca_out_params + ca_qkv_params + ca_other_params + dus_params

    optimizer = torch.optim.AdamW([
        {'params': ca_out_params, 'lr': 2e-4},     # снижен с 5e-4 для стабильности после warmup
        {'params': ca_qkv_params, 'lr': 8e-5},     # снижен с 2e-4
        {'params': ca_other_params, 'lr': 2.5e-5}, # снижен с 5e-5
        {'params': dus_params, 'lr': 2.5e-5}       # снижен с 5e-5
    ])

    if getattr(args, "run_diagnosis_first", False):
        diag_batch = next(iter(dataloader))
        run_nan_diagnosis(
            model,
            diag_batch,
            device,
            mesh,
            ca_params=ca_out_params + ca_qkv_params + ca_other_params,
            dus_params=dus_params
        )
        if getattr(args, "stop_after_diagnosis", False):
            print("[Diagnosis] stop_after_diagnosis=True: работа завершена для анализа логов.")
            return
        optimizer.zero_grad(set_to_none=True)
        xm.mark_step()

    model.train()
    global_step = 0
    metrics_history = []
    start_time = time.time()
    checkpoints_saved = 0
    time_limit_seconds = 9.0 * 3600  # 9 часов для Kaggle TPU
    base_lrs = [group['lr'] for group in optimizer.param_groups]
    session_step = 0

    if args.resume_from_checkpoint and args.gcs_checkpoint_dir:
        available_ckpts = get_gcs_checkpoints(args.gcs_checkpoint_dir)
        resumed = False
        for latest_ckpt, step in available_ckpts:
            local_ckpt = os.path.join(args.output_dir, f"resume_step_{step}.pth")
            try:
                downloaded = False
                client = get_gcs_client()
                if client is not None:
                    try:
                        clean_path = latest_ckpt.replace("gs://", "")
                        bucket_name, blob_name = clean_path.split("/", 1)
                        bucket = client.bucket(bucket_name)
                        blob = bucket.blob(blob_name)
                        blob.download_to_filename(local_ckpt)
                        downloaded = True
                    except Exception as e_gcs:
                        print(f"GCS client download error: {e_gcs}, falling back to gsutil...")
                if not downloaded:
                    import shutil
                    if shutil.which("gsutil"):
                        subprocess.run(["gsutil", "-q", "cp", latest_ckpt, local_ckpt], check=True)
                    else:
                        raise RuntimeError(f"GCS client failed and gsutil unavailable to download {latest_ckpt}")

                ckpt_state = torch.load(local_ckpt, map_location="cpu", weights_only=False)

                # ИНТЕЛЛЕКТУАЛЬНЫЙ МАППИНГ КЛЮЧЕЙ (FSDP / torch.compile proof)
                model_keys = list(model.state_dict().keys())

                def normalize_key(k):
                    return k.replace("._orig_module.", ".").replace("_orig_module.", "").replace(".module.", ".").replace("module.", "")

                norm_to_model = {normalize_key(k): k for k in model_keys}
                clean_state = {}

                raw_weights = ckpt_state.get("trainable_state", ckpt_state.get("model", ckpt_state)) if isinstance(ckpt_state, dict) else ckpt_state
                for ckpt_k, v in raw_weights.items():
                    if not torch.is_tensor(v):
                        continue
                    norm_k = normalize_key(ckpt_k)
                    if norm_k in norm_to_model:
                        clean_state[norm_to_model[norm_k]] = v
                    else:
                        clean_state[ckpt_k] = v

                if not is_state_dict_finite(clean_state):
                    print(f"[Resume WARN] Чекпоинт {latest_ckpt} (шаг {step}) содержит NaN/Inf! Отбрасываем поврежденный чекпоинт...")
                    if os.path.exists(local_ckpt):
                        os.remove(local_ckpt)
                    continue

                load_info = model.load_state_dict(clean_state, strict=False)

                # Проверка: все ли обучаемые параметры загружены?
                missing_trainable = [k for k in load_info.missing_keys if k in dict(model.named_parameters()) and dict(model.named_parameters())[k].requires_grad]
                if missing_trainable:
                    print(f"[Resume ERROR] Не загружены обучаемые веса (FSDP mismatch?): {missing_trainable[:5]}...")
                    sys.exit(1)

                ema_tracker = EMATracker(model, decay=args.ema_decay)
                global_step = step
                if isinstance(ckpt_state, dict) and "metrics_history" in ckpt_state:
                    metrics_history = ckpt_state["metrics_history"]
                    print(f"[Resume] Восстановлена история метрик ({len(metrics_history)} записей).")
                resumed = True
                print(f"Успешно возобновлено обучение с валидного чекпоинта {latest_ckpt} (шаг {global_step})")
                if os.path.exists(local_ckpt):
                    os.remove(local_ckpt)
                break
            except Exception as e:
                print(f"Failed to check/load {latest_ckpt}: {e}")
                if os.path.exists(local_ckpt):
                    try: os.remove(local_ckpt)
                    except OSError: pass

        if not resumed and available_ckpts:
            print("[Resume] Ни один из чекпоинтов на GCS не прошел проверку на валидность (все содержали NaN). Стартуем с шага 0.")

    nan_streak = 0
    for epoch in range(args.epochs):
        for batch in dataloader:
            if global_step >= args.max_steps: return

            session_step += 1
            # Ступенчатый прогрев LR и архитектурного сигнала (шаг 20)
            if session_step == 1 or global_step % 20 == 0:
                quantized_global = (global_step // 20) * 20
                quantized_session = (session_step // 20) * 20

                # Глобальный прогрев (зависит от номера шага)
                lr_warmup = min(1.0, max(0.05, quantized_global / args.warmup_steps))

                # Сессионный прогрев (ребилд AdamW-дисперсии после "амнезии" при resume)
                # Увеличена длительность до 300 шагов, старт с 1%
                session_warmup = min(1.0, max(0.01, quantized_session / 300.0))

                eff_lr_factor = min(lr_warmup, session_warmup)
                for group, base_lr in zip(optimizer.param_groups, base_lrs):
                    group['lr'] = base_lr * eff_lr_factor

            # Демпфируем архитектурный сигнал (CA/SA) на время прогрева оптимизатора
            global_warmup_factor = min(1.0, global_step / args.warmup_steps)
            eff_warmup_factor = min(global_warmup_factor, session_warmup)
            warmup_factor = torch.tensor(eff_warmup_factor, dtype=torch.float32, device=device)

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

            check_tensor_nan("dus_final", outputs["dus_final"], global_step)
            check_tensor_nan("loss", loss, global_step)

            if not torch.isfinite(loss):
                nan_streak += 1
                print(f"\n[WARN] Шаг {global_step}: Обнаружен не-конечный loss (streak: {nan_streak}). "
                      f"Пропуск backward и шага оптимизатора для защиты весов модели.")
                if nan_streak >= 20:
                    print(f"\n[ABORT] Превышен лимит последовательных не-конечных loss ({nan_streak} подряд). "
                          f"Прерывание обучения для сохранения квоты TPU.")
                    return
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                xm.mark_step()
                continue
            nan_streak = 0

            loss.backward()

            # SPMD All-Reduce для реплицированных и шардированных градиентов по мешу TPU
            xm.reduce_gradients(optimizer)

            # Проверка исходных градиентов на первых шагах (до санитизации)
            if global_step <= 10:
                for name, p in model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        check_tensor_nan(f"grad.{name}", p.grad, global_step)

            # In-Place поэлементная санитизация: изолирует единичные стохастические выбросы
            # и гарантирует, что остальные слои модели продолжают обучение
            for p in trainable_params:
                if p.grad is not None:
                    p.grad.nan_to_num_(nan=0.0, posinf=1.0, neginf=-1.0)

            # Клиппинг градиентов после редукции и санитизации
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            is_finite = torch.isfinite(grad_norm)
            is_safe = is_finite & (grad_norm <= 50.0)

            # Шаг оптимизатора (градиенты уже редуцированы через xm.reduce_gradients)
            optimizer.step()

            # Обновление EMA и PACE надежно изолировано через torch.where от аномальных спайков (> 50.0)
            ema_tracker.update(model, is_safe=is_safe)
            ema_tracker.pace_pullback(model, alpha=args.pace_alpha, is_safe=is_safe)

            xm.mark_step()

            cp, cb, cm = content_cos_stats(outputs)

            try:
                op12 = _ca_inner(model.ca_layers["12"]).out_proj.weight.detach().abs().mean()
                op36 = _ca_inner(model.ca_layers["36"]).out_proj.weight.detach().abs().mean()
                qp36 = _ca_inner(model.ca_layers["36"]).q_proj.weight.detach().abs().mean()
            except Exception:
                op12 = torch.tensor(0.0)
                op36 = torch.tensor(0.0)
                qp36 = torch.tensor(0.0)

            # ca_ratio_* = ||ca_out|| / ||A|| (тензоры, без .item() в графе XLA)
            ratio12 = _ca_inner(model.ca_layers["12"]).last_ca_ratio
            ratio24 = _ca_inner(model.ca_layers["24"]).last_ca_ratio
            ratio36 = _ca_inner(model.ca_layers["36"]).last_ca_ratio

            if global_step % args.log_steps == 0:
                def log_step(s, l, c, vc, cc, w, o12, o36, q36, cp_t, cm_t, tg_t, r12, r24, r36, gn_t, lr_out, lr_qkv, lr_dus):
                    log_dict = {
                        "loss": l.item(),
                        "cos_sim": c.item(),
                        "void_cos_sim": vc.item(),
                        "content_to_void_sim": cc.item(),
                        "grad_norm": gn_t.item(),
                        "warmup_factor": w.item(),
                        "lr_ca_out": float(lr_out),
                        "lr_ca_qkv": float(lr_qkv),
                        "lr_dus": float(lr_dus),
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

                    log_dict["step"] = s
                    metrics_history.append(dict(log_dict))
                    wandb.log(log_dict, step=s)

                lr_out_val = optimizer.param_groups[0]['lr']
                lr_qkv_val = optimizer.param_groups[1]['lr']
                lr_dus_val = optimizer.param_groups[3]['lr']
                xm.add_step_closure(log_step, args=(global_step, loss, avg_cos_sim, true_void_sim, content_to_void_sim, warmup_factor, op12, op36, qp36, cp.detach(), cm.detach(), outputs["t_global"].detach(), ratio12, ratio24, ratio36, grad_norm.detach(), lr_out_val, lr_qkv_val, lr_dus_val))

            if global_step <= 10 and GLOBAL_NAN_CHECKS:
                stacked_nan = torch.stack(GLOBAL_NAN_CHECKS)
                names_to_log = list(GLOBAL_NAN_NAMES)
                def print_nan_alerts(s, nan_tensor, names):
                    results = nan_tensor.cpu().tolist()
                    found_bad = False
                    for n, is_bad in zip(names, results):
                        if is_bad:
                            print(f"[NAN_ALERT] Step {s}: '{n}' contains NaN/Inf!", flush=True)
                            found_bad = True
                    if not found_bad:
                        print(f"[NAN_CHECK_OK] Step {s}: all checked tensors are finite.", flush=True)
                xm.add_step_closure(print_nan_alerts, args=(global_step, stacked_nan, names_to_log))
                GLOBAL_NAN_CHECKS.clear()
                GLOBAL_NAN_NAMES.clear()

            if val_dataloader is not None and global_step % args.val_steps == 0 and global_step > 0:
                run_validation(model, ema_tracker, val_dataloader, device, mesh, global_step, metrics_history=metrics_history)

            if global_step % args.save_steps == 0 and global_step > 0:
                ema_tracker.apply_shadow(model)
                ckpt_path = os.path.join(args.output_dir, f"phase6_ca_layers_step_{global_step}.pth")
                state_dict = model.state_dict()
                named_params = dict(model.named_parameters())
                trainable_state = {k: v for k, v in state_dict.items() if k in named_params and named_params[k].requires_grad}

                is_weights_finite = is_state_dict_finite(trainable_state)
                is_loss_finite = torch.isfinite(loss)
                save_payload = {
                    "trainable_state": trainable_state,
                    **trainable_state,
                    "step": global_step,
                    "metrics_history": metrics_history,
                }
                # Сохраняем чекпоинт:
                xm.save(save_payload, ckpt_path)
                ema_tracker.restore(model)
                if args.gcs_checkpoint_dir:
                    xm.add_step_closure(sync_to_gcs_and_delete, args=(ckpt_path, args.gcs_checkpoint_dir, True))

                if not is_weights_finite:
                    print(f"\n[ABORT] На шаге {global_step} обнаружены NaN/Inf в весах модели! "
                          f"Аварийный чекпоинт сохранен для диагностики. Обучение прервано для сохранения квоты TPU.")
                    return
                elif not is_loss_finite:
                    print(f"\n[WARN] На шаге {global_step} зафиксирован не-конечный loss текущего шага, но веса валидны. Обучение продолжается.")

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
