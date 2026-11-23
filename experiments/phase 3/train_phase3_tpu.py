# %% [markdown]
# # BEBLaDII Phase 3 Plan C (Kaggle TPU v5e-8 / v3-8, PyTorch/XLA)
# Адаптация условного ядра латентной диффузии на Грассманиане (`GrassmannianDiffusionCore`)
# под 8-чиповый Data Parallel на Kaggle TPU (16 ГБ HBM, `xmp.spawn` + `MpDeviceLoader`):
# - Мастер-веса модели, `OrthMuon`/`AdamW` и `EMAModel` строго в `float32` (БЕЗ `XLA_USE_BF16=1`).
# - Замороженная Система 1 (`GrassmannianEncoder` + `GrassmannianDecoder`) в `bfloat16` на XLA.
# - 100% статичные формы тензоров и детерминированный единый граф XLA (0 перекомпиляций):
#   - `sample_mixed_noise_schedule` и `apply_prompt_dropout` выполняются на CPU в `collate_fn`.
#   - `Two-Step Rollout` выполняется детерминированно для фиксированной доли `B_roll` батча на каждом шаге.
#   - `L_CE`, `L_adj`, `L_bary`, `L_sink`, `L_adaln` без динамических масок `tensor[mask]` и без `.item()`.
# - Полная совместимость чекпоинтов с GPU-версией (`phase3_step_*.pt` в GCS).

# %%
import argparse
import dataclasses
import glob
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# Отключаем TorchDynamo для предотвращения конфликтов с компилятором XLA
if hasattr(torch, "_dynamo"):
    torch._dynamo.disable()

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr


# %% [markdown]
# ## 1. Инициализация окружения, секретов Kaggle и путей

# %%
def cleanup_kaggle_checkpoints(checkpoint_dir: str) -> None:
    """Очищает старые локальные .pt чекпоинты в /kaggle/working для защиты от переполнения диска."""
    working_dir = Path("/kaggle/working")
    if not working_dir.exists():
        return
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        return
    deleted = 0
    for f in ckpt_path.glob("phase3_step_*.pt"):
        try:
            f.unlink()
            deleted += 1
        except OSError:
            pass
    if deleted > 0:
        _, _, free = shutil.disk_usage(working_dir)
        print(f"[Disk Cleanup] Удалено {deleted} старых чекпоинтов. Свободно: {free / (1024 ** 3):.2f} GB", flush=True)


def setup_kaggle_secrets(is_kaggle: bool) -> None:
    """Подгружает секреты WandB и GCP из Kaggle UserSecretsClient (вызывается на хосте до spawn)."""
    if not is_kaggle:
        return
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()

        try:
            wandb_key = user_secrets.get_secret("WANDB_API_KEY")
            if wandb_key:
                os.environ["WANDB_API_KEY"] = wandb_key.strip()
                print("[Kaggle Secrets] Ключ WANDB_API_KEY загружен.", flush=True)
        except Exception:
            pass

        try:
            gcp_json = user_secrets.get_secret("GCP_SA_JSON")
            if gcp_json and gcp_json.strip():
                sa_path = "/tmp/gcp_sa.json"
                with open(sa_path, "w", encoding="utf-8") as f:
                    f.write(gcp_json.strip())
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path
                print(f"[Kaggle Secrets] GCP_SA_JSON сохранен в {sa_path}.", flush=True)
                if shutil.which("gcloud"):
                    res = subprocess.run(
                        ["gcloud", "auth", "activate-service-account", f"--key-file={sa_path}"],
                        capture_output=True,
                        text=True,
                    )
                    if res.returncode == 0:
                        print("[Kaggle Secrets] gcloud CLI авторизован.", flush=True)
                    else:
                        print(f"[Kaggle Secrets Warning] Ошибка gcloud auth: {res.stderr.strip()}", flush=True)
                try:
                    from google.cloud import storage
                    _ = storage.Client()
                    print("[Kaggle Secrets] GCS Python SDK (google-cloud-storage) авторизован и готов.", flush=True)
                except Exception as e_sdk:
                    print(f"[Kaggle Secrets Warning] Ошибка инициализации google-cloud-storage: {e_sdk}", flush=True)
        except Exception as e_gcp:
            print(f"[Kaggle Secrets Warning] GCP_SA_JSON не загружен: {e_gcp}", flush=True)
    except ImportError:
        pass
    except Exception as e:
        print(f"[Kaggle Secrets Warning] Ошибка UserSecretsClient: {e}", flush=True)


def resolve_project_root() -> str:
    """Определяет корень репозитория BEBLaDII для локального запуска и Kaggle TPU."""
    if os.path.exists("/kaggle") or ("KAGGLE_KERNEL_RUN_TYPE" in os.environ):
        kaggle_root = "/kaggle/working/BEBLaDII"
        if os.path.exists(kaggle_root):
            return kaggle_root
    env_root = os.environ.get("PROJECT_ROOT", "")
    if env_root and os.path.exists(env_root):
        return env_root
    return str(Path(__file__).resolve().parents[2])


def resolve_file_path(filename: str, candidates: List[str], search_roots: List[str]) -> str:
    """Находит путь к файлу среди явных кандидатов или рекурсивно в корневых директориях."""
    for cand in candidates:
        if cand and os.path.exists(cand):
            return cand
    for root_dir in search_roots:
        if root_dir and os.path.exists(root_dir):
            matches = list(Path(root_dir).rglob(filename))
            if matches:
                return str(matches[0])
    return candidates[0] if candidates else filename


@dataclasses.dataclass
class EnvironmentConfig:
    """Единая конфигурация путей и параметров среды для Kaggle TPU / локального CPU smoke-test."""
    is_kaggle: bool
    project_root: str
    data_dir: str
    output_dir: str
    checkpoint_dir: str
    train_data_path: str
    val_data_path: str
    stopwords_path: str
    sep_concept_path: str
    void_concept_path: str
    encoder_weights_path: str
    decoder_weights_path: str
    gcs_checkpoints_dir: str
    wandb_mode: str
    num_workers: int

    @classmethod
    def detect(
        cls,
        override_output_dir: str = "",
        override_checkpoint_dir: str = "",
        setup_secrets: bool = False,
    ) -> "EnvironmentConfig":
        is_kaggle = os.path.exists("/kaggle") or ("KAGGLE_KERNEL_RUN_TYPE" in os.environ)
        if setup_secrets:
            setup_kaggle_secrets(is_kaggle)
        project_root = resolve_project_root()
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        gcs_checkpoints_dir = "gs://bebladii-weigths-us/planC/Phase3_Diffusion/checkpoints/"

        if is_kaggle:
            data_dir = "/kaggle/input"
            output_dir = "/kaggle/working/output"
            checkpoint_dir = "/kaggle/working/checkpoints/phase3"
            search_roots = [
                "/kaggle/input/datasets/bogdanbuliakov/bebladii-planc-phase2-data",
                "/kaggle/input/datasets/bogdanbuliakov/bebladii-planc-phase1-data",
                "/kaggle/input",
                project_root,
            ]
            train_data_path = resolve_file_path(
                "train_phase2.parquet",
                [
                    "/kaggle/input/datasets/bogdanbuliakov/bebladii-planc-phase2-data/data/train_phase2.parquet",
                    "/kaggle/input/datasets/bogdanbuliakov/bebladii-planc-phase2-data/train_phase2.parquet",
                    "/kaggle/input/bebladii-planc-phase2-data/data/train_phase2.parquet",
                    "/kaggle/input/bebladii-planc-phase2-data/train_phase2.parquet",
                ],
                search_roots,
            )
            val_data_path = resolve_file_path(
                "val_phase2.parquet",
                [
                    "/kaggle/input/datasets/bogdanbuliakov/bebladii-planc-phase2-data/data/val_phase2.parquet",
                    "/kaggle/input/datasets/bogdanbuliakov/bebladii-planc-phase2-data/val_phase2.parquet",
                    "/kaggle/input/bebladii-planc-phase2-data/data/val_phase2.parquet",
                    "/kaggle/input/bebladii-planc-phase2-data/val_phase2.parquet",
                ],
                search_roots,
            )
            stopwords_path = resolve_file_path(
                "tekken_stopwords_mask.pt",
                [
                    "/kaggle/input/datasets/bogdanbuliakov/bebladii-planc-phase1-data/data/tekken_stopwords_mask.pt",
                    "/kaggle/input/bebladii-planc-phase1-data/data/tekken_stopwords_mask.pt",
                    os.path.join(project_root, "data", "plan C", "tekken_stopwords_mask.pt"),
                ],
                search_roots,
            )
            encoder_weights_path = resolve_file_path(
                "encoder/weights.pt",
                [
                    os.path.join(project_root, "storage", "components", "encoder", "weights.pt"),
                    "/kaggle/input/datasets/bogdanbuliakov/bebladii-planc-phase1-data/encoder/weights.pt",
                    "/kaggle/input/bebladii-planc-phase1-data/encoder/weights.pt",
                ],
                search_roots,
            )
            decoder_search_roots = [
                "/kaggle/input/datasets/bogdanbuliakov/bebladii-planc-phase2-data",
                "/kaggle/input/bebladii-planc-phase2-data",
                "/kaggle/input",
            ]
            decoder_weights_path = resolve_file_path(
                "phase2_decoder_step_25000.pt",
                [
                    "/kaggle/input/datasets/bogdanbuliakov/bebladii-planc-phase2-data/decoder/phase2_decoder_step_25000.pt",
                    "/kaggle/input/bebladii-planc-phase2-data/decoder/phase2_decoder_step_25000.pt",
                    "/kaggle/input/datasets/bogdanbuliakov/bebladii-planc-phase2-data/phase2_decoder_step_25000.pt",
                    "/kaggle/input/bebladii-planc-phase2-data/phase2_decoder_step_25000.pt",
                    os.path.join(project_root, "storage", "components", "decoder", "weights.pt"),
                ],
                decoder_search_roots,
            )
            wandb_mode = "online" if "WANDB_API_KEY" in os.environ else "disabled"
            num_workers = 2
        else:
            data_dir = os.path.join(project_root, "data")
            output_dir = os.path.join(project_root, "experiments", "phase 3", "output")
            checkpoint_dir = os.path.join(project_root, "experiments", "phase 3", "checkpoints")
            train_data_path = resolve_file_path(
                "train_phase2.parquet",
                [os.path.join(project_root, "BEBLaDII-plan\u0421-Phase2-Data", "data", "train_phase2.parquet")],
                [project_root],
            )
            val_data_path = resolve_file_path(
                "val_phase2.parquet",
                [os.path.join(project_root, "BEBLaDII-plan\u0421-Phase2-Data", "data", "val_phase2.parquet")],
                [project_root],
            )
            stopwords_path = resolve_file_path(
                "tekken_stopwords_mask.pt",
                [os.path.join(project_root, "data", "plan C", "tekken_stopwords_mask.pt")],
                [project_root],
            )
            encoder_weights_path = os.path.join(project_root, "storage", "components", "encoder", "weights.pt")
            decoder_weights_path = os.path.join(project_root, "storage", "components", "decoder", "weights.pt")
            wandb_mode = "disabled"
            num_workers = 0

        sep_concept_path = os.path.join(project_root, "storage", "components", "sep_concept.pt")
        void_concept_path = os.path.join(project_root, "storage", "components", "void_concept.pt")

        if override_output_dir:
            output_dir = override_output_dir
        if override_checkpoint_dir:
            checkpoint_dir = override_checkpoint_dir

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        return cls(
            is_kaggle=is_kaggle,
            project_root=project_root,
            data_dir=data_dir,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            train_data_path=train_data_path,
            val_data_path=val_data_path,
            stopwords_path=stopwords_path,
            sep_concept_path=sep_concept_path,
            void_concept_path=void_concept_path,
            encoder_weights_path=encoder_weights_path,
            decoder_weights_path=decoder_weights_path,
            gcs_checkpoints_dir=gcs_checkpoints_dir,
            wandb_mode=wandb_mode,
            num_workers=num_workers,
        )


# %% [markdown]
# ## 2. Импорт Системы 1 и базовые примитивы Грассманиана

# %%
PROJECT_ROOT_INIT = resolve_project_root()
if PROJECT_ROOT_INIT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_INIT)

from storage.components.decoder import get_decoder
from storage.components.encoder import get_encoder
from storage.components.plan_c_layers import (
    RMSNorm,
    RotaryEmbedding128,
    SphereLinear,
    WordAwareElasticChunker,
    apply_rotary_pos_emb,
    canonicalize_cell_basis,
    clamp_to_unit_ball,
)
from storage.components.tokenizer import get_tokenizer

PAD_TOKEN_ID = 10
SIGMA_DATA = 0.05
N_PROMPT = 64
N_ANS = 127
N_CANVAS = 128
T_PROMPT = N_PROMPT * 4  # 256 токенов
T_ANS = N_ANS * 4        # 508 токенов
TAU_ADJ = 0.85
TAU_BARY = 0.50
VOID_ENERGY_THR = 0.35
VOID_SIM_THR = 0.80


def _autocast_ctx(device: torch.device, enabled: bool):
    """Безопасный контекст BF16 autocast для XLA (на CPU отключен)."""
    if enabled and device.type == "xla":
        return torch.amp.autocast(device_type="xla", dtype=torch.bfloat16, enabled=True)
    return torch.amp.autocast(device_type=device.type, enabled=False)


def sim_gr(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """O(8)-инвариантное косинусное сходство операторов плотности в float32."""
    A_f = A.float()
    B_f = B.float()
    cross = torch.matmul(A_f, B_f.transpose(-2, -1))
    Ga = torch.matmul(A_f, A_f.transpose(-2, -1))
    Gb = torch.matmul(B_f, B_f.transpose(-2, -1))
    num = (cross ** 2).sum(dim=(-2, -1))
    norm_a = torch.sqrt((Ga ** 2).sum(dim=(-2, -1)) + 1e-16)
    norm_b = torch.sqrt((Gb ** 2).sum(dim=(-2, -1)) + 1e-16)
    return num / (norm_a * norm_b + eps)


def compute_rk_eff(V: torch.Tensor) -> torch.Tensor:
    """Эффективный спектральный ранг клетки V in R^{..., 8, 128} в float32."""
    V_f = V.float()
    G = torch.matmul(V_f, V_f.transpose(-2, -1))
    tr_g = torch.diagonal(G, dim1=-2, dim2=-1).sum(dim=-1)
    g_frob_sq = (G ** 2).sum(dim=(-2, -1))
    return (tr_g ** 2) / (g_frob_sq + 1e-8)


def detect_void_mask(V_cells: torch.Tensor, void_ref: torch.Tensor) -> torch.Tensor:
    """Латентный детектор VOID-ячеек: (||V_i||_F^2 < 0.35) or (Sim_Gr(V_i, V_void) > 0.80)."""
    V_f = V_cells.float()
    energy = (V_f ** 2).sum(dim=(-2, -1))
    ref = void_ref.to(device=V_f.device, dtype=torch.float32)
    while ref.ndim < V_f.ndim:
        ref = ref.unsqueeze(0)
    ref = ref.expand_as(V_f)
    s_void = sim_gr(V_f, ref)
    return (energy < VOID_ENERGY_THR) | (s_void > VOID_SIM_THR)


@torch.no_grad()
def compute_radar_sigma(
    decoder: Optional[nn.Module],
    V_t: torch.Tensor,
    use_bf16: bool = True,
) -> torch.Tensor:
    """Поклеточная оценка NoiseRadarHead (s7) в float32 [0, 1], pos 0 (SEP) = 0.0."""
    B, N, _, _ = V_t.shape
    if decoder is None or N <= 1 or not hasattr(decoder, "estimate_noise"):
        return torch.zeros((B, N), device=V_t.device, dtype=torch.float32)
    if getattr(decoder, "noise_radar", None) is None:
        return torch.zeros((B, N), device=V_t.device, dtype=torch.float32)

    dec_dtype = next(decoder.parameters()).dtype
    v_ans = V_t[:, 1:].to(dtype=dec_dtype)
    with _autocast_ctx(V_t.device, enabled=use_bf16):
        sigma_cell, _, _ = decoder.estimate_noise(v_ans)
    zeros_sep = torch.zeros((B, 1), device=V_t.device, dtype=torch.float32)
    return torch.cat([zeros_sep, sigma_cell.float().clamp(0.0, 1.0)], dim=1)


class PerHeadRMSNorm(nn.Module):
    """По-головочный RMSNorm (8 x 128) с расчетом в float32 для QK-norm."""

    def __init__(self, num_heads: int = 8, head_dim: int = 128, eps: float = 1e-5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_heads, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = torch.nan_to_num(x.float(), nan=0.0, posinf=65000.0, neginf=-65000.0)
        var = torch.mean(x_f ** 2, dim=-1, keepdim=True)
        normed = x_f * torch.rsqrt(var + self.eps)
        w = self.weight.float().view(1, self.num_heads, 1, self.head_dim)
        return (normed * w).to(x.dtype)


# %% [markdown]
# ## 3. Архитектура Фазы 3: `ConditionEmbedder`, `PromptAdapter`, `DiffusionCore`, `OutputHead`

# %%
class SinusoidalTimeEmbedding(nn.Module):
    """Синусоидальное кодирование скалярного уровня шума t in [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / max(half - 1, 1))
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = t.float().unsqueeze(-1) * self.freqs.to(device=t.device, dtype=torch.float32)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ConditionEmbedder(nn.Module):
    """
    Вектор управления c_i in R^{B x N x 256}:
    глобальный t_global (64) + 8 чисто диагностических датчиков s0..s7 (ADR 139).
    """

    def __init__(self, void_ref: torch.Tensor, cond_dim: int = 256):
        super().__init__()
        self.register_buffer("void_ref", void_ref.float().clone())
        self.t_global_emb = SinusoidalTimeEmbedding(64)
        self.mlp = nn.Sequential(
            nn.Linear(72, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.LayerNorm(cond_dim),
        )

    @staticmethod
    def extract_t_global(t_mat: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        tf = t_mat.float()
        if tf.ndim == 2 and tf.shape[1] > 1:
            tf_ans = tf[:, 1:]
            active_ans = (tf_ans > 1e-3).float()
            active_cnt = active_ans.sum(dim=-1, keepdim=True).clamp(min=1.0)
            return ((tf_ans * active_ans).sum(dim=-1, keepdim=True) / active_cnt).expand(batch_size, seq_len)
        return tf.view(batch_size, 1).expand(batch_size, seq_len)

    def forward(
        self,
        V_t: torch.Tensor,
        t_mat: torch.Tensor,
        m_prompt: torch.Tensor,
        radar_sigma: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _, _ = V_t.shape
        device = V_t.device
        with torch.amp.autocast(device_type=device.type, enabled=False):
            Vf = V_t.float()
            mp = m_prompt.float()
            t_bar = self.extract_t_global(t_mat, B, N)

            G = torch.matmul(Vf, Vf.transpose(-2, -1))
            energy = torch.diagonal(G, dim1=-2, dim2=-1).sum(dim=-1)
            s0 = torch.clamp(energy / 2.52, max=2.0)
            s1 = (G ** 2).sum(dim=(-2, -1)) / ((energy + 1e-6) ** 2)
            s2 = 1.0 / (8.0 * torch.clamp(s1, min=1e-4))
            void_exp = self.void_ref.view(1, 1, 8, 128).expand(B, N, 8, 128)
            s3 = sim_gr(Vf, void_exp)

            pos_is_sep = (torch.arange(N, device=device) == 0).view(1, N)
            if radar_sigma is not None:
                s7 = torch.where(pos_is_sep, torch.zeros_like(s0), radar_sigma.float().clamp(0.0, 1.0))
                s4_raw = torch.clamp(1.0 - s7 / 0.05, min=0.0, max=1.0)
            else:
                s7 = torch.zeros_like(s0)
                s4_raw = torch.zeros_like(s0)
            s4 = torch.where(pos_is_sep, torch.ones_like(s0), s4_raw)
            s5 = mp.view(B, 1).expand(B, N)
            s6 = s4[:, 1:].mean(dim=-1, keepdim=True).expand(B, N) if N > 1 else s4

            sensors = torch.stack([s0, s1, s2, s3, s4, s5, s6, s7], dim=-1)
            e_glob = self.t_global_emb(t_bar)
            feat = torch.cat([e_glob, sensors], dim=-1)
            c_f32 = self.mlp(feat)
        return c_f32.to(V_t.dtype)


def _clamp_swiglu(gate_act: torch.Tensor, up_val: torch.Tensor) -> torch.Tensor:
    prod = F.silu(gate_act.float()) * up_val.float()
    return torch.clamp(prod, -65000.0, 65000.0).to(up_val.dtype)


def _clamp_safe(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x.float(), -65000.0, 65000.0).to(x.dtype)


class PromptAdapterLayer(nn.Module):
    """Слой PromptAdapter: Full SA (Shared KV + Split K/V Channel Gating) + SwiGLU FFN."""

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 2816,
        num_heads: int = 8,
        init_layer_scale: float = 0.02,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.attn_norm = RMSNorm(hidden_size)
        self.q_proj = SphereLinear(hidden_size, hidden_size, bias=False)
        self.kv_proj = SphereLinear(hidden_size, hidden_size, bias=False)
        self.o_proj = SphereLinear(hidden_size, hidden_size, bias=False)
        self.q_norm = PerHeadRMSNorm(num_heads, self.head_dim)
        self.k_norm = PerHeadRMSNorm(num_heads, self.head_dim)
        self.v_gate = nn.Parameter(torch.ones(1, num_heads, 1, self.head_dim))

        self.mlp_norm = RMSNorm(hidden_size)
        self.gate_proj = SphereLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = SphereLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = SphereLinear(intermediate_size, hidden_size, bias=False)

        self.ls_attn = nn.Parameter(torch.full((hidden_size,), init_layer_scale))
        self.ls_mlp = nn.Parameter(torch.full((hidden_size,), init_layer_scale))

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, C = x.shape
        h = self.attn_norm(x)

        q = self.q_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        u = self.kv_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(u)
        v = u * self.v_gate.to(u.dtype)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, L, C)
        x = x + self.ls_attn.to(x.dtype) * self.o_proj(attn_out)

        h_mlp = self.mlp_norm(x)
        gated = _clamp_swiglu(self.gate_proj(h_mlp), self.up_proj(h_mlp))
        mlp_out = _clamp_safe(self.down_proj(gated))
        x = x + self.ls_mlp.to(x.dtype) * mlp_out
        return x


class PromptAdapter(nn.Module):
    """2-слойный контекстуализатор чистого промпта в нормированном потоке RMS=1."""

    def __init__(
        self,
        num_layers: int = 2,
        hidden_size: int = 1024,
        intermediate_size: int = 2816,
        num_heads: int = 8,
        init_layer_scale: float = 0.02,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                PromptAdapterLayer(hidden_size, intermediate_size, num_heads, init_layer_scale)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        V_prompt: torch.Tensor,
        rope: RotaryEmbedding128,
        prompt_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, _, _ = V_prompt.shape
        x = (V_prompt.float() / SIGMA_DATA).to(V_prompt.dtype).reshape(B, L, -1)
        cos, sin = rope(x, L)

        attn_mask = None
        if prompt_valid_mask is not None:
            # Без ветвлений (.any()): если вся строка False, открываем pos 0 через torch.where
            has_any = prompt_valid_mask.any(dim=-1, keepdim=True)  # (B, 1)
            pos0_mask = (torch.arange(L, device=x.device) == 0).unsqueeze(0)  # (1, L)
            safe_mask = prompt_valid_mask | ((~has_any) & pos0_mask)
            attn_mask = torch.where(
                safe_mask[:, None, None, :],
                torch.zeros(B, 1, 1, L, device=x.device, dtype=x.dtype),
                torch.full((B, 1, 1, L), -10000.0, device=x.device, dtype=x.dtype),
            )

        for layer in self.layers:
            x = layer(x, cos, sin, attn_mask=attn_mask)
        return x


def _init_soft_gate_adaln(proj: nn.Linear, hidden_size: int, gate_bias: float = 0.02) -> None:
    nn.init.normal_(proj.weight, mean=0.0, std=1e-4)
    nn.init.zeros_(proj.bias)
    with torch.no_grad():
        proj.bias[2 * hidden_size : 3 * hidden_size] = gate_bias
        proj.bias[5 * hidden_size : 6 * hidden_size] = gate_bias


def _compute_adaln_stats_static(
    M: torch.Tensor,
    m_active: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    100% статичный расчёт статистик AdaLN без динамической индексации M[m_active]:
    Возвращает (sum_M: [1, 4096], sum_M2: [1, 4096], count: [1]) в float32.
    """
    mask_f = m_active.float().unsqueeze(-1)  # (B, N, 1)
    m_f = M.float() * mask_f
    sum_m = m_f.sum(dim=(0, 1)).unsqueeze(0)        # (1, 4096)
    sum_m2 = (m_f ** 2).sum(dim=(0, 1)).unsqueeze(0)  # (1, 4096)
    count = mask_f.sum().view(1)                     # (1,)
    return sum_m, sum_m2, count


class DiffusionSABlock(nn.Module):
    """Блок Self-Attention ядра (Full SA или Local SA r=8) с Shared KV, SwiGLU и Soft-Gate AdaLN."""

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 2816,
        num_heads: int = 8,
        local_radius: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.local_radius = local_radius

        self.attn_norm = RMSNorm(hidden_size)
        self.q_proj = SphereLinear(hidden_size, hidden_size, bias=False)
        self.kv_proj = SphereLinear(hidden_size, hidden_size, bias=False)
        self.o_proj = SphereLinear(hidden_size, hidden_size, bias=False)
        self.q_norm = PerHeadRMSNorm(num_heads, self.head_dim)
        self.k_norm = PerHeadRMSNorm(num_heads, self.head_dim)
        self.v_gate = nn.Parameter(torch.ones(1, num_heads, 1, self.head_dim))

        self.mlp_norm = RMSNorm(hidden_size)
        self.gate_proj = SphereLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = SphereLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = SphereLinear(intermediate_size, hidden_size, bias=False)

        self.adaln_proj = nn.Linear(256, 6 * hidden_size)
        _init_soft_gate_adaln(self.adaln_proj, hidden_size, gate_bias=0.02)

        if local_radius is not None:
            idx = torch.arange(N_CANVAS)
            allowed = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= local_radius
            allowed[0, :] = True
            allowed[:, 0] = True
            local_mask = torch.zeros(1, 1, N_CANVAS, N_CANVAS, dtype=torch.float32)
            local_mask = local_mask.masked_fill(~allowed.unsqueeze(0).unsqueeze(0), -10000.0)
            self.register_buffer("local_mask", local_mask, persistent=False)
        else:
            self.register_buffer("local_mask", None, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        m_active: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        B, N, C = x.shape
        mods = self.adaln_proj(c.to(self.adaln_proj.weight.dtype)).to(x.dtype)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods.chunk(6, dim=-1)

        M = torch.cat([shift_msa, scale_msa, shift_mlp, scale_mlp], dim=-1)
        stats = _compute_adaln_stats_static(M, m_active)

        h = self.attn_norm(x) * (1.0 + scale_msa) + shift_msa
        q = self.q_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        u = self.kv_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(u)
        v = u * self.v_gate.to(u.dtype)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_mask = None
        if self.local_mask is not None:
            attn_mask = self.local_mask[:, :, :N, :N].to(dtype=q.dtype, device=q.device)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        x = x + gate_msa * self.o_proj(attn_out)

        h_mlp = self.mlp_norm(x) * (1.0 + scale_mlp) + shift_mlp
        gated = _clamp_swiglu(self.gate_proj(h_mlp), self.up_proj(h_mlp))
        mlp_out = _clamp_safe(self.down_proj(gated))
        x = x + gate_mlp * mlp_out
        return x, stats


class CAPromptBlock(nn.Module):
    """Слои 2, 4, 6: Cross-Attention на промпт с Token-Centered Contrast, Zero-Vector Sink и Cross-Gated SwiGLU."""

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 2816,
        num_heads: int = 8,
        rope_offset: int = N_PROMPT,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rope_offset = rope_offset

        self.ca_norm = RMSNorm(hidden_size)
        self.prompt_norm = RMSNorm(hidden_size)

        self.q_proj = SphereLinear(hidden_size, hidden_size, bias=False)
        self.kv_proj = SphereLinear(hidden_size, hidden_size, bias=False)
        self.o_proj = SphereLinear(hidden_size, hidden_size, bias=False)
        self.q_norm = PerHeadRMSNorm(num_heads, self.head_dim)
        self.k_norm = PerHeadRMSNorm(num_heads, self.head_dim)
        self.v_gate = nn.Parameter(torch.ones(1, num_heads, 1, self.head_dim))

        self.k_sink = nn.Parameter(torch.randn(1, num_heads, 1, self.head_dim) * 0.02)

        self.mlp_norm = RMSNorm(hidden_size)
        self.gate_proj = SphereLinear(hidden_size, intermediate_size, bias=False)
        self.w_canvas_gate = SphereLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = SphereLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = SphereLinear(intermediate_size, hidden_size, bias=False)

        self.adaln_proj = nn.Linear(256, 6 * hidden_size)
        _init_soft_gate_adaln(self.adaln_proj, hidden_size, gate_bias=0.02)

    def forward(
        self,
        x_ans: torch.Tensor,
        x_prompt: torch.Tensor,
        c: torch.Tensor,
        rope: RotaryEmbedding128,
        prompt_attn_mask: torch.Tensor,
        m_active: torch.Tensor,
        return_sink_weights: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        B, N_a, C = x_ans.shape
        N_p = x_prompt.shape[1]

        mods = self.adaln_proj(c.to(self.adaln_proj.weight.dtype)).to(x_ans.dtype)
        shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = mods.chunk(6, dim=-1)

        M = torch.cat([shift_ca, scale_ca, shift_mlp, scale_mlp], dim=-1)
        stats = _compute_adaln_stats_static(M, m_active)

        h_ans_in = self.ca_norm(x_ans) * (1.0 + scale_ca) + shift_ca
        h_prompt_in = self.prompt_norm(x_prompt)

        q = self.q_proj(h_ans_in).view(B, N_a, self.num_heads, self.head_dim).transpose(1, 2)
        u_p = self.kv_proj(h_prompt_in).view(B, N_p, self.num_heads, self.head_dim).transpose(1, 2)

        mask_p_f = prompt_attn_mask.float().view(B, 1, N_p, 1)
        denom = mask_p_f.sum(dim=2, keepdim=True).clamp(min=1.0)
        u_p_mean = (u_p.float() * mask_p_f).sum(dim=2, keepdim=True) / denom
        u_p_centered = (u_p.float() - 0.5 * u_p_mean).to(u_p.dtype)

        q = self.q_norm(q)
        k_p = self.k_norm(u_p_centered)
        v_p = u_p * self.v_gate.to(u_p.dtype)

        cos_all, sin_all = rope(x_ans, self.rope_offset + N_a)
        cos_p, sin_p = cos_all[:N_p], sin_all[:N_p]
        cos_a, sin_a = cos_all[self.rope_offset : self.rope_offset + N_a], sin_all[self.rope_offset : self.rope_offset + N_a]

        q, _ = apply_rotary_pos_emb(q, q, cos_a, sin_a)
        _, k_p = apply_rotary_pos_emb(k_p, k_p, cos_p, sin_p)

        k_sink_exp = self.k_norm(self.k_sink.to(k_p.dtype)).expand(B, -1, -1, -1)
        k_full = torch.cat([k_p, k_sink_exp], dim=2)

        # Статичная маска: открыты активные клетки промпта и всегда открыт последний столбец k_sink
        zeros_p = torch.zeros(B, 1, 1, N_p, device=x_ans.device, dtype=torch.float32)
        neg_p = torch.full((B, 1, 1, N_p), -10000.0, device=x_ans.device, dtype=torch.float32)
        mask_p = torch.where(prompt_attn_mask[:, None, None, :], zeros_p, neg_p)
        mask_sink = torch.zeros(B, 1, 1, 1, device=x_ans.device, dtype=torch.float32)
        attn_mask = torch.cat([mask_p, mask_sink], dim=-1)

        scores = torch.matmul(q.float(), k_full.float().transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + attn_mask
        probs = F.softmax(scores, dim=-1)
        sink_weights = probs[:, :, :, -1].mean(dim=1) if return_sink_weights else None

        attn_out = torch.matmul(probs[:, :, :, :N_p].to(v_p.dtype), v_p)
        attn_out = attn_out.transpose(1, 2).reshape(B, N_a, C)
        h_ca = self.o_proj(attn_out)
        x_ans = x_ans + gate_ca * h_ca

        x_tilde_ans = self.mlp_norm(x_ans) * (1.0 + scale_mlp) + shift_mlp
        gate_pre = self.gate_proj(h_ca).float() + self.w_canvas_gate(x_tilde_ans).float()
        prod = F.silu(gate_pre) * self.up_proj(h_ca).float()
        gated = torch.clamp(prod, -65000.0, 65000.0).to(x_ans.dtype)
        ffn_out = _clamp_safe(self.down_proj(gated))
        x_ans = x_ans + gate_mlp * ffn_out

        return x_ans, stats, sink_weights


class GrassmannianOutputHead(nn.Module):
    """Выходная голова с тождественным якорем на позиции 0 (SEP) и радиальным сжатием в B^128."""

    def __init__(self, hidden_size: int = 1024, cond_dim: int = 256):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.adaln_out = nn.Linear(cond_dim, 2 * hidden_size)
        nn.init.normal_(self.adaln_out.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.adaln_out.bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        x_ans: torch.Tensor,
        V_t: torch.Tensor,
        t_mat: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _, _ = V_t.shape
        mods = self.adaln_out(c.to(self.adaln_out.weight.dtype)).to(x_ans.dtype)
        shift_out, scale_out = mods.chunk(2, dim=-1)
        h = self.norm(x_ans) * (1.0 + scale_out) + shift_out
        delta_x = self.out_proj(h.to(self.out_proj.weight.dtype)).view(B, N, 8, 128)

        t_bar = ConditionEmbedder.extract_t_global(t_mat, B, N)
        g_t_raw = torch.clamp(t_bar / 0.02, min=0.0, max=1.0)
        pos_is_sep = (torch.arange(N, device=V_t.device) == 0).view(1, N)
        g_t = torch.where(pos_is_sep, torch.zeros_like(g_t_raw), g_t_raw).unsqueeze(-1).unsqueeze(-1)
        V_0_raw = V_t.float() + SIGMA_DATA * g_t * delta_x.float()
        return clamp_to_unit_ball(V_0_raw)


class GrassmannianDiffusionCore(nn.Module):
    """Полное условное ядро диффузии Фазы 3 Plan C (~167.64M обучаемых параметров)."""

    def __init__(
        self,
        void_ref: torch.Tensor,
        hidden_size: int = 1024,
        intermediate_size: int = 2816,
        num_heads: int = 8,
        cond_dim: int = 256,
        local_radius: int = 8,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.rope = RotaryEmbedding128(dim=self.head_dim, max_seq_len=512)
        self.cond_embedder = ConditionEmbedder(void_ref=void_ref, cond_dim=cond_dim)
        self.prompt_adapter = PromptAdapter(
            num_layers=2,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            init_layer_scale=0.02,
        )

        self.layers = nn.ModuleList(
            [
                DiffusionSABlock(hidden_size, intermediate_size, num_heads, local_radius=None),          # 0: Full
                DiffusionSABlock(hidden_size, intermediate_size, num_heads, local_radius=local_radius),  # 1: Local
                CAPromptBlock(hidden_size, intermediate_size, num_heads, rope_offset=N_PROMPT),          # 2: CA_Prompt
                DiffusionSABlock(hidden_size, intermediate_size, num_heads, local_radius=None),          # 3: Full
                CAPromptBlock(hidden_size, intermediate_size, num_heads, rope_offset=N_PROMPT),          # 4: CA_Prompt
                DiffusionSABlock(hidden_size, intermediate_size, num_heads, local_radius=None),          # 5: Full
                CAPromptBlock(hidden_size, intermediate_size, num_heads, rope_offset=N_PROMPT),          # 6: CA_Prompt
                DiffusionSABlock(hidden_size, intermediate_size, num_heads, local_radius=None),          # 7: Full
                DiffusionSABlock(hidden_size, intermediate_size, num_heads, local_radius=local_radius),  # 8: Local
                DiffusionSABlock(hidden_size, intermediate_size, num_heads, local_radius=None),          # 9: Full
            ]
        )
        self.output_head = GrassmannianOutputHead(hidden_size=hidden_size, cond_dim=cond_dim)

    def encode_prompt(
        self,
        V_prompt: torch.Tensor,
        prompt_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.prompt_adapter(V_prompt, self.rope, prompt_valid_mask=prompt_valid_mask)

    def forward_with_cached_prompt(
        self,
        V_t: torch.Tensor,
        t_mat: torch.Tensor,
        x_prompt: torch.Tensor,
        prompt_attn_mask: torch.Tensor,
        m_prompt: torch.Tensor,
        radar_sigma: Optional[torch.Tensor] = None,
        return_sink_weights: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        B, N_a, _, _ = V_t.shape
        m_active = t_mat > 1e-3

        c = self.cond_embedder(V_t, t_mat, m_prompt, radar_sigma=radar_sigma)
        x_ans = (V_t.float() / SIGMA_DATA).to(V_t.dtype).reshape(B, N_a, self.hidden_size)
        cos_a, sin_a = self.rope(x_ans, N_a)

        sum_m_list, sum_m2_list, count_list = [], [], []
        sink_list = []

        for layer in self.layers:
            if isinstance(layer, CAPromptBlock):
                x_ans, stats, sw = layer(
                    x_ans,
                    x_prompt,
                    c,
                    self.rope,
                    prompt_attn_mask=prompt_attn_mask,
                    m_active=m_active,
                    return_sink_weights=return_sink_weights,
                )
                if sw is not None:
                    sink_list.append(sw)
            else:
                x_ans, stats = layer(x_ans, c, cos_a, sin_a, m_active=m_active)
            sum_m_list.append(stats[0])
            sum_m2_list.append(stats[1])
            count_list.append(stats[2])

        V_0_hat = self.output_head(x_ans, V_t, t_mat, c)
        sink_weights = torch.stack(sink_list, dim=1) if sink_list else None

        num_layers = len(self.layers)
        sum_m_pack = torch.stack(sum_m_list, dim=1)    # (1, num_layers, 4096)
        sum_m2_pack = torch.stack(sum_m2_list, dim=1)  # (1, num_layers, 4096)
        count_pack = torch.stack(count_list, dim=0).view(1, num_layers)
        return V_0_hat, (sum_m_pack, sum_m2_pack, count_pack), sink_weights

    def forward(
        self,
        V_t: torch.Tensor,
        t_mat: torch.Tensor,
        V_prompt: torch.Tensor,
        prompt_valid_mask: torch.Tensor,
        prompt_attn_mask: torch.Tensor,
        m_prompt: torch.Tensor,
        radar_sigma: Optional[torch.Tensor] = None,
        return_sink_weights: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        x_prompt = self.encode_prompt(V_prompt, prompt_valid_mask=prompt_valid_mask)
        return self.forward_with_cached_prompt(
            V_t=V_t,
            t_mat=t_mat,
            x_prompt=x_prompt,
            prompt_attn_mask=prompt_attn_mask,
            m_prompt=m_prompt,
            radar_sigma=radar_sigma,
            return_sink_weights=return_sink_weights,
        )


# %% [markdown]
# ## 4. Датасет `(Prompt, Answer)` и CPU-генерация расписания шума / Prompt Dropout

# %%
def encode_token_ids(tokenizer: Any, text: str) -> List[int]:
    if tokenizer is None:
        return [100 + (ord(ch) % 1000) for ch in text[:256]]
    out = tokenizer.encode(text, add_special_tokens=False)
    return out.ids if hasattr(out, "ids") else list(out)


def build_punctuation_id_set(tokenizer: Optional[Any]) -> set:
    punct_ids = {1046, 2247, 1063, 1047, 13, 28723, 28725, 120, 26, 29}
    if tokenizer is not None:
        try:
            vocab = tokenizer.get_vocab()
            for text, tid in vocab.items():
                if any(p in text for p in [".", "!", "?", "\n"]):
                    punct_ids.add(tid)
        except Exception:
            pass
    return punct_ids


def sample_mixed_noise_schedule_cpu(batch_size: int, i_last: torch.Tensor) -> torch.Tensor:
    """
    Языково-ориентированный Mixed Noise Curriculum Фазы 3 (ADR 140), выполняемый на CPU:
    - 25% High-Noise / Pure-Start
    - 20% Shifted Flat Diffusion (Sigmoid(N(1.0, 1.0)))
    - 20% Causal Prefix / Block-AR
    - 20% Cloze & Crystal Anchors
    - 15% Rolling Wave Gradient
    """
    t_mat = torch.zeros(batch_size, N_CANVAS, dtype=torch.float32)
    regimes = np.random.random(batch_size)

    for b in range(batch_size):
        r = float(regimes[b])
        if r < 0.25:
            if np.random.random() < 0.333:
                t_mat[b, 1:] = 1.0
            else:
                t_mat[b, 1:] = float(np.random.uniform(0.80, 1.0))
        elif r < 0.45:
            z = float(np.random.normal(1.0, 1.0))
            t_flat = min(1.0, max(0.01, 1.0 / (1.0 + math.exp(-z))))
            t_mat[b, 1:] = t_flat
        elif r < 0.65:
            max_cut = max(0, min(int(i_last[b]) - 1, N_ANS - 1))
            if max_cut > 0:
                if np.random.random() < 0.5 and max_cut >= 8:
                    step_blk = 16 if (max_cut >= 16 and np.random.random() < 0.5) else 8
                    choices = list(range(0, max_cut + 1, step_blk))
                    cut = choices[int(np.random.randint(0, len(choices)))]
                else:
                    cut = int(np.random.randint(0, max_cut + 1))
            else:
                cut = 0

            t_ans = torch.ones(N_ANS, dtype=torch.float32)
            if np.random.random() < 0.5:
                blk_len = int(np.random.randint(8, 25))
                blk_end = min(N_ANS, cut + blk_len)
                if blk_end > cut:
                    if np.random.random() < 0.5:
                        t_ans[cut:blk_end] = 1.0
                    else:
                        t_ans[cut:blk_end] = torch.from_numpy(
                            np.random.uniform(0.50, 1.0, size=(blk_end - cut,)).astype(np.float32)
                        )
            else:
                if np.random.random() < 0.5:
                    t_ans[cut:] = 1.0
                else:
                    t_ans[cut:] = float(np.random.uniform(0.50, 1.0))
            if cut > 0:
                t_ans[:cut] = 0.0
            t_mat[b, 1:] = t_ans
        elif r < 0.85:
            if np.random.random() < 0.5:
                t_ans = torch.ones(N_ANS, dtype=torch.float32)
            else:
                t_ans = torch.from_numpy(np.random.uniform(0.50, 1.0, size=(N_ANS,)).astype(np.float32))
            n_c = max(2, min(int(i_last[b]), N_ANS))
            keep_frac = 0.20 + 0.50 * float(np.random.random())
            n_anchors = max(1, min(n_c - 1, int(round(n_c * keep_frac))))
            anchor_idx = torch.from_numpy(np.random.permutation(n_c)[:n_anchors])
            if np.random.random() < 0.65:
                t_ans[anchor_idx] = 0.0
            else:
                t_ans[anchor_idx] = torch.from_numpy(
                    np.random.uniform(0.01, 0.15, size=(n_anchors,)).astype(np.float32)
                )
            t_mat[b, 1:] = t_ans
        else:
            t_start = 0.20 * float(np.random.random())
            t_end = 0.80 + 0.20 * float(np.random.random())
            grad_t = torch.linspace(t_start, t_end, N_ANS, dtype=torch.float32)
            jitter = torch.from_numpy(np.random.uniform(-0.03, 0.03, size=(N_ANS,)).astype(np.float32))
            t_mat[b, 1:] = (grad_t + jitter).clamp(0.01, 1.0)
    return t_mat


def apply_prompt_dropout_cpu(
    prompt_valid_mask: torch.Tensor,
    full_dropout_prob: float = 0.15,
    prune_prob: float = 0.50,
    prune_frac: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Двойная регуляризация промпта на CPU (15% Full Dropout + 50% Semantic Cell Pruning)."""
    B, _ = prompt_valid_mask.shape
    attn_mask = prompt_valid_mask.clone()
    m_prompt = torch.ones(B, dtype=torch.float32)

    for b in range(B):
        if np.random.random() < full_dropout_prob:
            attn_mask[b] = False
            m_prompt[b] = 0.0
        elif np.random.random() < prune_prob:
            valid_idx = np.where(attn_mask[b].numpy())[0]
            if len(valid_idx) > 1:
                n_drop = max(1, min(len(valid_idx) - 1, int(round(len(valid_idx) * prune_frac))))
                drop_sel = np.random.choice(valid_idx, size=n_drop, replace=False)
                attn_mask[b, drop_sel] = False
    return attn_mask, m_prompt


class PromptAnswerDataset(Dataset):
    """Датасет пар (Q, A) с WordAwareElasticChunker (ADR 137, elastic_drop_prob=0.0 по умолчанию)."""

    def __init__(
        self,
        parquet_path: Optional[str],
        tokenizer: Any,
        stopword_mask: torch.Tensor,
        max_prompt_tokens: int = T_PROMPT,
        max_ans_tokens: int = T_ANS,
        max_rows: Optional[int] = None,
        is_validation: bool = False,
        synthetic_len: int = 64,
        elastic_drop_prob: float = 0.0,
    ):
        self.tokenizer = tokenizer
        self.stopword_mask = stopword_mask.bool()
        self.max_prompt_tokens = max_prompt_tokens
        self.max_ans_tokens = max_ans_tokens
        self.is_validation = is_validation
        self.elastic_drop_prob = 0.0 if is_validation else float(elastic_drop_prob)
        self.punct_ids = build_punctuation_id_set(tokenizer)
        self.chunker = WordAwareElasticChunker(
            tokenizer=tokenizer,
            chunk_size=4,
            pad_token_id=PAD_TOKEN_ID,
            stopword_mask=self.stopword_mask,
        )

        self.pairs: List[Tuple[str, str]] = []
        if parquet_path and os.path.exists(parquet_path):
            df = pd.read_parquet(parquet_path, columns=["Q", "A"])
            if max_rows is not None and len(df) > max_rows:
                df = df.head(max_rows)
            df = df.dropna(subset=["Q", "A"])
            self.pairs = [
                (str(q), str(a))
                for q, a in zip(df["Q"].tolist(), df["A"].tolist())
                if str(q).strip() and str(a).strip()
            ]

        if not self.pairs:
            base_pairs = [
                (
                    "Explain why normalized Grassmannian density similarity stays in [0, 1].",
                    "By Cauchy-Schwarz, the Frobenius inner product of two density operators "
                    "divided by the product of their Frobenius norms is bounded between 0 and 1.",
                ),
                (
                    "Напиши краткое доказательство бесконечности множества простых чисел.",
                    "Предположим, что простых чисел конечное число. Перемножим их все и прибавим единицу.",
                ),
                (
                    "Write a Python function to compute factorial iteratively.",
                    "def factorial(n: int) -> int:\n    res = 1\n    for k in range(2, n + 1):\n        res *= k\n    return res\n",
                ),
                (
                    "What is the capital of France?",
                    "The capital of France is Paris, situated along the Seine River.",
                ),
            ]
            repeats = max(1, (synthetic_len + len(base_pairs) - 1) // len(base_pairs))
            self.pairs = (base_pairs * repeats)[:synthetic_len]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        q_text, a_text = self.pairs[idx]
        q_ids = encode_token_ids(self.tokenizer, q_text)[: self.max_prompt_tokens] or [1]

        a_ids_raw = encode_token_ids(self.tokenizer, a_text)
        if len(a_ids_raw) > self.max_ans_tokens:
            window = a_ids_raw[: self.max_ans_tokens]
            last_punct = -1
            for pos in range(len(window) - 1, max(31, len(window) // 3), -1):
                if window[pos] in self.punct_ids:
                    last_punct = pos
                    break
            a_ids = window[: last_punct + 1] if last_punct >= 32 else window
        else:
            a_ids = a_ids_raw
        if not a_ids:
            a_ids = [1]

        p_pack = self.chunker.chunk_tokens(q_ids, max_cells=N_PROMPT, elastic_drop_prob=0.0)
        a_pack = self.chunker.chunk_tokens(a_ids, max_cells=N_ANS, elastic_drop_prob=self.elastic_drop_prob)

        prompt_ids = p_pack["input_ids"]
        ans_ids = a_pack["input_ids"]
        n_real_prompt = (prompt_ids.view(N_PROMPT, 4) != PAD_TOKEN_ID).sum(dim=-1)
        n_real_ans = (ans_ids.view(N_ANS, 4) != PAD_TOKEN_ID).sum(dim=-1)

        w_cell = torch.zeros(N_CANVAS, dtype=torch.float32)
        n_content_ans = int(a_pack["n_content_cells"])
        i_last = max(1, min(N_ANS, n_content_ans))
        for i in range(1, N_CANVAS):
            if i < i_last:
                w_cell[i] = 1.0
            elif i == i_last or i == i_last + 1:
                w_cell[i] = 2.0
            else:
                w_cell[i] = 0.25

        return {
            "prompt_ids": prompt_ids,
            "ans_ids": ans_ids,
            "ans_targets": a_pack["targets"],
            "prompt_weights": p_pack["weights"],
            "ans_weights": a_pack["weights"],
            "n_real_prompt": n_real_prompt,
            "n_real_ans": n_real_ans,
            "i_last": torch.tensor(i_last, dtype=torch.long),
            "w_cell": w_cell,
        }


def make_train_collate_fn() -> Callable[[List[Dict[str, torch.Tensor]]], Dict[str, torch.Tensor]]:
    """
    Collate для train на CPU: собирает батч и заранее генерирует на CPU расписание шума t_mat,
    маски Prompt Dropout (p_attn_mask, m_prompt) и параметры t_hi/t_base для rollout.
    Устраняет любые Python-циклы и CPU-TPU синхронизации внутри шага обучения.
    """
    def _collate(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = {k: torch.stack([s[k] for s in samples], dim=0) for k in samples[0].keys()}
        B = batch["prompt_ids"].shape[0]
        p_valid = batch["n_real_prompt"] > 0
        p_attn_mask, m_prompt = apply_prompt_dropout_cpu(p_valid)
        t_mat = sample_mixed_noise_schedule_cpu(B, batch["i_last"])
        t_hi = torch.from_numpy(np.random.uniform(0.80, 1.0, size=(B, 1)).astype(np.float32))
        t_base = torch.from_numpy(np.random.uniform(0.15, 0.70, size=(B, 1)).astype(np.float32))

        batch["p_valid"] = p_valid
        batch["p_attn_mask"] = p_attn_mask
        batch["m_prompt"] = m_prompt
        batch["t_mat_sched"] = t_mat
        batch["t_hi_roll"] = t_hi
        batch["t_base_roll"] = t_base
        return batch
    return _collate


@torch.no_grad()
def build_rollout_slice_tpu(
    model: GrassmannianDiffusionCore,
    decoder: Optional[nn.Module],
    V_0: torch.Tensor,
    V_prompt: torch.Tensor,
    p_valid: torch.Tensor,
    p_attn_mask: torch.Tensor,
    m_prompt: torch.Tensor,
    sep_exp: torch.Tensor,
    t_hi: torch.Tensor,
    t_base: torch.Tensor,
    use_bf16: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Детерминированный Two-Step Rollout (alpha=1.0, ADR 140) для фиксированного среза батча B_roll.
    Никаких ветвлений и .item() — 100% статичный граф XLA.
    """
    B_r, N, _, _ = V_0.shape
    device = V_0.device
    was_training = model.training
    model.eval()

    zeros_col = torch.zeros((B_r, 1), device=device, dtype=torch.float32)
    t_hi_mat = torch.cat([zeros_col, t_hi.float().expand(B_r, N - 1)], dim=1)
    noise_1 = torch.randn_like(V_0, dtype=torch.float32) * SIGMA_DATA
    t_hi_exp = t_hi_mat.unsqueeze(-1).unsqueeze(-1)
    V_hi_ans = (1.0 - t_hi_exp[:, 1:]) * V_0[:, 1:].float() + t_hi_exp[:, 1:] * noise_1[:, 1:]
    V_hi = torch.cat([sep_exp.float(), V_hi_ans], dim=1)

    radar_hi = compute_radar_sigma(decoder, V_hi, use_bf16=use_bf16)
    with _autocast_ctx(device, enabled=use_bf16):
        V_draft, _, _ = model(
            V_t=V_hi,
            t_mat=t_hi_mat,
            V_prompt=V_prompt,
            prompt_valid_mask=p_valid,
            prompt_attn_mask=p_attn_mask,
            m_prompt=m_prompt,
            radar_sigma=radar_hi,
            return_sink_weights=False,
        )
    V_draft = torch.cat([sep_exp.float(), V_draft[:, 1:].float()], dim=1)

    sig_draft = compute_radar_sigma(decoder, V_draft, use_bf16=use_bf16)
    sig_ans = sig_draft[:, 1:].float()
    sig_mean = sig_ans.mean(dim=-1, keepdim=True).clamp(min=1e-4)
    rel_uncert = torch.clamp(sig_ans / sig_mean, min=0.35, max=1.45)

    t_base_f = t_base.float()
    t_ans = torch.clamp(t_base_f * rel_uncert, min=0.02, max=0.90)
    cryst_mask = (sig_ans < 0.035) & (t_base_f < 0.45)
    t_ans = torch.where(cryst_mask, torch.zeros_like(t_ans), t_ans)
    t_mat = torch.cat([zeros_col, t_ans], dim=1)

    noise_2 = torch.randn_like(V_0, dtype=torch.float32) * SIGMA_DATA
    t_exp = t_mat.unsqueeze(-1).unsqueeze(-1)
    V_t_ans = (1.0 - t_exp[:, 1:]) * V_draft[:, 1:] + t_exp[:, 1:] * noise_2[:, 1:]
    V_t = torch.cat([sep_exp.float(), V_t_ans], dim=1)

    radar_sigma = compute_radar_sigma(decoder, V_t, use_bf16=use_bf16)
    if was_training:
        model.train()
    return V_t, t_mat, radar_sigma


# %% [markdown]
# ## 5. Статичная гибридная функция потерь и сквозной `L_CE` (Zero-Recompilation, FP32)

# %%
def _xla_all_reduce_sum(x: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size > 1 and hasattr(xm, "all_reduce"):
        return xm.all_reduce(xm.REDUCE_SUM, x)
    return x


def compute_phase3_loss_tpu(
    V_0_hat: torch.Tensor,
    V_0: torch.Tensor,
    t_mat: torch.Tensor,
    w_cell: torch.Tensor,
    i_last: torch.Tensor,
    adaln_stats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    w_adaln_t: torch.Tensor,
    m_prompt: torch.Tensor,
    sink_weights: Optional[torch.Tensor],
    basis_weight: float = 0.50,
    gr_weight: float = 0.50,
    energy_weight: float = 0.15,
    adj_weight: float = 0.10,
    bary_weight: float = 0.10,
    sink_weight: float = 0.10,
    sink_content_max: float = 0.35,
    sink_content_min: float = 0.08,
    bary_k_per_sample: int = 4,
    world_size: int = 1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    100% статичная гибридная функция потерь Фазы 3 (FP32, без .item() и динамических масок).
    Возвращает (total_loss, dict_of_detached_tensors).
    """
    device = V_0_hat.device
    with torch.amp.autocast(device_type=device.type, enabled=False):
        Vh = V_0_hat.float()
        V0 = V_0.float()
        tf = t_mat.float()
        wc = w_cell.float()

        B, N, _, _ = Vh.shape
        m_active = tf > 1e-3
        w_snr = torch.clamp(1.0 / torch.clamp(tf, min=0.25), max=2.0)
        w_tilde = m_active.float() * wc * w_snr
        w_denom = w_tilde.sum().clamp(min=1e-6)

        # 1. Канонический лосс базиса (SmoothL1 beta=0.02)
        elem_basis = F.smooth_l1_loss(Vh, V0, beta=0.02, reduction="none").mean(dim=(-2, -1))
        l_basis = (w_tilde * elem_basis).sum() / w_denom

        # 2. Декомпозированный Грассманов лосс (угловой + энергия)
        Gh = torch.matmul(Vh, Vh.transpose(-2, -1))
        G0 = torch.matmul(V0, V0.transpose(-2, -1))
        cross = torch.matmul(Vh, V0.transpose(-2, -1))
        norm_gh = torch.sqrt((Gh ** 2).sum(dim=(-2, -1)) + 1e-12)
        norm_g0 = torch.sqrt((G0 ** 2).sum(dim=(-2, -1)) + 1e-12)
        sim_cell = (cross ** 2).sum(dim=(-2, -1)) / (norm_gh * norm_g0 + 1e-8)
        elem_gr_dir = torch.clamp(1.0 - sim_cell, min=0.0)
        l_gr = (w_tilde * elem_gr_dir).sum() / w_denom

        energy_h = torch.diagonal(Gh, dim1=-2, dim2=-1).sum(dim=-1)
        energy_0 = torch.diagonal(G0, dim1=-2, dim2=-1).sum(dim=-1)
        elem_energy = (
            F.smooth_l1_loss(energy_h, energy_0, beta=0.05, reduction="none")
            + 0.5 * F.smooth_l1_loss(norm_gh, norm_g0, beta=0.05, reduction="none")
        )
        l_energy = (w_tilde * elem_energy).sum() / w_denom

        # 3. Анти-коллапс смежных смысловых ячеек (статичная маска)
        pos_idx = torch.arange(N - 1, device=device).unsqueeze(0).expand(B, N - 1)
        adj_mask = (pos_idx >= 1) & (pos_idx < i_last.unsqueeze(1)) & m_active[:, :-1] & m_active[:, 1:]
        sim_adj = sim_gr(Vh[:, :-1], Vh[:, 1:])
        adj_w = adj_mask.float()
        l_adj = ((F.relu(sim_adj - TAU_ADJ) ** 2) * adj_w).sum() / adj_w.sum().clamp(min=1.0)

        # 4. Межпримерное отталкивание от барицентра (статичная выборка B * K_bary клеток)
        pos_full = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        content_active = (pos_full >= 1) & (pos_full <= i_last.unsqueeze(1)) & m_active
        K_b = min(bary_k_per_sample, N)
        bary_scores = torch.rand(B, N, device=device, dtype=torch.float32) + content_active.float() * 2.0
        _, top_pos = torch.topk(bary_scores, k=K_b, dim=1, sorted=False)  # (B, K_b)

        gather_idx = top_pos.unsqueeze(-1).unsqueeze(-1).expand(B, K_b, 8, 128)
        cells = torch.gather(Vh, dim=1, index=gather_idx).reshape(B * K_b, 8, 128)
        valid_c = torch.gather(content_active.float(), dim=1, index=top_pos).reshape(B * K_b)

        cross_b = torch.matmul(cells.unsqueeze(1), cells.unsqueeze(0).transpose(-2, -1))
        Gm = torch.matmul(cells, cells.transpose(-2, -1))
        norm_m = torch.sqrt((Gm ** 2).sum(dim=(-2, -1)) + 1e-16)
        sim_mat = (cross_b ** 2).sum(dim=(-2, -1)) / (norm_m.unsqueeze(1) * norm_m.unsqueeze(0) + 1e-8)

        b_ids = torch.arange(B, device=device).repeat_interleave(K_b)
        pair_mask = (b_ids.unsqueeze(1) != b_ids.unsqueeze(0)).float() * (valid_c.unsqueeze(1) * valid_c.unsqueeze(0))
        l_bary = ((F.relu(sim_mat - TAU_BARY) ** 2) * pair_mask).sum() / pair_mask.sum().clamp(min=1.0)

        # 5. Пол разнообразия AdaLN (статичный расчёт)
        sum_m_pack, sum_m2_pack, count_pack = adaln_stats
        sum_m_tot = _xla_all_reduce_sum(sum_m_pack.float().sum(dim=0), world_size)
        sum_m2_tot = _xla_all_reduce_sum(sum_m2_pack.float().sum(dim=0), world_size)
        count_tot = _xla_all_reduce_sum(count_pack.float().sum(dim=0), world_size)

        cnt = count_tot.clamp(min=1.0).unsqueeze(-1)
        mean_m = sum_m_tot / cnt
        var_m = torch.clamp(sum_m2_tot / cnt - mean_m ** 2, min=1e-12)
        sigma_layers = torch.sqrt(var_m).mean(dim=-1)
        has_active = (count_tot > 0).float()
        l_adaln = (F.relu(1e-2 - sigma_layers) * has_active).sum() / has_active.sum().clamp(min=1.0)
        sigma_mean_t = sigma_layers.mean()
        sigma_min_t = sigma_layers.min()

        # 6. Штраф на слив контентных ячеек в k_sink при m_prompt == 1 (статичная маска)
        if sink_weights is not None:
            sw = sink_weights.float()
            if sw.ndim == 2:
                sw = sw.unsqueeze(1)
            cond_active = (m_prompt.float() > 0.5).view(B, 1)
            sink_mask_3d = (content_active & cond_active).unsqueeze(1).expand_as(sw).float()
            pen_sink = (
                F.relu(sw - sink_content_max) ** 2
                + 0.25 * (F.relu(sink_content_min - sw) ** 2)
            )
            l_sink = (pen_sink * sink_mask_3d).sum() / sink_mask_3d.sum().clamp(min=1.0)
        else:
            l_sink = Vh.sum() * 0.0

        total_loss = (
            basis_weight * l_basis
            + gr_weight * l_gr
            + energy_weight * l_energy
            + adj_weight * l_adj
            + bary_weight * l_bary
            + sink_weight * l_sink
            + w_adaln_t * l_adaln
        )

        metrics = {
            "loss/total": total_loss.detach(),
            "loss/basis": l_basis.detach(),
            "loss/gr": l_gr.detach(),
            "loss/energy": l_energy.detach(),
            "loss/adj": l_adj.detach(),
            "loss/bary": l_bary.detach(),
            "loss/sink": l_sink.detach(),
            "loss/adaln_raw": l_adaln.detach(),
            "adaln/sigma_mean": sigma_mean_t.detach(),
            "adaln/sigma_min": sigma_min_t.detach(),
        }
        return total_loss, metrics


def compute_decoder_ce_loss_tpu(
    decoder: nn.Module,
    V_0_hat: torch.Tensor,
    t_mat: torch.Tensor,
    ans_targets: torch.Tensor,
    ce_max_tokens: int = 256,
    use_bf16: bool = True,
    use_grad_ckpt: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    100% статичный сквозной Cross-Entropy через замороженный GrassmannianDecoder (ADR 140):
    - Выбирает ровно M_sel = min(ce_max_tokens, B * 508) позиций через topk(rand + 2 * valid_mask).
    - Никаких torch.where(mask)[0] и .item() -> строго 1 граф XLA, логиты (256, 131072) = 134 MB FP32.
    """
    B = V_0_hat.shape[0]
    device = V_0_hat.device
    V_ans_hat = V_0_hat[:, 1:]  # (B, 127, 8, 128) с градиентом

    active_tok_mask = (t_mat[:, 1:] > 1e-3).unsqueeze(-1).expand(B, N_ANS, 4).reshape(B, -1)
    valid_flat = (active_tok_mask & (ans_targets != -100)).reshape(-1)
    M_sel = min(ce_max_tokens, valid_flat.numel())

    scores = torch.rand(valid_flat.numel(), device=device, dtype=torch.float32) + valid_flat.float() * 2.0
    _, sel_idx = torch.topk(scores, k=M_sel, sorted=False)
    sel_valid = valid_flat[sel_idx].float()

    dec_dtype = next(decoder.parameters()).dtype
    V_in = V_ans_hat.to(dtype=dec_dtype)
    with _autocast_ctx(device, enabled=use_bf16):
        if use_grad_ckpt:
            h_final = torch.utils.checkpoint.checkpoint(
                decoder.decode_hidden, V_in, use_reentrant=False
            )
        else:
            h_final = decoder.decode_hidden(V_in)

    h_sub = h_final.reshape(-1, h_final.shape[-1])[sel_idx]
    raw_tgts = ans_targets.reshape(-1)[sel_idx].long()
    safe_tgts = torch.where(valid_flat[sel_idx], raw_tgts, torch.zeros_like(raw_tgts))

    logits_sub = decoder.get_tied_head_logits(h_sub)  # float32 (M_sel, vocab_size)
    with torch.amp.autocast(device_type=device.type, enabled=False):
        per_tok_ce = F.cross_entropy(logits_sub.float(), safe_tgts, reduction="none")
        denom = sel_valid.sum().clamp(min=1.0)
        l_ce = (per_tok_ce * sel_valid).sum() / denom
        with torch.no_grad():
            tok_acc = ((logits_sub.argmax(dim=-1) == safe_tgts).float() * sel_valid).sum() / denom
    return l_ce, tok_acc.detach()


# %% [markdown]
# ## 6. Оптимизаторы `OrthMuon (d=1)` + `AdamW` и XLA-safe `EMAModel`

# %%
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Ортогонализация градиента Ньютона-Шульца 5-го порядка (BF16 на XLA MXU, FP32 на CPU)."""
    orig_dtype = G.dtype
    X = G.bfloat16() if G.device.type == "xla" else G.float()
    M, N = X.shape
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = X / (X.norm() + eps)
    if M > N:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if M > N:
        X = X.T
    return X.to(orig_dtype)


class OrthMuon(torch.optim.Optimizer):
    """OrthMuon (d=1, сферическая оптимизация строк *.q_weight на S^{d-1}, ADR 131)."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        sphere: bool = True,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            sphere=sphere,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group.get("weight_decay", 0.0)
            sphere = group.get("sphere", True)

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim != 2:
                    raise ValueError(f"OrthMuon accepts only 2D tensors, got {p.shape}")

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                elif state["momentum_buffer"].device != g.device:
                    state["momentum_buffer"] = state["momentum_buffer"].to(g.device)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                update = g.add(buf, alpha=momentum) if nesterov else buf
                update_ortho = zeropower_via_newtonschulz5(update, steps=ns_steps)
                scale = 0.2 * math.sqrt(max(p.size(0), p.size(1)))

                if sphere:
                    q_f32 = p.data.float()
                    u_f32 = update_ortho.float()
                    u_tan = u_f32 - (u_f32 * q_f32).sum(dim=-1, keepdim=True) * q_f32
                    a_f32 = q_f32 - (lr * scale) * u_tan
                    q_next = F.normalize(a_f32, p=2, dim=-1, eps=1e-8)
                    p.data.copy_(q_next.to(p.dtype))
                else:
                    if wd > 0.0:
                        p.data.mul_(1.0 - lr * wd)
                    p.data.add_(update_ortho, alpha=-lr * scale)
        return loss


def create_phase3_optimizers(
    model: nn.Module,
    muon_lr: float = 1e-3,
    adamw_lr: float = 3e-4,
    scale_1d_lr: float = 1.2e-3,
    weight_decay: float = 0.01,
) -> Tuple[Optional[OrthMuon], torch.optim.AdamW]:
    """Разделение параметров строго совпадает с GPU-версией для совместимости чекпоинтов."""
    sphere_params = []
    adamw_decay_params = []
    adamw_nodecay_params = []
    scale_1d_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".q_weight"):
            sphere_params.append(param)
        elif (
            name.endswith(".row_scale")
            or name.endswith(".v_gate")
            or name.endswith(".ls_attn")
            or name.endswith(".ls_mlp")
            or ("_norm." in name)
            or ("q_norm." in name)
            or ("k_norm." in name)
        ):
            scale_1d_params.append(param)
        elif name.endswith(".bias") or param.ndim < 2 or "mlp.3." in name:
            adamw_nodecay_params.append(param)
        else:
            adamw_decay_params.append(param)

    opt_muon = None
    if sphere_params:
        opt_muon = OrthMuon(
            [{"params": sphere_params, "lr": muon_lr, "weight_decay": 0.0, "sphere": True}],
            lr=muon_lr,
            momentum=0.95,
            ns_steps=5,
        )

    adamw_groups = []
    if adamw_decay_params:
        adamw_groups.append({"params": adamw_decay_params, "lr": adamw_lr, "base_lr": adamw_lr, "weight_decay": weight_decay})
    if adamw_nodecay_params:
        adamw_groups.append({"params": adamw_nodecay_params, "lr": adamw_lr, "base_lr": adamw_lr, "weight_decay": 0.0})
    if scale_1d_params:
        adamw_groups.append({"params": scale_1d_params, "lr": scale_1d_lr, "base_lr": scale_1d_lr, "weight_decay": 0.0})

    opt_adamw = torch.optim.AdamW(adamw_groups, lr=adamw_lr, betas=(0.9, 0.999))
    return opt_muon, opt_adamw


def get_cosine_lr_mult(step: int, max_steps: int, warmup_steps: int, min_ratio: float = 0.05) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return 0.05 + 0.95 * (float(step) / float(warmup_steps))
    if max_steps <= warmup_steps:
        return 1.0
    progress = min(1.0, max(0.0, float(step - warmup_steps) / float(max_steps - warmup_steps)))
    return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))


def update_optimizers_lr(
    opt_muon: Optional[OrthMuon],
    opt_adamw: torch.optim.AdamW,
    step: int,
    max_steps: int,
    warmup_steps: int,
    muon_lr: float,
) -> float:
    mult = get_cosine_lr_mult(step, max_steps, warmup_steps, min_ratio=0.05)
    if opt_muon is not None:
        for g in opt_muon.param_groups:
            g["lr"] = muon_lr * mult
    for g in opt_adamw.param_groups:
        g["lr"] = g["base_lr"] * mult
    return mult


class EMAModel:
    """
    XLA-safe FP32 EMA-трекер (ADR 132 + чеклист TPU):
    - Обновление через тензорный `one_minus_decay_t` без `lerp_` со скаляром;
    - `apply_shadow` сохраняет бэкап весов в CPU RAM (экономит ~0.67 ГБ HBM);
    - `load_state_dict` обновляет тени строго in-place через `.copy_()`.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, device: Optional[torch.device] = None):
        self.decay = float(decay)
        self.device = device or next(model.parameters()).device
        self.one_minus_decay_t = torch.tensor(1.0 - self.decay, device=self.device, dtype=torch.float32)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone().float().to(self.device)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        alpha_t = self.one_minus_decay_t
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                s = self.shadow[name]
                diff = param.detach().float() - s
                s.add_(diff * alpha_t)

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module) -> None:
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.detach().cpu().clone()
                val = self.shadow[name]
                if name.endswith(".q_weight"):
                    val = F.normalize(val.float(), p=2, dim=-1, eps=1e-8)
                param.data.copy_(val.to(device=param.device, dtype=param.dtype))

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name].to(device=param.device, dtype=param.dtype))
        self.backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        for k, v in state_dict.items():
            if k in self.shadow:
                self.shadow[k].copy_(v.to(device=self.shadow[k].device, dtype=torch.float32))


# %% [markdown]
# ## 7. Сэмплер Эйлера–Хьюна с CFG и Evaluation Gate (на EMA-весах)

# %%
def _apply_cfg_and_clamp(
    v0_cond: torch.Tensor,
    v0_uncond: torch.Tensor,
    cfg_scale: float,
    cfg_rescale: float = 0.7,
) -> torch.Tensor:
    vc = v0_cond.float()
    vu = v0_uncond.float()
    v_cfg = vu + cfg_scale * (vc - vu)
    if cfg_rescale > 0.0:
        norm_c = torch.sqrt((vc ** 2).sum(dim=(-2, -1), keepdim=True) + 1e-12)
        norm_cfg = torch.sqrt((v_cfg ** 2).sum(dim=(-2, -1), keepdim=True) + 1e-12)
        v_rescaled = v_cfg * (norm_c / norm_cfg)
        v_cfg = cfg_rescale * v_rescaled + (1.0 - cfg_rescale) * v_cfg
    return clamp_to_unit_ball(v_cfg)


@torch.no_grad()
def euler_heun_sample(
    core: GrassmannianDiffusionCore,
    V_prompt: torch.Tensor,
    prompt_valid_mask: torch.Tensor,
    sep_concept: torch.Tensor,
    num_steps: int = 20,
    cfg_scale: float = 3.0,
    cfg_rescale: float = 0.7,
    decoder: Optional[nn.Module] = None,
    use_bf16: bool = True,
    seed: Optional[int] = None,
) -> torch.Tensor:
    B = V_prompt.shape[0]
    device = V_prompt.device

    with _autocast_ctx(device, enabled=use_bf16):
        x_prompt = core.encode_prompt(V_prompt, prompt_valid_mask=prompt_valid_mask)

    sep_exp = sep_concept.to(device=device, dtype=torch.float32).view(1, 1, 8, 128).expand(B, 1, 8, 128)
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    noise = (torch.randn((B, N_ANS, 8, 128), generator=gen, dtype=torch.float32) * SIGMA_DATA).to(device)
    V_t = torch.cat([sep_exp, noise], dim=1)

    t_grid = np.linspace(1.0, 0.0, num=num_steps + 1)
    m_cond = torch.ones(B, device=device, dtype=torch.float32)
    m_uncond = torch.zeros(B, device=device, dtype=torch.float32)
    uncond_mask = torch.zeros_like(prompt_valid_mask)

    def predict_v0(V_in: torch.Tensor, t_val: float) -> torch.Tensor:
        t_mat = torch.full((B, N_CANVAS), t_val, device=device, dtype=torch.float32)
        t_mat[:, 0] = 0.0
        radar_sig = compute_radar_sigma(decoder, V_in, use_bf16=use_bf16)
        with _autocast_ctx(device, enabled=use_bf16):
            v0_cond, _, _ = core.forward_with_cached_prompt(
                V_t=V_in,
                t_mat=t_mat,
                x_prompt=x_prompt,
                prompt_attn_mask=prompt_valid_mask,
                m_prompt=m_cond,
                radar_sigma=radar_sig,
                return_sink_weights=False,
            )
            if abs(cfg_scale - 1.0) > 1e-4:
                v0_uncond, _, _ = core.forward_with_cached_prompt(
                    V_t=V_in,
                    t_mat=t_mat,
                    x_prompt=x_prompt,
                    prompt_attn_mask=uncond_mask,
                    m_prompt=m_uncond,
                    radar_sigma=radar_sig,
                    return_sink_weights=False,
                )
                v0 = _apply_cfg_and_clamp(v0_cond, v0_uncond, cfg_scale=cfg_scale, cfg_rescale=cfg_rescale)
            else:
                v0 = v0_cond.float()
        return torch.cat([sep_exp, v0[:, 1:]], dim=1)

    for k in range(num_steps):
        t_cur = float(t_grid[k])
        t_next = float(t_grid[k + 1])
        dt = t_cur - t_next

        v0_1 = predict_v0(V_t, t_cur)
        if t_next <= 1e-6:
            V_t = v0_1
            xm.mark_step()
            break

        d_1 = (V_t[:, 1:] - v0_1[:, 1:]) / max(t_cur, 1e-4)
        V_pred = torch.cat([sep_exp, V_t[:, 1:] - dt * d_1], dim=1)
        v0_2 = predict_v0(V_pred, t_next)
        d_2 = (V_pred[:, 1:] - v0_2[:, 1:]) / max(t_next, 1e-4)
        V_t = torch.cat([sep_exp, V_t[:, 1:] - 0.5 * dt * (d_1 + d_2)], dim=1)
        xm.mark_step()

    return V_t


@torch.no_grad()
def decode_latent_answers(
    V_canvas: torch.Tensor,
    decoder: nn.Module,
    tokenizer: Any,
    void_ref: torch.Tensor,
) -> Tuple[List[str], List[int]]:
    V_ans = V_canvas[:, 1:].float()
    is_void = detect_void_mask(V_ans, void_ref).cpu()
    B = V_ans.shape[0]
    texts: List[str] = []
    void_starts: List[int] = []
    dec_dtype = next(decoder.parameters()).dtype

    for b in range(B):
        void_idx = torch.where(is_void[b])[0]
        if len(void_idx) > 0:
            n_keep = int(void_idx[0])
            void_starts.append(n_keep + 1)
        else:
            n_keep = N_ANS
            void_starts.append(-1)

        if n_keep == 0:
            texts.append("")
            continue

        logits = decoder(V_ans[b : b + 1, :n_keep].to(dtype=dec_dtype))
        pred_ids = logits.argmax(dim=-1)[0].cpu()
        xm.mark_step()
        if tokenizer is not None:
            texts.append(
                WordAwareElasticChunker.detokenize_non_pad(
                    pred_ids, tokenizer=tokenizer, pad_token_id=PAD_TOKEN_ID, skip_special_tokens=True
                )
            )
        else:
            tok_ids = [tid for tid in pred_ids.tolist() if tid != PAD_TOKEN_ID]
            texts.append(" ".join(str(x) for x in tok_ids[:32]))
    return texts, void_starts


@torch.no_grad()
def evaluate_phase3_tpu(
    model: GrassmannianDiffusionCore,
    ema: Optional[EMAModel],
    encoder: nn.Module,
    decoder: nn.Module,
    tokenizer: Any,
    val_dataloader: DataLoader,
    sep_concept: torch.Tensor,
    void_concept: torch.Tensor,
    device: torch.device,
    use_bf16: bool = True,
    max_batches: int = 10,
    run_generation_check: bool = True,
    gen_steps: int = 20,
) -> Dict[str, Any]:
    """Валидация на master-процессе строго на EMA-весах с `xm.mark_step()` на каждом шаге."""
    was_training = model.training
    model.eval()
    if ema is not None:
        ema.apply_shadow(model)

    sep_cpu, _ = canonicalize_cell_basis(sep_concept.float())
    void_dev = void_concept.float().to(device)

    low_sim, low_cfg, low_sink_c, low_sink_v = [], [], [], []
    hi_sim, hi_cfg, hi_sink_c, hi_sink_v = [], [], [], []
    rk_vals, radar_vals, tok_acc_low_vals, tok_acc_hi_vals, ce_hi_vals = [], [], [], [], []
    tp, fp, fn = 0, 0, 0
    void_start_exact, void_start_pm1, void_start_tot = 0, 0, 0
    gen_samples: List[Dict[str, str]] = []

    enc_dtype = next(encoder.parameters()).dtype
    dec_dtype = next(decoder.parameters()).dtype

    try:
        for b_idx, batch in enumerate(val_dataloader):
            if b_idx >= max_batches:
                break
            prompt_ids = batch["prompt_ids"].to(device)
            ans_ids = batch["ans_ids"].to(device)
            n_real_prompt = batch["n_real_prompt"].to(device)
            n_real_ans = batch["n_real_ans"].to(device)
            w_p = batch["prompt_weights"].to(device)
            w_a = batch["ans_weights"].to(device)
            B = prompt_ids.shape[0]

            with _autocast_ctx(device, enabled=use_bf16):
                V_prompt_raw = encoder(prompt_ids, w_p.to(enc_dtype)).float()
                V_ans_raw = encoder(ans_ids, w_a.to(enc_dtype)).float()
            V_prompt, _ = canonicalize_cell_basis(V_prompt_raw)
            V_ans, _ = canonicalize_cell_basis(V_ans_raw)

            sep_exp = sep_cpu.to(device).view(1, 1, 8, 128).expand(B, 1, 8, 128)
            V_0 = torch.cat([sep_exp, V_ans], dim=1)
            p_valid = n_real_prompt > 0
            noise = torch.randn_like(V_0, dtype=torch.float32)

            def _run_win(t0_v: float, t1_v: float):
                t_val = torch.linspace(t0_v, t1_v, N_ANS, device=device).view(1, N_ANS).expand(B, N_ANS)
                t_mat = torch.cat([torch.zeros(B, 1, device=device), t_val], dim=1)
                t_exp = t_mat.unsqueeze(-1).unsqueeze(-1)
                V_t = torch.cat([sep_exp, (1.0 - t_exp[:, 1:]) * V_ans + t_exp[:, 1:] * SIGMA_DATA * noise[:, 1:]], dim=1)
                rad = compute_radar_sigma(decoder, V_t, use_bf16=use_bf16)
                with _autocast_ctx(device, enabled=use_bf16):
                    vh, _, sw = model(
                        V_t=V_t, t_mat=t_mat, V_prompt=V_prompt, prompt_valid_mask=p_valid,
                        prompt_attn_mask=p_valid, m_prompt=torch.ones(B, device=device),
                        radar_sigma=rad, return_sink_weights=True,
                    )
                    vu, _, _ = model(
                        V_t=V_t, t_mat=t_mat, V_prompt=V_prompt, prompt_valid_mask=p_valid,
                        prompt_attn_mask=torch.zeros_like(p_valid), m_prompt=torch.zeros(B, device=device),
                        radar_sigma=rad, return_sink_weights=False,
                    )
                return vh.float(), vu.float(), (sw.float() if sw is not None else None)

            V_hat_lo, V_unc_lo, sw_lo = _run_win(0.1, 0.5)
            V_hat_hi, V_unc_hi, sw_hi = _run_win(0.75, 0.95)

            is_content = n_real_ans > 0
            is_void = ~is_content
            c_f = is_content.float()
            v_f = is_void.float()
            c_den = c_f.sum().clamp(min=1.0)
            v_den = v_f.sum().clamp(min=1.0)

            s_lo = (sim_gr(V_hat_lo[:, 1:], V_ans) * c_f).sum() / c_den
            c_lo = (torch.sqrt(((V_hat_lo[:, 1:] - V_unc_lo[:, 1:]) ** 2).sum(dim=(-2, -1)) + 1e-12) * c_f).sum() / c_den
            s_hi = (sim_gr(V_hat_hi[:, 1:], V_ans) * c_f).sum() / c_den
            c_hi = (torch.sqrt(((V_hat_hi[:, 1:] - V_unc_hi[:, 1:]) ** 2).sum(dim=(-2, -1)) + 1e-12) * c_f).sum() / c_den
            rk_t = (compute_rk_eff(V_hat_lo[:, 1:]) * c_f).sum() / c_den

            sc_lo = (sw_lo.mean(dim=1)[:, 1:] * c_f).sum() / c_den if sw_lo is not None else torch.zeros((), device=device)
            sv_lo = (sw_lo.mean(dim=1)[:, 1:] * v_f).sum() / v_den if sw_lo is not None else torch.zeros((), device=device)
            sc_hi = (sw_hi.mean(dim=1)[:, 1:] * c_f).sum() / c_den if sw_hi is not None else torch.zeros((), device=device)
            sv_hi = (sw_hi.mean(dim=1)[:, 1:] * v_f).sum() / v_den if sw_hi is not None else torch.zeros((), device=device)

            sig_lo = compute_radar_sigma(decoder, V_hat_lo, use_bf16=use_bf16)[:, 1:]
            rad_t = (sig_lo * c_f).sum() / c_den

            # Токенные метрики (статичная выборка 256 реальных токенов)
            real_tok_flat = ((ans_ids.view(B, N_ANS, 4) != PAD_TOKEN_ID) & is_content.unsqueeze(-1)).reshape(-1)
            M_v = min(256, real_tok_flat.numel())
            _, idx_r = torch.topk(real_tok_flat.float(), k=M_v, sorted=False)
            sel_v = real_tok_flat[idx_r].float()
            sel_den = sel_v.sum().clamp(min=1.0)
            tgt_r = torch.where(real_tok_flat[idx_r], ans_ids.reshape(-1)[idx_r].long(), torch.zeros(M_v, device=device, dtype=torch.long))

            with _autocast_ctx(device, enabled=use_bf16):
                h_lo = decoder.decode_hidden(V_hat_lo[:, 1:].to(dec_dtype))
                h_hi = decoder.decode_hidden(V_hat_hi[:, 1:].to(dec_dtype))
            logits_lo = decoder.get_tied_head_logits(h_lo.reshape(-1, h_lo.shape[-1])[idx_r])
            logits_hi = decoder.get_tied_head_logits(h_hi.reshape(-1, h_hi.shape[-1])[idx_r])
            acc_lo_t = ((logits_lo.argmax(dim=-1) == tgt_r).float() * sel_v).sum() / sel_den
            acc_hi_t = ((logits_hi.argmax(dim=-1) == tgt_r).float() * sel_v).sum() / sel_den
            ce_hi_t = (F.cross_entropy(logits_hi.float(), tgt_r, reduction="none") * sel_v).sum() / sel_den

            pred_void_cpu = detect_void_mask(V_hat_lo[:, 1:], void_dev).cpu()
            true_void_cpu = is_void.cpu()

            packed = torch.stack([s_lo, c_lo, s_hi, c_hi, rk_t, sc_lo, sv_lo, sc_hi, sv_hi, rad_t, acc_lo_t, acc_hi_t, ce_hi_t])
            xm.mark_step()
            vals = packed.cpu().tolist()

            low_sim.append(vals[0]); low_cfg.append(vals[1])
            hi_sim.append(vals[2]); hi_cfg.append(vals[3])
            rk_vals.append(vals[4])
            low_sink_c.append(vals[5]); low_sink_v.append(vals[6])
            hi_sink_c.append(vals[7]); hi_sink_v.append(vals[8])
            radar_vals.append(vals[9])
            tok_acc_low_vals.append(vals[10]); tok_acc_hi_vals.append(vals[11]); ce_hi_vals.append(vals[12])

            tp += int((pred_void_cpu & true_void_cpu).sum())
            fp += int((pred_void_cpu & ~true_void_cpu).sum())
            fn += int((~pred_void_cpu & true_void_cpu).sum())

            for b in range(B):
                tv = torch.where(true_void_cpu[b])[0]
                pv = torch.where(pred_void_cpu[b])[0]
                if len(tv) > 0:
                    void_start_tot += 1
                    ti = int(tv[0])
                    pi = int(pv[0]) if len(pv) > 0 else N_ANS
                    if pi == ti:
                        void_start_exact += 1
                    if abs(pi - ti) <= 1:
                        void_start_pm1 += 1

            if run_generation_check and b_idx == 0 and B >= 1:
                n_gen = min(2, B)
                V_gen = euler_heun_sample(
                    model, V_prompt[:n_gen], p_valid[:n_gen], sep_cpu,
                    num_steps=gen_steps, cfg_scale=3.0, decoder=decoder, use_bf16=use_bf16, seed=1234,
                )
                gen_texts, gen_vstarts = decode_latent_answers(V_gen, decoder, tokenizer, void_dev)
                p_ids_cpu = prompt_ids[:n_gen].cpu()
                for g_i in range(n_gen):
                    q_str = WordAwareElasticChunker.detokenize_non_pad(
                        p_ids_cpu[g_i], tokenizer=tokenizer, pad_token_id=PAD_TOKEN_ID, skip_special_tokens=True
                    )
                    gen_samples.append({
                        "prompt": q_str[:160],
                        "generated": gen_texts[g_i][:300],
                        "void_start": str(gen_vstarts[g_i]),
                    })

            del V_hat_lo, V_unc_lo, V_hat_hi, V_unc_hi, h_lo, h_hi, logits_lo, logits_hi
            xm.mark_step()
    finally:
        if ema is not None:
            ema.restore(model)
        if was_training:
            model.train()

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    void_f1 = (2 * precision * recall) / max(1e-8, precision + recall) if (tp + fp + fn) > 0 else 1.0

    return {
        "val/sim_gr_denoising": float(np.mean(low_sim)) if low_sim else 0.0,
        "val/cfg_contrast": float(np.mean(low_cfg)) if low_cfg else 0.0,
        "val/sim_gr_hi": float(np.mean(hi_sim)) if hi_sim else 0.0,
        "val/cfg_contrast_hi": float(np.mean(hi_cfg)) if hi_cfg else 0.0,
        "val/rk_eff_content": float(np.mean(rk_vals)) if rk_vals else 0.0,
        "val/radar_sigma_cell": float(np.mean(radar_vals)) if radar_vals else 0.0,
        "val/tok_acc_low": float(np.mean(tok_acc_low_vals)) if tok_acc_low_vals else 0.0,
        "val/tok_acc_hi": float(np.mean(tok_acc_hi_vals)) if tok_acc_hi_vals else 0.0,
        "val/ce_loss_hi": float(np.mean(ce_hi_vals)) if ce_hi_vals else 0.0,
        "val/void_f1": float(void_f1),
        "val/void_start_exact": float(void_start_exact / max(1, void_start_tot)),
        "val/void_start_pm1": float(void_start_pm1 / max(1, void_start_tot)),
        "val/sink_attn_content": float(np.mean(low_sink_c)) if low_sink_c else 0.0,
        "val/sink_attn_void": float(np.mean(low_sink_v)) if low_sink_v else 0.0,
        "val/sink_attn_content_hi": float(np.mean(hi_sink_c)) if hi_sink_c else 0.0,
        "val/sink_attn_void_hi": float(np.mean(hi_sink_v)) if hi_sink_v else 0.0,
        "gen_samples": gen_samples,
    }


# %% [markdown]
# ## 8. Чекпоинты (совместимые с GPU/GCS) и цикл обучения на TPU

# %%
def _parse_gcs_uri(gcs_uri: str) -> Tuple[str, str]:
    """Разбирает gs://bucket/prefix -> (bucket_name, prefix_without_leading_slash)."""
    clean = gcs_uri.replace("gs://", "", 1)
    parts = clean.split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket_name, prefix


def get_gcs_client():
    """Возвращает google.cloud.storage.Client() (основной способ работы с GCS на Kaggle TPU)."""
    try:
        from google.cloud import storage
        return storage.Client()
    except Exception:
        return None


def sync_to_gcs(local_path: str, gcs_dir: str) -> bool:
    if not gcs_dir:
        return False
    target = gcs_dir.rstrip("/") + "/" + os.path.basename(local_path)
    client = get_gcs_client()
    if client is not None:
        try:
            bucket_name, blob_name = _parse_gcs_uri(target)
            client.bucket(bucket_name).blob(blob_name).upload_from_filename(local_path)
            print(f"[GCS] Выгружено (SDK): {target}", flush=True)
            return True
        except Exception as e:
            print(f"[GCS Warning] Ошибка SDK при выгрузке {target}: {e}", flush=True)
    if shutil.which("gsutil"):
        try:
            res = subprocess.run(["gsutil", "-q", "cp", local_path, target], capture_output=True, text=True)
            if res.returncode == 0:
                print(f"[GCS] Выгружено (gsutil): {target}", flush=True)
                return True
        except Exception as e:
            print(f"[GCS Warning] Ошибка gsutil при выгрузке {target}: {e}", flush=True)
    return False


def find_latest_checkpoint(checkpoint_dir: str, gcs_dir: Optional[str] = None) -> Optional[str]:
    local_files = glob.glob(os.path.join(checkpoint_dir, "phase3_step_*.pt"))
    if local_files:
        local_files.sort(key=lambda f: int(re.findall(r"\d+", os.path.basename(f))[-1]))
        return local_files[-1]

    if gcs_dir:
        client = get_gcs_client()
        if client is not None:
            try:
                bucket_name, prefix = _parse_gcs_uri(gcs_dir.rstrip("/") + "/")
                blobs = [
                    b for b in client.list_blobs(bucket_name, prefix=prefix)
                    if re.search(r"phase3_step_(\d+)\.pt$", os.path.basename(b.name))
                ]
                if blobs:
                    blobs.sort(key=lambda b: int(re.findall(r"\d+", os.path.basename(b.name))[-1]))
                    latest_blob = blobs[-1]
                    dest = os.path.join(checkpoint_dir, os.path.basename(latest_blob.name))
                    print(f"[GCS] Скачивание последнего чекпоинта: gs://{bucket_name}/{latest_blob.name} ...", flush=True)
                    latest_blob.download_to_filename(dest)
                    if os.path.exists(dest):
                        print(f"[GCS] Чекпоинт загружен: {dest}", flush=True)
                        return dest
            except Exception as e:
                print(f"[GCS Warning] Ошибка SDK при поиске чекпоинта в {gcs_dir}: {e}", flush=True)

        if shutil.which("gsutil"):
            try:
                res = subprocess.run(["gsutil", "ls", gcs_dir.rstrip("/") + "/phase3_step_*.pt"], capture_output=True, text=True)
                if res.returncode == 0 and res.stdout.strip():
                    lines = [ln.strip() for ln in res.stdout.strip().split("\n") if ln.strip()]
                    lines.sort(key=lambda f: int(re.findall(r"\d+", os.path.basename(f))[-1]))
                    latest = lines[-1]
                    dest = os.path.join(checkpoint_dir, os.path.basename(latest))
                    dl = subprocess.run(["gsutil", "cp", latest, dest], capture_output=True, text=True)
                    if dl.returncode == 0 and os.path.exists(dest):
                        return dest
            except Exception as e:
                print(f"[GCS Warning] Ошибка gsutil при поиске чекпоинта: {e}", flush=True)
    return None


def migrate_cond_embedder_state_dict(sd: Dict[str, torch.Tensor], expected_in_dim: int = 72) -> Dict[str, torch.Tensor]:
    key = "cond_embedder.mlp.0.weight"
    if key in sd and sd[key].ndim == 2 and sd[key].shape[1] == 200 and expected_in_dim == 72:
        sd = dict(sd)
        sd[key] = sd[key][:, 128:200].clone()
    return sd


def save_checkpoint_tpu(
    model: nn.Module,
    opt_muon: Optional[OrthMuon],
    opt_adamw: torch.optim.AdamW,
    step: int,
    metrics: Dict[str, float],
    path: str,
    ema: Optional[EMAModel] = None,
    sync_gcs_flag: bool = False,
    gcs_dir: str = "",
    keep_local: bool = False,
) -> None:
    """Безопасное сохранение чекпоинта на TPU через xm.save(..., master_only=True)."""
    payload: Dict[str, Any] = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "opt_adamw_state_dict": opt_adamw.state_dict(),
        "metrics": metrics,
        "ema_state_dict": ema.state_dict() if ema is not None else None,
    }
    if opt_muon is not None:
        payload["opt_muon_state_dict"] = opt_muon.state_dict()

    xm.save(payload, path, master_only=True)
    if xm.is_master_ordinal():
        print(f"[Checkpoint] Шаг {step} сохранен: {path}", flush=True)
        if sync_gcs_flag and gcs_dir:
            ok = sync_to_gcs(path, gcs_dir)
            if ok and not keep_local and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
    xm.rendezvous(f"save_ckpt_{step}")


def _mp_train_worker(index: int, args: argparse.Namespace, env: EnvironmentConfig, resume_path: Optional[str]) -> None:
    """Рабочий процесс одного TPU-чипа (запускается через xmp.spawn)."""
    device = xm.xla_device()
    world_size = xr.global_runtime_device_count() if hasattr(xr, "global_runtime_device_count") else 1
    rank = xm.get_ordinal()
    is_master = xm.is_master_ordinal()
    use_bf16 = device.type == "xla"

    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    if hasattr(xm, "set_rng_state"):
        xm.set_rng_state(42 + rank)

    sync_gcs_flag = (env.is_kaggle and not args.no_gcs) or (not args.no_gcs and args.gcs_dir != "")
    if args.smoke_test:
        sync_gcs_flag = False

    use_wandb = False
    wandb_mode = args.wandb if args.wandb != "auto" else env.wandb_mode
    if is_master and wandb_mode == "online":
        api_key = os.environ.get("WANDB_API_KEY", "").strip()
        if api_key:
            try:
                import wandb
                wandb.login(key=api_key)
                wandb.init(
                    project="BEBLaDII-Phase3-PlanC",
                    name=f"phase3-tpu-{int(time.time())}",
                    config={**vars(args), "world_size": world_size, "device_type": device.type},
                )
                use_wandb = True
            except Exception as e:
                print(f"[WandB Warning] {e}", flush=True)

    # 1. Загрузка замороженной Системы 1 (в BF16 на TPU, в FP32 на CPU smoke-test)
    frozen_dtype = torch.bfloat16 if use_bf16 else torch.float32
    tokenizer = get_tokenizer()
    encoder = get_encoder(weights_path=env.encoder_weights_path, device=device, dtype=frozen_dtype, eval_mode=True)
    decoder = get_decoder(
        weights_path=env.decoder_weights_path,
        encoder=encoder,
        device=device,
        dtype=frozen_dtype,
        eval_mode=True,
    )
    for p in list(encoder.parameters()) + list(decoder.parameters()):
        p.requires_grad_(False)

    sep_concept = torch.load(env.sep_concept_path, map_location="cpu", weights_only=False)
    void_concept = torch.load(env.void_concept_path, map_location="cpu", weights_only=False)
    if os.path.exists(env.stopwords_path):
        sw_obj = torch.load(env.stopwords_path, map_location="cpu", weights_only=False)
        stopword_mask = (sw_obj["mask"] if isinstance(sw_obj, dict) else sw_obj).bool()
    else:
        stopword_mask = torch.zeros(131072, dtype=torch.bool)

    sep_cpu, _ = canonicalize_cell_basis(sep_concept.float())
    sep_dev = sep_cpu.to(device=device, dtype=torch.float32)

    # 2. Создание модели на CPU, загрузка чекпоинта ДО переноса на TPU (Блок E), затем .to(device) в FP32
    model = GrassmannianDiffusionCore(void_ref=void_concept)
    start_step = 0
    ckpt_loaded = None
    if resume_path and os.path.exists(resume_path):
        ckpt_loaded = torch.load(resume_path, map_location="cpu", weights_only=False)
        sd = {k.replace("module.", ""): v for k, v in ckpt_loaded["model_state_dict"].items()}
        sd = migrate_cond_embedder_state_dict(sd, expected_in_dim=72)
        model.load_state_dict(sd, strict=True)
        start_step = int(ckpt_loaded.get("step", 0))
        if is_master:
            print(f"[Resume] Веса загружены на CPU с шага {start_step}: {resume_path}", flush=True)

    model = model.float().to(device)
    ema = EMAModel(model, decay=args.ema_decay, device=device) if not args.no_ema else None
    opt_muon, opt_adamw = create_phase3_optimizers(
        model, muon_lr=args.muon_lr, adamw_lr=args.adamw_lr, scale_1d_lr=args.scale_1d_lr
    )

    if ckpt_loaded is not None:
        if ema is not None and ckpt_loaded.get("ema_state_dict") is not None:
            ema_sd = migrate_cond_embedder_state_dict(ckpt_loaded["ema_state_dict"], expected_in_dim=72)
            ema.load_state_dict(ema_sd)
        try:
            opt_adamw.load_state_dict(ckpt_loaded["opt_adamw_state_dict"])
        except Exception as e:
            if is_master:
                print(f"[Resume Warning] AdamW state не загружен: {e}", flush=True)
        if opt_muon is not None and "opt_muon_state_dict" in ckpt_loaded:
            try:
                opt_muon.load_state_dict(ckpt_loaded["opt_muon_state_dict"])
            except Exception as e:
                if is_master:
                    print(f"[Resume Warning] OrthMuon state не загружен: {e}", flush=True)
        del ckpt_loaded

    xm.rendezvous("model_ready")

    # 3. Датасеты и DataLoader'ы (строго drop_last=True на train и val!)
    train_path = args.data_path if args.data_path != "auto" else env.train_data_path
    val_path = args.val_data_path if args.val_data_path != "auto" else env.val_data_path
    max_rows = 32 if args.smoke_test else None

    train_dataset = PromptAnswerDataset(
        parquet_path=None if args.smoke_test else train_path,
        tokenizer=tokenizer,
        stopword_mask=stopword_mask,
        max_rows=max_rows,
        is_validation=False,
        elastic_drop_prob=args.elastic_drop_prob,
    )
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    dl_kwargs: Dict[str, Any] = dict(
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=env.num_workers,
        collate_fn=make_train_collate_fn(),
    )
    if env.num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(train_dataset, drop_last=True, **dl_kwargs)

    val_loader = None
    if is_master:
        val_dataset = PromptAnswerDataset(
            parquet_path=None if args.smoke_test else (val_path if (val_path and os.path.exists(val_path)) else None),
            tokenizer=tokenizer,
            stopword_mask=stopword_mask,
            max_rows=16 if args.smoke_test else 256,
            is_validation=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(args.batch_size, 8),
            shuffle=False,
            num_workers=0,
            drop_last=True,
        )

    # Фиксированное число rollout-примеров в батче (единый граф XLA, 0 перекомпиляций)
    b_roll = int(round(args.batch_size * args.rollout_prob)) if args.rollout_prob > 0.0 else 0
    b_roll = min(args.batch_size, max(0, b_roll))

    # Тензорные гиперпараметры на устройстве (обновляются без перекомпиляции)
    w_ce_t = torch.tensor(0.0, device=device, dtype=torch.float32)
    w_adaln_t = torch.tensor(0.0, device=device, dtype=torch.float32)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    enc_dtype = next(encoder.parameters()).dtype

    if is_master:
        n_params = sum(p.numel() for p in trainable_params)
        print(
            f"=== Старт обучения Phase 3 Plan C на TPU (params={n_params/1e6:.2f}M, "
            f"batch_per_chip={args.batch_size}, world_size={world_size}, "
            f"eff_batch={args.batch_size * world_size}, b_roll={b_roll}/{args.batch_size}, "
            f"start_step={start_step}, max_steps={args.max_steps}) ===",
            flush=True,
        )

    model.train()
    global_step = start_step + 1
    epoch = 0
    lr_mult = update_optimizers_lr(opt_muon, opt_adamw, global_step, args.max_steps, args.lr_warmup_steps, args.muon_lr)
    last_metrics_py: Dict[str, float] = {}

    metric_keys = [
        "loss/total", "loss/ce", "train/ce_tok_acc",
        "loss/basis", "loss/gr", "loss/energy", "loss/adj", "loss/bary", "loss/sink",
        "adaln/sigma_mean", "adaln/sigma_min",
    ]

    while global_step <= args.max_steps:
        train_sampler.set_epoch(epoch)
        para_loader = pl.MpDeviceLoader(train_loader, device)

        for batch in para_loader:
            if global_step > args.max_steps:
                break
            t0 = time.time()

            # Обновление LR раз в 50 шагов (Чеклист B: Константный LR)
            if global_step % 50 == 1 or global_step == start_step + 1:
                lr_mult = update_optimizers_lr(
                    opt_muon, opt_adamw, global_step, args.max_steps, args.lr_warmup_steps, args.muon_lr
                )

            # Обновление тензорных весов прогрева in-place (без перекомпиляции)
            w_adaln_val = 0.05 * min(1.0, float(global_step) / 500.0)
            w_adaln_t.fill_(w_adaln_val)
            ce_warmup = max(1, int(args.ce_warmup_steps))
            w_ce_val = float(args.ce_weight) * min(1.0, float(max(1, global_step - start_step)) / float(ce_warmup))
            w_ce_t.fill_(w_ce_val)

            if opt_muon is not None:
                opt_muon.zero_grad(set_to_none=True)
            opt_adamw.zero_grad(set_to_none=True)

            prompt_ids = batch["prompt_ids"]
            ans_ids = batch["ans_ids"]
            ans_targets = batch["ans_targets"]
            w_p = batch["prompt_weights"]
            w_a = batch["ans_weights"]
            i_last = batch["i_last"]
            w_cell = batch["w_cell"]
            p_valid = batch["p_valid"]
            p_attn_mask = batch["p_attn_mask"]
            m_prompt = batch["m_prompt"]
            t_mat_sched = batch["t_mat_sched"]
            t_hi_roll = batch["t_hi_roll"]
            t_base_roll = batch["t_base_roll"]
            B = prompt_ids.shape[0]

            # 1. On-the-fly кодирование замороженным Энкодером + канонизация базиса
            with torch.no_grad():
                with _autocast_ctx(device, enabled=use_bf16):
                    V_prompt_raw = encoder(prompt_ids, w_p.to(enc_dtype))
                    V_ans_raw = encoder(ans_ids, w_a.to(enc_dtype))
                V_prompt, _ = canonicalize_cell_basis(V_prompt_raw.float())
                V_ans, _ = canonicalize_cell_basis(V_ans_raw.float())

                sep_exp = sep_dev.view(1, 1, 8, 128).expand(B, 1, 8, 128)
                V_0 = torch.cat([sep_exp, V_ans], dim=1)

                # Детерминированный Two-Step Rollout на первых b_roll примерах батча
                if b_roll > 0 and global_step >= args.rollout_start_step:
                    V_t_r, t_mat_r, rad_r = build_rollout_slice_tpu(
                        model=model,
                        decoder=decoder,
                        V_0=V_0[:b_roll],
                        V_prompt=V_prompt[:b_roll],
                        p_valid=p_valid[:b_roll],
                        p_attn_mask=p_attn_mask[:b_roll],
                        m_prompt=m_prompt[:b_roll],
                        sep_exp=sep_exp[:b_roll],
                        t_hi=t_hi_roll[:b_roll],
                        t_base=t_base_roll[:b_roll],
                        use_bf16=use_bf16,
                    )
                    if b_roll < B:
                        t_s = t_mat_sched[b_roll:]
                        noise_s = torch.randn_like(V_0[b_roll:], dtype=torch.float32) * SIGMA_DATA
                        t_exp_s = t_s.unsqueeze(-1).unsqueeze(-1)
                        V_t_s = torch.cat(
                            [sep_exp[b_roll:], (1.0 - t_exp_s[:, 1:]) * V_0[b_roll:, 1:] + t_exp_s[:, 1:] * noise_s[:, 1:]],
                            dim=1,
                        )
                        rad_s = compute_radar_sigma(decoder, V_t_s, use_bf16=use_bf16)
                        V_t = torch.cat([V_t_r, V_t_s], dim=0)
                        t_mat = torch.cat([t_mat_r, t_s], dim=0)
                        radar_sigma = torch.cat([rad_r, rad_s], dim=0)
                    else:
                        V_t, t_mat, radar_sigma = V_t_r, t_mat_r, rad_r
                else:
                    t_mat = t_mat_sched
                    noise = torch.randn_like(V_0, dtype=torch.float32) * SIGMA_DATA
                    t_exp = t_mat.unsqueeze(-1).unsqueeze(-1)
                    V_t = torch.cat([sep_exp, (1.0 - t_exp[:, 1:]) * V_0[:, 1:] + t_exp[:, 1:] * noise[:, 1:]], dim=1)
                    radar_sigma = compute_radar_sigma(decoder, V_t, use_bf16=use_bf16)

            # 2. Прямой проход ядра (BF16 autocast на TPU, FP32 мастер-веса)
            with _autocast_ctx(device, enabled=use_bf16):
                V_0_hat, adaln_stats, sink_weights = model(
                    V_t=V_t,
                    t_mat=t_mat,
                    V_prompt=V_prompt,
                    prompt_valid_mask=p_valid,
                    prompt_attn_mask=p_attn_mask,
                    m_prompt=m_prompt,
                    radar_sigma=radar_sigma,
                    return_sink_weights=True,
                )

            # 3. Гибридный геометрический лосс + сквозной токенный L_CE (FP32)
            loss_geo, m_dict = compute_phase3_loss_tpu(
                V_0_hat=V_0_hat,
                V_0=V_0,
                t_mat=t_mat,
                w_cell=w_cell,
                i_last=i_last,
                adaln_stats=adaln_stats,
                w_adaln_t=w_adaln_t,
                m_prompt=m_prompt,
                sink_weights=sink_weights,
                world_size=world_size,
            )

            if args.ce_weight > 0.0:
                l_ce, ce_acc = compute_decoder_ce_loss_tpu(
                    decoder=decoder,
                    V_0_hat=V_0_hat,
                    t_mat=t_mat,
                    ans_targets=ans_targets,
                    ce_max_tokens=args.ce_max_tokens,
                    use_bf16=use_bf16,
                    use_grad_ckpt=args.decoder_grad_ckpt,
                )
                loss = loss_geo + w_ce_t * l_ce
            else:
                l_ce = torch.zeros((), device=device, dtype=torch.float32)
                ce_acc = torch.zeros((), device=device, dtype=torch.float32)
                loss = loss_geo

            m_dict["loss/total"] = loss.detach()
            m_dict["loss/ce"] = l_ce.detach()
            m_dict["train/ce_tok_acc"] = ce_acc.detach()

            # 4. Обратный проход, усреднение градиентов на TPU, клиппинг, шаг оптимизаторов и EMA
            loss.backward()

            if opt_muon is not None:
                xm.reduce_gradients(opt_muon)
            xm.reduce_gradients(opt_adamw)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

            if opt_muon is not None:
                opt_muon.step()
            opt_adamw.step()

            if ema is not None:
                ema.update(model)

            # 5. Регистрация step_closure и очистка буферов ДО xm.mark_step() (Чеклист D)
            should_log = is_master and (global_step % args.log_every == 0 or global_step == start_step + 1)
            if should_log:
                stacked = torch.stack([m_dict[k] for k in metric_keys]).detach()
                step_snap = global_step
                lr_snap = lr_mult
                w_ce_snap = w_ce_val
                t0_snap = t0

                def _async_log(s_val: int, m_tensor: torch.Tensor, lr_v: float, wce_v: float, t_start: float) -> None:
                    vals = m_tensor.cpu().tolist()
                    md = {k: float(v) for k, v in zip(metric_keys, vals)}
                    md["loss/w_ce"] = wce_v
                    md["lr/mult"] = lr_v
                    dt_s = time.time() - t_start
                    last_metrics_py.clear()
                    last_metrics_py.update(md)
                    print(
                        f"[Step {s_val:6d}/{args.max_steps}] "
                        f"tot={md['loss/total']:.4f} | "
                        f"ce={md['loss/ce']:.3f} (acc={md['train/ce_tok_acc']:.3f}) | "
                        f"basis={md['loss/basis']:.4f} | "
                        f"gr={md['loss/gr']:.4f} | "
                        f"en={md['loss/energy']:.4f} | "
                        f"adj={md['loss/adj']:.4f} | "
                        f"sink={md['loss/sink']:.4f} | "
                        f"adaln_sig={md['adaln/sigma_mean']:.2e} | "
                        f"lr_m={lr_v:.3f} | {dt_s:.2f}s",
                        flush=True,
                    )
                    if use_wandb:
                        import wandb
                        wandb.log({**md, "time/step_sec": dt_s}, step=s_val)

                xm.add_step_closure(_async_log, args=(step_snap, stacked, lr_snap, w_ce_snap, t0_snap))

            del V_0_hat, adaln_stats, sink_weights, loss_geo, loss, l_ce, ce_acc, m_dict
            xm.mark_step()

            # 6. Валидация на EMA-весах (только master)
            if val_loader is not None and (global_step % args.eval_every == 0):
                if is_master:
                    val_res = evaluate_phase3_tpu(
                        model=model,
                        ema=ema,
                        encoder=encoder,
                        decoder=decoder,
                        tokenizer=tokenizer,
                        val_dataloader=val_loader,
                        sep_concept=sep_cpu,
                        void_concept=void_concept,
                        device=device,
                        use_bf16=use_bf16,
                        max_batches=args.eval_batches,
                        run_generation_check=True,
                        gen_steps=args.gen_steps,
                    )
                    gen_samples = val_res.pop("gen_samples", [])
                    print(
                        f"  [Eval @ {global_step}] Sim_Gr={val_res['val/sim_gr_denoising']:.4f} "
                        f"(hi={val_res['val/sim_gr_hi']:.4f}) | "
                        f"TokAcc={val_res['val/tok_acc_low']:.3f} "
                        f"(hi={val_res['val/tok_acc_hi']:.3f}, CE_hi={val_res['val/ce_loss_hi']:.2f}) | "
                        f"CFG={val_res['val/cfg_contrast']:.4f} (hi={val_res['val/cfg_contrast_hi']:.4f}) | "
                        f"VOID_F1={val_res['val/void_f1']:.4f} (pm1={val_res['val/void_start_pm1']:.3f}) | "
                        f"radar={val_res['val/radar_sigma_cell']:.3f} | "
                        f"sink_c={val_res['val/sink_attn_content']:.3f} (hi={val_res['val/sink_attn_content_hi']:.3f}) | "
                        f"rk={val_res['val/rk_eff_content']:.2f}",
                        flush=True,
                    )
                    for idx_g, gs in enumerate(gen_samples):
                        print(
                            f"    [Sample {idx_g}] Q: {gs['prompt']!r} -> void_start={gs['void_start']} | A: {gs['generated']!r}",
                            flush=True,
                        )
                    if use_wandb:
                        import wandb
                        wandb.log(val_res, step=global_step)
                xm.rendezvous(f"eval_done_{global_step}")

            # 7. Регулярный чекпоинт (xm.save на всех воркерах, запись и GCS на master)
            if global_step % args.save_every == 0:
                ckpt_path = os.path.join(env.checkpoint_dir, f"phase3_step_{global_step}.pt")
                save_checkpoint_tpu(
                    model=model,
                    opt_muon=opt_muon,
                    opt_adamw=opt_adamw,
                    step=global_step,
                    metrics=dict(last_metrics_py),
                    path=ckpt_path,
                    ema=ema,
                    sync_gcs_flag=sync_gcs_flag,
                    gcs_dir=env.gcs_checkpoints_dir,
                    keep_local=not env.is_kaggle,
                )

            global_step += 1

        epoch += 1

    if is_master:
        print("=== Обучение Фазы 3 Plan C (TPU) завершено ===", flush=True)


# %% [markdown]
# ## 9. CLI и точка входа (`xmp.spawn`)

# %%
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BEBLaDII Phase 3 Plan C Latent Diffusion Training (TPU/XLA)")
    parser.add_argument("--smoke-test", "--smoke_test", dest="smoke_test", action="store_true")
    parser.add_argument("--data-path", "--data_path", dest="data_path", type=str, default="auto")
    parser.add_argument("--val-data-path", "--val_data_path", dest="val_data_path", type=str, default="auto")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=8, help="Микро-батч на 1 TPU-чип (8 x 8 = 64)")
    parser.add_argument("--max-steps", "--max_steps", dest="max_steps", type=int, default=100000)
    parser.add_argument("--muon-lr", "--muon_lr", dest="muon_lr", type=float, default=1e-3)
    parser.add_argument("--adamw-lr", "--adamw_lr", dest="adamw_lr", type=float, default=3e-4)
    parser.add_argument("--scale-1d-lr", "--scale_1d_lr", dest="scale_1d_lr", type=float, default=1.2e-3)
    parser.add_argument("--lr-warmup-steps", "--lr_warmup_steps", dest="lr_warmup_steps", type=int, default=1000)
    parser.add_argument("--ema-decay", "--ema_decay", dest="ema_decay", type=float, default=0.999)
    parser.add_argument("--no-ema", "--no_ema", dest="no_ema", action="store_true")
    parser.add_argument("--log-every", "--log_every", dest="log_every", type=int, default=20)
    parser.add_argument("--eval-every", "--eval_every", dest="eval_every", type=int, default=500)
    parser.add_argument("--eval-batches", "--eval_batches", dest="eval_batches", type=int, default=10)
    parser.add_argument("--gen-steps", "--gen_steps", dest="gen_steps", type=int, default=20)
    parser.add_argument("--save-every", "--save_every", dest="save_every", type=int, default=500)
    parser.add_argument("--resume-ckpt", "--resume_ckpt", dest="resume_ckpt", type=str, default="auto")
    parser.add_argument("--checkpoint-dir", "--checkpoint_dir", dest="checkpoint_dir", type=str, default="")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default="")
    parser.add_argument("--gcs-dir", "--gcs_dir", dest="gcs_dir", type=str, default="")
    parser.add_argument("--no-gcs", "--no_gcs", dest="no_gcs", action="store_true")
    parser.add_argument("--wandb", type=str, default="auto", choices=["auto", "online", "disabled"])
    parser.add_argument("--elastic-drop-prob", "--elastic_drop_prob", dest="elastic_drop_prob", type=float, default=0.0)
    parser.add_argument("--rollout-prob", "--rollout_prob", dest="rollout_prob", type=float, default=0.35)
    parser.add_argument("--rollout-start-step", "--rollout_start_step", dest="rollout_start_step", type=int, default=1000)
    parser.add_argument("--ce-weight", "--ce_weight", dest="ce_weight", type=float, default=0.15)
    parser.add_argument("--ce-warmup-steps", "--ce_warmup_steps", dest="ce_warmup_steps", type=int, default=300)
    parser.add_argument("--ce-max-tokens", "--ce_max_tokens", dest="ce_max_tokens", type=int, default=256)
    parser.add_argument("--decoder-grad-ckpt", "--decoder_grad_ckpt", dest="decoder_grad_ckpt", action="store_true")
    parser.add_argument("--nprocs", type=int, default=0, help="Число процессов TPU (0 = авто)")
    args, _ = parser.parse_known_args()
    return args


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = parse_args()

    # Поддержка env-переменных валидатора run_tpu_smoke_test.py
    if os.environ.get("SMOKE_TEST") == "1":
        args.smoke_test = True

    env = EnvironmentConfig.detect(
        override_output_dir=args.output_dir,
        override_checkpoint_dir=args.checkpoint_dir,
        setup_secrets=True,
    )
    if args.gcs_dir:
        env.gcs_checkpoints_dir = args.gcs_dir
    if env.is_kaggle:
        cleanup_kaggle_checkpoints(env.checkpoint_dir)

    sync_gcs_flag = (env.is_kaggle and not args.no_gcs) or (not args.no_gcs and args.gcs_dir != "")

    if args.smoke_test:
        print("=== РЕЖИМ TPU SMOKE TEST (ФАЗА 3) АКТИВИРОВАН ===", flush=True)
        args.max_steps = int(os.environ.get("MAX_STEPS", "2"))
        args.batch_size = 2
        args.log_every = 1
        args.eval_every = args.max_steps
        args.eval_batches = int(os.environ.get("VAL_STEPS", "1"))
        args.save_every = args.max_steps
        args.lr_warmup_steps = 1
        args.ce_warmup_steps = 1
        args.ce_max_tokens = 32
        args.rollout_start_step = 1
        args.gen_steps = 2
        args.wandb = "disabled"
        args.resume_ckpt = ""
        args.nprocs = 1
        sync_gcs_flag = False

    # Скачиваем последний чекпоинт из GCS однократно на хосте ДО xmp.spawn
    resume_path = None
    if args.resume_ckpt:
        if args.resume_ckpt == "auto":
            resume_path = find_latest_checkpoint(env.checkpoint_dir, env.gcs_checkpoints_dir if sync_gcs_flag else None)
        else:
            resume_path = args.resume_ckpt

    nprocs = args.nprocs if args.nprocs > 0 else None
    start_method = "fork" if sys.platform != "win32" else "spawn"
    xmp.spawn(_mp_train_worker, args=(args, env, resume_path), nprocs=nprocs, start_method=start_method)


# %% [markdown]
# ## 10. Конфигурация запуска (Kaggle TPU v5e-8 / v3-8)

# %%
TRAINING_CONFIG = {
    "smoke_test": False,
    "data_path": "auto",
    "val_data_path": "auto",
    "batch_size": 8,          # 8 на чип x 8 TPU-чипов = эффективный батч 64
    "max_steps": 100000,
    "muon_lr": 1e-3,
    "adamw_lr": 3e-4,
    "scale_1d_lr": 1.2e-3,
    "lr_warmup_steps": 1000,
    "ema_decay": 0.999,
    "log_every": 20,
    "eval_every": 500,
    "eval_batches": 10,
    "gen_steps": 20,
    "save_every": 500,
    "resume_ckpt": "auto",
    "gcs_dir": "",
    "no_gcs": False,
    "wandb": "auto",
    "elastic_drop_prob": 0.0,
    "rollout_prob": 0.35,
    "rollout_start_step": 1000,
    "ce_weight": 0.15,
    "ce_warmup_steps": 300,
    "ce_max_tokens": 256,     # 256 на чип x 8 чипов = 2048 токенов на шаг (как на 2xT4)
    "decoder_grad_ckpt": False,
    "nprocs": 0,              # 0 = авто (все 8 чипов на Kaggle TPU)
}

if __name__ == "__main__":
    cli_args = parse_args()
    is_notebook = any("ipykernel" in m for m in sys.modules) or (
        not sys.argv[1:] and os.environ.get("SMOKE_TEST") != "1"
    )
    if is_notebook:
        for k, v in TRAINING_CONFIG.items():
            if hasattr(cli_args, k):
                setattr(cli_args, k, v)
    main(cli_args)
