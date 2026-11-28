# %% [markdown]
# # BEBLaDII Phase 3 Plan C — Условное ядро латентной диффузии на Грассманиане с `CA_Prompt` и `NoiseRadarHead`
# Архитектура:
# - Замороженный Энкодер Фазы 1 (`GrassmannianEncoder`) + обновлённый O(8)-инвариантный Декодер-Радар Фазы 2 (`GrassmannianDecoder` step_25000, ADR 137).
# - Словно-ориентированный эластичный чанкер `WordAwareElasticChunker` + канонизация базиса `canonicalize_cell_basis(V)` для таргетов V_0 и V_prompt.
# - Unit-RMS остаточный поток (`sigma_data = 0.05`) + `PromptAdapter` (`2 x Full SA`).
# - 10-слойное ядро `GrassmannianDiffusionCore`: `Full, Local(r=8), CA_Prompt, Full, CA_Prompt, Full, CA_Prompt, Full, Local(r=8), Full`.
# - `Shared KV Projection` + `Split K/V Channel Gating` во всех 12 слоях внимания.
# - `ConditionEmbedder` (локальное/глобальное время + 8 спектральных датчиков, включая `s7 = NoiseRadarHead.sigma_cell_hat`) + `Soft-Gate AdaLN`.
# - Детерминированный якорь `SEP` на позиции `0` ($t_0 = 0$) и формат ответа $N_{\text{ans}} = 127$ (до 508 токенов).
# - Целевая платформа: Kaggle 2xT4 (`sm_75`, FP16 + GradScaler, `DDP` или `DataParallel` с внешним расчетом лосса).

# %%
import argparse
import dataclasses
import glob
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


# %% [markdown]
# ## 1. Инициализация окружения, секретов и линейной алгебры

# %%
def setup_kaggle_secrets(is_kaggle: bool) -> None:
    """Подгружает секреты WandB и GCP из Kaggle UserSecretsClient."""
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
                res = subprocess.run(
                    ["gcloud", "auth", "activate-service-account", f"--key-file={sa_path}"],
                    capture_output=True,
                    text=True,
                )
                if res.returncode == 0:
                    print("[Kaggle Secrets] GCP Service Account успешно авторизован.", flush=True)
                else:
                    print(f"[Kaggle Secrets Warning] Ошибка авторизации gcloud: {res.stderr.strip()}", flush=True)
        except Exception:
            pass
    except ImportError:
        pass
    except Exception as e:
        print(f"[Kaggle Secrets Warning] Ошибка UserSecretsClient: {e}", flush=True)


def warmup_linalg_backends() -> None:
    """Последовательный прогрев cuSOLVER на всех GPU до запуска DataParallel/DDP."""
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        print(f"[Linalg Warmup] Последовательный прогрев cuSOLVER на {num_devices} GPU...", flush=True)
        for dev_idx in range(num_devices):
            dev = torch.device(f"cuda:{dev_idx}")
            dummy = torch.eye(32, device=dev, dtype=torch.float32)
            try:
                _ = torch.linalg.eigh(dummy)
            except Exception as e:
                print(f"[Linalg Warmup Warning] GPU {dev_idx}: {e}", flush=True)
        print("[Linalg Warmup] cuSOLVER инициализирован на всех GPU.", flush=True)


def resolve_project_root() -> str:
    """Определяет корень репозитория BEBLaDII для локального запуска и Kaggle."""
    if os.path.exists("/kaggle") or ("KAGGLE_KERNEL_RUN_TYPE" in os.environ):
        kaggle_root = "/kaggle/working/BEBLaDII"
        if os.path.exists(kaggle_root):
            return kaggle_root
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
    """Единая конфигурация путей, устройств и режимов обучения Фазы 2."""
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
    device: torch.device
    use_amp: bool
    amp_dtype: torch.dtype
    num_workers: int
    pin_memory: bool

    @classmethod
    def detect(
        cls,
        override_device: str = "auto",
        local_rank: int = 0,
        override_output_dir: str = "",
        override_checkpoint_dir: str = "",
    ) -> "EnvironmentConfig":
        is_kaggle = os.path.exists("/kaggle") or ("KAGGLE_KERNEL_RUN_TYPE" in os.environ)
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
                    os.path.join(project_root, "BEBLaDII-plan\u0421-Phase2-Data", "data", "train_phase2.parquet"),
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
                    os.path.join(project_root, "BEBLaDII-plan\u0421-Phase2-Data", "data", "val_phase2.parquet"),
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
                    "/kaggle/input/datasets/bogdanbuliakov/bebladii-planc-phase1-data/storage/components/encoder/weights.pt",
                    "/kaggle/input/bebladii-planc-phase1-data/storage/components/encoder/weights.pt",
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
                    "/kaggle/input/datasets/bogdanbuliakov/bebladii-planc-phase2-data/decoder/weights.pt",
                    "/kaggle/input/bebladii-planc-phase2-data/decoder/weights.pt",
                    os.path.join(project_root, "storage", "components", "decoder", "weights.pt"),
                ],
                decoder_search_roots,
            )
            wandb_mode = "online" if "WANDB_API_KEY" in os.environ else "disabled"
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

        sep_concept_path = os.path.join(project_root, "storage", "components", "sep_concept.pt")
        void_concept_path = os.path.join(project_root, "storage", "components", "void_concept.pt")

        if override_device == "cpu":
            device = torch.device("cpu")
        elif override_device == "cuda":
            device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        use_amp = device.type == "cuda"
        # На Kaggle 2xT4 (sm_75) отсутствует аппаратный bf16 -> используем FP16 + GradScaler (ADR 126)
        amp_dtype = torch.float16 if use_amp else torch.float32
        num_workers = 2 if is_kaggle else 0
        pin_memory = device.type == "cuda"

        if override_output_dir:
            output_dir = override_output_dir
        if override_checkpoint_dir:
            checkpoint_dir = override_checkpoint_dir

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        if not os.path.isdir(checkpoint_dir):
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
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            num_workers=num_workers,
            pin_memory=pin_memory,
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
SIGMA_NOISE = 0.05
SIGMA_DATA = SIGMA_NOISE  # Алиас обратной совместимости (std изотропного шума = 0.05)
DEFAULT_AXIS_SIGMA: Tuple[float, ...] = (
    0.08319, 0.06884, 0.05511, 0.04440, 0.03341, 0.02695, 0.02075, 0.01697,
)
N_PROMPT = 64
N_ANS = 127
N_CANVAS = 128
T_PROMPT = N_PROMPT * 4  # 256 токенов
T_ANS = N_ANS * 4        # 508 токенов
TAU_ADJ = 0.85
TAU_BARY = 0.50
VOID_ENERGY_THR = 0.35
VOID_SIM_THR = 0.80


def compute_axis_t(t_mat: torch.Tensor, delta: float = 0.0, num_axes: int = 8) -> torch.Tensor:
    """
    Поосевое расписание SD3 log-SNR shift (§2.1):
      t_k(t) = (s_k * t) / (1 + (s_k - 1) * t),  s_k = exp(linspace(-delta, +delta, 8)).
    Сохраняет точные границы t_k(0) = 0, t_k(1) = 1.
    Вход: t_mat произвольной формы (*S), выход: (*S, num_axes) в float32.
    """
    tf = t_mat.float().unsqueeze(-1)
    if abs(float(delta)) <= 1e-8:
        return tf.expand(*t_mat.shape, num_axes)
    s = torch.exp(
        torch.linspace(-float(delta), float(delta), num_axes, device=t_mat.device, dtype=torch.float32)
    )
    while s.ndim < tf.ndim:
        s = s.unsqueeze(0)
    return (s * tf) / (1.0 + (s - 1.0) * tf).clamp(min=1e-6)


def add_axis_shifted_noise(
    V_0: torch.Tensor,
    t_mat: torch.Tensor,
    noise_eps: torch.Tensor,
    delta: float = 0.0,
    sigma_noise: float = SIGMA_NOISE,
) -> torch.Tensor:
    """Зашумление V_0 с опциональным поосевым SD3-сдвигом t_k(t) при стандартном sigma_noise=0.05."""
    t_ax = compute_axis_t(t_mat, delta=delta, num_axes=V_0.shape[-2]).unsqueeze(-1)
    return (1.0 - t_ax) * V_0.float() + t_ax * (noise_eps.float() * float(sigma_noise))


def sim_gr(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Строго нормированное O(8)-инвариантное косинусное сходство операторов плотности:
    Sim_Gr(A, B) = ||A B^T||_F^2 / (||A A^T||_F * ||B B^T||_F + eps) in [0, 1].
    Всегда выполняется в float32 для защиты от переполнения FP16 (ADR 123, 126).
    """
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
    """
    Латентный детектор VOID-ячеек до вызова декодера (§1.3):
    is_void(i) <=> (||V_i||_F^2 < 0.35) or (Sim_Gr(V_i, V_void) > 0.80).
    """
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
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    active_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Быстрый расчёт поклеточного показания NoiseRadarHead (Выход А декодера Фазы 2) для датчика s7:
    Возвращает radar_sigma (B, N) в float32 в [0, 1], причём для якоря SEP (pos 0) всегда 0.0.
    При передаче `active_mask` (B, N) на GPU/CPU прогоняет тяжёлый 131K-словарь `compute_dict_stats`
    строго по незафиксированным (`active_mask[:, 1:] == True`) клеткам (§7.4).
    """
    B, N, _, _ = V_t.shape
    radar_full = torch.zeros((B, N), device=V_t.device, dtype=torch.float32)
    if decoder is None or N <= 1 or not hasattr(decoder, "estimate_noise"):
        return radar_full
    if getattr(decoder, "noise_radar", None) is None:
        return radar_full

    dec_dtype = next(decoder.parameters()).dtype
    v_ans = V_t[:, 1:].to(dtype=amp_dtype if use_amp else dec_dtype)

    if (
        active_mask is not None
        and V_t.device.type != "xla"
        and hasattr(decoder, "decode_stage1")
        and getattr(decoder, "noise_radar", None) is not None
    ):
        ans_act = active_mask[:, 1:].bool()  # (B, N_a)
        n_act = int(ans_act.sum().item())
        if n_act == 0:
            return radar_full
        n_tot = int(ans_act.numel())
        if n_act < int(0.95 * n_tot):
            with torch.autocast(device_type=V_t.device.type, dtype=amp_dtype, enabled=use_amp):
                stage1 = decoder.decode_stage1(v_ans)
            nr = decoder.noise_radar
            N_a = N - 1
            cs = int(getattr(nr, "chunk_size", 4))
            h_unp = stage1["h_unpacked"]  # (B, N_a*cs, D)
            D_h = h_unp.shape[-1]
            h_act = h_unp.view(B, N_a, cs, D_h)[ans_act].reshape(1, n_act * cs, D_h)
            ds_act = nr.compute_dict_stats(
                h_act,
                getattr(decoder, "lm_head_weight", None),
                getattr(decoder, "lm_head_inv_norms", None),
            ).view(n_act, cs, 4)

            with torch.autocast(device_type=V_t.device.type, enabled=False):
                eig_act = stage1["eigvals"][ans_act].float()
                h_cell_act = stage1["h_cell"][ans_act].float()
                energy = eig_act.sum(dim=-1, keepdim=True)
                purity = (eig_act ** 2).sum(dim=-1, keepdim=True)
                inv_rk = purity / (energy ** 2 + 1e-6)
                cell_hidden_feat = F.silu(nr.cell_proj(h_cell_act))
                cell_base = torch.cat([eig_act, energy, purity, inv_rk, cell_hidden_feat], dim=-1)
                dict_mean = ds_act.mean(dim=1)[..., :3]
                dict_min = ds_act.min(dim=1).values[..., :3]
                cell_in = torch.cat([cell_base, dict_mean, dict_min], dim=-1)
                sig_act = torch.sigmoid(nr.cell_mlp(cell_in).squeeze(-1)).clamp(0.0, 1.0)

            sig_ans = torch.zeros((B, N_a), device=V_t.device, dtype=torch.float32)
            sig_ans[ans_act] = sig_act
            radar_full[:, 1:] = sig_ans
            return radar_full

    with torch.autocast(device_type=V_t.device.type, dtype=amp_dtype, enabled=use_amp):
        sigma_cell, _, _ = decoder.estimate_noise(v_ans)
    radar_full[:, 1:] = sigma_cell.float().clamp(0.0, 1.0)
    return radar_full


class PerHeadRMSNorm(nn.Module):
    """По-головочный RMSNorm (8 x 128) с расчетом в float32 для QK-norm."""

    def __init__(self, num_heads: int = 8, head_dim: int = 128, eps: float = 1e-5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_heads, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, L, D)
        x_f = torch.nan_to_num(x.float(), nan=0.0, posinf=65000.0, neginf=-65000.0)
        var = torch.mean(x_f ** 2, dim=-1, keepdim=True)
        normed = x_f * torch.rsqrt(var + self.eps)
        w = self.weight.float().view(1, self.num_heads, 1, self.head_dim)
        return (normed * w).to(x.dtype)


# %% [markdown]
# ## 3. Архитектура Фазы 2: `ConditionEmbedder`, `PromptAdapter`, `DiffusionCore`, `OutputHead`

# %%
class SinusoidalTimeEmbedding(nn.Module):
    """Синусоидальное кодирование скалярного/поячеечного уровня шума t in [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = t.float().unsqueeze(-1) * freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FourierTimeEmbedding(nn.Module):
    """
    Log-spaced Fourier features времени t in [0, 1] (§3.1, EDM2-стиль):
    32 частоты от 0.5 до 16 циклов на [0, 1] -> [sin(w*t), cos(w*t)] в R^64.
    """

    def __init__(self, dim: int = 64, min_log2_cycles: float = -1.0, max_log2_cycles: float = 4.0):
        super().__init__()
        self.dim = int(dim)
        half = self.dim // 2
        freqs = 2.0 * math.pi * (
            2.0 ** torch.linspace(min_log2_cycles, max_log2_cycles, half, dtype=torch.float32)
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = t.float().unsqueeze(-1) * self.freqs.to(device=t.device, dtype=torch.float32)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ConditionEmbedderV2(nn.Module):
    """
    Усовершенствованный ConditionEmbedder (Этап 2 / §§3.1–3.2):
    - FourierTimeEmbedding(64) по глобальному t_bar (0.5..16 циклов);
    - 19 диагностических датчиков в float32 (0 новых matmul):
      [s0, s1, s3, s4, s5, s6, s7, offdiag_ratio, axis_energies(8), block_energies(3)];
    - MLP: 83 -> 256 -> SiLU -> 256 -> LayerNorm(256).
    """

    SENSOR_NAMES: Tuple[str, ...] = (
        "s0_energy", "s1_purity", "s3_void_sim", "s4_radar_clean",
        "s5_m_prompt", "s6_cryst_frac", "s7_radar_sigma", "offdiag_ratio",
        "ax_e0", "ax_e1", "ax_e2", "ax_e3", "ax_e4", "ax_e5", "ax_e6", "ax_e7",
        "blk_sem", "blk_syn", "blk_pos",
    )

    def __init__(self, void_ref: torch.Tensor, cond_dim: int = 256, fourier_dim: int = 64):
        super().__init__()
        self.fourier_dim = int(fourier_dim)
        self.sensors_dim = 19
        self.register_buffer("void_ref", void_ref.float().clone())
        self.t_global_emb = FourierTimeEmbedding(self.fourier_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.fourier_dim + self.sensors_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.LayerNorm(cond_dim),
        )
        self._capture_sensors: bool = False
        self._captured_sensors: List[torch.Tensor] = []

    def forward(
        self,
        V_t: torch.Tensor,
        t_mat: torch.Tensor,
        m_prompt: torch.Tensor,
        radar_sigma: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, K, D = V_t.shape
        device_type = V_t.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            Vf = V_t.float()
            mp = m_prompt.float()
            t_bar = ConditionEmbedder.extract_t_global(t_mat, B, N)

            G = torch.matmul(Vf, Vf.transpose(-2, -1))  # (B, N, 8, 8)
            diag_g = torch.diagonal(G, dim1=-2, dim2=-1)  # (B, N, 8)
            energy = diag_g.sum(dim=-1)
            g_frob_sq = (G ** 2).sum(dim=(-2, -1))
            s0 = torch.clamp(energy / 2.52, max=2.0)
            s1 = g_frob_sq / ((energy + 1e-6) ** 2)

            void_exp = self.void_ref.view(1, 1, K, D).expand(B, N, K, D)
            s3 = sim_gr(Vf, void_exp)

            if radar_sigma is not None:
                s7 = radar_sigma.float().clamp(0.0, 1.0)
                if N > 0:
                    s7 = s7.clone()
                    s7[:, 0] = 0.0
                s4 = torch.clamp(1.0 - s7 / SIGMA_NOISE, min=0.0, max=1.0)
            else:
                s7 = torch.zeros_like(s0)
                s4 = torch.zeros_like(s0)
            if N > 0:
                s4 = s4.clone()
                s4[:, 0] = 1.0
            s5 = mp.view(B, 1).expand(B, N)
            s6 = s4[:, 1:].mean(dim=-1, keepdim=True).expand(B, N) if N > 1 else s4

            # Внедиагональная неортогональность (1 скаляр)
            diag_sq_sum = (diag_g ** 2).sum(dim=-1)
            offdiag_sq = torch.clamp(g_frob_sq - diag_sq_sum, min=0.0)
            offdiag_ratio = torch.sqrt(offdiag_sq + 1e-12) / (torch.sqrt(g_frob_sq + 1e-12) + 1e-6)

            # Поосевые энергии diag(G)_k (8 скаляров)
            axis_energies = torch.log1p(torch.clamp(diag_g, min=0.0))  # (B, N, 8)

            # Доли энергии по 3 блокам [0:96 sem | 96:112 syn | 112:128 pos] (3 скаляра)
            e_denom = energy + 1e-6
            b_sem = torch.clamp((Vf[..., 0:96] ** 2).sum(dim=(-2, -1)) / e_denom, 0.0, 1.0)
            b_syn = torch.clamp((Vf[..., 96:112] ** 2).sum(dim=(-2, -1)) / e_denom, 0.0, 1.0)
            b_pos = torch.clamp((Vf[..., 112:128] ** 2).sum(dim=(-2, -1)) / e_denom, 0.0, 1.0)

            base_7 = torch.stack([s0, s1, s3, s4, s5, s6, s7, offdiag_ratio], dim=-1)  # (B, N, 8)
            blk_3 = torch.stack([b_sem, b_syn, b_pos], dim=-1)                          # (B, N, 3)
            sensors = torch.cat([base_7, axis_energies, blk_3], dim=-1)                 # (B, N, 19)

            if self._capture_sensors:
                self._captured_sensors.append(sensors.detach().reshape(-1, self.sensors_dim).cpu())

            e_glob = self.t_global_emb(t_bar)
            feat = torch.cat([e_glob, sensors], dim=-1)
            c_f32 = self.mlp(feat)
        return c_f32.to(V_t.dtype)

    @torch.no_grad()
    def compute_sensor_importance(self) -> Dict[str, float]:
        """Возвращает ||W[:, 64+j]||_2 * std_val(s_j) по накопленным val-датчикам (§3.3)."""
        if not self._captured_sensors:
            return {}
        s_all = torch.cat(self._captured_sensors, dim=0).float()
        s_std = s_all.std(dim=0)
        w_sens = self.mlp[0].weight.detach().float()[:, self.fourier_dim :]  # (cond_dim, 19)
        w_norms = w_sens.norm(p=2, dim=0).cpu()
        imp = w_norms * s_std
        return {name: float(imp[i].item()) for i, name in enumerate(self.SENSOR_NAMES)}


class AxisBlockAdaLN(nn.Module):
    """
    Компактная структурная AdaLN (Вариант D2, §3.3):
      c (B, N, 256) -> Linear(256, r) -> SiLU ->
        up_dim: (num_mods * 128)  [поканальная подстройка в R^128]
        up_ab:  (num_mods * 8 * 3) [управление парами (ось k=0..7, блок sem/syn/pos)]
        up_extra (опционально): дополнительные скаляры (например, 16 для OutputHeadV2).
    Сборка (без седловой точки и без искусственного сдвига вдоль 1_128):
      scale = (1 + sc_dim) * (1 + sc_ab) - 1
      shift = sh_dim * (1 + sh_ab)
      gate  = g_ab * (1 + g_dim)   [bias(g_ab) = gate_bias = 0.02]
    """

    def __init__(
        self,
        cond_dim: int = 256,
        num_mods: int = 6,
        bottleneck_r: int = 64,
        gate_indices: Tuple[int, ...] = (2, 5),
        gate_bias: float = 0.02,
        extra_dim: int = 0,
    ):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.num_mods = int(num_mods)
        self.bottleneck_r = int(bottleneck_r)
        self.gate_indices = tuple(gate_indices)
        self.extra_dim = int(extra_dim)
        self.in_features = self.cond_dim  # совместимость с проверками layer.adaln_proj.in_features

        self.down = nn.Linear(self.cond_dim, self.bottleneck_r)
        self.up_dim = nn.Linear(self.bottleneck_r, self.num_mods * 128)
        self.up_ab = nn.Linear(self.bottleneck_r, self.num_mods * 8 * 3)
        nn.init.normal_(self.up_dim.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.up_dim.bias)
        nn.init.normal_(self.up_ab.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.up_ab.bias)
        with torch.no_grad():
            ab_bias = self.up_ab.bias.view(self.num_mods, 8, 3)
            for gi in self.gate_indices:
                if 0 <= gi < self.num_mods:
                    ab_bias[gi].fill_(float(gate_bias))

        if self.extra_dim > 0:
            self.up_extra: Optional[nn.Linear] = nn.Linear(self.bottleneck_r, self.extra_dim)
            nn.init.normal_(self.up_extra.weight, mean=0.0, std=1e-4)
            nn.init.zeros_(self.up_extra.bias)
        else:
            self.up_extra = None

        self._capture_hr: bool = False
        self._captured_hr: List[torch.Tensor] = []

    @staticmethod
    def _expand_blocks(ab: torch.Tensor) -> torch.Tensor:
        # ab: (..., 8, 3) -> (..., 8, 128) по границам [96 sem | 16 syn | 16 pos]
        b0 = ab[..., 0:1].expand(*ab.shape[:-1], 96)
        b1 = ab[..., 1:2].expand(*ab.shape[:-1], 16)
        b2 = ab[..., 2:3].expand(*ab.shape[:-1], 16)
        return torch.cat([b0, b1, b2], dim=-1)

    def forward(
        self, c: torch.Tensor, out_dtype: torch.dtype
    ) -> Union[Tuple[torch.Tensor, ...], Tuple[Tuple[torch.Tensor, ...], torch.Tensor]]:
        B, N, _ = c.shape
        h_r = F.silu(self.down(c.to(self.down.weight.dtype)))  # (B, N, r)
        if self._capture_hr:
            self._captured_hr.append(h_r.detach().float().reshape(-1, self.bottleneck_r).cpu())

        m_dim = self.up_dim(h_r).view(B, N, self.num_mods, 1, 128).to(out_dtype)
        m_ab_raw = self.up_ab(h_r).view(B, N, self.num_mods, 8, 3).to(out_dtype)
        m_ab = self._expand_blocks(m_ab_raw)  # (B, N, num_mods, 8, 128)

        outs: List[torch.Tensor] = []
        if self.num_mods == 6:
            # Порядок: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
            for idx, kind in enumerate(("shift", "scale", "gate", "shift", "scale", "gate")):
                d_i = m_dim[:, :, idx]  # (B, N, 1, 128)
                a_i = m_ab[:, :, idx]   # (B, N, 8, 128)
                if kind == "shift":
                    outs.append(d_i * (1.0 + a_i))
                elif kind == "scale":
                    outs.append((1.0 + d_i) * (1.0 + a_i) - 1.0)
                else:
                    outs.append(a_i * (1.0 + d_i))
        elif self.num_mods == 2:
            # Порядок: shift, scale
            outs.append(m_dim[:, :, 0] * (1.0 + m_ab[:, :, 0]))
            outs.append((1.0 + m_dim[:, :, 1]) * (1.0 + m_ab[:, :, 1]) - 1.0)
        else:
            for idx in range(self.num_mods):
                outs.append(m_dim[:, :, idx] + m_ab[:, :, idx])

        if self.up_extra is not None:
            extra = self.up_extra(h_r).to(out_dtype)
            return tuple(outs), extra
        return tuple(outs)

    @torch.no_grad()
    def compute_rank_stats(self) -> Dict[str, float]:
        """Эффективный ранг r_eff, k_99, tail_ratio и dead_frac по накопленным h_r (§3.3)."""
        if not self._captured_hr:
            return {}
        H = torch.cat(self._captured_hr, dim=0).float()  # (M, r)
        if H.shape[0] < 4:
            return {}
        Hc = H - H.mean(dim=0, keepdim=True)
        cov = (Hc.T @ Hc) / max(1, H.shape[0] - 1)
        eigvals = torch.linalg.eigvalsh(cov).clamp(min=0.0).flip(dims=[0])
        total = eigvals.sum().clamp(min=1e-12)
        p = (eigvals / total).clamp(min=1e-12)
        r_eff = float(torch.exp(-(p * torch.log(p)).sum()).item())
        cum = torch.cumsum(eigvals / total, dim=0)
        k_99 = int((cum < 0.99).sum().item()) + 1
        k_99 = min(k_99, self.bottleneck_r)
        q_tail = max(1, self.bottleneck_r // 4)
        tail_ratio = float((eigvals[-q_tail:].sum() / total).item())
        dead_frac = float((H.abs().mean(dim=0) < 1e-3).float().mean().item())
        return {
            "r_eff": r_eff,
            "k_99": float(k_99),
            "tail_ratio": tail_ratio,
            "dead_frac": dead_frac,
        }


class ConditionEmbedder(nn.Module):
    """
    Формирует вектор управления c_i in R^{B x N x 256}:
    только управляющий глобальный шум t_global (64) + 8 чисто диагностических датчиков s0..s7 в float32.
    Оракульный поячеечный t_local удален: модель обязана диагностировать чистоту каждой ячейки по V_t и s0..s7.
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
        """Извлекает единый для всего полотна управляющий скаляр t_global in [0, 1] формы (B, N)."""
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
        device_type = V_t.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            Vf = V_t.float()
            mp = m_prompt.float()
            t_bar = self.extract_t_global(t_mat, B, N)

            G = torch.matmul(Vf, Vf.transpose(-2, -1))  # (B, N, 8, 8)
            energy = torch.diagonal(G, dim1=-2, dim2=-1).sum(dim=-1)
            s0 = torch.clamp(energy / 2.52, max=2.0)
            s1 = (G ** 2).sum(dim=(-2, -1)) / ((energy + 1e-6) ** 2)
            s2 = 1.0 / (8.0 * torch.clamp(s1, min=1e-4))
            void_exp = self.void_ref.view(1, 1, 8, 128).expand(B, N, 8, 128)
            s3 = sim_gr(Vf, void_exp)
            if radar_sigma is not None:
                s7 = radar_sigma.float().clamp(0.0, 1.0)
                if N > 0:
                    s7 = s7.clone()
                    s7[:, 0] = 0.0
                # Диагностический индикатор чистоты кристалла строго по радару s7 (без оракула t_local)
                s4 = torch.clamp(1.0 - s7 / 0.05, min=0.0, max=1.0)
            else:
                s7 = torch.zeros_like(s0)
                s4 = torch.zeros_like(s0)
            if N > 0:
                s4 = s4.clone()
                s4[:, 0] = 1.0
            s5 = mp.view(B, 1).expand(B, N)
            if N > 1:
                # Средняя диагностическая доля кристаллизации полотна по датчику s4
                s6 = s4[:, 1:].mean(dim=-1, keepdim=True).expand(B, N)
            else:
                s6 = s4
            sensors = torch.stack([s0, s1, s2, s3, s4, s5, s6, s7], dim=-1)

            e_glob = self.t_global_emb(t_bar)
            feat = torch.cat([e_glob, sensors], dim=-1)
            c_f32 = self.mlp(feat)
        return c_f32.to(V_t.dtype)


def _clamp_swiglu(gate_act: torch.Tensor, up_val: torch.Tensor) -> torch.Tensor:
    """Безопасное произведение SwiGLU с клампом [-65000, 65000] для FP16."""
    prod = F.silu(gate_act.float()) * up_val.float()
    return torch.clamp(prod, -65000.0, 65000.0).to(up_val.dtype)


def _clamp_fp16_safe(x: torch.Tensor) -> torch.Tensor:
    """Клэмп выхода down_proj перед умножением на gate_mlp (ADR 127 / §6.2)."""
    return torch.clamp(x.float(), -65000.0, 65000.0).to(x.dtype)


class PromptAdapterLayer(nn.Module):
    """
    Один слой PromptAdapter: Full SA (Shared KV + Split K/V Channel Gating) + SwiGLU FFN,
    без AdaLN(t), с обучаемым LayerScale (alpha_0 = 0.02).
    """

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
        v = u * self.v_gate.to(u.dtype)  # V никогда не вращается через RoPE

        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, L, C)
        x = x + self.ls_attn.to(x.dtype) * self.o_proj(attn_out)

        h_mlp = self.mlp_norm(x)
        gated = _clamp_swiglu(self.gate_proj(h_mlp), self.up_proj(h_mlp))
        mlp_out = _clamp_fp16_safe(self.down_proj(gated))
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
            # На случай полностью пустого промпта оставляем pos 0 открытой для корректного softmax
            safe_mask = prompt_valid_mask.clone()
            empty_rows = ~safe_mask.any(dim=-1)
            if empty_rows.any():
                safe_mask[empty_rows, 0] = True
            attn_mask = torch.zeros(B, 1, 1, L, device=x.device, dtype=x.dtype)
            attn_mask = attn_mask.masked_fill(~safe_mask[:, None, None, :], -10000.0)

        for layer in self.layers:
            x = layer(x, cos, sin, attn_mask=attn_mask)
        return x


def _init_soft_gate_adaln(proj: nn.Linear, hidden_size: int, gate_bias: float = 0.02) -> None:
    """Инициализация Soft-Gate AdaLN: weight ~ N(0, 1e-8), shift/scale bias=0, gate bias=0.02."""
    nn.init.normal_(proj.weight, mean=0.0, std=1e-4)
    nn.init.zeros_(proj.bias)
    with torch.no_grad():
        proj.bias[2 * hidden_size : 3 * hidden_size] = gate_bias
        proj.bias[5 * hidden_size : 6 * hidden_size] = gate_bias


def _compute_adaln_stats(
    M: torch.Tensor,
    m_active: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Возвращает компактные статистики модуляций M in R^{B x N x 4096} по зашумленным позициям:
    (sum_M: [1, 4096], sum_M2: [1, 4096], count: [1]) для безопасного gather в DataParallel/DDP.
    """
    m_flat = M[m_active].float()
    if m_flat.numel() == 0:
        dev = M.device
        d = M.size(-1)
        zero = (M.sum() * 0.0).float()
        return (
            torch.zeros(1, d, device=dev, dtype=torch.float32) + zero,
            torch.zeros(1, d, device=dev, dtype=torch.float32) + zero,
            torch.zeros(1, device=dev, dtype=torch.float32),
        )
    sum_m = m_flat.sum(dim=0, keepdim=True)
    sum_m2 = (m_flat ** 2).sum(dim=0, keepdim=True)
    count = torch.tensor([float(m_flat.shape[0])], device=M.device, dtype=torch.float32)
    return sum_m, sum_m2, count


class DiffusionSABlock(nn.Module):
    """
    Блок Self-Attention ядра (Full SA или Local SA r=8, W=17 + Global Pos 0)
    с Shared KV, Split K/V Channel Gating, SwiGLU FFN и Soft-Gate AdaLN.
    """

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
        stats = _compute_adaln_stats(M, m_active)

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
        mlp_out = _clamp_fp16_safe(self.down_proj(gated))
        x = x + gate_mlp * mlp_out
        return x, stats


class CAPromptBlock(nn.Module):
    """
    Слои 2, 4, 6: Cross-Attention на промпт с:
    - Shared KV + Split K/V Channel Gating;
    - Token-Centered Contrast только на ветке ключей K_p (ADR 118);
    - Фиксированным RoPE offset=64 для запросов холста (ADR 108);
    - Zero-Vector Attention Sink k_sink (без RoPE, v_sink=0);
    - Cross-Gated SwiGLU FFN (4 SphereLinear матрицы) + Soft-Gate AdaLN.
    """

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

        # Zero-Vector Attention Sink: обучаемый ключ, фиксированное нулевое значение (ADR 118)
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
        stats = _compute_adaln_stats(M, m_active)

        h_ans_in = self.ca_norm(x_ans) * (1.0 + scale_ca) + shift_ca
        h_prompt_in = self.prompt_norm(x_prompt)

        q = self.q_proj(h_ans_in).view(B, N_a, self.num_heads, self.head_dim).transpose(1, 2)
        u_p = self.kv_proj(h_prompt_in).view(B, N_p, self.num_heads, self.head_dim).transpose(1, 2)

        # Token-Centered Contrast только на ветке ключей K_p по активным (не-VOID) клеткам промпта
        mask_p_f = prompt_attn_mask.float().view(B, 1, N_p, 1)
        denom = mask_p_f.sum(dim=2, keepdim=True).clamp(min=1.0)
        u_p_mean = (u_p.float() * mask_p_f).sum(dim=2, keepdim=True) / denom
        u_p_centered = (u_p.float() - 0.5 * u_p_mean).to(u_p.dtype)

        q = self.q_norm(q)
        k_p = self.k_norm(u_p_centered)
        v_p = u_p * self.v_gate.to(u_p.dtype)

        # RoPE с фиксированным offset=64 для запросов ответа и [0..N_p-1] для ключей промпта
        cos_all, sin_all = rope(x_ans, self.rope_offset + N_a)
        cos_p, sin_p = cos_all[:N_p], sin_all[:N_p]
        cos_a, sin_a = cos_all[self.rope_offset : self.rope_offset + N_a], sin_all[self.rope_offset : self.rope_offset + N_a]

        q, _ = apply_rotary_pos_emb(q, q, cos_a, sin_a)
        _, k_p = apply_rotary_pos_emb(k_p, k_p, cos_p, sin_p)

        # Добавляем k_sink (без RoPE); v_sink тождественно равен 0 (Zero-Vector Sink)
        k_sink_exp = self.k_norm(self.k_sink.to(k_p.dtype)).expand(B, -1, -1, -1)
        k_full = torch.cat([k_p, k_sink_exp], dim=2)  # (B, H, N_p + 1, D)

        # Маска: открыты только активные не-VOID клетки промпта и всегда открыт k_sink (последний столбец)
        attn_mask = torch.zeros(B, 1, 1, N_p + 1, device=x_ans.device, dtype=torch.float32)
        attn_mask[:, :, :, :N_p] = attn_mask[:, :, :, :N_p].masked_fill(
            ~prompt_attn_mask[:, None, None, :], -10000.0
        )

        # Однократный расчёт внимания в float32 (исключает повторный проход SDPA и защищает от FP16 underflow)
        scores = torch.matmul(q.float(), k_full.float().transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + attn_mask
        probs = F.softmax(scores, dim=-1)
        sink_weights = probs[:, :, :, -1].mean(dim=1) if return_sink_weights else None

        # Так как v_sink == 0, вклад в контекст дают только первые N_p ключей промпта
        attn_out = torch.matmul(probs[:, :, :, :N_p].to(v_p.dtype), v_p)
        attn_out = attn_out.transpose(1, 2).reshape(B, N_a, C)
        h_ca = self.o_proj(attn_out)
        x_ans = x_ans + gate_ca * h_ca

        # Cross-Gated SwiGLU FFN: val строго из h_ca, gate смешивает h_ca и состояние холста
        x_tilde_ans = self.mlp_norm(x_ans) * (1.0 + scale_mlp) + shift_mlp
        gate_pre = self.gate_proj(h_ca).float() + self.w_canvas_gate(x_tilde_ans).float()
        prod = F.silu(gate_pre) * self.up_proj(h_ca).float()
        gated = torch.clamp(prod, -65000.0, 65000.0).to(x_ans.dtype)
        ffn_out = _clamp_fp16_safe(self.down_proj(gated))
        x_ans = x_ans + gate_mlp * ffn_out

        return x_ans, stats, sink_weights


class GrassmannianOutputHead(nn.Module):
    """
    Выходная голова:
    V_0_raw = V_t + 0.05 * g(t_i) * out_proj(norm(X) * (1 + scale_out) + shift_out),
    с последующим построчным сжатием в единичный шар B^128 в float32.
    """

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
        g_t = torch.clamp(t_bar / 0.02, min=0.0, max=1.0)
        if N > 0:
            g_t = g_t.clone()
            g_t[:, 0] = 0.0  # Точное тождество на ячейке 0 (SEP) без подглядывания в поячеечный t_local
        g_t = g_t.unsqueeze(-1).unsqueeze(-1)
        V_0_raw = V_t.float() + SIGMA_DATA * g_t * delta_x.float()
        return clamp_to_unit_ball(V_0_raw)


class GrassmannianDiffusionCore(nn.Module):
    """
    Полное условное ядро диффузии Фазы 2 Plan C (~167.64M обучаемых параметров).
    Включает PromptAdapter (2 слоя), ConditionEmbedder, 10 слоев ядра (CAPromptBlock на слоях 2, 4, 6)
    и GrassmannianOutputHead.
    """

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

        # Порядок слоев: 0:Full, 1:Local, 2:CA, 3:Full, 4:CA, 5:Full, 6:CA, 7:Full, 8:Local, 9:Full
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
        """Однократное кодирование промпта через PromptAdapter (кэшируется при инференсе)."""
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
        """
        Прямой проход ядра с уже вычисленным x_prompt.
        Возвращает (V_0_hat, (sum_M_stack, sum_M2_stack, count_vec), sink_weights (B, N_ca, N_a)).
        """
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
        sink_weights = torch.stack(sink_list, dim=1) if sink_list else None  # (B, N_ca, N_a)

        # Собираем в единые тензоры: (num_gpus, num_layers, 4096) после DataParallel gather
        num_layers = len(self.layers)
        sum_m_pack = torch.stack(sum_m_list, dim=1)    # (1, num_layers, 4096)
        sum_m2_pack = torch.stack(sum_m2_list, dim=1)  # (1, num_layers, 4096)
        count_pack = torch.stack(count_list, dim=0).view(1, num_layers)  # (1, num_layers)
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
# ## 4. Датасет `(Prompt, Answer)`, Граничные ячейки и Mixed Noise Curriculum

# %%
def encode_token_ids(tokenizer: Any, text: str) -> List[int]:
    """Унифицированная токенизация без добавления служебных BOS/EOS."""
    if tokenizer is None:
        return [100 + (ord(ch) % 1000) for ch in text[:256]]
    out = tokenizer.encode(text, add_special_tokens=False)
    return out.ids if hasattr(out, "ids") else list(out)


def build_punctuation_id_set(tokenizer: Optional[Any]) -> set:
    """Множество token_id со знаками завершения мысли (. ! ? \\n) для аккуратной обрезки."""
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


def make_pooling_weights(input_ids: torch.Tensor, stopword_mask: torch.Tensor) -> torch.Tensor:
    """Веса спектрального пулинга Фазы 1 (1.0 / 0.02 / 0.001, по-оконная нормировка)."""
    is_pad = input_ids == PAD_TOKEN_ID
    valid_idx = torch.clamp(input_ids, 0, len(stopword_mask) - 1)
    is_stop = stopword_mask[valid_idx] & (~is_pad)
    raw_w = torch.where(
        is_pad,
        torch.tensor(0.001, dtype=torch.float32, device=input_ids.device),
        torch.where(
            is_stop,
            torch.tensor(0.02, dtype=torch.float32, device=input_ids.device),
            torch.tensor(1.0, dtype=torch.float32, device=input_ids.device),
        ),
    )
    B, T = input_ids.shape
    raw_w = raw_w.view(B, T // 4, 4)
    norm_w = raw_w / raw_w.sum(dim=-1, keepdim=True)
    return norm_w.view(B, T)


class PromptAnswerDataset(Dataset):
    """
    Датасет пар (Q, A) с WordAwareElasticChunker (ADR 137):
    - Промпт Q: упаковывается по целым словам (1..4 токена) в N_PROMPT=64 клетки (256 слотов), elastic_drop_prob=0.0.
    - Ответ A: упаковывается по целым словам в N_ANS=127 клеток (508 слотов) с опциональным мягким
      elastic_drop_prob (по умолчанию 0.05 на train, 0.0 на val) и двухуровневыми весами пулера.
    """

    def __init__(
        self,
        parquet_path: Optional[str],
        tokenizer: Any,
        stopword_mask: torch.Tensor,
        max_prompt_tokens: int = T_PROMPT,
        max_ans_tokens: int = T_ANS,
        max_rows: Optional[int] = None,
        is_validation: bool = False,
        synthetic_len: int = 200,
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
            prefix = "[Val Dataset]" if is_validation else "[Train Dataset]"
            print(f"{prefix} Загрузка пар (Q, A) из: {parquet_path} ...", flush=True)
            df = pd.read_parquet(parquet_path, columns=["Q", "A"])
            if max_rows is not None and len(df) > max_rows:
                df = df.head(max_rows)
            df = df.dropna(subset=["Q", "A"])
            self.pairs = [
                (str(q), str(a))
                for q, a in zip(df["Q"].tolist(), df["A"].tolist())
                if str(q).strip() and str(a).strip()
            ]
            print(f"{prefix} Загружено {len(self.pairs)} пар (Q, A).", flush=True)

        if not self.pairs:
            print("[Dataset Info] Используется встроенный синтетический мини-корпус (Q, A).", flush=True)
            base_pairs = [
                (
                    "Explain why normalized Grassmannian density similarity stays in [0, 1].",
                    "By Cauchy-Schwarz, the Frobenius inner product of two density operators "
                    "divided by the product of their Frobenius norms is bounded between 0 and 1. "
                    "It equals 1 if and only if the subspaces and relative spectra match.",
                ),
                (
                    "Напиши краткое доказательство бесконечности множества простых чисел.",
                    "Предположим, что простых чисел конечное число. Перемножим их все и прибавим единицу. "
                    "Полученное число не делится ни на одно из известных простых чисел, значит у него есть новый простой делитель.",
                ),
                (
                    "Write a Python function to compute factorial iteratively.",
                    "def factorial(n: int) -> int:\n    res = 1\n    for k in range(2, n + 1):\n        res *= k\n    return res\n",
                ),
                (
                    "What is the capital of France and why is it historically significant?",
                    "The capital of France is Paris. Situated along the Seine River, it has served as "
                    "a major European center of science, diplomacy, literature, and architecture for centuries.",
                ),
            ]
            repeats = max(1, (synthetic_len + len(base_pairs) - 1) // len(base_pairs))
            self.pairs = (base_pairs * repeats)[:synthetic_len]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        q_text, a_text = self.pairs[idx]
        q_ids = encode_token_ids(self.tokenizer, q_text)[: self.max_prompt_tokens]
        if not q_ids:
            q_ids = [1]

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
        prompt_weights = p_pack["weights"]
        ans_weights = a_pack["weights"]

        # Разметка ячеек промпта (64) и ответа (127)
        n_real_prompt = (prompt_ids.view(N_PROMPT, 4) != PAD_TOKEN_ID).sum(dim=-1)
        n_real_ans = (ans_ids.view(N_ANS, 4) != PAD_TOKEN_ID).sum(dim=-1)

        # Маска содержательных не-стоп-клеток (доля не-стоп-токенов >= 0.5, §6) для L_bary / L_adj
        ans_cells_ids = ans_ids.view(N_ANS, 4)
        non_pad_tok = ans_cells_ids != PAD_TOKEN_ID
        safe_ans_ids = ans_cells_ids.clamp(0, self.stopword_mask.numel() - 1)
        non_stop_tok = non_pad_tok & (~self.stopword_mask[safe_ans_ids])
        n_non_pad = non_pad_tok.float().sum(dim=-1)
        n_non_stop = non_stop_tok.float().sum(dim=-1)
        is_content_word_ans = (n_non_pad > 0) & (n_non_stop >= 0.5 * n_non_pad.clamp(min=1.0))
        is_content_word_cell = torch.cat(
            [torch.zeros(1, dtype=torch.bool), is_content_word_ans], dim=0
        )

        # Базовые веса ячеек холста 0..127 (см. §5.1)
        w_cell = torch.zeros(N_CANVAS, dtype=torch.float32)
        n_content_ans = int(a_pack["n_content_cells"].item())
        i_last = max(1, min(N_ANS, n_content_ans))  # в индексах холста 1..127
        for i in range(1, N_CANVAS):
            if i < i_last:
                w_cell[i] = 1.0
            elif i == i_last:
                w_cell[i] = 2.0
            elif i == i_last + 1:
                w_cell[i] = 2.0
            else:
                w_cell[i] = 0.25

        return {
            "prompt_ids": prompt_ids,
            "ans_ids": ans_ids,
            "ans_targets": a_pack["targets"],
            "prompt_weights": prompt_weights,
            "ans_weights": ans_weights,
            "n_real_prompt": n_real_prompt,
            "n_real_ans": n_real_ans,
            "i_last": torch.tensor(i_last, dtype=torch.long),
            "w_cell": w_cell,
            "is_content_word_cell": is_content_word_cell,
        }


def sample_mixed_noise_schedule(
    batch_size: int,
    device: torch.device,
    i_last: Optional[torch.Tensor] = None,
    seq_len: int = N_CANVAS,
) -> torch.Tensor:
    """
    Языково-ориентированный Mixed Noise Curriculum Фазы 3 (для ячеек 0..seq_len-1, t_0 == 0.0):
    - 25% High-Noise / Pure-Start: 1/3 ровно t=1.0, 2/3 U(0.80, 1.0)
    - 20% Shifted Flat Diffusion: t ~ Sigmoid(N(1.0, 1.0)) (медиана ~0.73)
    - 20% Causal Prefix / Block-AR: чистый префикс 1..k (t=0), хвост/блок в t=1.0 (50%) или U(0.50, 1.0)
    - 20% Cloze & Crystal Anchors: 20-70% чистых/мягких якорей, остальные в t=1.0 (50%, прямой Cloze) или U(0.50, 1.0)
    - 15% Rolling Wave Gradient: монотонный фронт шума слева направо [0.0..0.20] -> [0.80..1.0].
    """
    n_ans = int(seq_len) - 1
    t_mat = torch.zeros(batch_size, int(seq_len), device=device, dtype=torch.float32)
    regimes = torch.rand(batch_size, device=device)

    for b in range(batch_size):
        r = float(regimes[b].item())
        if r < 0.25:
            if float(torch.rand(1, device=device).item()) < 0.333:
                t_mat[b, 1:] = 1.0
            else:
                t_hi = torch.empty(1, device=device).uniform_(0.80, 1.0)
                t_mat[b, 1:] = t_hi
        elif r < 0.45:
            t_flat = torch.sigmoid(torch.randn(1, device=device) * 1.0 + 1.0).clamp(0.01, 1.0)
            t_mat[b, 1:] = t_flat
        elif r < 0.65:
            max_cut = int(i_last[b].item()) - 1 if i_last is not None else n_ans - 1
            max_cut = max(0, min(max_cut, n_ans - 1))
            if max_cut > 0:
                if float(torch.rand(1, device=device).item()) < 0.5 and max_cut >= 8:
                    # Выравниваем срез под кратность блокам 8/16 для Block-AR
                    step_blk = 16 if (max_cut >= 16 and float(torch.rand(1, device=device).item()) < 0.5) else 8
                    choices = list(range(0, max_cut + 1, step_blk))
                    cut = choices[int(torch.randint(0, len(choices), (1,)).item())]
                else:
                    cut = int(torch.randint(0, max_cut + 1, (1,)).item())
            else:
                cut = 0

            # Префикс 1..cut чистый (t=0.0), хвост/блок в чистом шуме t=1.0 (50%) или U(0.50, 1.0)
            t_ans = torch.ones(n_ans, device=device, dtype=torch.float32)
            if float(torch.rand(1, device=device).item()) < 0.5:
                blk_len = int(torch.randint(8, 25, (1,)).item())
                blk_end = min(n_ans, cut + blk_len)
                if blk_end > cut:
                    if float(torch.rand(1, device=device).item()) < 0.5:
                        t_ans[cut:blk_end] = 1.0
                    else:
                        t_ans[cut:blk_end] = torch.empty(blk_end - cut, device=device).uniform_(0.50, 1.0)
            else:
                if float(torch.rand(1, device=device).item()) < 0.5:
                    t_ans[cut:] = 1.0
                else:
                    t_ans[cut:] = torch.empty(1, device=device).uniform_(0.50, 1.0)
            if cut > 0:
                t_ans[:cut] = 0.0
            t_mat[b, 1:] = t_ans
        elif r < 0.85:
            # Cloze & Crystal Anchors: 20-70% чистых/мягких якорей, остальные в t=1.0 (50%, прямой Cloze) или U(0.50, 1.0)
            if float(torch.rand(1, device=device).item()) < 0.5:
                t_ans = torch.ones(n_ans, device=device, dtype=torch.float32)
            else:
                t_ans = torch.empty(n_ans, device=device, dtype=torch.float32).uniform_(0.50, 1.0)
            n_c = int(i_last[b].item()) if i_last is not None else n_ans
            n_c = max(2, min(n_c, n_ans))
            keep_frac = 0.20 + 0.50 * float(torch.rand(1, device=device).item())
            n_anchors = max(1, min(n_c - 1, int(round(n_c * keep_frac))))
            anchor_idx = torch.randperm(n_c, device=device)[:n_anchors]
            if float(torch.rand(1, device=device).item()) < 0.65:
                t_ans[anchor_idx] = 0.0
            else:
                t_ans[anchor_idx] = torch.empty(n_anchors, device=device).uniform_(0.01, 0.15)
            t_mat[b, 1:] = t_ans
        else:
            t_start = 0.20 * float(torch.rand(1, device=device).item())
            t_end = 0.80 + 0.20 * float(torch.rand(1, device=device).item())
            grad_t = torch.linspace(t_start, t_end, n_ans, device=device)
            jitter = torch.empty(n_ans, device=device).uniform_(-0.03, 0.03)
            t_mat[b, 1:] = (grad_t + jitter).clamp(0.01, 1.0)
    return t_mat


@torch.no_grad()
def build_rollout_noisy_canvas(
    raw_model: GrassmannianDiffusionCore,
    decoder: Optional[nn.Module],
    V_0: torch.Tensor,
    V_prompt: torch.Tensor,
    p_valid: torch.Tensor,
    p_attn_mask: torch.Tensor,
    m_prompt: torch.Tensor,
    sep_exp: torch.Tensor,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Two-Step Rollout для обучения непрерывной Coarse-to-Fine фокусировке:
    1. Шаг 1 (высокий шум t_hi in [0.80, 1.0]): модель строит сглаженный черновик ответа V_0_draft.
    2. NoiseRadarHead оценивает истинную неуверенность каждой ячейки черновика sigma_draft.
    3. Шаг 2: чистый самочерновик V_draft (alpha=1.0, без подмешивания GT V_0) перезашумляется
       к среднему уровню t_base in [0.15, 0.70] с адаптивным профилем t_i ~ sigma_draft_i.
    Возвращает (V_t, t_mat, radar_sigma).
    """
    B, N, _, _ = V_0.shape
    device = V_0.device
    was_training = raw_model.training
    raw_model.eval()

    # Шаг 1: черновой прогноз на высоком глобальном шуме t_hi in [0.80, 1.0]
    axis_shift_delta = float(getattr(raw_model, "axis_shift_delta", 0.0))
    t_hi = torch.empty((B, 1), device=device, dtype=torch.float32).uniform_(0.80, 1.0)
    t_hi_mat = torch.cat([torch.zeros((B, 1), device=device, dtype=torch.float32), t_hi.expand(B, N - 1)], dim=1)
    V_hi = add_axis_shifted_noise(V_0, t_hi_mat, torch.randn_like(V_0, dtype=torch.float32), delta=axis_shift_delta)
    V_hi[:, 0] = sep_exp[:, 0].float()

    radar_hi = compute_radar_sigma(decoder, V_hi, use_amp=use_amp, amp_dtype=amp_dtype)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        V_draft, _, _ = raw_model(
            V_t=V_hi.to(amp_dtype if use_amp else torch.float32),
            t_mat=t_hi_mat,
            V_prompt=V_prompt.to(amp_dtype if use_amp else torch.float32),
            prompt_valid_mask=p_valid,
            prompt_attn_mask=p_attn_mask,
            m_prompt=m_prompt,
            radar_sigma=radar_hi,
            return_sink_weights=False,
        )
    V_draft = V_draft.float()
    V_draft[:, 0] = sep_exp[:, 0].float()

    # Оценка чистоты ячеек черновика радаром (sigma_draft in [0, 1])
    sig_draft = compute_radar_sigma(decoder, V_draft, use_amp=use_amp, amp_dtype=amp_dtype)
    if sig_draft is not None:
        sig_ans = sig_draft[:, 1:].float()
        sig_mean = sig_ans.mean(dim=-1, keepdim=True).clamp(min=1e-4)
        rel_uncert = torch.clamp(sig_ans / sig_mean, min=0.35, max=1.45)
    else:
        sig_ans = None
        rel_uncert = torch.ones((B, N - 1), device=device, dtype=torch.float32)

    # Шаг 2: адаптивное перезашумление чистого самочерновика (alpha=1.0, без подсказки от GT V_0)
    t_base = torch.empty((B, 1), device=device, dtype=torch.float32).uniform_(0.15, 0.70)
    t_ans = torch.clamp(t_base * rel_uncert, min=0.02, max=0.90)
    if sig_ans is not None:
        cryst_mask = (sig_ans < 0.035) & (t_base < 0.45)
        t_ans = torch.where(cryst_mask, torch.zeros_like(t_ans), t_ans)
    t_mat = torch.cat([torch.zeros((B, 1), device=device, dtype=torch.float32), t_ans], dim=1)

    V_t = add_axis_shifted_noise(V_draft, t_mat, torch.randn_like(V_0, dtype=torch.float32), delta=axis_shift_delta)
    V_t[:, 0] = sep_exp[:, 0].float()

    radar_sigma = compute_radar_sigma(decoder, V_t, use_amp=use_amp, amp_dtype=amp_dtype)
    if was_training:
        raw_model.train()
    return V_t, t_mat, radar_sigma


def apply_prompt_dropout(
    prompt_valid_mask: torch.Tensor,
    full_dropout_prob: float = 0.15,
    prune_prob: float = 0.50,
    prune_frac: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Двойная регуляризация промпта при обучении (§4.2 п.4):
    - 15% Full Prompt Dropout (CFG unconditional, m_prompt = 0);
    - 50% среди остальных: маскирование ~25% смысловых ячеек (Semantic Cell Pruning).
    """
    B, N_p = prompt_valid_mask.shape
    device = prompt_valid_mask.device
    attn_mask = prompt_valid_mask.clone()
    m_prompt = torch.ones(B, device=device, dtype=torch.float32)

    for b in range(B):
        if float(torch.rand(1, device=device).item()) < full_dropout_prob:
            attn_mask[b] = False
            m_prompt[b] = 0.0
        elif float(torch.rand(1, device=device).item()) < prune_prob:
            valid_idx = torch.where(attn_mask[b])[0]
            if len(valid_idx) > 1:
                n_drop = max(1, int(round(len(valid_idx) * prune_frac)))
                n_drop = min(n_drop, len(valid_idx) - 1)
                perm = torch.randperm(len(valid_idx), device=device)[:n_drop]
                attn_mask[b, valid_idx[perm]] = False
    return attn_mask, m_prompt


# %% [markdown]
# ## 5. Гибридная функция потерь (FP32, вне DataParallel)

# %%
class _DiffAllReduceSum(torch.autograd.Function):
    """Дифференцируемая сумма по DDP-рангам для точной глобальной статистики L_adaln."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


def _ddp_sum(x: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        return _DiffAllReduceSum.apply(x)
    return x


def compute_phase3_loss(
    V_0_hat: torch.Tensor,
    V_0: torch.Tensor,
    t_mat: torch.Tensor,
    w_cell: torch.Tensor,
    i_last: torch.Tensor,
    adaln_stats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    step: int,
    m_prompt: Optional[torch.Tensor] = None,
    sink_weights: Optional[torch.Tensor] = None,
    is_content_word_cell: Optional[torch.Tensor] = None,
    basis_weight: float = 0.50,
    gr_weight: float = 0.50,
    energy_weight: float = 0.15,
    spec_diag_weight: float = 0.05,
    orth_offdiag_weight: float = 0.05,
    anchor_weight: float = 0.0,
    adj_weight: float = 0.10,
    bary_weight: float = 0.10,
    sink_weight: float = 0.10,
    sink_content_max: float = 0.35,
    sink_content_min: float = 0.08,
    adaln_max_weight: float = 0.05,
    adaln_warmup_steps: int = 500,
    axis_shift_delta: float = 0.0,
    basis_high_t_mode: str = "none",
    static_xla_bary: bool = True,
    world_size: int = 1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Гибридная функция потерь Фазы 3 (канонический базис V_0 + O(8)-инвариантный декодер ADR 137):
    L_total = w_basis*L_basis + w_gr*L_Gr_dir + w_en*L_energy
            + w_spec*L_spec_diag + w_orth*L_orth_offdiag + w_anchor*L_anchor
            + w_adj*L_adj + w_bary*L_bary + w_sink*L_sink + w_adaln(s)*L_adaln.
    """
    device_type = V_0_hat.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        Vh = V_0_hat.float()
        V0 = V_0.float()
        tf = t_mat.float()
        wc = w_cell.float()

        B, N, K_ax, _ = Vh.shape
        m_active = tf > 1e-3
        # Смягченный SNR-вес [1.0..2.0], чтобы не подавлять обучение на высоком шуме t in [0.7, 1.0]
        w_snr = torch.clamp(1.0 / torch.clamp(tf, min=0.25), max=2.0)
        w_tilde = m_active.float() * wc * w_snr
        w_denom = w_tilde.sum() + 1e-6

        # 1. Канонический лосс базиса (SmoothL1 beta=0.02), с поддержкой поосевого t_k(t) и смягчения на высоком шуме
        t_ax = compute_axis_t(tf, delta=axis_shift_delta, num_axes=K_ax)  # (B, N, K)
        if basis_high_t_mode == "sign_only":
            # Знаковое выравнивание с detach на высоком шуме (t_k >= 0.5): снимает разрывность SignFix без SVD и без смешивания осей
            dot_ax = (Vh.detach() * V0).sum(dim=-1, keepdim=True)
            sgn_ax = torch.where(dot_ax >= 0.0, torch.ones_like(dot_ax), -torch.ones_like(dot_ax))
            v0_basis = torch.where((t_ax >= 0.5).unsqueeze(-1), V0 * sgn_ax, V0)
        else:
            v0_basis = V0

        elem_basis_ax = F.smooth_l1_loss(Vh, v0_basis, beta=0.02, reduction="none").mean(dim=-1)  # (B, N, K)
        if abs(float(axis_shift_delta)) > 1e-8:
            w_snr_ax = torch.clamp(1.0 / torch.clamp(t_ax, min=0.25), max=2.0)
            w_basis_ax = (m_active.float() * wc).unsqueeze(-1) * w_snr_ax
        else:
            w_basis_ax = w_tilde.unsqueeze(-1).expand(B, N, K_ax)

        if basis_high_t_mode == "decay":
            w_basis_ax = w_basis_ax * torch.clamp(1.0 - t_ax, min=0.1).pow(2)

        l_basis = (w_basis_ax * elem_basis_ax).sum() / (w_basis_ax.sum() + 1e-6)

        # 2. Декомпозированный Грассманов лосс + спектрально-ортогональные лоссы из уже вычисленной G (§6):
        Gh = torch.matmul(Vh, Vh.transpose(-2, -1))
        G0 = torch.matmul(V0, V0.transpose(-2, -1))
        cross = torch.matmul(Vh, V0.transpose(-2, -1))
        gh_frob_sq = (Gh ** 2).sum(dim=(-2, -1))
        g0_frob_sq = (G0 ** 2).sum(dim=(-2, -1))
        norm_gh = torch.sqrt(gh_frob_sq + 1e-12)
        norm_g0 = torch.sqrt(g0_frob_sq + 1e-12)
        sim_cell = (cross ** 2).sum(dim=(-2, -1)) / (norm_gh * norm_g0 + 1e-8)
        elem_gr_dir = torch.clamp(1.0 - sim_cell, min=0.0)
        l_gr = (w_tilde * elem_gr_dir).sum() / w_denom

        diag_gh = torch.diagonal(Gh, dim1=-2, dim2=-1)  # (B, N, K)
        diag_g0 = torch.diagonal(G0, dim1=-2, dim2=-1)  # (B, N, K)
        energy_h = diag_gh.sum(dim=-1)
        energy_0 = diag_g0.sum(dim=-1)
        elem_energy = (
            F.smooth_l1_loss(energy_h, energy_0, beta=0.05, reduction="none")
            + 0.5 * F.smooth_l1_loss(norm_gh, norm_g0, beta=0.05, reduction="none")
        )
        l_energy = (w_tilde * elem_energy).sum() / w_denom

        # 2б. Поосевой спектральный лосс diag(Gh) vs diag(G0) и лосс ортогональности (с затуханием (1 - t), §6)
        w_low_t = w_tilde * torch.clamp(1.0 - tf, min=0.0)
        w_low_denom = w_low_t.sum() + 1e-6
        elem_spec_diag = F.smooth_l1_loss(diag_gh, diag_g0, beta=0.05, reduction="none").mean(dim=-1)
        l_spec_diag = (w_low_t * elem_spec_diag).sum() / w_low_denom

        diag_gh_sq = (diag_gh ** 2).sum(dim=-1)
        offdiag_sq = torch.clamp(gh_frob_sq - diag_gh_sq, min=0.0)
        elem_orth = offdiag_sq / (gh_frob_sq + 1e-8)
        l_orth = (w_low_t * elem_orth).sum() / w_low_denom

        # 2в. Якорный лосс на чистых не-SEP клетках (t_i <= 1e-3, pos >= 1) для обучения lambda_keep (§2.2)
        pos_full = torch.arange(N, device=Vh.device).unsqueeze(0).expand(B, N)
        if anchor_weight > 0.0:
            clean_mask = (tf <= 1e-3) & (pos_full >= 1)
            clean_f = clean_mask.float()
            elem_anchor = F.smooth_l1_loss(Vh, V0, beta=0.01, reduction="none").mean(dim=(-2, -1))
            l_anchor = (clean_f * elem_anchor).sum() / clean_f.sum().clamp(min=1.0)
        else:
            l_anchor = Vh.sum() * 0.0

        # 3. Анти-коллапс смежных смысловых ячеек (1 <= i < i_last), XLA-safe с ослаблением на стоп-клетках (§6)
        pos_idx = torch.arange(N - 1, device=Vh.device).unsqueeze(0).expand(B, N - 1)
        adj_mask = (pos_idx >= 1) & (pos_idx < i_last.unsqueeze(1)) & m_active[:, :-1] & m_active[:, 1:]
        sim_adj = sim_gr(Vh[:, :-1], Vh[:, 1:])
        adj_w = adj_mask.float()
        if is_content_word_cell is not None:
            cw = is_content_word_cell.to(device=Vh.device, dtype=torch.bool)
            adj_cw_factor = torch.where(cw[:, :-1] & cw[:, 1:], torch.ones_like(adj_w), torch.full_like(adj_w, 0.25))
            adj_w = adj_w * adj_cw_factor
        l_adj = (adj_w * (F.relu(sim_adj - TAU_ADJ) ** 2)).sum() / adj_w.sum().clamp(min=1.0)

        # 4. Межпримерное отталкивание от барицентра (M <= 64, строго b_m != b_n, исключая стоп-клетки, XLA-safe §6)
        content_active = (pos_full >= 1) & (pos_full <= i_last.unsqueeze(1)) & m_active
        if is_content_word_cell is not None:
            bary_mask = content_active & is_content_word_cell.to(device=Vh.device, dtype=torch.bool)
        else:
            bary_mask = content_active

        if static_xla_bary:
            total_slots = B * N
            M_sel = min(64, total_slots)
            bary_flat = bary_mask.reshape(total_slots)
            with torch.no_grad():
                priority = bary_flat.float() * 2.0 + torch.rand(total_slots, device=Vh.device, dtype=torch.float32)
                sel = torch.topk(priority, k=M_sel, dim=0).indices
                b_sel = torch.div(sel, N, rounding_mode="floor")
                v_sel = bary_flat.index_select(0, sel)
                pair_mask = (b_sel.unsqueeze(1) != b_sel.unsqueeze(0)) & v_sel.unsqueeze(1) & v_sel.unsqueeze(0)
            cells = Vh.reshape(total_slots, K_ax, -1).index_select(0, sel)
            cross_b = torch.matmul(cells.unsqueeze(1), cells.unsqueeze(0).transpose(-2, -1))
            Gm = torch.matmul(cells, cells.transpose(-2, -1))
            norm_m = torch.sqrt((Gm ** 2).sum(dim=(-2, -1)) + 1e-16)
            sim_mat = (cross_b ** 2).sum(dim=(-2, -1)) / (norm_m.unsqueeze(1) * norm_m.unsqueeze(0) + 1e-8)
            pair_f = pair_mask.float()
            l_bary = ((F.relu(sim_mat - TAU_BARY) ** 2) * pair_f).sum() / pair_f.sum().clamp(min=1.0)
        else:
            b_idx, i_idx = torch.where(bary_mask)
            if len(b_idx) > 1 and len(torch.unique(b_idx)) > 1:
                if len(b_idx) > 64:
                    sel = torch.randperm(len(b_idx), device=Vh.device)[:64]
                    b_idx = b_idx[sel]
                    i_idx = i_idx[sel]
                cells = Vh[b_idx, i_idx]
                cross_b = torch.matmul(cells.unsqueeze(1), cells.unsqueeze(0).transpose(-2, -1))
                Gm = torch.matmul(cells, cells.transpose(-2, -1))
                norm_m = torch.sqrt((Gm ** 2).sum(dim=(-2, -1)) + 1e-16)
                sim_mat = (cross_b ** 2).sum(dim=(-2, -1)) / (norm_m.unsqueeze(1) * norm_m.unsqueeze(0) + 1e-8)
                diff_ex = b_idx.unsqueeze(1) != b_idx.unsqueeze(0)
                if diff_ex.any():
                    l_bary = (F.relu(sim_mat[diff_ex] - TAU_BARY) ** 2).mean()
                else:
                    l_bary = Vh.sum() * 0.0
            else:
                l_bary = Vh.sum() * 0.0

        # 5. Пол разнообразия AdaLN (суммирование по репликам DP/DDP и линейный warm-up 500 шагов)
        sum_m_pack, sum_m2_pack, count_pack = adaln_stats
        sum_m_tot = _ddp_sum(sum_m_pack.float().sum(dim=0))      # (num_layers, 4096)
        sum_m2_tot = _ddp_sum(sum_m2_pack.float().sum(dim=0))    # (num_layers, 4096)
        count_tot = _ddp_sum(count_pack.float().sum(dim=0))      # (num_layers,)

        valid_layers = count_tot > 0
        if valid_layers.any():
            cnt = count_tot.clamp(min=1.0).unsqueeze(-1)
            mean_m = sum_m_tot / cnt
            var_m = torch.clamp(sum_m2_tot / cnt - mean_m ** 2, min=1e-12)
            sigma_layers = torch.sqrt(var_m).mean(dim=-1)  # (num_layers,)
            l_adaln = F.relu(1e-2 - sigma_layers).mean()
            sigma_mean_val = float(sigma_layers.mean().item())
            sigma_min_val = float(sigma_layers.min().item())
        else:
            l_adaln = Vh.sum() * 0.0
            sigma_mean_val = 0.0
            sigma_min_val = 0.0

        # 6. Явный штраф на слив контентных ячеек в k_sink при m_prompt == 1 (VOID и uncond не штрафуются)
        if sink_weights is not None and m_prompt is not None:
            sw = sink_weights.float()
            if sw.ndim == 2:
                sw = sw.unsqueeze(1)  # (B, 1, N)
            cond_active = (m_prompt.float() > 0.5).view(B, 1)
            sink_mask = (content_active & cond_active).unsqueeze(1).expand_as(sw).float()
            sink_pen = F.relu(sw - sink_content_max) ** 2 + 0.25 * (F.relu(sink_content_min - sw) ** 2)
            l_sink = (sink_pen * sink_mask).sum() / sink_mask.sum().clamp(min=1.0)
        else:
            l_sink = Vh.sum() * 0.0

        # В DDP каждый ранг усредняет градиенты по world_size; масштабируем вклад DDP для L_adaln корректно
        w_adaln = adaln_max_weight * min(1.0, float(step) / float(max(1, adaln_warmup_steps)))
        adaln_term = (w_adaln * float(world_size)) * l_adaln if world_size > 1 else w_adaln * l_adaln

        total_loss = (
            basis_weight * l_basis
            + gr_weight * l_gr
            + energy_weight * l_energy
            + spec_diag_weight * l_spec_diag
            + orth_offdiag_weight * l_orth
            + anchor_weight * l_anchor
            + adj_weight * l_adj
            + bary_weight * l_bary
            + sink_weight * l_sink
            + adaln_term
        )

        metrics = {
            "loss/total": float(total_loss.item()),
            "loss/basis": float(l_basis.item()),
            "loss/gr": float(l_gr.item()),
            "loss/energy": float(l_energy.item()),
            "loss/spec_diag": float(l_spec_diag.item()),
            "loss/orth_offdiag": float(l_orth.item()),
            "loss/anchor": float(l_anchor.item()),
            "loss/adj": float(l_adj.item()),
            "loss/bary": float(l_bary.item()),
            "loss/sink": float(l_sink.item()),
            "loss/adaln_raw": float(l_adaln.item()),
            "loss/w_adaln": float(w_adaln),
            "adaln/sigma_mean": sigma_mean_val,
            "adaln/sigma_min": sigma_min_val,
        }
        return total_loss, metrics


compute_phase2_loss = compute_phase3_loss


def compute_decoder_ce_loss(
    decoder: nn.Module,
    V_0_hat: torch.Tensor,
    t_mat: torch.Tensor,
    ans_targets: torch.Tensor,
    ce_max_tokens: int = 512,
    ce_t_max: float = 0.50,
    stopword_mask: Optional[torch.Tensor] = None,
    ce_stopword_weight: float = 0.25,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, float]:
    """
    Сквозной токенный Cross-Entropy лосс через замороженный GrassmannianDecoder (eval, requires_grad=False):
    - Применяется только на низком/среднем шуме t_i in (0, ce_t_max) с гладким весом ((ce_t_max - t_i) / ce_t_max)^2,
      чтобы не вызывать unigram mode collapse (запятые/пробелы/\\n\\n) на высоком шуме;
    - Понижает вес стоп-слов/пунктуации до ce_stopword_weight;
    - Использует torch.utils.checkpoint.checkpoint(decoder.decode_hidden, ...) для минимизации VRAM на T4;
    - Считает логиты только на случайной подвыборке до ce_max_tokens активных контентных токенов
      (<= 512 x 131072 в FP32 = <= 268 MB VRAM), возвращая (l_ce, tok_acc).
    """
    B = V_0_hat.shape[0]
    device = V_0_hat.device
    V_ans_hat = V_0_hat[:, 1:]  # (B, n_ans, 8, 128) с градиентом
    n_ans = V_ans_hat.shape[1]

    t_tok = t_mat[:, 1:].float().unsqueeze(-1).expand(B, n_ans, 4).reshape(B, -1)
    active_tok_mask = (t_tok > 1e-3) & (t_tok < ce_t_max)
    valid_mask = active_tok_mask & (ans_targets != -100)
    flat_valid = torch.where(valid_mask.reshape(-1))[0]
    if flat_valid.numel() == 0:
        return V_0_hat.sum() * 0.0, 0.0

    if flat_valid.numel() > ce_max_tokens:
        perm = torch.randperm(flat_valid.numel(), device=device)[:ce_max_tokens]
        sel_idx = flat_valid[perm]
    else:
        sel_idx = flat_valid

    dec_dtype = next(decoder.parameters()).dtype
    V_in = V_ans_hat.to(amp_dtype if use_amp else dec_dtype)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        h_final = torch.utils.checkpoint.checkpoint(
            decoder.decode_hidden, V_in, use_reentrant=False
        )  # (B, 508, 1024)

    h_sub = h_final.reshape(-1, h_final.shape[-1])[sel_idx]
    tgt_sub = ans_targets.reshape(-1)[sel_idx].long()
    t_sub = t_tok.reshape(-1)[sel_idx]
    logits_sub = decoder.get_tied_head_logits(h_sub)  # float32 (M, vocab_size)
    with torch.autocast(device_type=device.type, enabled=False):
        per_tok_ce = F.cross_entropy(logits_sub.float(), tgt_sub, reduction="none")
        w_t = torch.clamp((ce_t_max - t_sub) / max(ce_t_max, 1e-6), min=0.0, max=1.0) ** 2
        if stopword_mask is not None:
            sw_idx = tgt_sub.clamp(0, stopword_mask.numel() - 1)
            w_sw = torch.where(
                stopword_mask[sw_idx],
                torch.full_like(w_t, float(ce_stopword_weight)),
                torch.ones_like(w_t),
            )
            w_tok = w_t * w_sw
        else:
            w_tok = w_t
        l_ce = (per_tok_ce * w_tok).sum() / w_tok.sum().clamp(min=1e-6)
        with torch.no_grad():
            tok_acc = float((logits_sub.argmax(dim=-1) == tgt_sub).float().mean().item())
    return l_ce, tok_acc


# %% [markdown]
# ## 6. Оптимизатор `OrthMuon (d=1)` + `AdamW` и `EMA` с проекцией на сферу

# %%
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Ортогонализация градиента Ньютона–Шульца 5-го порядка (Algorithm 2 Muon)."""
    orig_dtype = G.dtype
    X = G.bfloat16() if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else G.float()
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
    """
    OrthMuon (d=1, сферическая оптимизация направлений строк, ADR 131 + §5.2):
    - Для *.q_weight (sphere=True): тангенциальная проекция шага Muon и прямая ретракция на S^{d-1} в float32;
    - Сферически-корректный масштаб `sphere_scale=True`:
      `scale_sphere(M, N) = c_base * sqrt(M / min(M, N))` (опорное `c_base = 2.26` для N=128),
      выравнивающий угловую скорость поворота строк между матрицами разных форм `(32x128 .. 2048x1024)`.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        sphere: bool = True,
        sphere_scale: bool = True,
        sphere_c_base: float = 2.26,
        sphere_rms_norm: bool = False,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            sphere=sphere,
            sphere_scale=sphere_scale,
            sphere_c_base=sphere_c_base,
            sphere_rms_norm=sphere_rms_norm,
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
            use_sphere_scale = group.get("sphere_scale", True)
            c_base = float(group.get("sphere_c_base", 2.26))
            use_rms_norm = bool(group.get("sphere_rms_norm", False))

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
                M, N = p.size(0), p.size(1)

                if sphere:
                    q_f32 = p.data.float()
                    u_f32 = update_ortho.float()
                    u_tan = u_f32 - (u_f32 * q_f32).sum(dim=-1, keepdim=True) * q_f32
                    if use_sphere_scale:
                        if use_rms_norm:
                            mean_row_l2 = u_tan.norm(p=2, dim=-1).mean().clamp(min=1e-8)
                            u_tan = u_tan / mean_row_l2
                            scale = c_base
                        else:
                            scale = c_base * math.sqrt(float(M) / float(min(M, N)))
                    else:
                        scale = 0.2 * math.sqrt(max(M, N))
                    a_f32 = q_f32 - (lr * scale) * u_tan
                    q_next = F.normalize(a_f32, p=2, dim=-1, eps=1e-8)
                    p.data.copy_(q_next.to(p.dtype))
                else:
                    scale = 0.2 * math.sqrt(max(M, N))
                    if wd > 0.0:
                        p.data.mul_(1.0 - lr * wd)
                    p.data.add_(update_ortho, alpha=-lr * scale)
        return loss


def clip_phase3_grads(
    opt_muon: Optional[OrthMuon],
    opt_adamw: torch.optim.AdamW,
    adamw_max_norm: float = 1.0,
    sphere_max_norm: Optional[float] = 10.0,
) -> Tuple[float, float]:
    """
    Раздельный клиппинг градиентов (§5.1):
    - AdamW-параметры клиппируются строго по `adamw_max_norm` (1.0);
    - Сферические `q_weight` (OrthMuon) не смешиваются с AdamW и клиппируются только мягким
      страховочным порогом `sphere_max_norm` (10.0) от экстремального спайка до буфера моментума.
    """
    adamw_params = [p for g in opt_adamw.param_groups for p in g["params"] if p.grad is not None]
    adamw_gnorm = (
        float(torch.nn.utils.clip_grad_norm_(adamw_params, max_norm=float(adamw_max_norm)).item())
        if adamw_params
        else 0.0
    )
    sphere_gnorm = 0.0
    if opt_muon is not None and sphere_max_norm is not None and float(sphere_max_norm) > 0.0:
        sphere_params = [p for g in opt_muon.param_groups for p in g["params"] if p.grad is not None]
        if sphere_params:
            sphere_gnorm = float(
                torch.nn.utils.clip_grad_norm_(sphere_params, max_norm=float(sphere_max_norm)).item()
            )
    return adamw_gnorm, sphere_gnorm


def create_phase2_optimizers(
    model: nn.Module,
    muon_lr: float = 1e-3,
    adamw_lr: float = 3e-4,
    scale_1d_lr: float = 1.2e-3,
    weight_decay: float = 0.01,
) -> Tuple[Optional[OrthMuon], torch.optim.AdamW]:
    """
    Разделение параметров по §6.3:
    1. OrthMuon (lr=1e-3, wd=0.0, sphere=True): все *.q_weight (SphereLinear).
    2. AdamW (lr=3e-4, wd=0.01 / 0.0 для bias и LayerNorm): AdaLN, ConditionEmbedder, k_sink, OutputHead.
    3. AdamW (lr=1.2e-3, wd=0.0): row_scale, v_gate, RMSNorm/QK-norm, layer_scale (ls_attn, ls_mlp).
    """
    raw_model = getattr(model, "module", model)
    sphere_params = []
    adamw_decay_params = []
    adamw_nodecay_params = []
    scale_1d_params = []

    for name, param in raw_model.named_parameters():
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
            # Сдвиги и LayerNorm в ConditionEmbedder/AdaLN без weight decay
            adamw_nodecay_params.append(param)
        else:
            # 2D веса AdaLN, ConditionEmbedder, OutputHead, а также k_sink (4D, 1x8x1x128)
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
    """Косинусный множитель LR с линейным разогревом."""
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
    FP32 EMA-трекер с warmup decay (§5.3) и обязательной сферической проекцией *.q_weight
    при apply_shadow (ADR 132). Поддерживает обратную совместимость со старыми чекпоинтами.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: Optional[torch.device] = None,
        warmup: bool = True,
    ):
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.num_updates: int = 0
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        raw_model = getattr(model, "module", model)
        for name, param in raw_model.named_parameters():
            if param.requires_grad:
                t = param.detach().clone().float()
                if device is not None:
                    t = t.to(device)
                self.shadow[name] = t

    def current_decay(self) -> float:
        if not self.warmup:
            return self.decay
        return min(self.decay, (1.0 + float(self.num_updates)) / (10.0 + float(self.num_updates)))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        eff_decay = self.current_decay()
        one_minus_decay = 1.0 - eff_decay
        raw_model = getattr(model, "module", model)
        for name, param in raw_model.named_parameters():
            if param.requires_grad and name in self.shadow:
                s = self.shadow[name]
                if s.device != param.device:
                    s = s.to(param.device)
                    self.shadow[name] = s
                # XLA-safe обновление без lerp_(..., scalar)
                s.mul_(eff_decay).add_(param.detach().float(), alpha=one_minus_decay)

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module) -> None:
        self.backup = {}
        raw_model = getattr(model, "module", model)
        for name, param in raw_model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.detach().cpu().clone()
                val = self.shadow[name]
                if name.endswith(".q_weight"):
                    val = F.normalize(val.float(), p=2, dim=-1, eps=1e-8)
                param.data.copy_(val.to(device=param.device, dtype=param.dtype))

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        raw_model = getattr(model, "module", model)
        for name, param in raw_model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name].to(device=param.device, dtype=param.dtype))
        self.backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        out = {k: v.cpu() for k, v in self.shadow.items()}
        out["__num_updates__"] = torch.tensor(self.num_updates, dtype=torch.long)
        return out

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if "__num_updates__" in state_dict:
            nu = state_dict["__num_updates__"]
            self.num_updates = int(nu.item()) if isinstance(nu, torch.Tensor) else int(nu)
        for k, v in state_dict.items():
            if k in self.shadow and isinstance(v, torch.Tensor):
                self.shadow[k].copy_(v.to(self.shadow[k].device))


# %% [markdown]
# ## 7. Сэмплер Эйлера–Хьюна с CFG и Evaluation Gate

# %%
def procrustes_align_basis(v_src: torch.Tensor, v_ref: torch.Tensor) -> torch.Tensor:
    """
    8x8 ортогональное Procrustes-выравнивание базиса v_src к v_ref строго в float32 (§2.3):
      M = v_src @ v_ref^T = U Sigma W^T  =>  R* = W U^T in O(8),  v_aligned = R* v_src.
    На XLA/TPU возвращается v_src без SVD, чтобы не вызывать CPU-fallback.
    """
    if v_src.device.type == "xla":
        return v_src.float()
    a = v_src.float()
    b = v_ref.float()
    m = torch.matmul(a, b.transpose(-2, -1))
    u, _, vh = torch.linalg.svd(m)
    r_opt = torch.matmul(vh.transpose(-2, -1), u.transpose(-2, -1))
    return torch.matmul(r_opt, a)


def _apply_cfg_and_clamp(
    v0_cond: torch.Tensor,
    v0_uncond: torch.Tensor,
    cfg_scale: float,
    cfg_rescale: float = 0.7,
    procrustes: bool = True,
) -> torch.Tensor:
    """
    Classifier-Free Guidance с опциональным 8x8 Procrustes-выравниванием базиса v0_uncond к v0_cond (§2.3),
    по-ячеечной стабилизацией Фробениусовой нормы (CFG Rescale) и проекцией строк в единичный шар B^128.
    """
    vc = v0_cond.float()
    vu = procrustes_align_basis(v0_uncond, vc) if procrustes else v0_uncond.float()
    v_cfg = vu + cfg_scale * (vc - vu)
    if cfg_rescale > 0.0:
        norm_c = torch.sqrt((vc ** 2).sum(dim=(-2, -1), keepdim=True) + 1e-12)
        norm_cfg = torch.sqrt((v_cfg ** 2).sum(dim=(-2, -1), keepdim=True) + 1e-12)
        v_rescaled = v_cfg * (norm_c / norm_cfg)
        v_cfg = cfg_rescale * v_rescaled + (1.0 - cfg_rescale) * v_cfg
    return clamp_to_unit_ball(v_cfg)


def build_t_grid(
    num_steps: int,
    device: torch.device,
    mode: str = "sd3",
    shift: float = 0.55,
    gamma: float = 1.5,
) -> torch.Tensor:
    """
    Сетка шагов диффузии t in [1.0 -> 0.0] (§2.4):
    - 'linear': равномерная сетка;
    - 'sd3': t(u) = s*u / (1 + (s - 1)*u), s in [0.4, 0.7] сгущает шаги к t -> 0;
    - 'power': t(u) = u^gamma, gamma in [1.3, 2.0].
    """
    u = torch.linspace(1.0, 0.0, steps=int(num_steps) + 1, device=device, dtype=torch.float32)
    if mode == "sd3":
        s = float(shift)
        t_grid = (s * u) / (1.0 + (s - 1.0) * u).clamp(min=1e-6)
    elif mode == "power":
        t_grid = u.pow(float(gamma))
    else:
        t_grid = u
    t_grid[0] = 1.0
    t_grid[-1] = 0.0
    return t_grid


def compute_limited_cfg_scale(
    t_val: float,
    cfg_scale: float,
    t_range: Optional[Tuple[float, float]] = (0.25, 0.95),
) -> float:
    """
    Limited-Interval CFG (§2.4, Kynkäänniemi et al., NeurIPS 2024):
    - При t in [t_low, t_high]: полный cfg_scale;
    - При t > t_high или t <= 0.5 * t_low: строго 1.0 (экономит 2x compute, uncond не вызывается);
    - При t in (0.5 * t_low, t_low): плавный линейный спад к 1.0.
    """
    if t_range is None or abs(float(cfg_scale) - 1.0) <= 1e-4:
        return float(cfg_scale)
    t_low, t_high = float(t_range[0]), float(t_range[1])
    t = float(t_val)
    if t > t_high:
        return 1.0
    if t >= t_low:
        return float(cfg_scale)
    t_floor = 0.5 * t_low
    if t <= t_floor:
        return 1.0
    alpha = (t - t_floor) / max(t_low - t_floor, 1e-6)
    return 1.0 + (float(cfg_scale) - 1.0) * alpha


def _prepare_sampler_prompt_cache(
    raw_core: nn.Module,
    V_prompt: torch.Tensor,
    prompt_valid_mask: torch.Tensor,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Tuple[torch.Tensor, Any]:
    """Готовит x_prompt и опциональный послойный KV-кэш промпта (Этап 1 / §7.1)."""
    with torch.autocast(device_type=V_prompt.device.type, dtype=amp_dtype, enabled=use_amp):
        if hasattr(raw_core, "prepare_prompt_cache"):
            p_cache = raw_core.prepare_prompt_cache(V_prompt, prompt_valid_mask=prompt_valid_mask)
            return p_cache["x_prompt"], p_cache
        x_prompt = raw_core.encode_prompt(V_prompt, prompt_valid_mask=prompt_valid_mask)
        return x_prompt, None


def _forward_sampler_v0(
    raw_core: nn.Module,
    V_in: torch.Tensor,
    t_mat: torch.Tensor,
    x_prompt: torch.Tensor,
    prompt_cache: Any,
    prompt_valid_mask: torch.Tensor,
    uncond_mask: torch.Tensor,
    m_cond: torch.Tensor,
    m_uncond: torch.Tensor,
    radar_sig: Optional[torch.Tensor],
    eff_cfg_scale: float,
    cfg_rescale: float,
    procrustes_cfg: bool,
    use_amp: bool,
    amp_dtype: torch.dtype,
    want_sink: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Единый вызов модели в сэмплерах: 2B батчевый CFG (§7.3) + кэш KV (§7.1) + пропуск CA на uncond (§7.2)."""
    in_dtype = amp_dtype if use_amp else torch.float32
    v_cast = V_in.to(in_dtype)
    need_cfg = abs(float(eff_cfg_scale) - 1.0) > 1e-4
    with torch.autocast(device_type=V_in.device.type, dtype=amp_dtype, enabled=use_amp):
        if need_cfg and hasattr(raw_core, "forward_cfg_batched"):
            v0_cond, v0_uncond, sw = raw_core.forward_cfg_batched(
                V_t=v_cast,
                t_mat=t_mat,
                x_prompt=x_prompt,
                prompt_attn_mask=prompt_valid_mask,
                prompt_cache=prompt_cache,
                radar_sigma=radar_sig,
                return_sink_weights=want_sink,
            )
            v0 = _apply_cfg_and_clamp(
                v0_cond, v0_uncond, cfg_scale=eff_cfg_scale, cfg_rescale=cfg_rescale, procrustes=procrustes_cfg
            )
            return v0, sw

        fw_kwargs: Dict[str, Any] = dict(
            V_t=v_cast,
            t_mat=t_mat,
            x_prompt=x_prompt,
            prompt_attn_mask=prompt_valid_mask,
            m_prompt=m_cond,
            radar_sigma=radar_sig,
            return_sink_weights=want_sink,
        )
        if prompt_cache is not None:
            fw_kwargs["prompt_cache"] = prompt_cache
        v0_cond, _, sw = raw_core.forward_with_cached_prompt(**fw_kwargs)
        if need_cfg:
            fw_u = dict(fw_kwargs)
            fw_u["prompt_attn_mask"] = uncond_mask
            fw_u["m_prompt"] = m_uncond
            fw_u["return_sink_weights"] = False
            v0_uncond, _, _ = raw_core.forward_with_cached_prompt(**fw_u)
            v0 = _apply_cfg_and_clamp(
                v0_cond, v0_uncond, cfg_scale=eff_cfg_scale, cfg_rescale=cfg_rescale, procrustes=procrustes_cfg
            )
        else:
            v0 = v0_cond.float()
    return v0, sw


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
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    seed: Optional[int] = None,
    t_grid_mode: str = "sd3",
    t_grid_shift: float = 0.55,
    limited_cfg_t_range: Optional[Tuple[float, float]] = (0.25, 0.95),
    procrustes_cfg: bool = True,
    axis_shift_delta: Optional[float] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Генерация с нуля (t = 1 -> 0) методом Эйлера–Хьюна 2-го порядка с поосевым bridge-шагом (§2.1),
    Procrustes CFG (§2.3), неравномерной сеткой t_grid и Limited-Interval CFG (§2.4),
    кэшированием KV промпта (§7.1) и 2B-батчевым CFG с пропуском CA на uncond (§§7.2–7.3).
    """
    raw_core = getattr(core, "module", core)
    was_training = raw_core.training
    raw_core.eval()

    B = V_prompt.shape[0]
    device = V_prompt.device
    delta = float(getattr(raw_core, "axis_shift_delta", 0.0)) if axis_shift_delta is None else float(axis_shift_delta)

    gen = torch.Generator(device=device) if seed is not None else None
    if gen is not None:
        gen.manual_seed(seed)

    # Однократный прогон PromptAdapter + кэширование (k_p_rot, v_p, attn_mask) по CA-слоям
    x_prompt, prompt_cache = _prepare_sampler_prompt_cache(
        raw_core, V_prompt, prompt_valid_mask, use_amp=use_amp, amp_dtype=amp_dtype
    )

    sep_exp = sep_concept.to(device=device, dtype=torch.float32).view(1, 1, 8, 128).expand(B, 1, 8, 128)
    noise = torch.randn((B, N_ANS, 8, 128), generator=gen, device=device, dtype=torch.float32) * SIGMA_NOISE
    V_t = torch.cat([sep_exp, noise], dim=1)

    t_grid = build_t_grid(num_steps, device=device, mode=t_grid_mode, shift=t_grid_shift)
    m_cond = torch.ones(B, device=device, dtype=torch.float32)
    m_uncond = torch.zeros(B, device=device, dtype=torch.float32)
    uncond_mask = torch.zeros_like(prompt_valid_mask)
    last_sink_weights = None

    def predict_v0(V_in: torch.Tensor, t_val: float, want_sink: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        t_mat = torch.full((B, N_CANVAS), t_val, device=device, dtype=torch.float32)
        t_mat[:, 0] = 0.0
        radar_sig = compute_radar_sigma(decoder, V_in, use_amp=use_amp, amp_dtype=amp_dtype)
        eff_cfg = compute_limited_cfg_scale(t_val, cfg_scale, t_range=limited_cfg_t_range)
        v0, sw = _forward_sampler_v0(
            raw_core=raw_core,
            V_in=V_in,
            t_mat=t_mat,
            x_prompt=x_prompt,
            prompt_cache=prompt_cache,
            prompt_valid_mask=prompt_valid_mask,
            uncond_mask=uncond_mask,
            m_cond=m_cond,
            m_uncond=m_uncond,
            radar_sig=radar_sig,
            eff_cfg_scale=eff_cfg,
            cfg_rescale=cfg_rescale,
            procrustes_cfg=procrustes_cfg,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            want_sink=want_sink,
        )
        v0[:, 0] = sep_exp[:, 0]
        return v0, sw

    for k in range(num_steps):
        t_cur = float(t_grid[k].item())
        t_next = float(t_grid[k + 1].item())

        v0_1, sw_1 = predict_v0(V_t, t_cur, want_sink=(k == num_steps - 1))
        if k == num_steps - 1:
            last_sink_weights = sw_1

        if t_next <= 1e-6:
            V_t = v0_1
            break

        # Поосевой bridge-шаг (при delta=0 тождественен линейному шагу Эйлера/Хьюна)
        tc_ax = compute_axis_t(torch.tensor(t_cur, device=device), delta=delta).view(1, 1, 8, 1).clamp(min=1e-4)
        tn_ax = compute_axis_t(torch.tensor(t_next, device=device), delta=delta).view(1, 1, 8, 1)
        ratio_ax = tn_ax / tc_ax

        V_pred_ans = v0_1[:, 1:] + ratio_ax * (V_t[:, 1:] - v0_1[:, 1:])
        V_pred = torch.cat([sep_exp, V_pred_ans], dim=1)

        v0_2, _ = predict_v0(V_pred, t_next, want_sink=False)
        v0_bar_ans = 0.5 * (v0_1[:, 1:] + v0_2[:, 1:])
        V_next_ans = v0_bar_ans + ratio_ax * (V_t[:, 1:] - v0_bar_ans)
        V_t = torch.cat([sep_exp, V_next_ans], dim=1)

    if was_training:
        raw_core.train()
    return V_t, last_sink_weights


@torch.no_grad()
def rolling_wave_sample(
    core: GrassmannianDiffusionCore,
    V_prompt: torch.Tensor,
    prompt_valid_mask: torch.Tensor,
    sep_concept: torch.Tensor,
    void_ref: Optional[torch.Tensor] = None,
    num_steps: int = 40,
    window: float = 56.0,
    cfg_scale: float = 3.0,
    cfg_rescale: float = 0.7,
    decoder: Optional[nn.Module] = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    seed: Optional[int] = None,
    limited_cfg_t_range: Optional[Tuple[float, float]] = (0.25, 0.95),
    procrustes_cfg: bool = True,
    axis_shift_delta: Optional[float] = None,
) -> torch.Tensor:
    """
    Полу-AR волновая генерация (Rolling Diffusion) с поосевым bridge-шагом,
    кэшем KV промпта, 2B CFG и early-stop при кристаллизации устойчивого VOID-хвоста (>= 8 подряд).
    """
    raw_core = getattr(core, "module", core)
    was_training = raw_core.training
    raw_core.eval()

    B = V_prompt.shape[0]
    device = V_prompt.device
    delta = float(getattr(raw_core, "axis_shift_delta", 0.0)) if axis_shift_delta is None else float(axis_shift_delta)
    gen = torch.Generator(device=device) if seed is not None else None
    if gen is not None:
        gen.manual_seed(seed)

    x_prompt, prompt_cache = _prepare_sampler_prompt_cache(
        raw_core, V_prompt, prompt_valid_mask, use_amp=use_amp, amp_dtype=amp_dtype
    )

    sep_exp = sep_concept.to(device=device, dtype=torch.float32).view(1, 1, 8, 128).expand(B, 1, 8, 128)
    void_exp = void_ref.to(device=device, dtype=torch.float32).view(1, 1, 8, 128) if void_ref is not None else None
    noise = torch.randn((B, N_ANS, 8, 128), generator=gen, device=device, dtype=torch.float32) * SIGMA_NOISE
    V_t = torch.cat([sep_exp, noise], dim=1)

    cell_pos = torch.arange(N_ANS, device=device, dtype=torch.float32)
    u_grid = torch.linspace(0.0, 1.0, steps=num_steps + 1, device=device, dtype=torch.float32)
    m_cond = torch.ones(B, device=device, dtype=torch.float32)
    m_uncond = torch.zeros(B, device=device, dtype=torch.float32)
    uncond_mask = torch.zeros_like(prompt_valid_mask)
    active_rows = torch.ones(B, dtype=torch.bool, device=device)

    def get_t_ans(u: float) -> torch.Tensor:
        front = -window + u * (float(N_ANS) + window)
        return torch.clamp((cell_pos - front) / window, 0.0, 1.0)

    for k in range(num_steps):
        if not active_rows.any():
            break
        t_cur_ans = get_t_ans(float(u_grid[k].item()))
        t_next_ans = get_t_ans(float(u_grid[k + 1].item()))
        t_mat = torch.cat([torch.zeros(B, 1, device=device), t_cur_ans.unsqueeze(0).expand(B, -1)], dim=1)
        radar_sig = compute_radar_sigma(decoder, V_t, use_amp=use_amp, amp_dtype=amp_dtype)

        act_t = t_cur_ans[t_cur_ans > 1e-3]
        t_eff_mean = float(act_t.mean().item()) if act_t.numel() > 0 else 0.0
        eff_cfg = compute_limited_cfg_scale(t_eff_mean, cfg_scale, t_range=limited_cfg_t_range)
        v0, _ = _forward_sampler_v0(
            raw_core=raw_core,
            V_in=V_t,
            t_mat=t_mat,
            x_prompt=x_prompt,
            prompt_cache=prompt_cache,
            prompt_valid_mask=prompt_valid_mask,
            uncond_mask=uncond_mask,
            m_cond=m_cond,
            m_uncond=m_uncond,
            radar_sig=radar_sig,
            eff_cfg_scale=eff_cfg,
            cfg_rescale=cfg_rescale,
            procrustes_cfg=procrustes_cfg,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            want_sink=False,
        )

        dt_mask = (t_cur_ans - t_next_ans) > 1e-6
        active_cell = ((t_cur_ans > 1e-4) & dt_mask).view(1, N_ANS, 1, 1)
        tc_ax = compute_axis_t(t_cur_ans, delta=delta).view(1, N_ANS, 8, 1).clamp(min=1e-4)
        tn_ax = compute_axis_t(t_next_ans, delta=delta).view(1, N_ANS, 8, 1)
        V_next_ans = v0[:, 1:] + (tn_ax / tc_ax) * (V_t[:, 1:] - v0[:, 1:])
        snap = (t_next_ans <= 1e-5).view(1, N_ANS, 1, 1)
        updated_ans = torch.where(snap, v0[:, 1:], torch.where(active_cell, V_next_ans, V_t[:, 1:]))
        V_t[active_rows, 1:] = updated_ans[active_rows]

        # Early-stop для примеров, где в уже кристаллизованном префиксе сформировался блок из >= 8 VOID подряд
        if void_exp is not None:
            n_cryst = int((t_next_ans <= 1e-5).sum().item())
            if n_cryst >= 8:
                cryst_void = detect_void_mask(V_t[:, 1 : 1 + n_cryst], void_exp)
                tail_8_void = cryst_void[:, -8:].all(dim=-1)
                newly_done = active_rows & tail_8_void
                if newly_done.any():
                    V_t[newly_done, 1 + n_cryst :] = void_exp
                    active_rows = active_rows & (~newly_done)

    if was_training:
        raw_core.train()
    return V_t


@torch.no_grad()
def block_ar_sample(
    core: GrassmannianDiffusionCore,
    V_prompt: torch.Tensor,
    prompt_valid_mask: torch.Tensor,
    sep_concept: torch.Tensor,
    void_ref: torch.Tensor,
    block_size: int = 16,
    steps_per_block: int = 8,
    cfg_scale: float = 3.0,
    cfg_rescale: float = 0.7,
    decoder: Optional[nn.Module] = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    seed: Optional[int] = None,
    t_grid_mode: str = "sd3",
    t_grid_shift: float = 0.55,
    limited_cfg_t_range: Optional[Tuple[float, float]] = (0.25, 0.95),
    procrustes_cfg: bool = True,
    axis_shift_delta: Optional[float] = None,
) -> torch.Tensor:
    """Блочная полу-AR диффузия (Эйлер–Хьюн 2-го порядка + bridge-шаг + 2B CFG + кэш KV) с early-stop по VOID."""
    raw_core = getattr(core, "module", core)
    was_training = raw_core.training
    raw_core.eval()

    B = V_prompt.shape[0]
    device = V_prompt.device
    delta = float(getattr(raw_core, "axis_shift_delta", 0.0)) if axis_shift_delta is None else float(axis_shift_delta)
    gen = torch.Generator(device=device) if seed is not None else None
    if gen is not None:
        gen.manual_seed(seed)

    x_prompt, prompt_cache = _prepare_sampler_prompt_cache(
        raw_core, V_prompt, prompt_valid_mask, use_amp=use_amp, amp_dtype=amp_dtype
    )

    sep_exp = sep_concept.to(device=device, dtype=torch.float32).view(1, 1, 8, 128).expand(B, 1, 8, 128)
    noise = torch.randn((B, N_ANS, 8, 128), generator=gen, device=device, dtype=torch.float32) * SIGMA_NOISE
    V_t = torch.cat([sep_exp, noise], dim=1)

    t_grid = build_t_grid(steps_per_block, device=device, mode=t_grid_mode, shift=t_grid_shift)
    m_cond = torch.ones(B, device=device, dtype=torch.float32)
    m_uncond = torch.zeros(B, device=device, dtype=torch.float32)
    uncond_mask = torch.zeros_like(prompt_valid_mask)
    void_exp = void_ref.to(device=device, dtype=torch.float32).view(1, 1, 8, 128)
    active_rows = torch.ones(B, dtype=torch.bool, device=device)

    def predict_block_v0(V_in: torch.Tensor, s_idx: int, e_idx: int, t_val: float) -> torch.Tensor:
        t_mat = torch.ones(B, N_CANVAS, device=device, dtype=torch.float32)
        t_mat[:, :s_idx] = 0.0
        t_mat[:, s_idx:e_idx] = t_val
        radar_sig = compute_radar_sigma(decoder, V_in, use_amp=use_amp, amp_dtype=amp_dtype)
        eff_cfg = compute_limited_cfg_scale(t_val, cfg_scale, t_range=limited_cfg_t_range)
        v0, _ = _forward_sampler_v0(
            raw_core=raw_core,
            V_in=V_in,
            t_mat=t_mat,
            x_prompt=x_prompt,
            prompt_cache=prompt_cache,
            prompt_valid_mask=prompt_valid_mask,
            uncond_mask=uncond_mask,
            m_cond=m_cond,
            m_uncond=m_uncond,
            radar_sig=radar_sig,
            eff_cfg_scale=eff_cfg,
            cfg_rescale=cfg_rescale,
            procrustes_cfg=procrustes_cfg,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            want_sink=False,
        )
        return v0

    for s in range(1, N_CANVAS, block_size):
        if not active_rows.any():
            V_t[:, s:] = void_exp
            break
        e = min(N_CANVAS, s + block_size)
        for k in range(steps_per_block):
            tc = float(t_grid[k].item())
            tn = float(t_grid[k + 1].item())

            v0_1 = predict_block_v0(V_t, s, e, tc)
            if tn <= 1e-6:
                V_t[active_rows, s:e] = v0_1[active_rows, s:e]
                break

            tc_ax = compute_axis_t(torch.tensor(tc, device=device), delta=delta).view(1, 1, 8, 1).clamp(min=1e-4)
            tn_ax = compute_axis_t(torch.tensor(tn, device=device), delta=delta).view(1, 1, 8, 1)
            ratio_ax = tn_ax / tc_ax

            V_pred = V_t.clone()
            V_pred[active_rows, s:e] = v0_1[active_rows, s:e] + ratio_ax * (V_t[active_rows, s:e] - v0_1[active_rows, s:e])

            v0_2 = predict_block_v0(V_pred, s, e, tn)
            v0_bar = 0.5 * (v0_1[active_rows, s:e] + v0_2[active_rows, s:e])
            V_t[active_rows, s:e] = v0_bar + ratio_ax * (V_t[active_rows, s:e] - v0_bar)

        blk_void = detect_void_mask(V_t[:, s:e], void_exp)
        finished = blk_void.all(dim=-1)
        active_rows = active_rows & (~finished)

    if was_training:
        raw_core.train()
    return V_t


@torch.no_grad()
def crystal_radar_sample(
    core: GrassmannianDiffusionCore,
    V_prompt: torch.Tensor,
    prompt_valid_mask: torch.Tensor,
    sep_concept: torch.Tensor,
    decoder: Optional[nn.Module] = None,
    num_steps: int = 20,
    cfg_scale: float = 3.0,
    cfg_rescale: float = 0.7,
    crystal_thr: float = 0.04,
    crystal_start_t: float = 0.65,
    crystal_lock_max_t: Optional[float] = 0.15,
    void_ref: Optional[torch.Tensor] = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    seed: Optional[int] = None,
    t_grid_mode: str = "sd3",
    t_grid_shift: float = 0.55,
    limited_cfg_t_range: Optional[Tuple[float, float]] = (0.25, 0.95),
    procrustes_cfg: bool = True,
    axis_shift_delta: Optional[float] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Адаптивная кристаллизация под управлением NoiseRadarHead (§§2.1, 7.1–7.5):
    - Радар прогоняется только по незафиксированным (`~locked`) клеткам (§7.4);
    - Досрочный выход `if locked.all(): break` + опциональный early-stop по устойчивому VOID-хвосту (§7.5);
    - При `axis_shift_delta > 0` действует страховка `crystal_lock_max_t = 0.15` против раннего лока хвоста осей 4..7 (§9).
    """
    raw_core = getattr(core, "module", core)
    was_training = raw_core.training
    raw_core.eval()

    B = V_prompt.shape[0]
    device = V_prompt.device
    delta = float(getattr(raw_core, "axis_shift_delta", 0.0)) if axis_shift_delta is None else float(axis_shift_delta)
    effective_lock_t = (
        min(float(crystal_start_t), float(crystal_lock_max_t))
        if (crystal_lock_max_t is not None and delta > 1e-6)
        else float(crystal_start_t)
    )

    gen = torch.Generator(device=device) if seed is not None else None
    if gen is not None:
        gen.manual_seed(seed)

    x_prompt, prompt_cache = _prepare_sampler_prompt_cache(
        raw_core, V_prompt, prompt_valid_mask, use_amp=use_amp, amp_dtype=amp_dtype
    )

    sep_exp = sep_concept.to(device=device, dtype=torch.float32).view(1, 1, 8, 128).expand(B, 1, 8, 128)
    void_exp = void_ref.to(device=device, dtype=torch.float32).view(1, 1, 8, 128) if void_ref is not None else None
    noise = torch.randn((B, N_ANS, 8, 128), generator=gen, device=device, dtype=torch.float32) * SIGMA_NOISE
    V_t = torch.cat([sep_exp, noise], dim=1)

    t_grid = build_t_grid(num_steps, device=device, mode=t_grid_mode, shift=t_grid_shift)
    m_cond = torch.ones(B, device=device, dtype=torch.float32)
    m_uncond = torch.zeros(B, device=device, dtype=torch.float32)
    uncond_mask = torch.zeros_like(prompt_valid_mask)
    locked = torch.zeros((B, N_CANVAS), dtype=torch.bool, device=device)
    locked[:, 0] = True
    t_cell = torch.ones((B, N_CANVAS), device=device, dtype=torch.float32)
    t_cell[:, 0] = 0.0
    last_sink_weights = None

    for k in range(num_steps):
        if bool(locked.all()):
            break
        t_cur = float(t_grid[k].item())
        t_next = float(t_grid[k + 1].item())
        dt_base = t_cur - t_next

        t_global_mat = torch.full((B, N_CANVAS), t_cur, device=device, dtype=torch.float32)
        t_global_mat[:, 0] = 0.0
        radar_sig = compute_radar_sigma(
            decoder, V_t, use_amp=use_amp, amp_dtype=amp_dtype, active_mask=(~locked)
        )
        if radar_sig is not None:
            radar_sig = torch.where(locked, torch.zeros_like(radar_sig), radar_sig)

        eff_cfg = compute_limited_cfg_scale(t_cur, cfg_scale, t_range=limited_cfg_t_range)
        v0, sw = _forward_sampler_v0(
            raw_core=raw_core,
            V_in=V_t,
            t_mat=t_global_mat,
            x_prompt=x_prompt,
            prompt_cache=prompt_cache,
            prompt_valid_mask=prompt_valid_mask,
            uncond_mask=uncond_mask,
            m_cond=m_cond,
            m_uncond=m_uncond,
            radar_sig=radar_sig,
            eff_cfg_scale=eff_cfg,
            cfg_rescale=cfg_rescale,
            procrustes_cfg=procrustes_cfg,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            want_sink=(k == num_steps - 1),
        )
        if k == num_steps - 1:
            last_sink_weights = sw
        v0[:, 0] = sep_exp[:, 0]

        if t_next <= 1e-6:
            V_t = torch.where(locked.unsqueeze(-1).unsqueeze(-1), V_t, v0)
            break

        # Оцениваем чистоту предсказанного v0 строго на незафиксированных клетках (~locked)
        sig_v0 = compute_radar_sigma(
            decoder, v0, use_amp=use_amp, amp_dtype=amp_dtype, active_mask=(~locked)
        )
        if sig_v0 is not None:
            sig_ans = sig_v0[:, 1:].float()
            act_ans = (~locked[:, 1:]).float()
            sig_mean = (sig_ans * act_ans).sum(dim=-1, keepdim=True) / act_ans.sum(dim=-1, keepdim=True).clamp(min=1.0)
            sig_mean = sig_mean.clamp(min=1e-4)
            cool_speed = torch.clamp(sig_mean / sig_ans.clamp(min=1e-4), min=0.70, max=1.50)
            dt_mat = torch.cat([torch.zeros((B, 1), device=device), dt_base * cool_speed], dim=1)
        else:
            dt_mat = torch.full((B, N_CANVAS), dt_base, device=device, dtype=torch.float32)
            dt_mat[:, 0] = 0.0

        dt_eff = torch.minimum(dt_mat, t_cell)
        t_cell_next = (t_cell - dt_eff).clamp(min=0.0)
        tc_ax = compute_axis_t(t_cell, delta=delta).unsqueeze(-1).clamp(min=1e-4)
        tn_ax = compute_axis_t(t_cell_next, delta=delta).unsqueeze(-1)
        V_next = v0 + (tn_ax / tc_ax) * (V_t - v0)

        # Кристаллизуем клетки, где предсказание v0 уже чистое (sig_v0 < crystal_thr при t_next <= effective_lock_t) или t_cell дошел до 0
        newly_locked = (~locked) & (t_cell_next <= 1e-5)
        if sig_v0 is not None and t_next <= effective_lock_t:
            newly_locked = newly_locked | ((~locked) & (sig_v0 < crystal_thr))
        newly_locked[:, 0] = False

        V_t = torch.where(locked.unsqueeze(-1).unsqueeze(-1), V_t, V_next)
        if newly_locked.any():
            V_t = torch.where(newly_locked.unsqueeze(-1).unsqueeze(-1), v0, V_t)
            locked = locked | newly_locked
            t_cell_next = torch.where(locked, torch.zeros_like(t_cell_next), t_cell_next)

        # Устойчивый VOID-хвост: если все клетки с позиции j до конца уже залочены и являются VOID, фиксируем
        if void_exp is not None and t_next <= 0.35:
            is_v_pred = detect_void_mask(v0[:, 1:], void_exp)
            locked_void = locked[:, 1:] & is_v_pred
            for b in range(B):
                if bool(locked[b].all()):
                    continue
                # Ищем >= 8 подряд залоченных VOID-клеток -> фиксируем остаток хвоста как VOID
                run = 0
                cut_pos = -1
                for j in range(N_ANS):
                    if bool(locked_void[b, j].item()):
                        run += 1
                        if run >= 8:
                            cut_pos = j + 1
                            break
                    else:
                        run = 0
                if cut_pos >= 0 and cut_pos < N_ANS:
                    V_t[b, 1 + cut_pos :] = void_exp[0, 0]
                    locked[b, 1 + cut_pos :] = True
                    t_cell_next[b, 1 + cut_pos :] = 0.0

        t_cell = t_cell_next
        V_t[:, 0] = sep_exp[:, 0]
        if bool(locked.all()):
            break

    if was_training:
        raw_core.train()
    return V_t, last_sink_weights


@torch.no_grad()
def decode_latent_answers(
    V_canvas: torch.Tensor,
    decoder: nn.Module,
    tokenizer: Any,
    void_ref: torch.Tensor,
) -> Tuple[List[str], List[int]]:
    """
    Декодирование ответа после генерации:
    1. Отбрасываем ячейку 0 (SEP) -> V_ans (B, 127, 8, 128).
    2. Латентный фильтр VOID до декодера (§1.3).
    3. Вызов GrassmannianDecoder (eval) и сборка текста через WordAwareElasticChunker.detokenize_non_pad (ADR 137).
    """
    V_ans = V_canvas[:, 1:].float()
    is_void = detect_void_mask(V_ans, void_ref)
    B = V_ans.shape[0]
    texts: List[str] = []
    void_starts: List[int] = []

    for b in range(B):
        void_idx = torch.where(is_void[b])[0]
        if len(void_idx) > 0:
            n_keep = int(void_idx[0].item())
            void_starts.append(n_keep + 1)  # в координатах холста 1..127
        else:
            n_keep = N_ANS
            void_starts.append(-1)

        if n_keep == 0:
            texts.append("")
            continue

        logits = decoder(V_ans[b : b + 1, :n_keep].to(next(decoder.parameters()).dtype))
        pred_ids = logits.argmax(dim=-1)[0]
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
def _eval_denoising_window(
    raw_model: GrassmannianDiffusionCore,
    V_0: torch.Tensor,
    V_prompt: torch.Tensor,
    p_valid: torch.Tensor,
    t_range: Tuple[float, float],
    noise: torch.Tensor,
    use_amp: bool,
    amp_dtype: torch.dtype,
    decoder: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
    """Выполняет условный и безусловный проходы денойзинга в заданном диапазоне шума t (с датчиком s7)."""
    B = V_0.shape[0]
    device = V_0.device
    delta = float(getattr(raw_model, "axis_shift_delta", 0.0))
    t_val = torch.linspace(t_range[0], t_range[1], N_ANS, device=device).view(1, N_ANS).expand(B, N_ANS)
    t_mat = torch.cat([torch.zeros(B, 1, device=device), t_val], dim=1)
    V_t = add_axis_shifted_noise(V_0, t_mat, noise, delta=delta, sigma_noise=SIGMA_NOISE)
    V_t[:, 0] = V_0[:, 0]

    radar_sig = compute_radar_sigma(decoder, V_t, use_amp=use_amp, amp_dtype=amp_dtype)
    m_cond = torch.ones(B, device=device)
    m_uncond = torch.zeros(B, device=device)
    v_in = V_t.to(amp_dtype if use_amp else torch.float32)
    vp_in = V_prompt.to(amp_dtype if use_amp else torch.float32)

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        x_prompt, prompt_cache = _prepare_sampler_prompt_cache(
            raw_model, vp_in, p_valid, use_amp=use_amp, amp_dtype=amp_dtype
        )
        fw_c: Dict[str, Any] = dict(
            V_t=v_in,
            t_mat=t_mat,
            x_prompt=x_prompt,
            prompt_attn_mask=p_valid,
            m_prompt=m_cond,
            radar_sigma=radar_sig,
            return_sink_weights=True,
        )
        if prompt_cache is not None:
            fw_c["prompt_cache"] = prompt_cache
        V_0_hat, stats, sink_w = raw_model.forward_with_cached_prompt(**fw_c)
        fw_u = dict(fw_c)
        fw_u["prompt_attn_mask"] = torch.zeros_like(p_valid)
        fw_u["m_prompt"] = m_uncond
        fw_u["return_sink_weights"] = False
        V_0_uncond, _, _ = raw_model.forward_with_cached_prompt(**fw_u)
    return V_0_hat.float(), V_0_uncond.float(), stats, (sink_w.float() if sink_w is not None else None)


def _record_window_metrics(
    V_hat_ans: torch.Tensor,
    V_uncond_ans: torch.Tensor,
    V_gt_ans: torch.Tensor,
    sink_w: Optional[torch.Tensor],
    is_content: torch.Tensor,
    is_true_void: torch.Tensor,
    bucket: Dict[str, List[float]],
) -> None:
    """Накапливает Sim_Gr, CFG contrast, sink-доли, спектрально-ортогональные и блочные ошибки (§§1, 6)."""
    if is_content.any():
        bucket["sim"].append(float(sim_gr(V_hat_ans, V_gt_ans)[is_content].mean().item()))
        diff_frob = torch.sqrt(((V_hat_ans - V_uncond_ans) ** 2).sum(dim=(-2, -1)) + 1e-12)
        bucket["cfg"].append(float(diff_frob[is_content].mean().item()))
        if "orth_err" in bucket:
            vh_c = V_hat_ans[is_content].float()
            v0_c = V_gt_ans[is_content].float()
            Gh = torch.matmul(vh_c, vh_c.transpose(-2, -1))
            G0 = torch.matmul(v0_c, v0_c.transpose(-2, -1))
            diag_gh = torch.diagonal(Gh, dim1=-2, dim2=-1)
            diag_g0 = torch.diagonal(G0, dim1=-2, dim2=-1)
            gh_frob_sq = (Gh ** 2).sum(dim=(-2, -1))
            offdiag_sq = torch.clamp(gh_frob_sq - (diag_gh ** 2).sum(dim=-1), min=0.0)
            orth_err = torch.sqrt(offdiag_sq + 1e-12) / (torch.sqrt(gh_frob_sq + 1e-12) + 1e-8)
            bucket["orth_err"].append(float(orth_err.mean().item()))
            bucket["spec_l1"].append(float((diag_gh - diag_g0).abs().mean().item()))
            diff_c = (vh_c - v0_c).abs()
            bucket["err_sem"].append(float(diff_c[..., 0:96].mean().item()))
            bucket["err_syn"].append(float(diff_c[..., 96:112].mean().item()))
            bucket["err_pos"].append(float(diff_c[..., 112:128].mean().item()))
    if sink_w is not None:
        sw_mean = sink_w.mean(dim=1) if sink_w.ndim == 3 else sink_w
        sw_ans = sw_mean[:, 1:]
        if is_content.any():
            bucket["sink_c"].append(float(sw_ans[is_content].mean().item()))
        if is_true_void.any():
            bucket["sink_v"].append(float(sw_ans[is_true_void].mean().item()))


@torch.no_grad()
def evaluate_phase3(
    model: GrassmannianDiffusionCore,
    encoder: nn.Module,
    decoder: nn.Module,
    tokenizer: Any,
    val_dataloader: DataLoader,
    stopword_mask: torch.Tensor,
    sep_concept: torch.Tensor,
    void_concept: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    max_batches: int = 20,
    run_generation_check: bool = True,
    gen_steps: int = 20,
) -> Dict[str, Any]:
    """Полный расчет метрик Evaluation Gate Фазы 3 на низком [0.1, 0.5] и высоком [0.75, 0.95] шуме + NoiseRadarHead."""
    raw_model = getattr(model, "module", model)
    was_training = raw_model.training
    raw_model.eval()

    low_b: Dict[str, List[float]] = {
        "sim": [], "cfg": [], "sink_c": [], "sink_v": [],
        "orth_err": [], "spec_l1": [], "err_sem": [], "err_syn": [], "err_pos": [],
    }
    hi_b: Dict[str, List[float]] = {"sim": [], "cfg": [], "sink_c": [], "sink_v": []}
    rk_vals: List[float] = []
    radar_sig_vals: List[float] = []
    dict_cos_vals: List[float] = []
    tok_acc_low_vals: List[float] = []
    tok_acc_hi_vals: List[float] = []
    ce_loss_hi_vals: List[float] = []
    tp, fp, fn = 0, 0, 0
    void_start_exact, void_start_pm1, void_start_tot = 0, 0, 0
    full_canvas_ok, full_canvas_tot = 0, 0
    sum_m_acc, sum_m2_acc, cnt_acc = None, None, None
    gen_samples: List[Dict[str, str]] = []

    sep_cpu, _ = canonicalize_cell_basis(sep_concept.float())
    void_cpu = void_concept.float()
    sw_dev = stopword_mask.to(device)

    for b_idx, batch in enumerate(val_dataloader):
        if b_idx >= max_batches:
            break
        _t_b = time.time()
        print(f"    [eval_p3] batch {b_idx + 1}/{max_batches} start...", flush=True)
        prompt_ids = batch["prompt_ids"].to(device)
        ans_ids = batch["ans_ids"].to(device)
        n_real_prompt = batch["n_real_prompt"].to(device)
        n_real_ans = batch["n_real_ans"].to(device)
        B = prompt_ids.shape[0]

        w_p = batch["prompt_weights"].to(device) if "prompt_weights" in batch else make_pooling_weights(prompt_ids, sw_dev)
        w_a = batch["ans_weights"].to(device) if "ans_weights" in batch else make_pooling_weights(ans_ids, sw_dev)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            V_prompt_raw = encoder(prompt_ids, w_p).float()
            V_ans_raw = encoder(ans_ids, w_a).float()
        V_prompt, _ = canonicalize_cell_basis(V_prompt_raw)
        V_ans, _ = canonicalize_cell_basis(V_ans_raw)

        sep_exp = sep_cpu.to(device).view(1, 1, 8, 128).expand(B, 1, 8, 128)
        V_0 = torch.cat([sep_exp, V_ans], dim=1)
        p_valid = n_real_prompt > 0
        noise = torch.randn_like(V_0)

        V_0_hat, V_0_uncond, stats, sink_w = _eval_denoising_window(
            raw_model, V_0, V_prompt, p_valid, (0.1, 0.5), noise, use_amp, amp_dtype, decoder=decoder
        )
        V_0_hat_hi, V_0_uncond_hi, _, sink_w_hi = _eval_denoising_window(
            raw_model, V_0, V_prompt, p_valid, (0.75, 0.95), noise, use_amp, amp_dtype, decoder=decoder
        )

        # Накопление AdaLN статистик по всему валидационному прогону
        sm, sm2, cnt = stats
        sm = sm.float().sum(dim=0)
        sm2 = sm2.float().sum(dim=0)
        cnt = cnt.float().sum(dim=0)
        if sum_m_acc is None:
            sum_m_acc, sum_m2_acc, cnt_acc = sm, sm2, cnt
        else:
            sum_m_acc = sum_m_acc + sm
            sum_m2_acc = sum_m2_acc + sm2
            cnt_acc = cnt_acc + cnt

        V_hat_ans = V_0_hat[:, 1:]
        V_gt_ans = V_0[:, 1:]
        is_content = n_real_ans > 0
        is_true_void = ~is_content

        _record_window_metrics(V_hat_ans, V_0_uncond[:, 1:], V_gt_ans, sink_w, is_content, is_true_void, low_b)
        _record_window_metrics(V_0_hat_hi[:, 1:], V_0_uncond_hi[:, 1:], V_gt_ans, sink_w_hi, is_content, is_true_void, hi_b)
        if is_content.any():
            rk_vals.append(float(compute_rk_eff(V_hat_ans)[is_content].mean().item()))
            if decoder is not None and hasattr(decoder, "estimate_noise") and getattr(decoder, "noise_radar", None) is not None:
                dec_dtype = next(decoder.parameters()).dtype
                sig_c, _, rad_dict = decoder.estimate_noise(V_hat_ans.to(dec_dtype))
                radar_sig_vals.append(float(sig_c.float()[is_content].mean().item()))
                if "dict_stats" in rad_dict:
                    cos_c = rad_dict["dict_stats"].float().view(B, N_ANS, 4, 4)[..., 0].mean(dim=-1)
                    dict_cos_vals.append(float(cos_c[is_content].mean().item()))

            if decoder is not None:
                dec_dtype = next(decoder.parameters()).dtype
                ans_tgts = batch["ans_targets"].to(device) if "ans_targets" in batch else ans_ids
                real_tok_m = (ans_ids.view(B, N_ANS, 4) != PAD_TOKEN_ID) & is_content.unsqueeze(-1)
                real_tok_flat = real_tok_m.reshape(-1)
                if real_tok_flat.any():
                    h_low = decoder.decode_hidden(V_hat_ans.to(dec_dtype))
                    h_hi = decoder.decode_hidden(V_0_hat_hi[:, 1:].to(dec_dtype))
                    idx_r = torch.where(real_tok_flat)[0]
                    if idx_r.numel() > 512:
                        idx_r = idx_r[:512]
                    tgt_r = ans_ids.reshape(-1)[idx_r].long()
                    logits_low = decoder.get_tied_head_logits(h_low.reshape(-1, h_low.shape[-1])[idx_r])
                    logits_hi = decoder.get_tied_head_logits(h_hi.reshape(-1, h_hi.shape[-1])[idx_r])
                    tok_acc_low_vals.append(float((logits_low.argmax(dim=-1) == tgt_r).float().mean().item()))
                    tok_acc_hi_vals.append(float((logits_hi.argmax(dim=-1) == tgt_r).float().mean().item()))
                    ce_loss_hi_vals.append(float(F.cross_entropy(logits_hi.float(), tgt_r).item()))

        pred_void = detect_void_mask(V_hat_ans, void_cpu)
        tp += int((pred_void & is_true_void).sum().item())
        fp += int((pred_void & ~is_true_void).sum().item())
        fn += int((~pred_void & is_true_void).sum().item())

        for b in range(B):
            true_v = torch.where(is_true_void[b])[0]
            pred_v = torch.where(pred_void[b])[0]
            if len(true_v) > 0:
                void_start_tot += 1
                t_idx = int(true_v[0].item())
                p_idx = int(pred_v[0].item()) if len(pred_v) > 0 else N_ANS
                if p_idx == t_idx:
                    void_start_exact += 1
                if abs(p_idx - t_idx) <= 1:
                    void_start_pm1 += 1
            else:
                full_canvas_tot += 1
                if len(pred_v) == 0:
                    full_canvas_ok += 1

        # Генерация с нуля на первом батче (2 примера) + декодирование текста
        if run_generation_check and b_idx == 0 and B >= 1:
            n_gen = min(2, B)
            _t_gen = time.time()
            print(f"    [eval_p3] euler_heun_sample start (n_gen={n_gen}, steps={gen_steps})...", flush=True)
            V_gen, _ = euler_heun_sample(
                raw_model,
                V_prompt[:n_gen],
                p_valid[:n_gen],
                sep_cpu,
                num_steps=gen_steps,
                cfg_scale=3.0,
                decoder=decoder,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                seed=1234,
            )
            print(f"    [eval_p3] euler_heun_sample done in {time.time() - _t_gen:.2f}s; crystal_radar_sample start...", flush=True)
            _t_cryst = time.time()
            V_gen_cryst, _ = crystal_radar_sample(
                raw_model,
                V_prompt[:n_gen],
                p_valid[:n_gen],
                sep_cpu,
                decoder=decoder,
                void_ref=void_cpu,
                num_steps=gen_steps,
                cfg_scale=3.0,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                seed=1234,
            )
            print(f"    [eval_p3] crystal_radar_sample done in {time.time() - _t_cryst:.2f}s; decode_latent_answers start...", flush=True)
            _t_dec = time.time()
            gen_texts, gen_vstarts = decode_latent_answers(V_gen, decoder, tokenizer, void_cpu)
            cryst_texts, cryst_vstarts = decode_latent_answers(V_gen_cryst, decoder, tokenizer, void_cpu)
            print(f"    [eval_p3] decode_latent_answers done in {time.time() - _t_dec:.2f}s", flush=True)
            for g_i in range(n_gen):
                q_str = WordAwareElasticChunker.detokenize_non_pad(
                    prompt_ids[g_i], tokenizer=tokenizer, pad_token_id=PAD_TOKEN_ID, skip_special_tokens=True
                )
                gen_samples.append(
                    {
                        "prompt": q_str[:160],
                        "generated": gen_texts[g_i][:300],
                        "void_start": str(gen_vstarts[g_i]),
                        "generated_crystal": cryst_texts[g_i][:300],
                        "void_start_crystal": str(cryst_vstarts[g_i]),
                    }
                )
        print(f"    [eval_p3] batch {b_idx + 1}/{max_batches} done in {time.time() - _t_b:.2f}s", flush=True)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    void_f1 = (2 * precision * recall) / max(1e-8, precision + recall) if (tp + fp + fn) > 0 else 1.0

    sigma_mean, sigma_min = 0.0, 0.0
    if cnt_acc is not None and (cnt_acc > 0).any():
        cnt_c = cnt_acc.clamp(min=1.0).unsqueeze(-1)
        mean_m = sum_m_acc / cnt_c
        var_m = torch.clamp(sum_m2_acc / cnt_c - mean_m ** 2, min=1e-12)
        sig_l = torch.sqrt(var_m).mean(dim=-1)
        sigma_mean = float(sig_l.mean().item())
        sigma_min = float(sig_l.min().item())

    if was_training:
        raw_model.train()

    return {
        "val/sim_gr_denoising": float(np.mean(low_b["sim"])) if low_b["sim"] else 0.0,
        "val/cfg_contrast": float(np.mean(low_b["cfg"])) if low_b["cfg"] else 0.0,
        "val/sim_gr_hi": float(np.mean(hi_b["sim"])) if hi_b["sim"] else 0.0,
        "val/cfg_contrast_hi": float(np.mean(hi_b["cfg"])) if hi_b["cfg"] else 0.0,
        "val/rk_eff_content": float(np.mean(rk_vals)) if rk_vals else 0.0,
        "val/radar_sigma_cell": float(np.mean(radar_sig_vals)) if radar_sig_vals else 0.0,
        "val/dict_cos_max": float(np.mean(dict_cos_vals)) if dict_cos_vals else 0.0,
        "val/tok_acc_low": float(np.mean(tok_acc_low_vals)) if tok_acc_low_vals else 0.0,
        "val/tok_acc_hi": float(np.mean(tok_acc_hi_vals)) if tok_acc_hi_vals else 0.0,
        "val/ce_loss_hi": float(np.mean(ce_loss_hi_vals)) if ce_loss_hi_vals else 0.0,
        "val/void_f1": float(void_f1),
        "val/void_start_exact": float(void_start_exact / max(1, void_start_tot)),
        "val/void_start_pm1": float(void_start_pm1 / max(1, void_start_tot)),
        "val/full_canvas_no_void_acc": float(full_canvas_ok / max(1, full_canvas_tot)),
        "val/sink_attn_content": float(np.mean(low_b["sink_c"])) if low_b["sink_c"] else 0.0,
        "val/sink_attn_void": float(np.mean(low_b["sink_v"])) if low_b["sink_v"] else 0.0,
        "val/sink_attn_content_hi": float(np.mean(hi_b["sink_c"])) if hi_b["sink_c"] else 0.0,
        "val/sink_attn_void_hi": float(np.mean(hi_b["sink_v"])) if hi_b["sink_v"] else 0.0,
        "val/sigma_adaln_mean": sigma_mean,
        "val/sigma_adaln_min": sigma_min,
        "val/orth_err": float(np.mean(low_b["orth_err"])) if low_b["orth_err"] else 0.0,
        "val/spec_l1": float(np.mean(low_b["spec_l1"])) if low_b["spec_l1"] else 0.0,
        "val/err_sem": float(np.mean(low_b["err_sem"])) if low_b["err_sem"] else 0.0,
        "val/err_syn": float(np.mean(low_b["err_syn"])) if low_b["err_syn"] else 0.0,
        "val/err_pos": float(np.mean(low_b["err_pos"])) if low_b["err_pos"] else 0.0,
        "gen_samples": gen_samples,
    }


evaluate_phase2 = evaluate_phase3


# %% [markdown]
# ## 8. Чекпоинты, GCS и цикл обучения

# %%
def sync_to_gcs(local_path: str, gcs_dir: str) -> bool:
    if not gcs_dir:
        return False
    target = gcs_dir.rstrip("/") + "/" + os.path.basename(local_path)
    try:
        res = subprocess.run(["gsutil", "-q", "cp", local_path, target], capture_output=True, text=True)
        if res.returncode == 0:
            print(f"[GCS] Выгружено: {target}", flush=True)
            return True
    except Exception:
        pass
    return False


def verify_gcs_write_access(gcs_dir: str) -> bool:
    """Тестовое сохранение небольшого файла в GCS при старте для проверки прав и доступности бакета."""
    if not gcs_dir:
        return False
    test_name = "_gcs_write_test.txt"
    local_tmp = os.path.join("/tmp" if os.path.isdir("/tmp") else ".", test_name)
    target = gcs_dir.rstrip("/") + "/" + test_name
    try:
        with open(local_tmp, "w", encoding="utf-8") as f:
            f.write(f"BEBLaDII Phase 3 GCS write test at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
        res = subprocess.run(["gsutil", "-q", "cp", local_tmp, target], capture_output=True, text=True)
        if res.returncode == 0:
            print(f"[GCS Test] OK: тестовый файл успешно сохранён в {target}", flush=True)
            return True
        print(f"[GCS Test Warning] Ошибка записи в {target}: {res.stderr.strip()}", flush=True)
    except Exception as e:
        print(f"[GCS Test Warning] Исключение при тестовой записи в {target}: {e}", flush=True)
    finally:
        if os.path.exists(local_tmp):
            try:
                os.remove(local_tmp)
            except OSError:
                pass
    return False


def find_latest_checkpoint(checkpoint_dir: str, gcs_dir: Optional[str] = None) -> Optional[str]:
    local_files = glob.glob(os.path.join(checkpoint_dir, "phase3_step_*.pt"))
    if local_files:
        local_files.sort(key=lambda f: int(re.findall(r"\d+", os.path.basename(f))[-1]))
        return local_files[-1]

    if gcs_dir:
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
        except Exception:
            pass
    return None


def save_checkpoint(
    model: nn.Module,
    opt_muon: Optional[OrthMuon],
    opt_adamw: torch.optim.AdamW,
    scaler: Optional[torch.amp.GradScaler],
    step: int,
    metrics: Dict[str, float],
    path: str,
    ema: Optional[EMAModel] = None,
    sync_gcs_flag: bool = False,
    gcs_dir: str = "",
    keep_local: bool = False,
) -> None:
    raw_model = getattr(model, "module", model)
    payload: Dict[str, Any] = {
        "step": step,
        "model_state_dict": raw_model.state_dict(),
        "opt_adamw_state_dict": opt_adamw.state_dict(),
        "metrics": metrics,
        "ema_state_dict": ema.state_dict() if ema is not None else None,
    }
    if opt_muon is not None:
        payload["opt_muon_state_dict"] = opt_muon.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    torch.save(payload, path)
    print(f"[Checkpoint] Шаг {step} сохранен: {path}", flush=True)

    if sync_gcs_flag and gcs_dir:
        ok = sync_to_gcs(path, gcs_dir)
        if ok and not keep_local:
            try:
                os.remove(path)
            except Exception:
                pass


def migrate_cond_embedder_state_dict(
    sd: Dict[str, torch.Tensor],
    expected_in_dim: int = 72,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Автоматическая миграция старых чекпоинтов (где ConditionEmbedder.mlp[0] имел вход 200 = 128 t_local + 64 t_global + 8 sensors)
    в новый формат 72 = 64 t_global + 8 sensors через срез [:, 128:200].
    """
    key = "cond_embedder.mlp.0.weight"
    if key in sd and sd[key].ndim == 2 and sd[key].shape[1] == 200 and expected_in_dim == 72:
        sd = dict(sd)
        sd[key] = sd[key][:, 128:200].clone()
        if verbose:
            print(
                f"[Checkpoint Migration] {key}: (256, 200) -> (256, 72) "
                f"(оракульный t_local отсечен, веса t_global и s0..s7 сохранены).",
                flush=True,
            )
    return sd


def load_checkpoint(
    path: str,
    model: nn.Module,
    opt_muon: Optional[OrthMuon],
    opt_adamw: torch.optim.AdamW,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    ema: Optional[EMAModel] = None,
) -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw_model = getattr(model, "module", model)
    expected_in_dim = int(raw_model.cond_embedder.mlp[0].weight.shape[1])
    sd = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    migrated = (
        "cond_embedder.mlp.0.weight" in sd
        and sd["cond_embedder.mlp.0.weight"].ndim == 2
        and sd["cond_embedder.mlp.0.weight"].shape[1] == 200
        and expected_in_dim == 72
    )
    sd = migrate_cond_embedder_state_dict(sd, expected_in_dim=expected_in_dim, verbose=True)
    raw_model.load_state_dict(sd, strict=True)

    try:
        adamw_sd = ckpt["opt_adamw_state_dict"]
        if migrated and isinstance(adamw_sd, dict) and "state" in adamw_sd:
            for st in adamw_sd["state"].values():
                if isinstance(st, dict):
                    for buf_name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                        buf = st.get(buf_name)
                        if isinstance(buf, torch.Tensor) and buf.ndim == 2 and buf.shape == (256, 200):
                            st[buf_name] = buf[:, 128:200].clone()
        opt_adamw.load_state_dict(adamw_sd)
    except Exception as e:
        print(f"[Resume Warning] AdamW state не загружен: {e}", flush=True)
    if opt_muon is not None and "opt_muon_state_dict" in ckpt:
        try:
            opt_muon.load_state_dict(ckpt["opt_muon_state_dict"])
        except Exception as e:
            print(f"[Resume Warning] OrthMuon state не загружен: {e}", flush=True)
    if scaler is not None and "scaler_state_dict" in ckpt:
        try:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        except Exception:
            pass
    if ema is not None and ckpt.get("ema_state_dict") is not None:
        ema_sd = migrate_cond_embedder_state_dict(
            ckpt["ema_state_dict"], expected_in_dim=expected_in_dim, verbose=False
        )
        ema.load_state_dict(ema_sd)

    step = int(ckpt.get("step", 0))
    print(f"[Resume] Восстановлено состояние с шага {step} ({path}).", flush=True)
    return step


def run_training_loop(
    model: nn.Module,
    encoder: nn.Module,
    decoder: nn.Module,
    tokenizer: Any,
    dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    sampler: Optional[DistributedSampler],
    env: EnvironmentConfig,
    stopword_mask: torch.Tensor,
    sep_concept: torch.Tensor,
    void_concept: torch.Tensor,
    opt_muon: Optional[OrthMuon],
    opt_adamw: torch.optim.AdamW,
    ema: Optional[EMAModel],
    args: argparse.Namespace,
    start_step: int,
    is_main_process: bool,
    world_size: int,
    sync_gcs_flag: bool,
    use_wandb: bool,
) -> None:
    raw_model = getattr(model, "module", model)
    scaler = torch.amp.GradScaler("cuda", init_scale=16384.0, growth_interval=1000) if env.use_amp else None

    sw_dev = stopword_mask.to(env.device)
    sep_cpu, _ = canonicalize_cell_basis(sep_concept.float())
    void_cpu = void_concept.float()

    model.train()
    step = start_step
    epoch = 0
    best_sim_gr = -1.0
    data_iter = iter(dataloader)

    if is_main_process:
        n_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        print(
            f"=== Старт обучения Фазы 3 Plan C (params={n_params/1e6:.2f}M, "
            f"micro_batch={args.batch_size}, grad_accum={args.grad_accum_steps}, "
            f"world_size={world_size}, eff_batch={args.batch_size * args.grad_accum_steps * world_size}) ===",
            flush=True,
        )

    while step < args.max_steps:
        step += 1
        t0 = time.time()
        lr_mult = update_optimizers_lr(
            opt_muon, opt_adamw, step, args.max_steps, args.lr_warmup_steps, args.muon_lr
        )
        if opt_muon is not None:
            opt_muon.zero_grad(set_to_none=True)
        opt_adamw.zero_grad(set_to_none=True)

        accum_metrics: Dict[str, float] = {}
        for micro_idx in range(args.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                data_iter = iter(dataloader)
                batch = next(data_iter)

            prompt_ids = batch["prompt_ids"].to(env.device, non_blocking=True)
            ans_ids = batch["ans_ids"].to(env.device, non_blocking=True)
            ans_targets = (
                batch["ans_targets"].to(env.device, non_blocking=True)
                if "ans_targets" in batch
                else ans_ids
            )
            n_real_prompt = batch["n_real_prompt"].to(env.device, non_blocking=True)
            i_last = batch["i_last"].to(env.device, non_blocking=True)
            w_cell = batch["w_cell"].to(env.device, non_blocking=True)
            is_cw_cell = (
                batch["is_content_word_cell"].to(env.device, non_blocking=True)
                if "is_content_word_cell" in batch
                else None
            )
            B = prompt_ids.shape[0]

            # 1. On-the-fly кодирование замороженным Энкодером + канонизация базиса + NoiseRadarHead (s7)
            with torch.no_grad():
                w_p = batch["prompt_weights"].to(env.device, non_blocking=True) if "prompt_weights" in batch else make_pooling_weights(prompt_ids, sw_dev)
                w_a = batch["ans_weights"].to(env.device, non_blocking=True) if "ans_weights" in batch else make_pooling_weights(ans_ids, sw_dev)
                with torch.autocast(device_type=env.device.type, dtype=env.amp_dtype, enabled=env.use_amp):
                    V_prompt_raw = encoder(prompt_ids, w_p)
                    V_ans_raw = encoder(ans_ids, w_a)
                V_prompt, _ = canonicalize_cell_basis(V_prompt_raw.float())
                V_ans, _ = canonicalize_cell_basis(V_ans_raw.float())

                sep_exp = sep_cpu.to(device=env.device, dtype=torch.float32).view(1, 1, 8, 128).expand(B, 1, 8, 128)
                V_0 = torch.cat([sep_exp, V_ans], dim=1)

                p_valid = n_real_prompt > 0
                p_attn_mask, m_prompt = apply_prompt_dropout(p_valid)

                use_rollout = (
                    step >= getattr(args, "rollout_start_step", 1000)
                    and float(getattr(args, "rollout_prob", 0.35)) > 0.0
                    and float(torch.rand(1, device=env.device).item()) < float(getattr(args, "rollout_prob", 0.35))
                )
                if use_rollout:
                    V_t, t_mat, radar_sigma = build_rollout_noisy_canvas(
                        raw_model=raw_model,
                        decoder=decoder,
                        V_0=V_0,
                        V_prompt=V_prompt,
                        p_valid=p_valid,
                        p_attn_mask=p_attn_mask,
                        m_prompt=m_prompt,
                        sep_exp=sep_exp,
                        use_amp=env.use_amp,
                        amp_dtype=env.amp_dtype,
                    )
                else:
                    t_mat = sample_mixed_noise_schedule(B, env.device, i_last=i_last)
                    noise = torch.randn_like(V_0, dtype=torch.float32) * SIGMA_DATA
                    V_t = (1.0 - t_mat.unsqueeze(-1).unsqueeze(-1)) * V_0.float() + t_mat.unsqueeze(-1).unsqueeze(-1) * noise
                    V_t[:, 0] = sep_exp[:, 0].float()

                    # Датчик s7: оценка поклеточного шума от NoiseRadarHead (Выход А декодера Фазы 2)
                    radar_sigma = compute_radar_sigma(decoder, V_t, use_amp=env.use_amp, amp_dtype=env.amp_dtype)

            # 2. Прямой проход модели (DataParallel возвращает (V_0_hat, adaln_stats, sink_weights) без расчёта лосса внутри)
            ddp_sync = (micro_idx == args.grad_accum_steps - 1)
            no_sync_ctx = model.no_sync() if (isinstance(model, DDP) and not ddp_sync) else None
            if no_sync_ctx is not None:
                no_sync_ctx.__enter__()

            try:
                with torch.autocast(device_type=env.device.type, dtype=env.amp_dtype, enabled=env.use_amp):
                    V_0_hat, adaln_stats, sink_weights = model(
                        V_t=V_t.to(env.amp_dtype if env.use_amp else torch.float32),
                        t_mat=t_mat,
                        V_prompt=V_prompt.to(env.amp_dtype if env.use_amp else torch.float32),
                        prompt_valid_mask=p_valid,
                        prompt_attn_mask=p_attn_mask,
                        m_prompt=m_prompt,
                        radar_sigma=radar_sigma,
                        return_sink_weights=True,
                    )

                # 3. Гибридный лосс строго в fp32 и снаружи DataParallel на полном микро-батче
                loss, m_dict = compute_phase3_loss(
                    V_0_hat=V_0_hat,
                    V_0=V_0,
                    t_mat=t_mat,
                    w_cell=w_cell,
                    i_last=i_last,
                    adaln_stats=adaln_stats,
                    step=step,
                    m_prompt=m_prompt,
                    sink_weights=sink_weights,
                    is_content_word_cell=is_cw_cell,
                    world_size=world_size,
                )

                # 3б. Сквозной токенный Cross-Entropy через замороженный Декодер (только на t < ce_t_max и вне rollout)
                ce_max_w = float(getattr(args, "ce_weight", 0.05))
                ce_warmup = max(1, int(getattr(args, "ce_warmup_steps", 300)))
                ce_max_toks = int(getattr(args, "ce_max_tokens", 512))
                ce_t_max = float(getattr(args, "ce_t_max", 0.50))
                ce_sw_w = float(getattr(args, "ce_stopword_weight", 0.25))
                w_ce = ce_max_w * min(1.0, float(max(1, step - start_step)) / float(ce_warmup))
                if decoder is not None and ce_max_w > 0.0 and (not use_rollout):
                    l_ce, ce_tok_acc = compute_decoder_ce_loss(
                        decoder=decoder,
                        V_0_hat=V_0_hat,
                        t_mat=t_mat,
                        ans_targets=ans_targets,
                        ce_max_tokens=ce_max_toks,
                        ce_t_max=ce_t_max,
                        stopword_mask=sw_dev,
                        ce_stopword_weight=ce_sw_w,
                        use_amp=env.use_amp,
                        amp_dtype=env.amp_dtype,
                    )
                    loss = loss + w_ce * l_ce
                    m_dict["loss/ce"] = float(l_ce.item())
                    m_dict["loss/w_ce"] = float(w_ce)
                    m_dict["train/ce_tok_acc"] = float(ce_tok_acc)
                    m_dict["loss/total"] = float(loss.item())
                else:
                    m_dict["loss/ce"] = 0.0
                    m_dict["loss/w_ce"] = 0.0
                    m_dict["train/ce_tok_acc"] = 0.0

                scaled_loss = loss / float(args.grad_accum_steps)
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
            finally:
                if no_sync_ctx is not None:
                    no_sync_ctx.__exit__(None, None, None)

            for k, v in m_dict.items():
                accum_metrics[k] = accum_metrics.get(k, 0.0) + v / float(args.grad_accum_steps)

        # 4. Шаг оптимизаторов с GradScaler + unscale + раздельный clip_grad_norm (§5.1)
        if scaler is not None:
            if opt_muon is not None:
                scaler.unscale_(opt_muon)
            scaler.unscale_(opt_adamw)
            clip_phase3_grads(opt_muon, opt_adamw, adamw_max_norm=1.0, sphere_max_norm=10.0)
            if opt_muon is not None:
                scaler.step(opt_muon)
            scaler.step(opt_adamw)
            scaler.update()
        else:
            clip_phase3_grads(opt_muon, opt_adamw, adamw_max_norm=1.0, sphere_max_norm=10.0)
            if opt_muon is not None:
                opt_muon.step()
            opt_adamw.step()

        if ema is not None:
            ema.update(raw_model)

        dt = time.time() - t0
        if is_main_process and (step % args.log_every == 0 or step == 1):
            print(
                f"[Step {step:6d}/{args.max_steps}] "
                f"tot={accum_metrics['loss/total']:.4f} | "
                f"ce={accum_metrics.get('loss/ce', 0.0):.3f} (acc={accum_metrics.get('train/ce_tok_acc', 0.0):.3f}) | "
                f"basis={accum_metrics['loss/basis']:.4f} | "
                f"gr={accum_metrics['loss/gr']:.4f} | "
                f"en={accum_metrics['loss/energy']:.4f} | "
                f"adj={accum_metrics['loss/adj']:.4f} | "
                f"sink={accum_metrics['loss/sink']:.4f} | "
                f"adaln_sig={accum_metrics['adaln/sigma_mean']:.2e} | "
                f"lr_m={lr_mult:.3f} | {dt:.2f}s",
                flush=True,
            )
            if use_wandb:
                import wandb
                wandb.log({**accum_metrics, "lr/mult": lr_mult, "time/step_sec": dt}, step=step)

        # 5. Валидация на EMA-весах (только rank 0)
        if is_main_process and val_dataloader is not None and (step % args.eval_every == 0):
            if ema is not None:
                ema.apply_shadow(raw_model)
            val_res = evaluate_phase3(
                model=raw_model,
                encoder=encoder,
                decoder=decoder,
                tokenizer=tokenizer,
                val_dataloader=val_dataloader,
                stopword_mask=sw_dev,
                sep_concept=sep_cpu,
                void_concept=void_cpu,
                device=env.device,
                use_amp=env.use_amp,
                amp_dtype=env.amp_dtype,
                max_batches=args.eval_batches,
                run_generation_check=True,
                gen_steps=getattr(args, "gen_steps", 20),
            )
            if ema is not None:
                ema.restore(raw_model)

            gen_samples = val_res.pop("gen_samples", [])
            print(
                f"  [Eval @ {step}] Sim_Gr={val_res['val/sim_gr_denoising']:.4f} (hi={val_res['val/sim_gr_hi']:.4f}) | "
                f"TokAcc={val_res.get('val/tok_acc_low', 0.0):.3f} (hi={val_res.get('val/tok_acc_hi', 0.0):.3f}, CE_hi={val_res.get('val/ce_loss_hi', 0.0):.2f}) | "
                f"CFG={val_res['val/cfg_contrast']:.4f} (hi={val_res['val/cfg_contrast_hi']:.4f}) | "
                f"VOID_F1={val_res['val/void_f1']:.4f} (pm1={val_res['val/void_start_pm1']:.3f}) | "
                f"radar={val_res['val/radar_sigma_cell']:.3f} | "
                f"sink_c={val_res['val/sink_attn_content']:.3f} (hi={val_res['val/sink_attn_content_hi']:.3f}) | "
                f"rk={val_res['val/rk_eff_content']:.2f} | "
                f"adaln_min={val_res['val/sigma_adaln_min']:.2e}",
                flush=True,
            )
            for idx_g, gs in enumerate(gen_samples):
                print(
                    f"    [Sample {idx_g} Full] Q: {gs['prompt']!r} -> void_start={gs['void_start']} | A: {gs['generated']!r}",
                    flush=True,
                )
                if "generated_crystal" in gs:
                    print(
                        f"    [Sample {idx_g} Cryst] void_start={gs['void_start_crystal']} | A: {gs['generated_crystal']!r}",
                        flush=True,
                    )
            if use_wandb:
                import wandb
                wandb.log(val_res, step=step)

            if val_res["val/sim_gr_denoising"] > best_sim_gr:
                best_sim_gr = val_res["val/sim_gr_denoising"]
                best_path = os.path.join(env.checkpoint_dir, "phase3_best_model.pt")
                save_checkpoint(
                    model=raw_model,
                    opt_muon=opt_muon,
                    opt_adamw=opt_adamw,
                    scaler=scaler,
                    step=step,
                    metrics=val_res,
                    path=best_path,
                    ema=ema,
                    sync_gcs_flag=sync_gcs_flag,
                    gcs_dir=env.gcs_checkpoints_dir,
                    keep_local=True,
                )

        # 6. Регулярный чекпоинт (только rank 0)
        if is_main_process and (step % args.save_every == 0):
            ckpt_path = os.path.join(env.checkpoint_dir, f"phase3_step_{step}.pt")
            save_checkpoint(
                model=raw_model,
                opt_muon=opt_muon,
                opt_adamw=opt_adamw,
                scaler=scaler,
                step=step,
                metrics=accum_metrics,
                path=ckpt_path,
                ema=ema,
                sync_gcs_flag=sync_gcs_flag,
                gcs_dir=env.gcs_checkpoints_dir,
                keep_local=not env.is_kaggle,
            )

    if is_main_process:
        print("=== Обучение Фазы 3 Plan C завершено ===", flush=True)


# %% [markdown]
# ## 9. CLI и точка входа

# %%
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BEBLaDII Phase 2 Plan C Latent Diffusion Training")
    parser.add_argument("--smoke-test", "--smoke_test", dest="smoke_test", action="store_true", help="Быстрый smoke-test на 3 шага")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Устройство исполнения")
    parser.add_argument("--use-dataparallel", "--use_dataparallel", dest="use_dataparallel", action="store_true", default=True)
    parser.add_argument("--no-dataparallel", "--no_dataparallel", dest="no_dataparallel", action="store_true")
    parser.add_argument("--data-path", "--data_path", dest="data_path", type=str, default="auto")
    parser.add_argument("--val-data-path", "--val_data_path", dest="val_data_path", type=str, default="auto")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=16, help="Микро-батч на ноду (8 на T4 x2)")
    parser.add_argument("--grad-accum-steps", "--grad_accum_steps", dest="grad_accum_steps", type=int, default=4)
    parser.add_argument("--max-steps", "--max_steps", dest="max_steps", type=int, default=100000)
    parser.add_argument("--muon-lr", "--muon_lr", dest="muon_lr", type=float, default=1e-3)
    parser.add_argument("--adamw-lr", "--adamw_lr", dest="adamw_lr", type=float, default=3e-4)
    parser.add_argument("--scale-1d-lr", "--scale_1d_lr", dest="scale_1d_lr", type=float, default=1.2e-3)
    parser.add_argument("--lr-warmup-steps", "--lr_warmup_steps", dest="lr_warmup_steps", type=int, default=1000)
    parser.add_argument("--ema-decay", "--ema_decay", dest="ema_decay", type=float, default=0.999)
    parser.add_argument("--no-ema", "--no_ema", dest="no_ema", action="store_true")
    parser.add_argument("--log-every", "--log_every", dest="log_every", type=int, default=20)
    parser.add_argument("--eval-every", "--eval_every", dest="eval_every", type=int, default=500)
    parser.add_argument("--eval-batches", "--eval_batches", dest="eval_batches", type=int, default=20)
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
    parser.add_argument("--ce-weight", "--ce_weight", dest="ce_weight", type=float, default=0.05)
    parser.add_argument("--ce-warmup-steps", "--ce_warmup_steps", dest="ce_warmup_steps", type=int, default=300)
    parser.add_argument("--ce-max-tokens", "--ce_max_tokens", dest="ce_max_tokens", type=int, default=512)
    parser.add_argument("--ce-t-max", "--ce_t_max", dest="ce_t_max", type=float, default=0.50)
    parser.add_argument("--ce-stopword-weight", "--ce_stopword_weight", dest="ce_stopword_weight", type=float, default=0.25)
    args, _ = parser.parse_known_args()
    return args


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = parse_args()

    is_ddp = ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ) and torch.cuda.is_available()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1")) if is_ddp else 1
    rank = int(os.environ.get("RANK", "0")) if is_ddp else 0
    is_main_process = rank == 0

    if is_ddp:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    env = EnvironmentConfig.detect(
        override_device=args.device,
        local_rank=local_rank,
        override_output_dir=args.output_dir,
        override_checkpoint_dir=args.checkpoint_dir,
    )
    if args.gcs_dir:
        env.gcs_checkpoints_dir = args.gcs_dir
    sync_gcs_flag = (env.is_kaggle and not args.no_gcs) or (not args.no_gcs and args.gcs_dir != "")

    if args.smoke_test:
        if is_main_process:
            print("=== РЕЖИМ SMOKE TEST (ФАЗА 3) АКТИВИРОВАН ===", flush=True)
        args.max_steps = 2
        args.batch_size = 2
        args.grad_accum_steps = 1
        args.log_every = 1
        args.eval_every = 2
        args.eval_batches = 1
        args.save_every = 2
        args.lr_warmup_steps = 1
        args.gen_steps = 2
        args.wandb = "disabled"
        args.resume_ckpt = ""
        sync_gcs_flag = False

    if is_main_process and sync_gcs_flag:
        verify_gcs_write_access(env.gcs_checkpoints_dir)

    use_wandb = False
    wandb_mode = args.wandb if args.wandb != "auto" else env.wandb_mode
    if is_main_process and wandb_mode == "online":
        api_key = os.environ.get("WANDB_API_KEY", "").strip()
        if api_key:
            try:
                import wandb
                wandb.login(key=api_key)
                wandb.init(
                    project="BEBLaDII-Phase3-PlanC",
                    name=f"phase3-{'kaggle' if env.is_kaggle else 'local'}-{int(time.time())}",
                    config=vars(args),
                )
                use_wandb = True
            except Exception as e:
                print(f"[WandB Warning] {e}", flush=True)

    # 1. Загрузка замороженной Системы 1 (read-only) и опорных концептов
    tokenizer = get_tokenizer()
    encoder = get_encoder(weights_path=env.encoder_weights_path, device=env.device, eval_mode=True)
    decoder = get_decoder(weights_path=env.decoder_weights_path, encoder=encoder, device=env.device, eval_mode=True)
    for p in list(encoder.parameters()) + list(decoder.parameters()):
        p.requires_grad_(False)

    sep_concept = torch.load(env.sep_concept_path, map_location="cpu", weights_only=False)
    void_concept = torch.load(env.void_concept_path, map_location="cpu", weights_only=False)
    if os.path.exists(env.stopwords_path):
        sw_obj = torch.load(env.stopwords_path, map_location="cpu", weights_only=False)
        stopword_mask = (sw_obj["mask"] if isinstance(sw_obj, dict) else sw_obj).bool()
    else:
        stopword_mask = torch.zeros(131072, dtype=torch.bool)

    # 2. Создание обучаемого ядра диффузии (~148.50M)
    model = GrassmannianDiffusionCore(void_ref=void_concept).to(env.device)
    ema = EMAModel(model, decay=args.ema_decay, device=env.device) if not args.no_ema else None
    opt_muon, opt_adamw = create_phase2_optimizers(
        model, muon_lr=args.muon_lr, adamw_lr=args.adamw_lr, scale_1d_lr=args.scale_1d_lr
    )

    start_step = 0
    resume_path = None
    if args.resume_ckpt:
        if args.resume_ckpt == "auto":
            resume_path = find_latest_checkpoint(env.checkpoint_dir, env.gcs_checkpoints_dir if sync_gcs_flag else None)
        else:
            resume_path = args.resume_ckpt
    if resume_path and os.path.exists(resume_path):
        start_step = load_checkpoint(resume_path, model, opt_muon, opt_adamw, None, env.device, ema=ema)

    # 3. Датасеты
    train_path = args.data_path if args.data_path != "auto" else env.train_data_path
    val_path = args.val_data_path if args.val_data_path != "auto" else env.val_data_path
    max_rows = 32 if args.smoke_test else None

    train_dataset = PromptAnswerDataset(
        parquet_path=train_path,
        tokenizer=tokenizer,
        stopword_mask=stopword_mask,
        max_rows=max_rows,
        is_validation=False,
        elastic_drop_prob=args.elastic_drop_prob,
    )
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=env.num_workers,
        pin_memory=env.pin_memory,
        drop_last=True,
    )

    val_loader = None
    if is_main_process and val_path and os.path.exists(val_path):
        val_dataset = PromptAnswerDataset(
            parquet_path=val_path,
            tokenizer=tokenizer,
            stopword_mask=stopword_mask,
            max_rows=16 if args.smoke_test else 512,
            is_validation=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(args.batch_size, 8),
            shuffle=False,
            num_workers=env.num_workers,
            pin_memory=env.pin_memory,
        )

    # 4. Параллелизм на 2xT4 (DDP основной режим, DataParallel fallback)
    if env.device.type == "cuda" and torch.cuda.device_count() > 1:
        warmup_linalg_backends()

    wrapped_model: nn.Module = model
    if is_ddp:
        wrapped_model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    elif env.device.type == "cuda" and torch.cuda.device_count() > 1 and args.use_dataparallel and not args.no_dataparallel:
        if is_main_process:
            print(f"[DataParallel] Обнаружено {torch.cuda.device_count()} GPU! Включение DataParallel...", flush=True)
        wrapped_model = nn.DataParallel(model)

    try:
        run_training_loop(
            model=wrapped_model,
            encoder=encoder,
            decoder=decoder,
            tokenizer=tokenizer,
            dataloader=train_loader,
            val_dataloader=val_loader,
            sampler=sampler,
            env=env,
            stopword_mask=stopword_mask,
            sep_concept=sep_concept,
            void_concept=void_concept,
            opt_muon=opt_muon,
            opt_adamw=opt_adamw,
            ema=ema,
            args=args,
            start_step=start_step,
            is_main_process=is_main_process,
            world_size=world_size,
            sync_gcs_flag=sync_gcs_flag,
            use_wandb=use_wandb,
        )
    finally:
        if is_ddp and dist.is_initialized():
            dist.destroy_process_group()


# %% [markdown]
# ## 10. Конфигурация запуска (Kaggle / Локально)

# %%
TRAINING_CONFIG = {
    "smoke_test": False,
    "device": "auto",
    "use_dataparallel": True,
    "data_path": "auto",
    "val_data_path": "auto",
    "batch_size": 16,
    "grad_accum_steps": 4,
    "max_steps": 100000,
    "muon_lr": 1e-3,
    "adamw_lr": 3e-4,
    "scale_1d_lr": 1.2e-3,
    "lr_warmup_steps": 1000,
    "ema_decay": 0.999,
    "log_every": 20,
    "eval_every": 500,
    "eval_batches": 20,
    "save_every": 500,
    "resume_ckpt": "auto",
    "gcs_dir": "",
    "no_gcs": False,
    "wandb": "auto",
    "elastic_drop_prob": 0.0,
    "rollout_prob": 0.35,
    "rollout_start_step": 1000,
    "ce_weight": 0.05,
    "ce_warmup_steps": 300,
    "ce_max_tokens": 512,
    "ce_t_max": 0.50,
    "ce_stopword_weight": 0.25,
}

if __name__ == "__main__":
    cli_args = parse_args()
    is_notebook = any("ipykernel" in m for m in sys.modules) or not sys.argv[1:]
    if is_notebook:
        for k, v in TRAINING_CONFIG.items():
            if hasattr(cli_args, k):
                setattr(cli_args, k, v)
    main(cli_args)
