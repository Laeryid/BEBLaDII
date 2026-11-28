"""
BEBLaDII Phase 3 Alt (Plan C) — Гибридное Грассманианово ядро диффузии с 2D-векторным
вниманием в R^128, 2x Грассманиановым CA_Prompt и 3-слотовым DCA-пробросом.

Ключевые отличия от базового `experiments/phase 3/train_phase3.py` (~167.9M):
1. Слои Грассманианового векторного внимания (`GrassmannianSABlock` в R^128):
   - Холст хранится как набор из 8 независимых базисных векторов на клетку `(B, N, 8, 128)`.
   - Каждый вектор v_{i,k} in R^128 видит ВСЕ 8 векторов соседних клеток v_{j,0..7}
     (в локальном окне `r=6` + глобальный `SEP`, или по всему холсту в `GrassFull`)
     через общие проекции 128x128 (`Shared KV + Split K/V Channel Gating`).
   - Внутри одной клетки (j == i) вектор v_{i,k} автоматически взаимодействует со всеми
     7 остальными векторами своей клетки; RoPE задает единую позицию клетки i для всех 8 осей.
   - Поосевой `SwiGLU FFN` (128 -> grass_ffn -> 128) без перемешивания 8 осей.
2. Слои полного внимания (`FullSABlock` на 1024):
   - Собирают 8 векторов в целостное представление клетки `1024` для глобального синтеза,
     работы с составными концептами и большой памяти FFN.
3. 2 слоя Грассманианового Cross-Attention (`GrassmannianCABlock` в R^128):
   - Каждый базисный вектор ответа v_{i,k} in R^128 напрямую обращается ко всем 8*N_p=512
     базисным векторам промпта (из легкого `GrassmannianPromptAdapter` в R^128) + `k_sink`.
   - Сохранены все инварианты ADR 135/136: Shared KV, Token-Centered Contrast только на K_p,
     V_p без RoPE, однократный float32 softmax с `v_sink == 0`, Cross-Gated SwiGLU FFN.
4. 3-слотовый поосевой DCA-проброс (`PerAxisGRNv3`):
   - Стек `[x_in (вход итерации), h_ca (выход ближайшего CA), x_prev (предыдущий слой)]`.
   - Каждый из 8 векторов (128d) смешивается строго с самим собой по этим 3 слотам.
   - Два независимых GRN на блок (`dca_q` и `dca_kv`, совместимо с Shared KV; в CA — `dca_q`).
   - Использование `SiLU` при `w=0`, `b=[0, 0, 1]` дает строгое тождество `out == x_prev`
     на старте и гарантированно ненулевой градиент по `w` с 1-го шага.
5. Компактность (~38.6M параметров в пресете `standard` против ~167.9M, ~4.3x легче) +
   поддержка CPU/GPU, опциональное кэширование латентов энкодера (`--cache-latents`) и
   быстрый CPU-пресет (`--cpu-fast`).
"""

import argparse
import dataclasses
import datetime
import glob
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


# %% Настройка путей и импорт общих утилит Plan C / Phase 3
if "__file__" in globals():
    ALT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = ALT_DIR.parents[1]
elif os.path.exists("/kaggle/working/BEBLaDII"):
    PROJECT_ROOT = Path("/kaggle/working/BEBLaDII")
    ALT_DIR = PROJECT_ROOT / "experiments" / "phase 3 alt"
else:
    PROJECT_ROOT = Path.cwd()
    ALT_DIR = PROJECT_ROOT / "experiments" / "phase 3 alt"
PHASE3_DIR = PROJECT_ROOT / "experiments" / "phase 3"
for _p in (str(PROJECT_ROOT), str(PHASE3_DIR), str(ALT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from storage.components.decoder import get_decoder
from storage.components.encoder import get_encoder
from storage.components.quantization import estimate_model_memory_mb
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

from train_phase3 import (
    AxisBlockAdaLN,
    ConditionEmbedder,
    ConditionEmbedderV2,
    DEFAULT_AXIS_SIGMA,
    EMAModel,
    FourierTimeEmbedding,
    GrassmannianOutputHead,
    N_ANS,
    N_CANVAS,
    N_PROMPT,
    OrthMuon,
    PAD_TOKEN_ID,
    PerHeadRMSNorm,
    PromptAnswerDataset,
    SIGMA_DATA,
    SIGMA_NOISE,
    _clamp_fp16_safe,
    _clamp_swiglu,
    _compute_adaln_stats,
    _init_soft_gate_adaln,
    add_axis_shifted_noise,
    apply_prompt_dropout,
    block_ar_sample,
    build_rollout_noisy_canvas,
    clip_phase3_grads,
    compute_axis_t,
    compute_decoder_ce_loss,
    compute_phase3_loss,
    compute_radar_sigma,
    compute_rk_eff,
    crystal_radar_sample,
    decode_latent_answers,
    detect_void_mask,
    euler_heun_sample,
    evaluate_phase3,
    make_pooling_weights,
    resolve_file_path,
    rolling_wave_sample,
    sample_mixed_noise_schedule,
    save_checkpoint,
    setup_kaggle_secrets,
    sim_gr,
    update_optimizers_lr,
    verify_gcs_write_access,
    warmup_linalg_backends,
)
from vocab_optim import FastVocabOptimizer

NUM_AXES = 8
AXIS_DIM = 128
HIDDEN_SIZE = NUM_AXES * AXIS_DIM  # 1024


# %% 1. Конфигурация окружения Phase 3 Alt (CPU / GPU / Kaggle)
@dataclasses.dataclass
class AltEnvironmentConfig:
    """Конфигурация путей и устройства, изолированная в experiments/phase 3 alt."""
    is_kaggle: bool
    project_root: str
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
    use_grad_scaler: bool
    num_workers: int
    pin_memory: bool

    @classmethod
    def detect(
        cls,
        override_device: str = "auto",
        local_rank: int = 0,
        override_output_dir: str = "",
        override_checkpoint_dir: str = "",
        cpu_bf16: bool = False,
        amp_dtype_str: str = "auto",
    ) -> "AltEnvironmentConfig":
        is_kaggle = os.path.exists("/kaggle") or ("KAGGLE_KERNEL_RUN_TYPE" in os.environ)
        setup_kaggle_secrets(is_kaggle)
        project_root = str(PROJECT_ROOT)
        gcs_checkpoints_dir = "gs://bebladii-weigths-us/planC/Phase3_Alt_Diffusion/checkpoints/"

        if is_kaggle:
            output_dir = "/kaggle/working/output_phase3_alt"
            checkpoint_dir = "/kaggle/working/checkpoints/phase3_alt"
            search_roots = [
                "/kaggle/input/datasets/bogdanbuliakov/bebladii-planc-phase2-data",
                "/kaggle/input/datasets/bogdanbuliakov/bebladii-planc-phase1-data",
                "/kaggle/input/bebladii-planc-phase2-data",
                "/kaggle/input/bebladii-planc-phase1-data",
                "/kaggle/input",
                project_root,
            ]
            train_data_path = resolve_file_path(
                "train_phase6.parquet",
                [
                    "/kaggle/input/datasets/bogdanbuliakov/bebladii-planc-phase2-data/data/train_phase2.parquet",
                    "/kaggle/input/bebladii-planc-phase2-data/data/train_phase2.parquet",
                    os.path.join(project_root, "BEBLaDII-planB-Phase6-Data", "phase 6", "data", "train_phase6.parquet"),
                ],
                search_roots,
            )
            val_data_path = resolve_file_path(
                "val_phase6.parquet",
                [
                    "/kaggle/input/datasets/bogdanbuliakov/bebladii-planc-phase2-data/data/val_phase2.parquet",
                    "/kaggle/input/bebladii-planc-phase2-data/data/val_phase2.parquet",
                    os.path.join(project_root, "BEBLaDII-planB-Phase6-Data", "phase 6", "data", "val_phase6.parquet"),
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
                    "/kaggle/input/datasets/bogdanbuliakov/bebladii-planc-phase1-data/encoder/weights_emb_int8_fp16.pt",
                    "/kaggle/input/bebladii-planc-phase1-data/encoder/weights_emb_int8_fp16.pt",
                    os.path.join(project_root, "BEBLaDII-plan\u0421-Phase1-Data", "encoder", "weights.pt"),
                ],
                search_roots,
            )
            decoder_weights_path = resolve_file_path(
                "phase2_decoder_step_25000.pt",
                [
                    "/kaggle/input/datasets/bogdanbuliakov/bebladii-planc-phase2-data/decoder/phase2_decoder_step_25000.pt",
                    "/kaggle/input/bebladii-planc-phase2-data/decoder/phase2_decoder_step_25000.pt",
                    "/kaggle/input/datasets/bogdanbuliakov/bebladii-planc-phase2-data/decoder/weights_fp16.pt",
                    "/kaggle/input/bebladii-planc-phase2-data/decoder/weights_fp16.pt",
                    "/kaggle/input/datasets/bogdanbuliakov/bebladii-planc-phase2-data/decoder/weights.pt",
                    "/kaggle/input/bebladii-planc-phase2-data/decoder/weights.pt",
                    os.path.join(project_root, "storage", "components", "decoder", "weights.pt"),
                    os.path.join(project_root, "BEBLaDII-plan\u0421-Phase2-Data", "decoder", "phase2_decoder_step_25000.pt"),
                ],
                search_roots,
            )
            wandb_mode = "online" if "WANDB_API_KEY" in os.environ else "disabled"
        else:
            output_dir = str(ALT_DIR / "output")
            checkpoint_dir = str(ALT_DIR / "checkpoints")
            # Приоритетно используем указанный пользователем датасет Phase 6, затем fallback на Phase 2
            train_data_path = resolve_file_path(
                "train_phase6.parquet",
                [
                    os.path.join(project_root, "BEBLaDII-planB-Phase6-Data", "phase 6", "data", "train_phase6.parquet"),
                    os.path.join(project_root, "BEBLaDII-plan\u0421-Phase2-Data", "data", "train_phase2.parquet"),
                ],
                [],
            )
            val_data_path = resolve_file_path(
                "val_phase6.parquet",
                [
                    os.path.join(project_root, "BEBLaDII-planB-Phase6-Data", "phase 6", "data", "val_phase6.parquet"),
                    os.path.join(project_root, "BEBLaDII-plan\u0421-Phase2-Data", "data", "val_phase2.parquet"),
                ],
                [],
            )
            stopwords_path = resolve_file_path(
                "tekken_stopwords_mask.pt",
                [
                    os.path.join(project_root, "data", "plan C", "tekken_stopwords_mask.pt"),
                    os.path.join(project_root, "BEBLaDII-plan\u0421-Phase1-Data", "data", "tekken_stopwords_mask.pt"),
                ],
                [],
            )
            encoder_weights_path = resolve_file_path(
                "weights.pt",
                [
                    os.path.join(project_root, "storage", "components", "encoder", "weights.pt"),
                    os.path.join(project_root, "BEBLaDII-plan\u0421-Phase1-Data", "encoder", "weights.pt"),
                    os.path.join(project_root, "BEBLaDII-plan\u0421-Phase1-Data", "encoder", "weights_emb_int8_fp16.pt"),
                    os.path.join(project_root, "BEBLaDII-plan\u0421-Phase1-Data", "encoder", "weights_int8.pt"),
                ],
                [],
            )
            decoder_weights_path = resolve_file_path(
                "weights.pt",
                [
                    os.path.join(project_root, "storage", "components", "decoder", "weights.pt"),
                    os.path.join(project_root, "BEBLaDII-plan\u0421-Phase2-Data", "decoder", "weights_fp16.pt"),
                    os.path.join(project_root, "BEBLaDII-plan\u0421-Phase2-Data", "decoder", "weights_int8.pt"),
                    os.path.join(project_root, "BEBLaDII-plan\u0421-Phase2-Data", "decoder", "phase2_decoder_step_25000.pt"),
                ],
                [],
            )
            wandb_mode = "disabled"

        sep_concept_path = os.path.join(project_root, "storage", "components", "sep_concept.pt")
        void_concept_path = os.path.join(project_root, "storage", "components", "void_concept.pt")

        if override_device == "cpu":
            device = torch.device("cpu")
        elif override_device == "cuda":
            device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        if device.type == "cuda":
            if amp_dtype_str == "fp32":
                use_amp = False
                amp_dtype = torch.float32
                use_grad_scaler = False
            elif amp_dtype_str == "bf16" or (
                amp_dtype_str == "auto" and torch.cuda.is_bf16_supported()
            ):
                use_amp = True
                amp_dtype = torch.bfloat16
                use_grad_scaler = False
            else:
                # T4 (sm_75) и совместимые GPU: FP16 + GradScaler
                use_amp = True
                amp_dtype = torch.float16
                use_grad_scaler = True
        else:
            use_amp = bool(cpu_bf16)
            amp_dtype = torch.bfloat16 if cpu_bf16 else torch.float32
            use_grad_scaler = False

        num_workers = 2 if is_kaggle else 0
        pin_memory = device.type == "cuda"

        if override_output_dir:
            output_dir = override_output_dir
        if override_checkpoint_dir:
            checkpoint_dir = override_checkpoint_dir

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        return cls(
            is_kaggle=is_kaggle,
            project_root=project_root,
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
            use_grad_scaler=use_grad_scaler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )


# %% 2. Поосевой 3-слотовый DCA-проброс (`PerAxisGRNv3`)
class PerAxisGRNv3(nn.Module):
    """
    Поосевой (per-axis, per-channel) DeepCrossAttention / GRN-v3 на фиксированный стек слотов:
    - До первого CA-слоя (`has_ca=False`): 2 слота `[x_in, x_prev]`.
    - После первого CA-слоя (`has_ca=True`): 3 слота `[x_in, h_ca, x_prev]`.

    Каждый из 8 базисных векторов (128d) взаимодействует СТРОГО сам с собой по слотам
    (без перемешивания между 8 осями):
      dyn[b, n, k, s] = SiLU( sum_d w[k, d] * RMSNorm_128(S)[b, n, k, d, s] )
      out[b, n, k, d] = sum_s S[b, n, k, d, s] * (b[k, d, s] + dyn[b, n, k, s])

    Инициализация: b[..., -1] = 1.0 (x_prev), остальные b = 0.0, w = 0.0.
    - При w = 0: SiLU(0) == 0 -> строгое тождество `out == x_prev` на шаге 0;
    - При w = 0: SiLU'(0) == 0.5 -> гарантированный ненулевой градиент по `w` с 1-го шага.
    """

    def __init__(self, num_axes: int = NUM_AXES, axis_dim: int = AXIS_DIM, has_ca: bool = True):
        super().__init__()
        self.num_axes = num_axes
        self.axis_dim = axis_dim
        self.has_ca = has_ca
        self.num_slots = 3 if has_ca else 2

        b_init = torch.zeros(num_axes, axis_dim, self.num_slots)
        b_init[..., -1] = 1.0  # Последний слот всегда x_prev -> строгое тождество на старте
        self.b = nn.Parameter(b_init)
        self.w = nn.Parameter(torch.zeros(num_axes, axis_dim))

    def forward(
        self,
        x_in: torch.Tensor,
        h_ca: Optional[torch.Tensor],
        x_prev: torch.Tensor,
    ) -> torch.Tensor:
        if self.has_ca:
            if h_ca is None:
                h_ca = torch.zeros_like(x_prev)
            stack = torch.stack([x_in, h_ca, x_prev], dim=-1)  # (B, N, K, D, 3)
        else:
            stack = torch.stack([x_in, x_prev], dim=-1)        # (B, N, K, D, 2)

        orig_dtype = x_prev.dtype
        s_f = stack.float()
        # RMSNorm вдоль 128d независимо для каждого слота и каждой оси
        s_norm = s_f * torch.rsqrt(s_f.pow(2).mean(dim=-2, keepdim=True) + 1e-5)
        # Поосевой (k in 0..7) input-dependent вес для каждого слота
        logits = torch.einsum("bnkds,kd->bnks", s_norm, self.w.float())
        dyn = F.silu(logits)
        weights = self.b.float().unsqueeze(0).unsqueeze(0) + dyn.unsqueeze(-2)
        out = (s_f * weights).sum(dim=-1)
        return out.to(orig_dtype)


def _repeat_adaln_stats_to_4096(
    stats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Выравнивает размерность статистик AdaLN слоя (512 = 4*128 -> 4096 = 4*1024)
    повторением x8, сохраняя точное равенство `sigma_layer = mean(sqrt(var))` и полную
    совместимость с `compute_phase3_loss` и DataParallel/DDP gather.
    """
    sm, sm2, cnt = stats
    target = 4 * HIDDEN_SIZE
    d = sm.size(-1)
    if d == target:
        return sm, sm2, cnt
    rep = target // d
    return sm.repeat(1, rep), sm2.repeat(1, rep), cnt


def _safe_log_axis_norm(x: torch.Tensor) -> torch.Tensor:
    """Безопасный log1p(||x||_2) в float32 с +1e-8 под корнем (защита от NaN-градиента при x == 0, §3.4)."""
    return torch.log1p(torch.sqrt(x.float().pow(2).sum(dim=-1, keepdim=True) + 1e-8))


def _axis_rad_feat(x: torch.Tensor) -> torch.Tensor:
    """2D-радиусный признак оси [abs_ln, rel_ln] для Energy-Gated SwiGLU (§3.4)."""
    abs_ln = _safe_log_axis_norm(x)
    rel_ln = abs_ln - abs_ln.mean(dim=-2, keepdim=True)
    return torch.cat([abs_ln, rel_ln], dim=-1).to(x.dtype)


@torch.no_grad()
def _init_blockdiag_sphere(
    layer: SphereLinear,
    blocks: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    alpha: float = 0.15,
    extra_out_range: Optional[Tuple[int, int]] = None,
    extra_out_scale: float = 0.10,
) -> None:
    """
    Блочно-диагональная инициализация SphereLinear по границам [96 sem | 16 syn | 16 pos] (§1):
    внутри каждого блока — ортогональная матрица с весом sqrt(1 - alpha^2),
    вне блоков — случайный cross-talk с весом alpha, с проекцией строк на единичную сферу.
    """
    M, N = int(layer.out_features), int(layer.in_features)
    dev = layer.q_weight.device
    dtype = layer.q_weight.dtype

    diag_w = math.sqrt(max(0.0, 1.0 - float(alpha) ** 2))
    W_blk = torch.zeros(M, N, dtype=torch.float32)
    mask_blk = torch.zeros(M, N, dtype=torch.bool)
    for (o0, o1), (i0, i1) in blocks:
        m_b, n_b = o1 - o0, i1 - i0
        mat = torch.empty(m_b, n_b, dtype=torch.float32)
        nn.init.orthogonal_(mat)
        # Нормируем строки блока до единичной нормы (при m_b > n_b orthogonal_ даёт норму < 1)
        mat = F.normalize(mat, p=2, dim=-1, eps=1e-8)
        W_blk[o0:o1, i0:i1] = mat
        mask_blk[o0:o1, i0:i1] = True

    W_off = torch.randn(M, N, dtype=torch.float32)
    W_off = W_off.masked_fill(mask_blk, 0.0)
    off_norms = W_off.norm(p=2, dim=-1, keepdim=True)
    has_off = off_norms > 1e-8
    W_off = torch.where(has_off, W_off / off_norms.clamp(min=1e-8), torch.zeros_like(W_off))

    has_blk = mask_blk.any(dim=-1, keepdim=True)
    W = torch.where(has_blk, diag_w * W_blk + float(alpha) * W_off, W_off)
    # Если строка полностью без блока и без off (краевой случай), используем случайный единичный вектор
    row_n = W.norm(p=2, dim=-1, keepdim=True)
    W = torch.where(row_n > 1e-8, W / row_n.clamp(min=1e-8), F.normalize(torch.randn(M, N), p=2, dim=-1, eps=1e-8))

    layer.q_weight.data.copy_(W.to(device=dev, dtype=dtype))
    if extra_out_range is not None:
        r0, r1 = extra_out_range
        layer.row_scale.data[r0:r1].mul_(float(extra_out_scale))


def _apply_cell_rope_to_vectors(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Применяет RoPE с ОДИНАКОВЫМ позиционным углом клетки i ко всем K=8 векторам этой клетки.
    x: (B, H, N, K, d_h); cos/sin: (N, d_h) или батчевые (B, N, d_h) для правого выравнивания промпта (§4).
    """
    if cos.ndim == 2:
        cos_b = cos.to(dtype=x.dtype).view(1, 1, x.shape[2], 1, x.shape[4])
        sin_b = sin.to(dtype=x.dtype).view(1, 1, x.shape[2], 1, x.shape[4])
    else:
        cos_b = cos.to(dtype=x.dtype).view(x.shape[0], 1, x.shape[2], 1, x.shape[4])
        sin_b = sin.to(dtype=x.dtype).view(x.shape[0], 1, x.shape[2], 1, x.shape[4])
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rot = torch.cat((-x2, x1), dim=-1)
    return (x * cos_b) + (rot * sin_b)


def _get_right_aligned_prompt_rope(
    rope_grass: RotaryEmbedding128,
    x: torch.Tensor,
    prompt_valid_mask: Optional[torch.Tensor],
    rope_offset: int = N_PROMPT,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Правое выравнивание RoPE-позиций промпта вплотную к SEP (§4):
    реальные клетки 0..L_p-1 получают позиции (rope_offset - L_p)..(rope_offset - 1),
    т.е. последний токен промпта всегда на позиции 63, SEP на 64, ответ с 65 (инвариантно к Dynamic Crop!).
    """
    N_p = x.shape[1]
    max_pos = max(int(rope_offset), N_p)
    cos_all, sin_all = rope_grass(x, max_pos)
    if prompt_valid_mask is None:
        return cos_all[:N_p], sin_all[:N_p]
    L_p = prompt_valid_mask.long().sum(dim=-1, keepdim=True).clamp(min=1, max=max_pos)  # (B, 1)
    shift = max_pos - L_p
    pos_p = (torch.arange(N_p, device=x.device).unsqueeze(0) + shift).clamp(0, max_pos - 1)  # (B, N_p)
    return cos_all[pos_p], sin_all[pos_p]


class PHMLinear(nn.Module):
    """
    Факторизованный 2D-линейный слой над ячейкой X in R^{K_in x D_in}:
        Y = sum_{m=1..r} A_m @ X @ F_m^T  in R^{K_out x D_out}
    Вычисляется за 2 плотных GEMM, полностью совместим со SphereLinear, OrthMuon и EMA.
    """

    def __init__(
        self,
        in_axes: int = NUM_AXES,
        in_dim: int = AXIS_DIM,
        out_axes: int = NUM_AXES,
        out_dim: int = AXIS_DIM,
        rank: int = 4,
        cross_alpha: float = 0.25,
    ):
        super().__init__()
        self.in_axes = int(in_axes)
        self.in_dim = int(in_dim)
        self.out_axes = int(out_axes)
        self.out_dim = int(out_dim)
        self.rank = int(rank)
        self.in_features = self.in_axes * self.in_dim
        self.out_features = self.out_axes * self.out_dim

        self.f_proj = SphereLinear(self.in_dim, self.rank * self.out_dim, bias=False)
        self.axis_mix = nn.Parameter(torch.empty(self.out_axes, self.in_axes * self.rank))
        self._init_axis_mix(cross_alpha=cross_alpha)

    @torch.no_grad()
    def _init_axis_mix(self, cross_alpha: float = 0.25) -> None:
        A = torch.zeros(self.out_axes, self.in_axes, self.rank, dtype=torch.float32)
        if self.rank == 1 or cross_alpha <= 0.0:
            if self.out_axes == self.in_axes:
                A[:, :, 0] = torch.eye(self.out_axes, dtype=torch.float32)
            else:
                nn.init.orthogonal_(A[:, :, 0])
        else:
            main_w = math.sqrt(max(0.0, 1.0 - cross_alpha ** 2))
            if self.out_axes == self.in_axes:
                A[:, :, 0] = main_w * torch.eye(self.out_axes, dtype=torch.float32)
            else:
                nn.init.orthogonal_(A[:, :, 0])
                A[:, :, 0].mul_(main_w)
            noise = torch.randn(self.out_axes, self.in_axes, self.rank - 1, dtype=torch.float32)
            noise_norm = noise.flatten(1).norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6).view(self.out_axes, 1, 1)
            A[:, :, 1:] = (noise / noise_norm) * cross_alpha
        self.axis_mix.copy_(A.reshape(self.out_axes, self.in_axes * self.rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        was_3d = x.ndim == 3
        if was_3d:
            B, N, _ = x.shape
            x4 = x.view(B, N, self.in_axes, self.in_dim)
        else:
            B, N, _, _ = x.shape
            x4 = x

        h = self.f_proj(x4)
        h_mat = h.reshape(B * N, self.in_axes * self.rank, self.out_dim)
        a_mat = self.axis_mix.to(dtype=h_mat.dtype)
        y4 = torch.matmul(a_mat, h_mat).view(B, N, self.out_axes, self.out_dim)
        if was_3d:
            return y4.reshape(B, N, self.out_features)
        return y4


class GrassmannianOutputHeadV2(nn.Module):
    """
    EDM-подобная поосевая выходная голова Phase 3 Alt v2 (§2.2):
    - Аналитическая Винеровская база c_skip_base(t_k), c_out_base(t_k) по буферу axis_sigma (8,) и SIGMA_NOISE;
    - Обучаемые per-axis `lambda_keep` (bias -3.0, защита чистых якорей t_i=0 без оракула t_local)
      и `out_gain_mod` в AxisBlockAdaLN(num_mods=2, extra_dim=16);
    - Компактный PHMLinear(8, 128, 8, 128, rank=8) со стартовым output_gain=0.1 (~170K вместо 1.57M).
    """

    def __init__(
        self,
        cond_dim: int = 256,
        adaln_bottleneck_r: int = 64,
        axis_sigma: Tuple[float, ...] = DEFAULT_AXIS_SIGMA,
        sigma_noise: float = SIGMA_NOISE,
        axis_shift_delta: float = 0.35,
        phm_rank: int = 8,
        keep_bias: float = -3.0,
        output_gain_init: float = 0.1,
    ):
        super().__init__()
        self.sigma_noise = float(sigma_noise)
        self.axis_shift_delta = float(axis_shift_delta)
        self.keep_bias = float(keep_bias)
        self.register_buffer("axis_sigma", torch.tensor(list(axis_sigma), dtype=torch.float32), persistent=True)

        self.norm = RMSNorm(AXIS_DIM)
        self.adaln_out = AxisBlockAdaLN(
            cond_dim=cond_dim,
            num_mods=2,
            bottleneck_r=adaln_bottleneck_r,
            gate_indices=(),
            extra_dim=2 * NUM_AXES,
        )
        self.out_proj = PHMLinear(
            in_axes=NUM_AXES,
            in_dim=AXIS_DIM,
            out_axes=NUM_AXES,
            out_dim=AXIS_DIM,
            rank=int(phm_rank),
            cross_alpha=0.25,
        )
        self.output_gain = nn.Parameter(torch.full((NUM_AXES, 1), float(output_gain_init)))

    def forward(
        self,
        x_ans: torch.Tensor,
        V_t: torch.Tensor,
        t_mat: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        B, N, K, D = V_t.shape
        x4 = x_ans.view(B, N, K, D) if x_ans.ndim == 3 else x_ans
        (shift_out, scale_out), extra = self.adaln_out(c, x4.dtype)
        keep_logit, out_gain_mod = extra.chunk(2, dim=-1)  # каждый (B, N, 8)

        h = self.norm(x4) * (1.0 + scale_out) + shift_out
        delta = self.out_proj(h).float() * self.output_gain.float().view(1, 1, K, 1)

        t_bar = ConditionEmbedder.extract_t_global(t_mat, B, N)  # (B, N)
        t_k = compute_axis_t(t_bar, delta=self.axis_shift_delta, num_axes=K)  # (B, N, 8)
        s_k = self.axis_sigma.to(device=V_t.device, dtype=torch.float32).view(1, 1, K)
        s_n = self.sigma_noise

        denom_sq = ((1.0 - t_k) * s_k) ** 2 + (t_k * s_n) ** 2 + 1e-12
        c_skip_base = (1.0 - t_k) * (s_k ** 2) / denom_sq
        c_out_base = (t_k * s_k * s_n) / torch.sqrt(denom_sq)

        lam_keep = torch.sigmoid(keep_logit.float() + self.keep_bias)
        c_skip = c_skip_base + (1.0 - c_skip_base) * lam_keep
        c_out = c_out_base * (1.0 - lam_keep) * (1.0 + 0.5 * torch.tanh(out_gain_mod.float()))

        if N > 0:
            # Точное тождество на ячейке 0 (SEP) без оракульного t_local
            sep_mask = torch.zeros((1, N, 1), device=V_t.device, dtype=torch.bool)
            sep_mask[:, 0, :] = True
            c_skip = torch.where(sep_mask, torch.ones_like(c_skip), c_skip)
            c_out = torch.where(sep_mask, torch.zeros_like(c_out), c_out)

        V_0_raw = c_skip.unsqueeze(-1) * V_t.float() + c_out.unsqueeze(-1) * delta
        return clamp_to_unit_ball(V_0_raw)


# %% 3. Блоки Грассманианового внимания в R^128 и Full-1024 внимания
_SEM_SYN_POS_BLOCKS_128: List[Tuple[Tuple[int, int], Tuple[int, int]]] = [
    ((0, 96), (0, 96)),
    ((96, 112), (96, 112)),
    ((112, 128), (112, 128)),
]
_SEM_SYN_BLOCKS_112: List[Tuple[Tuple[int, int], Tuple[int, int]]] = [
    ((0, 96), (0, 96)),
    ((96, 112), (96, 112)),
]


class GrassmannianPromptAdapterLayer(nn.Module):
    """
    Один слой легкого Грассманианового PromptAdapter в R^128:
    Full векторное внимание по всем 8*N_p базисным векторам промпта (общие 128x128 веса)
    + Shared Axis-wise (Energy-Gated) SwiGLU FFN + LayerScale.
    """

    def __init__(
        self,
        axis_dim: int = AXIS_DIM,
        ffn_dim: int = 512,
        num_sub_heads: int = 4,
        init_layer_scale: float = 0.02,
        use_energy_gate: bool = False,
        blockdiag_alpha: float = 0.0,
    ):
        super().__init__()
        self.axis_dim = axis_dim
        self.num_sub_heads = num_sub_heads
        self.sub_head_dim = axis_dim // num_sub_heads
        self.use_energy_gate = bool(use_energy_gate)

        self.attn_norm = RMSNorm(axis_dim)
        self.norm_emb = nn.Linear(1, axis_dim)
        nn.init.zeros_(self.norm_emb.weight)
        nn.init.zeros_(self.norm_emb.bias)

        self.q_proj = SphereLinear(axis_dim, axis_dim, bias=False)
        self.kv_proj = SphereLinear(axis_dim, axis_dim, bias=False)
        self.o_proj = SphereLinear(axis_dim, axis_dim, bias=False)
        if blockdiag_alpha > 0.0 and axis_dim == 128:
            _init_blockdiag_sphere(self.q_proj, _SEM_SYN_POS_BLOCKS_128, alpha=blockdiag_alpha)
            _init_blockdiag_sphere(self.kv_proj, _SEM_SYN_POS_BLOCKS_128, alpha=blockdiag_alpha)
            _init_blockdiag_sphere(self.o_proj, _SEM_SYN_POS_BLOCKS_128, alpha=blockdiag_alpha)

        self.q_norm = PerHeadRMSNorm(num_sub_heads, self.sub_head_dim)
        self.k_norm = PerHeadRMSNorm(num_sub_heads, self.sub_head_dim)
        self.v_gate = nn.Parameter(torch.ones(1, num_sub_heads, 1, self.sub_head_dim))

        self.mlp_norm = RMSNorm(axis_dim)
        self.gate_proj = SphereLinear(axis_dim, ffn_dim, bias=False)
        self.up_proj = SphereLinear(axis_dim, ffn_dim, bias=False)
        self.down_proj = SphereLinear(ffn_dim, axis_dim, bias=False)
        if self.use_energy_gate:
            self.w_rad: Optional[nn.Linear] = nn.Linear(2, ffn_dim)
            nn.init.zeros_(self.w_rad.weight)
            nn.init.zeros_(self.w_rad.bias)
        else:
            self.w_rad = None

        self.ls_attn = nn.Parameter(torch.full((axis_dim,), init_layer_scale))
        self.ls_mlp = nn.Parameter(torch.full((axis_dim,), init_layer_scale))

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (B, N_p, K, 128)
        B, N_p, K, D = x.shape
        H, dh = self.num_sub_heads, self.sub_head_dim

        # Сохраняем информацию об энергии оси ||v_k|| перед RMSNorm (O(8)-эквивариантно, без NaN при x==0)
        log_norm = _safe_log_axis_norm(x).to(x.dtype)
        h = self.attn_norm(x) + self.norm_emb(log_norm)

        q = self.q_proj(h).view(B, N_p, K, H, dh).permute(0, 3, 1, 2, 4)
        u = self.kv_proj(h).view(B, N_p, K, H, dh).permute(0, 3, 1, 2, 4)

        q = self.q_norm(q.reshape(B, H, N_p * K, dh)).view(B, H, N_p, K, dh)
        k = self.k_norm(u.reshape(B, H, N_p * K, dh)).view(B, H, N_p, K, dh)
        v = (u.reshape(B, H, N_p * K, dh) * self.v_gate.to(u.dtype))  # V без RoPE

        q = _apply_cell_rope_to_vectors(q, cos, sin).reshape(B, H, N_p * K, dh)
        k = _apply_cell_rope_to_vectors(k, cos, sin).reshape(B, H, N_p * K, dh)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )
        attn_out = attn_out.view(B, H, N_p, K, dh).permute(0, 2, 3, 1, 4).reshape(B, N_p, K, D)
        x = x + self.ls_attn.to(x.dtype) * self.o_proj(attn_out)

        h_mlp = self.mlp_norm(x)
        gate_pre = self.gate_proj(h_mlp)
        if self.w_rad is not None:
            gate_pre = gate_pre + self.w_rad(_axis_rad_feat(x))
        gated = _clamp_swiglu(gate_pre, self.up_proj(h_mlp))
        mlp_out = _clamp_fp16_safe(self.down_proj(gated))
        x = x + self.ls_mlp.to(x.dtype) * mlp_out
        return x


class GrassmannianPromptAdapter(nn.Module):
    """2-слойный Грассманианов контекстуализатор чистого промпта в R^128 (~0.5M вместо 23.6M)."""

    def __init__(
        self,
        num_layers: int = 2,
        axis_dim: int = AXIS_DIM,
        ffn_dim: int = 512,
        num_sub_heads: int = 4,
        init_layer_scale: float = 0.02,
        prompt_right_align: bool = False,
        rope_offset: int = N_PROMPT,
        use_energy_gate: bool = False,
        blockdiag_alpha: float = 0.0,
    ):
        super().__init__()
        self.prompt_right_align = bool(prompt_right_align)
        self.rope_offset = int(rope_offset)
        self.layers = nn.ModuleList(
            [
                GrassmannianPromptAdapterLayer(
                    axis_dim=axis_dim,
                    ffn_dim=ffn_dim,
                    num_sub_heads=num_sub_heads,
                    init_layer_scale=init_layer_scale,
                    use_energy_gate=use_energy_gate,
                    blockdiag_alpha=blockdiag_alpha,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        V_prompt: torch.Tensor,
        rope_grass: RotaryEmbedding128,
        prompt_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # V_prompt: (B, N_p, 8, 128) -> x: (B, N_p, 8, 128)
        B, N_p, K, _ = V_prompt.shape
        x = (V_prompt.float() / SIGMA_NOISE).to(V_prompt.dtype)
        if self.prompt_right_align and prompt_valid_mask is not None:
            cos, sin = _get_right_aligned_prompt_rope(
                rope_grass, x, prompt_valid_mask, rope_offset=self.rope_offset
            )
        else:
            cos, sin = rope_grass(x, N_p)

        attn_mask = None
        if prompt_valid_mask is not None:
            safe_mask = prompt_valid_mask.clone()
            empty_rows = ~safe_mask.any(dim=-1)
            if empty_rows.any():
                safe_mask[empty_rows, 0] = True
            vec_mask = safe_mask.unsqueeze(-1).expand(B, N_p, K).reshape(B, N_p * K)
            attn_mask = torch.zeros(B, 1, 1, N_p * K, device=x.device, dtype=x.dtype)
            attn_mask = attn_mask.masked_fill(~vec_mask[:, None, None, :], -10000.0)

        for layer in self.layers:
            x = layer(x, cos, sin, attn_mask=attn_mask)
        return x


class GrassmannianSABlock(nn.Module):
    """
    Блок Грассманианового векторного внимания в R^128:
    - Вход: стек `[x_in, h_ca, x_prev]` формы `(B, N, 8, 128)` -> `dca_q` и `dca_kv` (PerAxisGRNv3).
    - Поддерживает v1 (4x32d + Linear AdaLN) и v2 (8x16d [6 sem | 1 syn | 1 pos] + блочно-диагональный
      старт + AxisBlockAdaLN D2 r=64 + Energy-Gated SwiGLU).
    """

    def __init__(
        self,
        axis_dim: int = AXIS_DIM,
        ffn_dim: int = 512,
        num_sub_heads: int = 4,
        local_radius: Optional[int] = 6,
        has_ca: bool = True,
        cond_dim: int = 256,
        adaln_type: str = "linear",
        adaln_bottleneck_r: int = 64,
        use_energy_gate: bool = False,
        blockdiag_alpha: float = 0.0,
    ):
        super().__init__()
        self.axis_dim = axis_dim
        self.num_sub_heads = num_sub_heads
        self.sub_head_dim = axis_dim // num_sub_heads
        self.local_radius = local_radius
        self.adaln_type = adaln_type
        self.adaln_bottleneck_r = int(adaln_bottleneck_r)
        self.use_energy_gate = bool(use_energy_gate)

        self.dca_q = PerAxisGRNv3(NUM_AXES, axis_dim, has_ca=has_ca)
        self.dca_kv = PerAxisGRNv3(NUM_AXES, axis_dim, has_ca=has_ca)

        self.attn_norm_q = RMSNorm(axis_dim)
        self.attn_norm_kv = RMSNorm(axis_dim)
        self.norm_emb_q = nn.Linear(1, axis_dim)
        self.norm_emb_kv = nn.Linear(1, axis_dim)
        nn.init.zeros_(self.norm_emb_q.weight)
        nn.init.zeros_(self.norm_emb_q.bias)
        nn.init.zeros_(self.norm_emb_kv.weight)
        nn.init.zeros_(self.norm_emb_kv.bias)

        self.q_proj = SphereLinear(axis_dim, axis_dim, bias=False)
        self.kv_proj = SphereLinear(axis_dim, axis_dim, bias=False)
        self.o_proj = SphereLinear(axis_dim, axis_dim, bias=False)
        if blockdiag_alpha > 0.0 and axis_dim == 128:
            _init_blockdiag_sphere(self.q_proj, _SEM_SYN_POS_BLOCKS_128, alpha=blockdiag_alpha)
            _init_blockdiag_sphere(self.kv_proj, _SEM_SYN_POS_BLOCKS_128, alpha=blockdiag_alpha)
            _init_blockdiag_sphere(self.o_proj, _SEM_SYN_POS_BLOCKS_128, alpha=blockdiag_alpha)

        self.q_norm = PerHeadRMSNorm(num_sub_heads, self.sub_head_dim)
        self.k_norm = PerHeadRMSNorm(num_sub_heads, self.sub_head_dim)
        self.v_gate = nn.Parameter(torch.ones(1, num_sub_heads, 1, self.sub_head_dim))

        self.mlp_norm = RMSNorm(axis_dim)
        self.gate_proj = SphereLinear(axis_dim, ffn_dim, bias=False)
        self.up_proj = SphereLinear(axis_dim, ffn_dim, bias=False)
        self.down_proj = SphereLinear(ffn_dim, axis_dim, bias=False)
        if self.use_energy_gate:
            self.w_rad: Optional[nn.Linear] = nn.Linear(2, ffn_dim)
            nn.init.zeros_(self.w_rad.weight)
            nn.init.zeros_(self.w_rad.bias)
        else:
            self.w_rad = None

        if adaln_type == "axis_block":
            self.adaln_proj: nn.Module = AxisBlockAdaLN(
                cond_dim=cond_dim,
                num_mods=6,
                bottleneck_r=self.adaln_bottleneck_r,
                gate_indices=(2, 5),
                gate_bias=0.02,
            )
        else:
            self.adaln_proj = nn.Linear(cond_dim, 6 * axis_dim)
            _init_soft_gate_adaln(self.adaln_proj, axis_dim, gate_bias=0.02)

        if local_radius is not None:
            self.register_buffer(
                "win_mask_default",
                self._build_window_mask(N_CANVAS, NUM_AXES, local_radius, torch.device("cpu")),
                persistent=False,
            )
        else:
            self.register_buffer("win_mask_default", None, persistent=False)

    def _compute_adaln_mods(
        self, c: torch.Tensor, out_dtype: torch.dtype, m_active: torch.Tensor
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        B, N, _ = c.shape
        if isinstance(self.adaln_proj, AxisBlockAdaLN):
            s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = self.adaln_proj(c, out_dtype)
            M = torch.cat([s_msa, sc_msa, s_mlp, sc_mlp], dim=-1).reshape(B, N, 4 * HIDDEN_SIZE)
            stats = _compute_adaln_stats(M, m_active)
            return s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp, stats
        mods = self.adaln_proj(c.to(self.adaln_proj.weight.dtype)).to(out_dtype)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods.chunk(6, dim=-1)
        M = torch.cat([shift_msa, scale_msa, shift_mlp, scale_mlp], dim=-1)
        stats = _repeat_adaln_stats_to_4096(_compute_adaln_stats(M, m_active))
        return (
            shift_msa.unsqueeze(2), scale_msa.unsqueeze(2), gate_msa.unsqueeze(2),
            shift_mlp.unsqueeze(2), scale_mlp.unsqueeze(2), gate_mlp.unsqueeze(2),
            stats,
        )

    @staticmethod
    def _build_window_mask(
        N: int,
        K: int,
        r: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Маска (1, 1, N, 1, (2*r + 2) * K) в float32:
        - Первые (2*r + 1)*K позиций: валидные клетки j = i + w - r in [0, N-1].
        - Последние K позиций (глобальный якорь SEP, j=0): открыты строго для i > r,
          чтобы не дублировать SEP для клеток i <= r (где j=0 уже внутри окна).
        """
        W = 2 * r + 1
        cell_i = torch.arange(N, device=device).unsqueeze(1)             # (N, 1)
        win_off = torch.arange(-r, r + 1, device=device).unsqueeze(0)    # (1, W)
        j_idx = cell_i + win_off                                         # (N, W)
        valid_win = (j_idx >= 0) & (j_idx < N)                           # (N, W)
        valid_sep = (cell_i > r)                                         # (N, 1)
        valid_cells = torch.cat([valid_win, valid_sep], dim=1)           # (N, W + 1)
        valid_vecs = valid_cells.unsqueeze(-1).expand(N, W + 1, K).reshape(1, 1, N, 1, (W + 1) * K)
        mask = torch.zeros_like(valid_vecs, dtype=torch.float32)
        return mask.masked_fill(~valid_vecs, -10000.0)

    def _local_window_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Явное локальное Грассманово внимание (B, H, N, K, dh):
        каждый из K=8 векторов клетки i видит все (2*r + 1)*K соседних векторов + K векторов SEP.
        """
        B, H, N, K, dh = q.shape
        r = int(self.local_radius)
        W = 2 * r + 1

        # Паддинг по оси N на r слева и справа + unfold в окна длины W
        k_pad = F.pad(k, (0, 0, 0, 0, r, r))  # (B, H, N + 2r, K, dh)
        v_pad = F.pad(v, (0, 0, 0, 0, r, r))
        k_win = k_pad.unfold(dimension=2, size=W, step=1).permute(0, 1, 2, 5, 3, 4).reshape(B, H, N, W * K, dh)
        v_win = v_pad.unfold(dimension=2, size=W, step=1).permute(0, 1, 2, 5, 3, 4).reshape(B, H, N, W * K, dh)

        # Добавляем глобальный якорь SEP (клетка 0)
        k_sep = k[:, :, 0:1, :, :].expand(B, H, N, K, dh)
        v_sep = v[:, :, 0:1, :, :].expand(B, H, N, K, dh)
        k_all = torch.cat([k_win, k_sep], dim=3)  # (B, H, N, (W+1)*K, dh)
        v_all = torch.cat([v_win, v_sep], dim=3)

        if self.win_mask_default is not None and N == N_CANVAS and K == NUM_AXES:
            win_mask = self.win_mask_default.to(device=q.device)
        else:
            win_mask = self._build_window_mask(N, K, r, q.device)

        scores = torch.matmul(q.float(), k_all.float().transpose(-2, -1)) / math.sqrt(dh)
        scores = scores + win_mask
        probs = F.softmax(scores, dim=-1).to(v_all.dtype)
        return torch.matmul(probs, v_all)  # (B, H, N, K, dh)

    def forward(
        self,
        x_in: torch.Tensor,
        h_ca: Optional[torch.Tensor],
        x_prev: torch.Tensor,
        c: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        m_active: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        B, N, K, D = x_prev.shape
        H, dh = self.num_sub_heads, self.sub_head_dim

        s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp, stats = self._compute_adaln_mods(c, x_prev.dtype, m_active)

        # 3-слотовый поосевой DCA: отдельные композиции для Q и Shared KV
        x_q = self.dca_q(x_in, h_ca, x_prev)
        x_kv = self.dca_kv(x_in, h_ca, x_prev)

        ln_q = _safe_log_axis_norm(x_q).to(x_q.dtype)
        ln_kv = _safe_log_axis_norm(x_kv).to(x_kv.dtype)
        h_q = (self.attn_norm_q(x_q) + self.norm_emb_q(ln_q)) * (1.0 + sc_msa) + s_msa
        h_kv = (self.attn_norm_kv(x_kv) + self.norm_emb_kv(ln_kv)) * (1.0 + sc_msa) + s_msa

        q = self.q_proj(h_q).view(B, N, K, H, dh).permute(0, 3, 1, 2, 4)
        u = self.kv_proj(h_kv).view(B, N, K, H, dh).permute(0, 3, 1, 2, 4)

        q = self.q_norm(q.reshape(B, H, N * K, dh)).view(B, H, N, K, dh)
        k = self.k_norm(u.reshape(B, H, N * K, dh)).view(B, H, N, K, dh)
        v_flat = u.reshape(B, H, N * K, dh) * self.v_gate.to(u.dtype)

        q = _apply_cell_rope_to_vectors(q, cos, sin)
        k = _apply_cell_rope_to_vectors(k, cos, sin)

        if self.local_radius is not None:
            v = v_flat.view(B, H, N, K, dh)
            attn_out = self._local_window_attention(q, k, v)
        else:
            q_f = q.reshape(B, H, N * K, dh)
            k_f = k.reshape(B, H, N * K, dh)
            attn_out = F.scaled_dot_product_attention(
                q_f, k_f, v_flat, attn_mask=None, dropout_p=0.0, is_causal=False
            ).view(B, H, N, K, dh)

        attn_out = attn_out.permute(0, 2, 3, 1, 4).reshape(B, N, K, D)
        x = x_prev + g_msa * self.o_proj(attn_out)

        h_mlp = self.mlp_norm(x) * (1.0 + sc_mlp) + s_mlp
        gate_pre = self.gate_proj(h_mlp)
        if self.w_rad is not None:
            gate_pre = gate_pre + self.w_rad(_axis_rad_feat(x))
        gated = _clamp_swiglu(gate_pre, self.up_proj(h_mlp))
        mlp_out = _clamp_fp16_safe(self.down_proj(gated))
        x = x + g_mlp * mlp_out
        return x, stats


class GrassmannianCABlock(nn.Module):
    """
    Грассманианов блок Cross-Attention на промпт в R^128 (v1: 4x32d=128d; v2: 7x16d=112d без pos):
    - Кэширование (k_full, v_p, attn_mask) промпта до цикла диффузии (§7.1);
    - Пропуск вычислений на безусловной ветке при инференсе и в 2B-батчевом CFG (§§7.2–7.3);
    - Правое выравнивание RoPE-позиций промпта вплотную к SEP (§4);
    - AxisBlockAdaLN D2 r=64 (§3.3) и Energy-Gated Cross-Gated SwiGLU (§3.4, 100% физическая изоляция CFG).
    """

    def __init__(
        self,
        axis_dim: int = AXIS_DIM,
        ffn_dim: int = 512,
        num_sub_heads: int = 4,
        qk_dim: Optional[int] = None,
        rope_offset: int = N_PROMPT,
        has_ca: bool = False,
        cond_dim: int = 256,
        prompt_right_align: bool = False,
        adaln_type: str = "linear",
        adaln_bottleneck_r: int = 64,
        use_energy_gate: bool = False,
        blockdiag_alpha: float = 0.0,
    ):
        super().__init__()
        self.axis_dim = int(axis_dim)
        self.qk_dim = int(axis_dim if qk_dim is None else qk_dim)
        self.num_sub_heads = int(num_sub_heads)
        if self.qk_dim % self.num_sub_heads != 0:
            raise ValueError(f"qk_dim={self.qk_dim} не делится на num_sub_heads={self.num_sub_heads}")
        self.sub_head_dim = self.qk_dim // self.num_sub_heads
        self.rope_offset = int(rope_offset)
        self.prompt_right_align = bool(prompt_right_align)
        self.adaln_type = adaln_type
        self.adaln_bottleneck_r = int(adaln_bottleneck_r)
        self.use_energy_gate = bool(use_energy_gate)

        self.dca_q = PerAxisGRNv3(NUM_AXES, axis_dim, has_ca=has_ca)

        self.ca_norm = RMSNorm(axis_dim)
        self.prompt_norm = RMSNorm(axis_dim)
        self.norm_emb_q = nn.Linear(1, axis_dim)
        nn.init.zeros_(self.norm_emb_q.weight)
        nn.init.zeros_(self.norm_emb_q.bias)

        self.q_proj = SphereLinear(self.qk_dim, self.qk_dim, bias=False)
        self.kv_proj = SphereLinear(self.qk_dim, self.qk_dim, bias=False)
        self.o_proj = SphereLinear(self.qk_dim, self.axis_dim, bias=False)
        if blockdiag_alpha > 0.0:
            if self.qk_dim == 112:
                _init_blockdiag_sphere(self.q_proj, _SEM_SYN_BLOCKS_112, alpha=blockdiag_alpha)
                _init_blockdiag_sphere(self.kv_proj, _SEM_SYN_BLOCKS_112, alpha=blockdiag_alpha)
                _init_blockdiag_sphere(
                    self.o_proj,
                    _SEM_SYN_BLOCKS_112,
                    alpha=blockdiag_alpha,
                    extra_out_range=(112, 128),
                    extra_out_scale=0.10,
                )
            elif self.qk_dim == 128:
                _init_blockdiag_sphere(self.q_proj, _SEM_SYN_POS_BLOCKS_128, alpha=blockdiag_alpha)
                _init_blockdiag_sphere(self.kv_proj, _SEM_SYN_POS_BLOCKS_128, alpha=blockdiag_alpha)
                _init_blockdiag_sphere(self.o_proj, _SEM_SYN_POS_BLOCKS_128, alpha=blockdiag_alpha)

        self.q_norm = PerHeadRMSNorm(self.num_sub_heads, self.sub_head_dim)
        self.k_norm = PerHeadRMSNorm(self.num_sub_heads, self.sub_head_dim)
        self.v_gate = nn.Parameter(torch.ones(1, self.num_sub_heads, 1, self.sub_head_dim))

        # Zero-Vector Attention Sink (без RoPE, v_sink == 0)
        self.k_sink = nn.Parameter(torch.randn(1, self.num_sub_heads, 1, self.sub_head_dim) * 0.02)

        self.mlp_norm = RMSNorm(axis_dim)
        self.gate_proj = SphereLinear(axis_dim, ffn_dim, bias=False)
        self.w_canvas_gate = SphereLinear(axis_dim, ffn_dim, bias=False)
        self.up_proj = SphereLinear(axis_dim, ffn_dim, bias=False)
        self.down_proj = SphereLinear(ffn_dim, axis_dim, bias=False)
        if self.use_energy_gate:
            self.w_rad: Optional[nn.Linear] = nn.Linear(2, ffn_dim)
            nn.init.zeros_(self.w_rad.weight)
            nn.init.zeros_(self.w_rad.bias)
        else:
            self.w_rad = None

        if adaln_type == "axis_block":
            self.adaln_proj: nn.Module = AxisBlockAdaLN(
                cond_dim=cond_dim,
                num_mods=6,
                bottleneck_r=self.adaln_bottleneck_r,
                gate_indices=(2, 5),
                gate_bias=0.02,
            )
        else:
            self.adaln_proj = nn.Linear(cond_dim, 6 * axis_dim)
            _init_soft_gate_adaln(self.adaln_proj, axis_dim, gate_bias=0.02)

    def _compute_adaln_mods(
        self, c: torch.Tensor, out_dtype: torch.dtype, m_active: torch.Tensor
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        B, N, _ = c.shape
        if isinstance(self.adaln_proj, AxisBlockAdaLN):
            s_ca, sc_ca, g_ca, s_mlp, sc_mlp, g_mlp = self.adaln_proj(c, out_dtype)
            M = torch.cat([s_ca, sc_ca, s_mlp, sc_mlp], dim=-1).reshape(B, N, 4 * HIDDEN_SIZE)
            stats = _compute_adaln_stats(M, m_active)
            return s_ca, sc_ca, g_ca, s_mlp, sc_mlp, g_mlp, stats
        mods = self.adaln_proj(c.to(self.adaln_proj.weight.dtype)).to(out_dtype)
        shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = mods.chunk(6, dim=-1)
        M = torch.cat([shift_ca, scale_ca, shift_mlp, scale_mlp], dim=-1)
        stats = _repeat_adaln_stats_to_4096(_compute_adaln_stats(M, m_active))
        return (
            shift_ca.unsqueeze(2), scale_ca.unsqueeze(2), gate_ca.unsqueeze(2),
            shift_mlp.unsqueeze(2), scale_mlp.unsqueeze(2), gate_mlp.unsqueeze(2),
            stats,
        )

    def prepare_prompt_kv(
        self,
        x_prompt: torch.Tensor,
        rope_grass: RotaryEmbedding128,
        prompt_attn_mask: torch.Tensor,
        prompt_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Однократно готовит (k_full, v_p, attn_mask) промпта для данного CA-слоя (§7.1)."""
        B, N_p, K, _ = x_prompt.shape
        H, dh = self.num_sub_heads, self.sub_head_dim

        h_prompt_in = self.prompt_norm(x_prompt)[..., : self.qk_dim]
        u_p = self.kv_proj(h_prompt_in).view(B, N_p, K, H, dh).permute(0, 3, 1, 2, 4)
        u_p_flat = u_p.reshape(B, H, N_p * K, dh)

        p_vec_mask = prompt_attn_mask.unsqueeze(-1).expand(B, N_p, K).reshape(B, N_p * K)
        mask_p_f = p_vec_mask.float().view(B, 1, N_p * K, 1)
        denom = mask_p_f.sum(dim=2, keepdim=True).clamp(min=1.0)
        u_p_mean = (u_p_flat.float() * mask_p_f).sum(dim=2, keepdim=True) / denom
        u_p_centered = (u_p_flat.float() - 0.5 * u_p_mean).to(u_p_flat.dtype)

        k_p = self.k_norm(u_p_centered).view(B, H, N_p, K, dh)
        v_p = u_p_flat * self.v_gate.to(u_p_flat.dtype)  # V никогда не проходит через RoPE

        if self.prompt_right_align:
            rope_mask = prompt_valid_mask if prompt_valid_mask is not None else prompt_attn_mask
            cos_p, sin_p = _get_right_aligned_prompt_rope(
                rope_grass, x_prompt, rope_mask, rope_offset=self.rope_offset
            )
        else:
            cos_all, sin_all = rope_grass(x_prompt, max(self.rope_offset, N_p))
            cos_p, sin_p = cos_all[:N_p], sin_all[:N_p]

        k_p_rot = _apply_cell_rope_to_vectors(k_p, cos_p, sin_p).reshape(B, H, N_p * K, dh)
        k_sink_exp = self.k_norm(self.k_sink.to(k_p_rot.dtype)).expand(B, -1, -1, -1)
        k_full = torch.cat([k_p_rot, k_sink_exp], dim=2)  # (B, H, N_p*K + 1, dh)

        attn_mask = torch.zeros(B, 1, 1, N_p * K + 1, device=x_prompt.device, dtype=torch.float32)
        attn_mask[:, :, :, : N_p * K] = attn_mask[:, :, :, : N_p * K].masked_fill(
            ~p_vec_mask[:, None, None, :], -10000.0
        )
        return {
            "k_full": k_full,
            "v_p": v_p,
            "attn_mask": attn_mask,
            "n_p_k": N_p * K,
        }

    def _forward_core(
        self,
        x_in: torch.Tensor,
        h_ca_prev: Optional[torch.Tensor],
        x_prev: torch.Tensor,
        x_prompt: torch.Tensor,
        c: torch.Tensor,
        rope_grass: RotaryEmbedding128,
        prompt_attn_mask: torch.Tensor,
        m_active: torch.Tensor,
        return_sink_weights: bool = False,
        prompt_valid_mask: Optional[torch.Tensor] = None,
        prompt_kv_cache: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        B, N_a, K, D = x_prev.shape
        H, dh = self.num_sub_heads, self.sub_head_dim

        s_ca, sc_ca, g_ca, s_mlp, sc_mlp, g_mlp, stats = self._compute_adaln_mods(c, x_prev.dtype, m_active)

        x_q = self.dca_q(x_in, h_ca_prev, x_prev)
        ln_q = _safe_log_axis_norm(x_q).to(x_q.dtype)
        h_ans_in = (self.ca_norm(x_q) + self.norm_emb_q(ln_q)) * (1.0 + sc_ca) + s_ca

        q = self.q_proj(h_ans_in[..., : self.qk_dim]).view(B, N_a, K, H, dh).permute(0, 3, 1, 2, 4)
        q = self.q_norm(q.reshape(B, H, N_a * K, dh)).view(B, H, N_a, K, dh)

        cos_all, sin_all = rope_grass(x_prev, self.rope_offset + N_a)
        cos_a = cos_all[self.rope_offset : self.rope_offset + N_a]
        sin_a = sin_all[self.rope_offset : self.rope_offset + N_a]
        q_rot = _apply_cell_rope_to_vectors(q, cos_a, sin_a).reshape(B, H, N_a * K, dh)

        if prompt_kv_cache is not None:
            k_full = prompt_kv_cache["k_full"]
            v_p = prompt_kv_cache["v_p"]
            attn_mask = prompt_kv_cache["attn_mask"]
            n_p_k = int(prompt_kv_cache["n_p_k"])
        else:
            kv_dict = self.prepare_prompt_kv(
                x_prompt=x_prompt,
                rope_grass=rope_grass,
                prompt_attn_mask=prompt_attn_mask,
                prompt_valid_mask=prompt_valid_mask,
            )
            k_full = kv_dict["k_full"]
            v_p = kv_dict["v_p"]
            attn_mask = kv_dict["attn_mask"]
            n_p_k = int(kv_dict["n_p_k"])

        # Однократный расчёт softmax в float32 (ADR 136)
        scores = torch.matmul(q_rot.float(), k_full.float().transpose(-2, -1)) / math.sqrt(dh)
        scores = scores + attn_mask
        probs = F.softmax(scores, dim=-1)

        sink_weights = None
        if return_sink_weights:
            sink_weights = probs[:, :, :, -1].view(B, H, N_a, K).mean(dim=(1, 3))

        # v_sink == 0 -> учитываем только первые n_p_k ключей промпта
        attn_out = torch.matmul(probs[:, :, :, :n_p_k].to(v_p.dtype), v_p)
        attn_out = attn_out.view(B, H, N_a, K, dh).permute(0, 2, 3, 1, 4).reshape(B, N_a, K, self.qk_dim)
        h_ca = self.o_proj(attn_out)
        delta_attn = g_ca * h_ca
        x_mid = x_prev + delta_attn

        # Cross-Gated (Energy-Gated) SwiGLU FFN в R^128: val строго из h_ca (физическая изоляция при CFG dropout)
        x_tilde = self.mlp_norm(x_mid) * (1.0 + sc_mlp) + s_mlp
        gate_pre = self.gate_proj(h_ca).float() + self.w_canvas_gate(x_tilde).float()
        if self.w_rad is not None:
            gate_pre = gate_pre + self.w_rad(_axis_rad_feat(x_mid)).float()
        prod = F.silu(gate_pre) * self.up_proj(h_ca).float()
        gated = torch.clamp(prod, -65000.0, 65000.0).to(x_prev.dtype)
        ffn_out = _clamp_fp16_safe(self.down_proj(gated))
        delta_mlp = g_mlp * ffn_out

        ca_delta = delta_attn + delta_mlp
        x_out = x_prev + ca_delta
        return x_out, ca_delta, stats, sink_weights

    def forward(
        self,
        x_in: torch.Tensor,
        h_ca_prev: Optional[torch.Tensor],
        x_prev: torch.Tensor,
        x_prompt: torch.Tensor,
        c: torch.Tensor,
        rope_grass: RotaryEmbedding128,
        prompt_attn_mask: torch.Tensor,
        m_active: torch.Tensor,
        return_sink_weights: bool = False,
        prompt_valid_mask: Optional[torch.Tensor] = None,
        prompt_kv_cache: Optional[Dict[str, Any]] = None,
        infer_cond_half: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        B, N_a, _, _ = x_prev.shape

        # §§7.2–7.3: на инференсе (вне XLA) пропускаем CA на безусловной ветке (математически тождественно 0)
        if (not self.training) and (x_prev.device.type != "xla"):
            if infer_cond_half:
                b_half = B // 2
                h_ca_c = h_ca_prev[:b_half] if h_ca_prev is not None else None
                xp_c = x_prompt[:b_half] if x_prompt.shape[0] > b_half else x_prompt
                pvm_c = (
                    prompt_valid_mask[:b_half]
                    if (prompt_valid_mask is not None and prompt_valid_mask.shape[0] > b_half)
                    else prompt_valid_mask
                )
                pam_c = prompt_attn_mask[:b_half] if prompt_attn_mask.shape[0] > b_half else prompt_attn_mask
                x_out_c, ca_delta_c, stats_c, sw_c = self._forward_core(
                    x_in=x_in[:b_half],
                    h_ca_prev=h_ca_c,
                    x_prev=x_prev[:b_half],
                    x_prompt=xp_c,
                    c=c[:b_half],
                    rope_grass=rope_grass,
                    prompt_attn_mask=pam_c,
                    m_active=m_active[:b_half],
                    return_sink_weights=return_sink_weights,
                    prompt_valid_mask=pvm_c,
                    prompt_kv_cache=prompt_kv_cache,
                )
                x_out_u = x_prev[b_half:]
                ca_delta_u = torch.zeros_like(x_out_u)
                x_out = torch.cat([x_out_c, x_out_u], dim=0)
                ca_delta = torch.cat([ca_delta_c, ca_delta_u], dim=0)
                if return_sink_weights and sw_c is not None:
                    sw_u = torch.ones((B - b_half, N_a), device=x_prev.device, dtype=sw_c.dtype)
                    sw = torch.cat([sw_c, sw_u], dim=0)
                else:
                    sw = None
                return x_out, ca_delta, stats_c, sw

            if not bool(prompt_attn_mask.any()):
                dev = x_prev.device
                zero_stats = (
                    torch.zeros(1, 4 * HIDDEN_SIZE, device=dev, dtype=torch.float32),
                    torch.zeros(1, 4 * HIDDEN_SIZE, device=dev, dtype=torch.float32),
                    torch.zeros(1, device=dev, dtype=torch.float32),
                )
                sw_ones = (
                    torch.ones((B, N_a), device=dev, dtype=torch.float32) if return_sink_weights else None
                )
                return x_prev, torch.zeros_like(x_prev), zero_stats, sw_ones

        return self._forward_core(
            x_in=x_in,
            h_ca_prev=h_ca_prev,
            x_prev=x_prev,
            x_prompt=x_prompt,
            c=c,
            rope_grass=rope_grass,
            prompt_attn_mask=prompt_attn_mask,
            m_active=m_active,
            return_sink_weights=return_sink_weights,
            prompt_valid_mask=prompt_valid_mask,
            prompt_kv_cache=prompt_kv_cache,
        )


class FullSABlock(nn.Module):
    """
    Полный интеграторский слой (1024d):
    - Получает стек `[x_in, h_ca, x_prev]` формы `(B, N, 8, 128)` -> 2x `PerAxisGRNv3` (`dca_q`, `dca_kv`).
    - Поддерживает как плотный `Linear` AdaLN (v1), так и `AxisBlockAdaLN` D2 r=64 (v2).
    """

    def __init__(
        self,
        hidden_size: int = HIDDEN_SIZE,
        intermediate_size: int = 2048,
        num_heads: int = 8,
        has_ca: bool = True,
        cond_dim: int = 256,
        adaln_type: str = "linear",
        adaln_bottleneck_r: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.adaln_type = adaln_type
        self.adaln_bottleneck_r = int(adaln_bottleneck_r)

        self.dca_q = PerAxisGRNv3(NUM_AXES, AXIS_DIM, has_ca=has_ca)
        self.dca_kv = PerAxisGRNv3(NUM_AXES, AXIS_DIM, has_ca=has_ca)

        self.attn_norm_q = RMSNorm(hidden_size)
        self.attn_norm_kv = RMSNorm(hidden_size)
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

        if adaln_type == "axis_block":
            self.adaln_proj: nn.Module = AxisBlockAdaLN(
                cond_dim=cond_dim,
                num_mods=6,
                bottleneck_r=self.adaln_bottleneck_r,
                gate_indices=(2, 5),
                gate_bias=0.02,
            )
        else:
            self.adaln_proj = nn.Linear(cond_dim, 6 * hidden_size)
            _init_soft_gate_adaln(self.adaln_proj, hidden_size, gate_bias=0.02)

    def forward(
        self,
        x_in: torch.Tensor,
        h_ca: Optional[torch.Tensor],
        x_prev: torch.Tensor,
        c: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        m_active: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        B, N, K, D = x_prev.shape
        C = K * D

        if isinstance(self.adaln_proj, AxisBlockAdaLN):
            s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = self.adaln_proj(c, x_prev.dtype)
            shift_msa = s_msa.reshape(B, N, C)
            scale_msa = sc_msa.reshape(B, N, C)
            gate_msa = g_msa.reshape(B, N, C)
            shift_mlp = s_mlp.reshape(B, N, C)
            scale_mlp = sc_mlp.reshape(B, N, C)
            gate_mlp = g_mlp.reshape(B, N, C)
        else:
            mods = self.adaln_proj(c.to(self.adaln_proj.weight.dtype)).to(x_prev.dtype)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods.chunk(6, dim=-1)
        M = torch.cat([shift_msa, scale_msa, shift_mlp, scale_mlp], dim=-1)
        stats = _compute_adaln_stats(M, m_active)

        x_q_flat = self.dca_q(x_in, h_ca, x_prev).reshape(B, N, C)
        x_kv_flat = self.dca_kv(x_in, h_ca, x_prev).reshape(B, N, C)
        x_flat = x_prev.reshape(B, N, C)

        h_q = self.attn_norm_q(x_q_flat) * (1.0 + scale_msa) + shift_msa
        h_kv = self.attn_norm_kv(x_kv_flat) * (1.0 + scale_msa) + shift_msa

        q = self.q_proj(h_q).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        u = self.kv_proj(h_kv).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(u)
        v = u * self.v_gate.to(u.dtype)  # V без RoPE

        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        x_flat = x_flat + gate_msa * self.o_proj(attn_out)

        h_mlp = self.mlp_norm(x_flat) * (1.0 + scale_mlp) + shift_mlp
        gated = _clamp_swiglu(self.gate_proj(h_mlp), self.up_proj(h_mlp))
        mlp_out = _clamp_fp16_safe(self.down_proj(gated))
        x_flat = x_flat + gate_mlp * mlp_out
        return x_flat.view(B, N, K, D), stats


# %% 4. Полное гибридное ядро `GrassmannianAltDiffusionCore`
ARCH_PRESETS: Dict[str, Dict[str, Any]] = {
    # ~38.6M параметров (4.35x меньше базового 167.9M): баланс скорости на CPU и емкости Full FFN
    "standard": {
        "full_ffn_dim": 2048,
        "grass_ffn_dim": 512,
        "num_sub_heads": 4,
        "local_radius": 6,
        "prompt_layers": 2,
        "stack": [
            "grass_full",   # 0: глобальный обзор всего холста в R^128 до CA #1 (осмысленный запрос Q)
            "grass_ca",     # 1: CA #1 (ранний макро-промпт -> DCA слот h_ca для 2..5)
            "full",         # 2: глобальный интегратор 1024 + память FFN
            "grass_local",  # 3: локальная стыковка осей в R^128
            "grass_full",   # 4: глобальное векторное внимание в R^128 перед CA #2
            "grass_ca",     # 5: CA #2 (уточненный запрос к промпту -> обновляет h_ca для 6..9)
            "full",         # 6: основной смысловой синтез 1024 + память FFN
            "grass_local",  # 7: тонкая кристаллизация в R^128
            "grass_local",  # 8: выравнивание фаз соседних слов в R^128
            "full",         # 9: финальный глобальный интегратор 1024 перед OutputHead
        ],
    },
    # ~59.5M параметров: расширенная память под GPU (4x FullSA с FFN=2816)
    "wide": {
        "full_ffn_dim": 2816,
        "grass_ffn_dim": 768,
        "num_sub_heads": 4,
        "local_radius": 6,
        "prompt_layers": 2,
        "stack": [
            "grass_full",   # 0: глобальный обзор всего холста в R^128 до CA #1
            "grass_ca",     # 1: CA #1
            "full",         # 2
            "grass_local",  # 3
            "full",         # 4
            "grass_ca",     # 5: CA #2
            "grass_full",   # 6
            "full",         # 7
            "grass_local",  # 8
            "full",         # 9
        ],
    },
    # ~27.9M параметров: сверхбыстрый режим для отладки и экспериментов на CPU
    "tiny": {
        "full_ffn_dim": 1024,
        "grass_ffn_dim": 384,
        "num_sub_heads": 4,
        "local_radius": 4,
        "prompt_layers": 2,
        "stack": [
            "grass_full",   # 0: глобальный обзор всего холста в R^128 до CA #1
            "grass_ca",     # 1 (CA #1)
            "full",         # 2
            "grass_local",  # 3
            "grass_ca",     # 4 (CA #2)
            "full",         # 5
            "grass_local",  # 6
            "full",         # 7
        ],
    },
}


class GrassmannianAltDiffusionCore(nn.Module):
    """
    Альтернативное условное ядро диффузии Фазы 3 Plan C (поддержка v1 и v2 по §9):
    - v2 (по умолчанию):
      * 3-блочная структура [96 sem | 16 syn | 16 pos]: 8x16d в GrassSA/PA + 7x16d (112d) без pos в GrassCA;
      * Блочно-диагональный старт (alpha=0.15) в GrassSA/PA/CA;
      * Правое выравнивание промпта (последний токен на pos=63, SEP=64) + RoPE base=500;
      * ConditionEmbedderV2 (Fourier 64d + 19 датчиков) + AxisBlockAdaLN (D2, r=64);
      * Energy-Gated SwiGLU (rad_feat = [abs_ln, rel_ln]) во всех поосевых FFN;
      * GrassmannianOutputHeadV2 (EDM поосевой c_skip/c_out + lambda_keep + PHMLinear r=8);
      * Поосевой SD3 log-SNR shift (axis_shift_delta = 0.35).
    - Этап 1 инференс (работает и для v1, и для v2):
      * prepare_prompt_cache (кэш KV промпта по всем CA-слоям);
      * forward_cfg_batched (2B батч с пропуском GrassCA на uncond).
    """

    def __init__(
        self,
        void_ref: torch.Tensor,
        preset: str = "standard",
        full_ffn_dim: Optional[int] = None,
        grass_ffn_dim: Optional[int] = None,
        num_sub_heads: Optional[int] = None,
        local_radius: Optional[int] = None,
        cond_dim: int = 256,
        arch_version: str = "v2",
        axis_shift_delta: Optional[float] = None,
        adaln_bottleneck_r: int = 64,
        prompt_right_align: Optional[bool] = None,
        blockdiag_alpha: Optional[float] = None,
        output_head_phm_rank: int = 8,
    ):
        super().__init__()
        if preset not in ARCH_PRESETS:
            raise ValueError(f"Неизвестный пресет архитектуры: {preset}. Доступны: {list(ARCH_PRESETS)}")
        if arch_version not in ("v1", "v2"):
            raise ValueError(f"Неизвестная версия архитектуры: {arch_version}. Доступны: ('v1', 'v2')")

        is_v2 = arch_version == "v2"
        cfg = dict(ARCH_PRESETS[preset])
        if is_v2 and num_sub_heads is None:
            cfg["num_sub_heads"] = 8  # 8 x 16d = [6 sem | 1 syn | 1 pos]
        if full_ffn_dim is not None:
            cfg["full_ffn_dim"] = int(full_ffn_dim)
        if grass_ffn_dim is not None:
            cfg["grass_ffn_dim"] = int(grass_ffn_dim)
        if num_sub_heads is not None:
            cfg["num_sub_heads"] = int(num_sub_heads)
        if local_radius is not None:
            cfg["local_radius"] = int(local_radius)

        self.preset = preset
        self.arch_version = arch_version
        self.arch_cfg = cfg
        self.hidden_size = HIDDEN_SIZE
        self.num_sub_heads = cfg["num_sub_heads"]
        self.sub_head_dim = AXIS_DIM // self.num_sub_heads
        self.stack_spec: List[str] = list(cfg["stack"])

        self.axis_shift_delta = (0.35 if is_v2 else 0.0) if axis_shift_delta is None else float(axis_shift_delta)
        self.prompt_right_align = is_v2 if prompt_right_align is None else bool(prompt_right_align)
        self.blockdiag_alpha = (0.15 if is_v2 else 0.0) if blockdiag_alpha is None else float(blockdiag_alpha)
        self.adaln_type = "axis_block" if is_v2 else "linear"
        self.adaln_bottleneck_r = int(adaln_bottleneck_r)
        self.use_energy_gate = is_v2
        self.ca_num_sub_heads = 7 if (is_v2 and self.sub_head_dim == 16) else self.num_sub_heads
        self.ca_qk_dim = 112 if (is_v2 and self.sub_head_dim == 16) else AXIS_DIM

        rope_grass_base = 500.0 if is_v2 else 10000.0
        self.rope_full = RotaryEmbedding128(dim=AXIS_DIM, max_seq_len=512)
        self.rope_grass = RotaryEmbedding128(dim=self.sub_head_dim, max_seq_len=512, base=rope_grass_base)

        if is_v2:
            self.cond_embedder: nn.Module = ConditionEmbedderV2(void_ref=void_ref, cond_dim=cond_dim)
        else:
            self.cond_embedder = ConditionEmbedder(void_ref=void_ref, cond_dim=cond_dim)

        self.prompt_adapter = GrassmannianPromptAdapter(
            num_layers=cfg["prompt_layers"],
            axis_dim=AXIS_DIM,
            ffn_dim=cfg["grass_ffn_dim"],
            num_sub_heads=self.num_sub_heads,
            init_layer_scale=0.02,
            prompt_right_align=self.prompt_right_align,
            rope_offset=N_PROMPT,
            use_energy_gate=self.use_energy_gate,
            blockdiag_alpha=self.blockdiag_alpha,
        )

        layers: List[nn.Module] = []
        seen_ca = False
        for kind in self.stack_spec:
            if kind == "grass_local":
                layers.append(
                    GrassmannianSABlock(
                        axis_dim=AXIS_DIM,
                        ffn_dim=cfg["grass_ffn_dim"],
                        num_sub_heads=self.num_sub_heads,
                        local_radius=cfg["local_radius"],
                        has_ca=seen_ca,
                        cond_dim=cond_dim,
                        adaln_type=self.adaln_type,
                        adaln_bottleneck_r=self.adaln_bottleneck_r,
                        use_energy_gate=self.use_energy_gate,
                        blockdiag_alpha=self.blockdiag_alpha,
                    )
                )
            elif kind == "grass_full":
                layers.append(
                    GrassmannianSABlock(
                        axis_dim=AXIS_DIM,
                        ffn_dim=cfg["grass_ffn_dim"],
                        num_sub_heads=self.num_sub_heads,
                        local_radius=None,
                        has_ca=seen_ca,
                        cond_dim=cond_dim,
                        adaln_type=self.adaln_type,
                        adaln_bottleneck_r=self.adaln_bottleneck_r,
                        use_energy_gate=self.use_energy_gate,
                        blockdiag_alpha=self.blockdiag_alpha,
                    )
                )
            elif kind == "grass_ca":
                layers.append(
                    GrassmannianCABlock(
                        axis_dim=AXIS_DIM,
                        ffn_dim=cfg["grass_ffn_dim"],
                        num_sub_heads=self.ca_num_sub_heads,
                        qk_dim=self.ca_qk_dim,
                        rope_offset=N_PROMPT,
                        has_ca=seen_ca,
                        cond_dim=cond_dim,
                        prompt_right_align=self.prompt_right_align,
                        adaln_type=self.adaln_type,
                        adaln_bottleneck_r=self.adaln_bottleneck_r,
                        use_energy_gate=self.use_energy_gate,
                        blockdiag_alpha=self.blockdiag_alpha,
                    )
                )
                seen_ca = True
            elif kind == "full":
                layers.append(
                    FullSABlock(
                        hidden_size=HIDDEN_SIZE,
                        intermediate_size=cfg["full_ffn_dim"],
                        num_heads=NUM_AXES,
                        has_ca=seen_ca,
                        cond_dim=cond_dim,
                        adaln_type=self.adaln_type,
                        adaln_bottleneck_r=self.adaln_bottleneck_r,
                    )
                )
            else:
                raise ValueError(f"Неизвестный тип слоя: {kind}")

        self.layers = nn.ModuleList(layers)
        # Финальный DCA-миксер перед OutputHead: видит [x_in, последний h_ca, x_last]
        self.final_dca = PerAxisGRNv3(NUM_AXES, AXIS_DIM, has_ca=seen_ca)
        if is_v2:
            self.output_head: nn.Module = GrassmannianOutputHeadV2(
                cond_dim=cond_dim,
                adaln_bottleneck_r=self.adaln_bottleneck_r,
                axis_sigma=DEFAULT_AXIS_SIGMA,
                sigma_noise=SIGMA_NOISE,
                axis_shift_delta=self.axis_shift_delta,
                phm_rank=int(output_head_phm_rank),
                keep_bias=-3.0,
                output_gain_init=0.1,
            )
        else:
            self.output_head = GrassmannianOutputHead(hidden_size=HIDDEN_SIZE, cond_dim=cond_dim)

    def encode_prompt(
        self,
        V_prompt: torch.Tensor,
        prompt_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Однократное кодирование промпта в (B, N_p, 8, 128) через GrassmannianPromptAdapter."""
        return self.prompt_adapter(V_prompt, self.rope_grass, prompt_valid_mask=prompt_valid_mask)

    def prepare_prompt_cache(
        self,
        V_prompt: torch.Tensor,
        prompt_valid_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Однократно кодирует промпт и кэширует (k_full, v_p, attn_mask) для каждого GrassmannianCABlock (§7.1).
        На 20-40 шагах инференса вся проекция промпта выполняется ровно 1 раз вместо 40-80.
        """
        x_prompt = self.encode_prompt(V_prompt, prompt_valid_mask=prompt_valid_mask)
        ca_kv: Dict[int, Dict[str, Any]] = {}
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, GrassmannianCABlock):
                ca_kv[idx] = layer.prepare_prompt_kv(
                    x_prompt=x_prompt,
                    rope_grass=self.rope_grass,
                    prompt_attn_mask=prompt_valid_mask,
                    prompt_valid_mask=prompt_valid_mask,
                )
        return {
            "x_prompt": x_prompt,
            "prompt_valid_mask": prompt_valid_mask,
            "ca_kv": ca_kv,
        }

    def forward_with_cached_prompt(
        self,
        V_t: torch.Tensor,
        t_mat: torch.Tensor,
        x_prompt: torch.Tensor,
        prompt_attn_mask: torch.Tensor,
        m_prompt: torch.Tensor,
        radar_sigma: Optional[torch.Tensor] = None,
        return_sink_weights: bool = False,
        prompt_valid_mask: Optional[torch.Tensor] = None,
        prompt_cache: Optional[Dict[str, Any]] = None,
        infer_cond_half: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        B, N_a, K, D = V_t.shape
        m_active = t_mat > 1e-3

        c = self.cond_embedder(V_t, t_mat, m_prompt, radar_sigma=radar_sigma)
        x_in = (V_t.float() / SIGMA_NOISE).to(V_t.dtype)  # (B, N_a, 8, 128)
        x_cur = x_in
        h_ca: Optional[torch.Tensor] = None

        cos_full, sin_full = self.rope_full(x_in, N_a)
        cos_grass, sin_grass = self.rope_grass(x_in, N_a)

        ca_kv_map = prompt_cache.get("ca_kv") if isinstance(prompt_cache, dict) else None
        p_valid_eff = (
            prompt_valid_mask
            if prompt_valid_mask is not None
            else (prompt_cache.get("prompt_valid_mask") if isinstance(prompt_cache, dict) else None)
        )

        sum_m_list, sum_m2_list, count_list = [], [], []
        sink_list = []

        for idx, layer in enumerate(self.layers):
            if isinstance(layer, GrassmannianCABlock):
                kv_c = None
                if ca_kv_map is not None and idx in ca_kv_map:
                    # Кэш валиден для условного батча или для cond-половины в режиме 2B
                    if infer_cond_half or bool(prompt_attn_mask.any()):
                        kv_c = ca_kv_map[idx]
                x_cur, ca_delta, stats, sw = layer(
                    x_in=x_in,
                    h_ca_prev=h_ca,
                    x_prev=x_cur,
                    x_prompt=x_prompt,
                    c=c,
                    rope_grass=self.rope_grass,
                    prompt_attn_mask=prompt_attn_mask,
                    m_active=m_active,
                    return_sink_weights=return_sink_weights,
                    prompt_valid_mask=p_valid_eff,
                    prompt_kv_cache=kv_c,
                    infer_cond_half=infer_cond_half,
                )
                h_ca = ca_delta
                if sw is not None:
                    sink_list.append(sw)
            elif isinstance(layer, GrassmannianSABlock):
                x_cur, stats = layer(
                    x_in=x_in,
                    h_ca=h_ca,
                    x_prev=x_cur,
                    c=c,
                    cos=cos_grass,
                    sin=sin_grass,
                    m_active=m_active,
                )
            else:
                x_cur, stats = layer(
                    x_in=x_in,
                    h_ca=h_ca,
                    x_prev=x_cur,
                    c=c,
                    cos=cos_full,
                    sin=sin_full,
                    m_active=m_active,
                )
            sum_m_list.append(stats[0])
            sum_m2_list.append(stats[1])
            count_list.append(stats[2])

        x_out_4d = self.final_dca(x_in, h_ca, x_cur)
        if isinstance(self.output_head, GrassmannianOutputHeadV2):
            V_0_hat = self.output_head(x_out_4d, V_t, t_mat, c)
        else:
            V_0_hat = self.output_head(x_out_4d.reshape(B, N_a, K * D), V_t, t_mat, c)
        sink_weights = torch.stack(sink_list, dim=1) if sink_list else None  # (B, 2, N_a)

        num_layers = len(self.layers)
        sum_m_pack = torch.stack(sum_m_list, dim=1)                       # (1, L, 4096)
        sum_m2_pack = torch.stack(sum_m2_list, dim=1)                     # (1, L, 4096)
        count_pack = torch.stack(count_list, dim=0).view(1, num_layers)   # (1, L)
        return V_0_hat, (sum_m_pack, sum_m2_pack, count_pack), sink_weights

    def forward_cfg_batched(
        self,
        V_t: torch.Tensor,
        t_mat: torch.Tensor,
        x_prompt: torch.Tensor,
        prompt_attn_mask: torch.Tensor,
        prompt_cache: Optional[Dict[str, Any]] = None,
        radar_sigma: Optional[torch.Tensor] = None,
        return_sink_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Единый батч 2B = [cond || uncond] на инференсе (§§7.2–7.3):
        - Все SA / FullSA / ConditionEmbedder / OutputHead работают плотным батчем 2B;
        - Внутри GrassmannianCABlock вычисляется строго первая половина [:B] (cond) с кэшем KV,
          а для [B:] (uncond) CA пропускается с математически тождественным нулём.
        """
        B = V_t.shape[0]
        device = V_t.device
        V_t_2b = torch.cat([V_t, V_t], dim=0)
        t_mat_2b = torch.cat([t_mat, t_mat], dim=0)
        m_2b = torch.cat(
            [torch.ones(B, device=device, dtype=torch.float32), torch.zeros(B, device=device, dtype=torch.float32)],
            dim=0,
        )
        rad_2b = torch.cat([radar_sigma, radar_sigma], dim=0) if radar_sigma is not None else None

        v0_2b, _, sw_2b = self.forward_with_cached_prompt(
            V_t=V_t_2b,
            t_mat=t_mat_2b,
            x_prompt=x_prompt,
            prompt_attn_mask=prompt_attn_mask,
            m_prompt=m_2b,
            radar_sigma=rad_2b,
            return_sink_weights=return_sink_weights,
            prompt_cache=prompt_cache,
            infer_cond_half=(not self.training) and (device.type != "xla"),
        )
        v0_cond = v0_2b[:B]
        v0_uncond = v0_2b[B:]
        sw_cond = sw_2b[:B] if sw_2b is not None else None
        return v0_cond, v0_uncond, sw_cond

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
            prompt_valid_mask=prompt_valid_mask,
        )

    def set_diagnostics_capture(self, enabled: bool) -> None:
        """Включает/сбрасывает сбор активаций h_r (AxisBlockAdaLN) и 19 датчиков для val-диагностики (§3.3)."""
        if isinstance(self.cond_embedder, ConditionEmbedderV2):
            self.cond_embedder._capture_sensors = bool(enabled)
            if enabled:
                self.cond_embedder._captured_sensors = []
        for mod in self.modules():
            if isinstance(mod, AxisBlockAdaLN):
                mod._capture_hr = bool(enabled)
                if enabled:
                    mod._captured_hr = []

    @torch.no_grad()
    def collect_v2_diagnostics(self) -> Dict[str, float]:
        """Возвращает агрегированную диагностику эффективного ранга AxisBlockAdaLN и датчиков (§3.3)."""
        r_effs, k99s, tails, deads = [], [], [], []
        for mod in self.modules():
            if isinstance(mod, AxisBlockAdaLN):
                st = mod.compute_rank_stats()
                if st:
                    r_effs.append(st["r_eff"])
                    k99s.append(st["k_99"])
                    tails.append(st["tail_ratio"])
                    deads.append(st["dead_frac"])
                mod._captured_hr = []
        out: Dict[str, float] = {}
        if r_effs:
            out["val/adaln_r_eff_mean"] = float(np.mean(r_effs))
            out["val/adaln_r_eff_max"] = float(np.max(r_effs))
            out["val/adaln_k99_max"] = float(np.max(k99s))
            out["val/adaln_tail_mean"] = float(np.mean(tails))
            out["val/adaln_dead_mean"] = float(np.mean(deads))
        if isinstance(self.cond_embedder, ConditionEmbedderV2):
            simp = self.cond_embedder.compute_sensor_importance()
            self.cond_embedder._captured_sensors = []
            if simp:
                out["val/sensor_imp_max"] = float(max(simp.values()))
                out["val/sensor_imp_min"] = float(min(simp.values()))
                out["val/sensor_imp_offdiag"] = float(simp.get("offdiag_ratio", 0.0))
                out["val/sensor_imp_blk_sem"] = float(simp.get("blk_sem", 0.0))
        return out

    @torch.no_grad()
    def get_dca_summary(self) -> Dict[str, float]:
        """Возвращает средние абсолютные веса DCA по слотам [x_in, h_ca, x_prev] для мониторинга."""
        b_in, b_ca, b_prev, w_abs = [], [], [], []
        for mod in self.modules():
            if isinstance(mod, PerAxisGRNv3):
                bf = mod.b.detach().float()
                b_in.append(float(bf[..., 0].abs().mean().item()))
                if mod.has_ca:
                    b_ca.append(float(bf[..., 1].abs().mean().item()))
                b_prev.append(float(bf[..., -1].mean().item()))
                w_abs.append(float(mod.w.detach().float().abs().mean().item()))
        return {
            "dca/b_in_abs": float(np.mean(b_in)) if b_in else 0.0,
            "dca/b_ca_abs": float(np.mean(b_ca)) if b_ca else 0.0,
            "dca/b_prev_mean": float(np.mean(b_prev)) if b_prev else 1.0,
            "dca/w_abs": float(np.mean(w_abs)) if w_abs else 0.0,
        }


# %% 5. Оптимизаторы под гибридную архитектуру + DCA (без weight decay на DCA b/w)
def create_alt_optimizers(
    model: nn.Module,
    muon_lr: float = 1e-3,
    adamw_lr: float = 3e-4,
    scale_1d_lr: float = 1.2e-3,
    weight_decay: float = 0.01,
    orthmuon_sphere_scale: bool = True,
    orthmuon_c_base: float = 2.26,
) -> Tuple[Optional[OrthMuon], torch.optim.AdamW]:
    """
    Распределение параметров:
    1. OrthMuon (sphere=True, sphere_scale=True, c_base=2.26, wd=0.0): все *.q_weight.
    2. AdamW (scale_1d_lr, wd=0.0): row_scale, v_gate, RMSNorm/QK-norm, ls_attn/ls_mlp, score_temp, output_gain.
    3. AdamW (adamw_lr, wd=0.0): DCA, norm_emb, w_rad, axis_mix, bias, LayerNorm.
    4. AdamW (adamw_lr, wd=weight_decay): AdaLN, ConditionEmbedder, OutputHead, k_sink.
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
            or name.endswith(".score_temp")
            or name.endswith(".output_gain")
            or ("_norm" in name and "norm_emb" not in name)
            or ("q_norm." in name)
            or ("k_norm." in name)
            or name.endswith("output_head.norm.weight")
        ):
            scale_1d_params.append(param)
        elif (
            "dca_q." in name
            or "dca_kv." in name
            or "final_dca." in name
            or "norm_emb" in name
            or ".w_rad." in name
            or "axis_mix" in name
            or name.endswith(".bias")
            or param.ndim < 2
            or "mlp.3." in name
        ):
            adamw_nodecay_params.append(param)
        else:
            adamw_decay_params.append(param)

    opt_muon = None
    if sphere_params:
        opt_muon = OrthMuon(
            [
                {
                    "params": sphere_params,
                    "lr": muon_lr,
                    "weight_decay": 0.0,
                    "sphere": True,
                    "sphere_scale": bool(orthmuon_sphere_scale),
                    "sphere_c_base": float(orthmuon_c_base),
                }
            ],
            lr=muon_lr,
            momentum=0.95,
            ns_steps=5,
            sphere_scale=bool(orthmuon_sphere_scale),
            sphere_c_base=float(orthmuon_c_base),
        )

    adamw_groups = []
    if adamw_decay_params:
        adamw_groups.append(
            {"params": adamw_decay_params, "lr": adamw_lr, "base_lr": adamw_lr, "weight_decay": weight_decay}
        )
    if adamw_nodecay_params:
        adamw_groups.append(
            {"params": adamw_nodecay_params, "lr": adamw_lr, "base_lr": adamw_lr, "weight_decay": 0.0}
        )
    if scale_1d_params:
        adamw_groups.append(
            {"params": scale_1d_params, "lr": scale_1d_lr, "base_lr": scale_1d_lr, "weight_decay": 0.0}
        )

    opt_adamw = torch.optim.AdamW(adamw_groups, lr=adamw_lr, betas=(0.9, 0.999))
    return opt_muon, opt_adamw


# %% 6. Опциональный кэш предкодированных латентов для быстрого тестирования на CPU
class PrecomputedLatentDataset(Dataset):
    """
    Держит предвычисленные канонизированные латенты `(V_prompt, V_ans)` в FP16 в RAM,
    убирая повторный прогон замороженного `GrassmannianEncoder` на каждом шаге CPU.
    Память на 1000 примеров: ~390 MB (FP16).
    """

    def __init__(self, items: List[Dict[str, torch.Tensor]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.items[idx]


@torch.no_grad()
def build_or_load_latent_cache(
    raw_dataset: PromptAnswerDataset,
    encoder: nn.Module,
    stopword_mask: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    num_samples: int,
    cache_path: Optional[str] = None,
    encode_batch_size: int = 8,
) -> PrecomputedLatentDataset:
    """Предкодирует `num_samples` примеров энкодером (+ канонизация) и сохраняет кэш на диск."""
    if cache_path and os.path.exists(cache_path):
        print(f"[Latent Cache] Загрузка предвычисленных латентов из {cache_path}...", flush=True)
        items = torch.load(cache_path, map_location="cpu", weights_only=False)
        print(f"[Latent Cache] Загружено {len(items)} примеров.", flush=True)
        return PrecomputedLatentDataset(items)

    n = min(num_samples, len(raw_dataset))
    print(f"[Latent Cache] Предкодирование {n} примеров через GrassmannianEncoder...", flush=True)
    t0 = time.time()
    subset = torch.utils.data.Subset(raw_dataset, list(range(n)))
    loader = DataLoader(subset, batch_size=encode_batch_size, shuffle=False, num_workers=0)
    sw_dev = stopword_mask.to(device)
    items: List[Dict[str, torch.Tensor]] = []

    for b_idx, batch in enumerate(loader):
        p_ids = batch["prompt_ids"].to(device)
        a_ids = batch["ans_ids"].to(device)
        w_p = batch["prompt_weights"].to(device) if "prompt_weights" in batch else make_pooling_weights(p_ids, sw_dev)
        w_a = batch["ans_weights"].to(device) if "ans_weights" in batch else make_pooling_weights(a_ids, sw_dev)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            vp_raw = encoder(p_ids, w_p)
            va_raw = encoder(a_ids, w_a)
        vp, _ = canonicalize_cell_basis(vp_raw.float())
        va, _ = canonicalize_cell_basis(va_raw.float())
        vp_cpu = vp.to(torch.float16).cpu()
        va_cpu = va.to(torch.float16).cpu()

        for i in range(p_ids.shape[0]):
            item_dict: Dict[str, torch.Tensor] = {
                "V_prompt": vp_cpu[i],
                "V_ans": va_cpu[i],
                "ans_targets": batch["ans_targets"][i],
                "n_real_prompt": batch["n_real_prompt"][i],
                "i_last": batch["i_last"][i],
                "w_cell": batch["w_cell"][i],
            }
            if "is_content_word_cell" in batch:
                item_dict["is_content_word_cell"] = batch["is_content_word_cell"][i]
            items.append(item_dict)
        if (b_idx + 1) % 10 == 0 or len(items) == n:
            print(f"  [Latent Cache] {len(items)}/{n} ({time.time() - t0:.1f}s)", flush=True)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(items, cache_path)
        print(f"[Latent Cache] Сохранено в {cache_path}.", flush=True)
    return PrecomputedLatentDataset(items)


# %% 7. Чекпоинты Phase 3 Alt и основной цикл обучения
def find_latest_alt_checkpoint(checkpoint_dir: str, gcs_dir: Optional[str] = None) -> Optional[str]:
    local_files = glob.glob(os.path.join(checkpoint_dir, "phase3_alt_step_*.pt"))
    if local_files:
        local_files.sort(key=lambda f: int(re.findall(r"\d+", os.path.basename(f))[-1]))
        return local_files[-1]

    if gcs_dir:
        try:
            res = subprocess.run(
                ["gsutil", "ls", gcs_dir.rstrip("/") + "/phase3_alt_step_*.pt"],
                capture_output=True,
                text=True,
            )
            if res.returncode == 0 and res.stdout.strip():
                remote_files = [x.strip() for x in res.stdout.strip().splitlines() if x.strip()]
                remote_files.sort(key=lambda f: int(re.findall(r"\d+", os.path.basename(f))[-1]))
                latest = remote_files[-1]
                os.makedirs(checkpoint_dir, exist_ok=True)
                dest = os.path.join(checkpoint_dir, os.path.basename(latest))
                print(f"[GCS Resume] Скачивание последнего чекпоинта: {latest} -> {dest}", flush=True)
                dl = subprocess.run(["gsutil", "cp", latest, dest], capture_output=True, text=True)
                if dl.returncode == 0 and os.path.exists(dest):
                    return dest
        except Exception as e:
            print(f"[GCS Resume Warning] {e}", flush=True)
    return None


def load_alt_checkpoint(
    path: str,
    model: nn.Module,
    opt_muon: Optional[OrthMuon],
    opt_adamw: torch.optim.AdamW,
    device: torch.device,
    ema: Optional[EMAModel] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw_model = getattr(model, "module", model)
    sd = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    raw_model.load_state_dict(sd, strict=True)

    try:
        opt_adamw.load_state_dict(ckpt["opt_adamw_state_dict"])
    except Exception as e:
        print(f"[Resume Warning] AdamW state не загружен: {e}", flush=True)
    if opt_muon is not None and "opt_muon_state_dict" in ckpt:
        try:
            opt_muon.load_state_dict(ckpt["opt_muon_state_dict"])
        except Exception as e:
            print(f"[Resume Warning] OrthMuon state не загружен: {e}", flush=True)
    if ema is not None and ckpt.get("ema_state_dict") is not None:
        ema.load_state_dict(ckpt["ema_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        try:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        except Exception as e:
            print(f"[Resume Warning] GradScaler state не загружен: {e}", flush=True)

    step = int(ckpt.get("step", 0))
    print(f"[Resume] Восстановлено состояние Phase 3 Alt с шага {step} ({path}).", flush=True)
    return step


def crop_batch_for_gpu(
    V_prompt: torch.Tensor,
    V_0: torch.Tensor,
    p_valid: torch.Tensor,
    i_last: torch.Tensor,
    w_cell: torch.Tensor,
    ans_targets: torch.Tensor,
    is_cw_cell: Optional[torch.Tensor] = None,
    full_prob: float = 0.25,
    align: int = 8,
    void_pad: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    GPU/CPU Dynamic Crop батча по фактической длине (L_p, i_last) (§7.5):
    - С вероятностью `full_prob` (25%) оставляет полный холст (64 / 128) против сдвига распределения длины;
    - Иначе обрезает промпт до кратного `align` (min 8) и холст до `min(128, max(i_last) + void_pad)` (кратно `align`);
    - Отключается на XLA/TPU для сохранения 100% статического графа.
    """
    if V_0.device.type == "xla" or float(torch.rand(1).item()) < float(full_prob):
        return V_prompt, V_0, p_valid, w_cell, ans_targets, is_cw_cell

    max_lp = max(1, int(p_valid.sum(dim=-1).max().item()))
    n_p_crop = min(V_prompt.shape[1], max(align, ((max_lp + align - 1) // align) * align))

    max_la = max(1, int(i_last.max().item()))
    n_a_raw = max_la + 1 + void_pad  # +1 за SEP на pos 0
    n_a_crop = min(V_0.shape[1], max(align * 2, ((n_a_raw + align - 1) // align) * align))

    V_prompt_c = V_prompt[:, :n_p_crop]
    p_valid_c = p_valid[:, :n_p_crop]
    V_0_c = V_0[:, :n_a_crop]
    w_cell_c = w_cell[:, :n_a_crop]
    ans_targets_c = ans_targets[:, : (n_a_crop - 1) * 4]
    is_cw_c = is_cw_cell[:, :n_a_crop] if is_cw_cell is not None else None
    return V_prompt_c, V_0_c, p_valid_c, w_cell_c, ans_targets_c, is_cw_c


class LengthBucketBatchSampler:
    """Группирует примеры близкой длины внутри мега-чанков для максимальной экономии GPU Dynamic Crop (§7.5)."""

    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        drop_last: bool = True,
        megabatch_mult: int = 16,
    ):
        self.lengths = list(lengths)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.megabatch_mult = max(1, int(megabatch_mult))
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        gen = torch.Generator()
        gen.manual_seed(1337 + self.epoch)
        n = len(self.lengths)
        perm = torch.randperm(n, generator=gen).tolist()
        mega_sz = self.batch_size * self.megabatch_mult
        batches: List[List[int]] = []
        for s in range(0, n, mega_sz):
            chunk = perm[s : s + mega_sz]
            chunk.sort(key=lambda idx: self.lengths[idx])
            for b in range(0, len(chunk), self.batch_size):
                batch = chunk[b : b + self.batch_size]
                if len(batch) == self.batch_size or (not self.drop_last and len(batch) > 0):
                    batches.append(batch)
        b_perm = torch.randperm(len(batches), generator=gen).tolist()
        for bi in b_perm:
            yield batches[bi]

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def _infer_content_word_mask_from_targets(
    ans_targets: torch.Tensor,
    stopword_mask: torch.Tensor,
) -> torch.Tensor:
    """Fallback-восстановление маски is_content_word_cell (B, N_canvas) из ans_targets для старых кэшей."""
    B, T = ans_targets.shape
    n_ans = T // 4
    cells = ans_targets.view(B, n_ans, 4)
    non_pad = (cells != PAD_TOKEN_ID) & (cells != -100)
    safe_ids = cells.clamp(0, stopword_mask.numel() - 1)
    non_stop = non_pad & (~stopword_mask[safe_ids])
    n_np = non_pad.float().sum(dim=-1)
    n_ns = non_stop.float().sum(dim=-1)
    is_cw_ans = (n_np > 0) & (n_ns >= 0.5 * n_np.clamp(min=1.0))
    return torch.cat([torch.zeros((B, 1), device=ans_targets.device, dtype=torch.bool), is_cw_ans], dim=1)


def run_alt_training_loop(
    model: nn.Module,
    encoder: nn.Module,
    decoder: Optional[nn.Module],
    tokenizer: Any,
    dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    sampler: Optional[DistributedSampler],
    env: AltEnvironmentConfig,
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
    fast_vocab: Optional[FastVocabOptimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> None:
    raw_model = getattr(model, "module", model)
    if scaler is None and env.use_grad_scaler:
        scaler = torch.amp.GradScaler("cuda", init_scale=16384.0, growth_interval=1000)

    sw_dev = stopword_mask.to(env.device)
    sep_cpu, _ = canonicalize_cell_basis(sep_concept.float())
    void_cpu = void_concept.float()
    radar_decoder = None if args.no_radar else decoder

    model.train()
    step = start_step
    epoch = 0
    best_sim_gr = -1.0
    data_iter = iter(dataloader)

    if is_main_process:
        n_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        muon_p = sum(p.numel() for g in opt_muon.param_groups for p in g["params"]) if opt_muon else 0
        adam_p = sum(p.numel() for g in opt_adamw.param_groups for p in g["params"])
        print(
            f"=== Старт обучения Phase 3 Alt (preset={args.arch_preset}, matrix={args.matrix_mode}, "
            f"init={args.init_mode}(damp={args.wiener_damp}), device={env.device}, "
            f"amp={env.use_amp}/{env.amp_dtype}, params={n_params/1e6:.2f}M "
            f"[Muon={muon_p/1e6:.2f}M, AdamW={adam_p/1e6:.2f}M], "
            f"micro_batch={args.batch_size}, grad_accum={args.grad_accum_steps}, "
            f"eff_batch={args.batch_size * args.grad_accum_steps * world_size}, "
            f"radar={'OFF' if args.no_radar else 'ON'}, ce_w={args.ce_weight}, "
            f"fast_vocab={'ON' if fast_vocab is not None else 'OFF'}, "
            f"rollout={args.rollout_prob}) ===",
            flush=True,
        )
        print(f"    Стек слоев ({len(raw_model.stack_spec)}): {raw_model.stack_spec}", flush=True)

    # Тестовый запрос к val_dataloader на старте, чтобы сразу выявить зависание DataLoader/воркеров
    if is_main_process and val_dataloader is not None:
        _t_val_probe = time.time()
        _val_probe_batch = next(iter(val_dataloader))
        _val_probe_bs = int(_val_probe_batch["prompt_ids"].shape[0])
        del _val_probe_batch
        print(
            f"[Val Probe OK] val_dataloader отдал 1 батч (B={_val_probe_bs}) за {time.time() - _t_val_probe:.2f}s",
            flush=True,
        )
    if world_size > 1 and dist.is_initialized():
        dist.barrier()

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

            n_real_prompt = batch["n_real_prompt"].to(env.device, non_blocking=True)
            i_last = batch["i_last"].to(env.device, non_blocking=True)
            w_cell = batch["w_cell"].to(env.device, non_blocking=True)
            ans_targets = batch["ans_targets"].to(env.device, non_blocking=True)
            is_cw_cell = (
                batch["is_content_word_cell"].to(env.device, non_blocking=True)
                if "is_content_word_cell" in batch
                else _infer_content_word_mask_from_targets(ans_targets, sw_dev)
            )
            B = n_real_prompt.shape[0]
            axis_shift_delta = float(getattr(raw_model, "axis_shift_delta", 0.0))

            # 1. Получаем канонизированные латенты (из кэша RAM или on-the-fly через Encoder)
            with torch.no_grad():
                if "V_prompt" in batch and "V_ans" in batch:
                    V_prompt = batch["V_prompt"].to(env.device, dtype=torch.float32, non_blocking=True)
                    V_ans = batch["V_ans"].to(env.device, dtype=torch.float32, non_blocking=True)
                else:
                    prompt_ids = batch["prompt_ids"].to(env.device, non_blocking=True)
                    ans_ids = batch["ans_ids"].to(env.device, non_blocking=True)
                    w_p = (
                        batch["prompt_weights"].to(env.device, non_blocking=True)
                        if "prompt_weights" in batch
                        else make_pooling_weights(prompt_ids, sw_dev)
                    )
                    w_a = (
                        batch["ans_weights"].to(env.device, non_blocking=True)
                        if "ans_weights" in batch
                        else make_pooling_weights(ans_ids, sw_dev)
                    )
                    with torch.autocast(device_type=env.device.type, dtype=env.amp_dtype, enabled=env.use_amp):
                        V_prompt_raw = encoder(prompt_ids, w_p)
                        V_ans_raw = encoder(ans_ids, w_a)
                    V_prompt, _ = canonicalize_cell_basis(V_prompt_raw.float())
                    V_ans, _ = canonicalize_cell_basis(V_ans_raw.float())

                sep_exp = sep_cpu.to(device=env.device, dtype=torch.float32).view(1, 1, 8, 128).expand(B, 1, 8, 128)
                V_0 = torch.cat([sep_exp, V_ans], dim=1)
                p_valid = n_real_prompt > 0

                if getattr(args, "gpu_dynamic_crop", False) and env.device.type != "xla":
                    V_prompt, V_0, p_valid, w_cell, ans_targets, is_cw_cell = crop_batch_for_gpu(
                        V_prompt=V_prompt,
                        V_0=V_0,
                        p_valid=p_valid,
                        i_last=i_last,
                        w_cell=w_cell,
                        ans_targets=ans_targets,
                        is_cw_cell=is_cw_cell,
                        full_prob=float(getattr(args, "gpu_dynamic_crop_full_prob", 0.25)),
                    )

                p_attn_mask, m_prompt = apply_prompt_dropout(p_valid)

                use_rollout = (
                    step >= args.rollout_start_step
                    and float(args.rollout_prob) > 0.0
                    and float(torch.rand(1, device=env.device).item()) < float(args.rollout_prob)
                )
                if use_rollout:
                    V_t, t_mat, radar_sigma = build_rollout_noisy_canvas(
                        raw_model=raw_model,
                        decoder=radar_decoder,
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
                    t_mat = sample_mixed_noise_schedule(B, env.device, i_last=i_last, seq_len=V_0.shape[1])
                    V_t = add_axis_shifted_noise(
                        V_0,
                        t_mat,
                        torch.randn_like(V_0, dtype=torch.float32),
                        delta=axis_shift_delta,
                        sigma_noise=SIGMA_NOISE,
                    )
                    V_t[:, 0] = sep_exp[:, 0].float()
                    radar_sigma = (
                        compute_radar_sigma(radar_decoder, V_t, use_amp=env.use_amp, amp_dtype=env.amp_dtype)
                        if radar_decoder is not None
                        else None
                    )

            ddp_sync = micro_idx == args.grad_accum_steps - 1
            no_sync_ctx = model.no_sync() if (isinstance(model, DDP) and not ddp_sync) else None
            if no_sync_ctx is not None:
                no_sync_ctx.__enter__()

            try:
                in_dtype = env.amp_dtype if env.use_amp else torch.float32
                with torch.autocast(device_type=env.device.type, dtype=env.amp_dtype, enabled=env.use_amp):
                    V_0_hat, adaln_stats, sink_weights = model(
                        V_t=V_t.to(in_dtype),
                        t_mat=t_mat,
                        V_prompt=V_prompt.to(in_dtype),
                        prompt_valid_mask=p_valid,
                        prompt_attn_mask=p_attn_mask,
                        m_prompt=m_prompt,
                        radar_sigma=radar_sigma,
                        return_sink_weights=True,
                    )

                # Гибридный лосс строго в float32 снаружи DataParallel
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
                    spec_diag_weight=float(getattr(args, "spec_diag_weight", 0.05)),
                    orth_offdiag_weight=float(getattr(args, "orth_offdiag_weight", 0.05)),
                    anchor_weight=float(getattr(args, "anchor_weight", 0.15 if raw_model.arch_version == "v2" else 0.0)),
                    axis_shift_delta=axis_shift_delta,
                    basis_high_t_mode=str(getattr(args, "basis_high_t_mode", "sign_only" if raw_model.arch_version == "v2" else "none")),
                    static_xla_bary=True,
                    world_size=world_size,
                )

                # Сквозной токенный CE через замороженный Декодер (если включен)
                w_ce = float(args.ce_weight) * min(
                    1.0, float(max(1, step)) / float(max(1, args.ce_warmup_steps))
                )
                if decoder is not None and float(args.ce_weight) > 0.0 and (not use_rollout):
                    if fast_vocab is not None:
                        l_ce, ce_tok_acc = fast_vocab.compute_decoder_ce_loss(
                            decoder=decoder,
                            V_0_hat=V_0_hat,
                            t_mat=t_mat,
                            ans_targets=ans_targets,
                            ce_max_tokens=args.ce_max_tokens,
                            ce_t_max=args.ce_t_max,
                            stopword_mask=sw_dev,
                            ce_stopword_weight=args.ce_stopword_weight,
                            use_amp=env.use_amp,
                            amp_dtype=env.amp_dtype,
                            ce_batch_fraction=args.ce_batch_fraction,
                        )
                    else:
                        l_ce, ce_tok_acc = compute_decoder_ce_loss(
                            decoder=decoder,
                            V_0_hat=V_0_hat,
                            t_mat=t_mat,
                            ans_targets=ans_targets,
                            ce_max_tokens=args.ce_max_tokens,
                            ce_t_max=args.ce_t_max,
                            stopword_mask=sw_dev,
                            ce_stopword_weight=args.ce_stopword_weight,
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

        # Шаг оптимизаторов с раздельным grad_clip (AdamW=1.0, Sphere=10.0, §5.1)
        if scaler is not None:
            if opt_muon is not None:
                scaler.unscale_(opt_muon)
            scaler.unscale_(opt_adamw)
            clip_phase3_grads(
                opt_muon,
                opt_adamw,
                adamw_max_norm=float(getattr(args, "adamw_grad_clip", 1.0)),
                sphere_max_norm=float(getattr(args, "sphere_grad_clip", 10.0)),
            )
            if opt_muon is not None:
                scaler.step(opt_muon)
            scaler.step(opt_adamw)
            scaler.update()
        else:
            clip_phase3_grads(
                opt_muon,
                opt_adamw,
                adamw_max_norm=float(getattr(args, "adamw_grad_clip", 1.0)),
                sphere_max_norm=float(getattr(args, "sphere_grad_clip", 10.0)),
            )
            if opt_muon is not None:
                opt_muon.step()
            opt_adamw.step()

        if ema is not None:
            ema.update(raw_model)

        dt = time.time() - t0
        if is_main_process and (step % args.log_every == 0 or step == 1):
            dca_stat = raw_model.get_dca_summary()
            print(
                f"[Alt Step {step:6d}/{args.max_steps}] "
                f"tot={accum_metrics['loss/total']:.4f} | "
                f"ce={accum_metrics.get('loss/ce', 0.0):.3f} (acc={accum_metrics.get('train/ce_tok_acc', 0.0):.3f}) | "
                f"basis={accum_metrics['loss/basis']:.4f} | "
                f"gr={accum_metrics['loss/gr']:.4f} | "
                f"en={accum_metrics['loss/energy']:.4f} | "
                f"adj={accum_metrics['loss/adj']:.4f} | "
                f"sink={accum_metrics['loss/sink']:.4f} | "
                f"dca(in={dca_stat['dca/b_in_abs']:.3f},ca={dca_stat['dca/b_ca_abs']:.3f},"
                f"prev={dca_stat['dca/b_prev_mean']:.3f},w={dca_stat['dca/w_abs']:.3f}) | "
                f"adaln={accum_metrics['adaln/sigma_mean']:.2e} | "
                f"lr_m={lr_mult:.3f} | {dt:.2f}s",
                flush=True,
            )
            if use_wandb:
                import wandb
                wandb.log({**accum_metrics, **dca_stat, "lr/mult": lr_mult, "time/step_sec": dt}, step=step)

        # Регулярный чекпоинт (до валидации, чтобы прогресс не терялся при таймауте/сбое eval)
        if is_main_process and (step % args.save_every == 0):
            ckpt_path = os.path.join(env.checkpoint_dir, f"phase3_alt_step_{step}.pt")
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

        # Валидация на EMA-весах
        if is_main_process and val_dataloader is not None and (step % args.eval_every == 0):
            if ema is not None:
                ema.apply_shadow(raw_model)
            raw_model.set_diagnostics_capture(True)
            eval_dec = decoder if (not args.no_eval_decoder) else None
            val_res = evaluate_phase3(
                model=raw_model,
                encoder=encoder,
                decoder=eval_dec,
                tokenizer=tokenizer,
                val_dataloader=val_dataloader,
                stopword_mask=sw_dev,
                sep_concept=sep_cpu,
                void_concept=void_cpu,
                device=env.device,
                use_amp=env.use_amp,
                amp_dtype=env.amp_dtype,
                max_batches=args.eval_batches,
                run_generation_check=(eval_dec is not None and not args.no_gen_check),
                gen_steps=args.gen_steps,
            )
            raw_model.set_diagnostics_capture(False)
            val_res.update(raw_model.collect_v2_diagnostics())
            if int(args.cfg_dsim_batches) > 0:
                val_res.update(
                    evaluate_cfg_dsim(
                        model=raw_model,
                        encoder=encoder,
                        val_dataloader=val_dataloader,
                        stopword_mask=sw_dev,
                        sep_concept=sep_cpu,
                        device=env.device,
                        use_amp=env.use_amp,
                        amp_dtype=env.amp_dtype,
                        max_batches=int(args.cfg_dsim_batches),
                    )
                )
            if ema is not None:
                ema.restore(raw_model)

            gen_samples = val_res.pop("gen_samples", [])
            print(
                f"  [Alt Eval @ {step}] Sim_Gr={val_res['val/sim_gr_denoising']:.4f} (hi={val_res['val/sim_gr_hi']:.4f}) | "
                f"TokAcc={val_res.get('val/tok_acc_low', 0.0):.3f} (hi={val_res.get('val/tok_acc_hi', 0.0):.3f}, CE_hi={val_res.get('val/ce_loss_hi', 0.0):.2f}) | "
                f"CFG={val_res['val/cfg_contrast']:.4f} (hi={val_res['val/cfg_contrast_hi']:.4f}) | "
                f"CFG_dSim(t50/t75/t95)={val_res.get('val/cfg_dsim_t50', 0.0):+.4f}/"
                f"{val_res.get('val/cfg_dsim_t75', 0.0):+.4f}/{val_res.get('val/cfg_dsim_t95', 0.0):+.4f} | "
                f"VOID_F1={val_res['val/void_f1']:.4f} (pm1={val_res['val/void_start_pm1']:.3f}) | "
                f"orth={val_res.get('val/orth_err', 0.0):.3f} spec={val_res.get('val/spec_l1', 0.0):.4f} | "
                f"blk(s/y/p)={val_res.get('val/err_sem', 0.0):.4f}/{val_res.get('val/err_syn', 0.0):.4f}/{val_res.get('val/err_pos', 0.0):.4f} | "
                f"radar={val_res['val/radar_sigma_cell']:.3f} | "
                f"sink_c={val_res['val/sink_attn_content']:.3f} (hi={val_res['val/sink_attn_content_hi']:.3f}) | "
                f"rk={val_res['val/rk_eff_content']:.2f} | "
                f"adaln_min={val_res['val/sigma_adaln_min']:.2e}"
                + (
                    f" (r_eff={val_res['val/adaln_r_eff_mean']:.1f}/k99={val_res['val/adaln_k99_max']:.0f})"
                    if "val/adaln_r_eff_mean" in val_res
                    else ""
                ),
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
                best_path = os.path.join(env.checkpoint_dir, "phase3_alt_best_model.pt")
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

        if world_size > 1 and dist.is_initialized() and ((step % args.eval_every == 0) or (step % args.save_every == 0)):
            dist.barrier()

    if is_main_process:
        print("=== Обучение Phase 3 Alt завершено ===", flush=True)


# %% 8. CLI и точка входа
@torch.no_grad()
def evaluate_cfg_dsim(
    model: nn.Module,
    encoder: nn.Module,
    val_dataloader: DataLoader,
    stopword_mask: torch.Tensor,
    sep_concept: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    max_batches: int = 4,
    t_levels: Tuple[float, ...] = (0.50, 0.75, 0.95),
    seed: int = 4242,
) -> Dict[str, float]:
    """
    Полезность условия (CFG-ΔSim): ΔSim(t) = SimGr(V̂_cond, V_0) − SimGr(V̂_uncond, V_0) на контентных ячейках.
    В отличие от `cfg_contrast` (норма ||V̂_cond − V̂_uncond||, меряет только расхождение веток) эта метрика
    положительна лишь тогда, когда промпт реально приближает предсказание к цели. Шум фиксирован seed'ом.
    """
    raw_model = getattr(model, "module", model)
    was_training = raw_model.training
    raw_model.eval()
    sw_dev = stopword_mask.to(device)
    sep_c, _ = canonicalize_cell_basis(sep_concept.float())
    gen = torch.Generator(device="cpu").manual_seed(seed)
    acc: Dict[float, List[float]] = {t: [] for t in t_levels}
    sim_c: Dict[float, List[float]] = {t: [] for t in t_levels}
    in_dtype = amp_dtype if use_amp else torch.float32

    for b_idx, batch in enumerate(val_dataloader):
        if b_idx >= max_batches:
            break
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
        sep_exp = sep_c.to(device).view(1, 1, NUM_AXES, AXIS_DIM).expand(B, 1, NUM_AXES, AXIS_DIM)
        V_0 = torch.cat([sep_exp, V_ans], dim=1)
        is_content = n_real_ans > 0
        if not bool(is_content.any()):
            continue
        p_valid = n_real_prompt > 0
        delta = float(getattr(raw_model, "axis_shift_delta", 0.0))
        noise_eps = torch.randn(tuple(V_0.shape), generator=gen).to(device)

        for t_val in t_levels:
            t_mat = torch.full((B, V_0.shape[1]), float(t_val), device=device, dtype=torch.float32)
            t_mat[:, 0] = 0.0
            V_t = add_axis_shifted_noise(V_0, t_mat, noise_eps, delta=delta, sigma_noise=SIGMA_NOISE)
            V_t[:, 0] = sep_exp[:, 0]
            outs = []
            for cond in (True, False):
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    vh, _, _ = raw_model(
                        V_t=V_t.to(in_dtype),
                        t_mat=t_mat,
                        V_prompt=V_prompt.to(in_dtype),
                        prompt_valid_mask=p_valid,
                        prompt_attn_mask=p_valid if cond else torch.zeros_like(p_valid),
                        m_prompt=torch.ones(B, device=device) if cond else torch.zeros(B, device=device),
                        radar_sigma=None,
                        return_sink_weights=False,
                    )
                outs.append(sim_gr(vh.float()[:, 1:], V_0[:, 1:])[is_content])
            sim_c[t_val].append(float(outs[0].mean().item()))
            acc[t_val].append(float((outs[0] - outs[1]).mean().item()))

    if was_training:
        raw_model.train()
    res: Dict[str, float] = {}
    for t_val in t_levels:
        tag = f"{int(round(t_val * 100)):02d}"
        res[f"val/cfg_dsim_t{tag}"] = float(np.mean(acc[t_val])) if acc[t_val] else 0.0
        res[f"val/sim_cond_t{tag}"] = float(np.mean(sim_c[t_val])) if sim_c[t_val] else 0.0
    return res


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BEBLaDII Phase 3 Alt: Grassmannian 2D Attention (R^128) + Full SA + 2x GrassCA + 3-slot DCA"
    )
    parser.add_argument("--smoke-test", "--smoke_test", dest="smoke_test", action="store_true", help="Быстрый smoke-test на 2 шага")
    parser.add_argument("--cpu-fast", "--cpu_fast", dest="cpu_fast", action="store_true", help="Пресет быстрого тестирования на CPU (без тяжелых прогонов 131k-словаря на каждом шаге)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Устройство исполнения")
    parser.add_argument("--amp-dtype", "--amp_dtype", dest="amp_dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--cpu-bf16", "--cpu_bf16", dest="cpu_bf16", action="store_true", help="Включить bfloat16 autocast на CPU")
    parser.add_argument(
        "--sys1-quant", "--sys1_quant", dest="sys1_quant", type=str, default="auto",
        choices=["auto", "fp32", "half", "fp16", "bf16", "emb_int8", "int8"],
        help="Режим квантизации замороженной Системы 1 (Encoder+Decoder): auto (по ускорителю и --sys1-mode), fp32, half, emb_int8, int8",
    )
    parser.add_argument(
        "--sys1-mode", "--sys1_mode", dest="sys1_mode", type=str, default="train",
        choices=["train", "inference"],
        help="Режим работы Системы 1 для --sys1-quant=auto: train (безопасный backward CE + emb_int8) или inference (полный int8)",
    )
    parser.add_argument("--num-threads", "--num_threads", dest="num_threads", type=int, default=0, help="Ограничить число CPU-потоков PyTorch (0 = по умолчанию)")

    # Архитектура и матричные оптимизации
    parser.add_argument("--arch-preset", "--arch_preset", dest="arch_preset", type=str, default="standard", choices=list(ARCH_PRESETS.keys()))
    parser.add_argument("--arch-version", "--arch_version", dest="arch_version", type=str, default="v2", choices=["v1", "v2"])
    parser.add_argument("--axis-shift-delta", "--axis_shift_delta", dest="axis_shift_delta", type=float, default=None, help="SD3-shift по осям (по умолчанию 0.35 в v2, 0.0 в v1)")
    parser.add_argument("--adaln-bottleneck-r", "--adaln_bottleneck_r", dest="adaln_bottleneck_r", type=int, default=64)
    parser.add_argument("--output-head-phm-rank", "--output_head_phm_rank", dest="output_head_phm_rank", type=int, default=8)
    parser.add_argument("--spec-diag-weight", "--spec_diag_weight", dest="spec_diag_weight", type=float, default=0.05)
    parser.add_argument("--orth-offdiag-weight", "--orth_offdiag_weight", dest="orth_offdiag_weight", type=float, default=0.05)
    parser.add_argument("--anchor-weight", "--anchor_weight", dest="anchor_weight", type=float, default=0.15)
    parser.add_argument("--basis-high-t-mode", "--basis_high_t_mode", dest="basis_high_t_mode", type=str, default="sign_only", choices=["none", "decay", "sign_only"])
    parser.add_argument("--adamw-grad-clip", "--adamw_grad_clip", dest="adamw_grad_clip", type=float, default=1.0)
    parser.add_argument("--sphere-grad-clip", "--sphere_grad_clip", dest="sphere_grad_clip", type=float, default=10.0)
    parser.add_argument("--orthmuon-c-base", "--orthmuon_c_base", dest="orthmuon_c_base", type=float, default=2.26)
    parser.add_argument("--gpu-dynamic-crop", "--gpu_dynamic_crop", dest="gpu_dynamic_crop", action="store_true", default=False)
    parser.add_argument("--gpu-dynamic-crop-full-prob", "--gpu_dynamic_crop_full_prob", dest="gpu_dynamic_crop_full_prob", type=float, default=0.25)
    parser.add_argument(
        "--matrix-mode", "--matrix_mode", dest="matrix_mode", type=str, default="safe",
        choices=["dense", "safe", "ultra"],
        help="Режим матричных оптимизаций: dense (базовые FullSA/GrassFull), safe (PHM Q/KV/O + GrassAlign + L*X*R, плотный FFN), ultra (PHM также в FFN)",
    )
    parser.add_argument("--phm-rank", "--phm_rank", dest="phm_rank", type=int, default=4)
    parser.add_argument("--phm-alpha", "--phm_alpha", dest="phm_alpha", type=float, default=0.25)
    parser.add_argument(
        "--init-mode", "--init_mode", dest="init_mode", type=str, default="all_5",
        choices=["none", "compat", "all_geo", "all_5", "idea4_wiener_head", "idea5_kmeans_ffn"],
        help="Геометрическая/data-driven инициализация весов при старте с нуля",
    )
    parser.add_argument("--wiener-damp", "--wiener_damp", dest="wiener_damp", type=float, default=0.7, help="Демпфирование Винеровского фильтра beta_k в OutputHead")
    parser.add_argument("--init-samples", "--init_samples", dest="init_samples", type=int, default=128, help="Число примеров для data-driven инициализации (Wiener + K-Means)")
    parser.add_argument("--init-seed", "--init_seed", dest="init_seed", type=int, default=1000)
    parser.add_argument("--full-ffn-dim", "--full_ffn_dim", dest="full_ffn_dim", type=int, default=None)
    parser.add_argument("--grass-ffn-dim", "--grass_ffn_dim", dest="grass_ffn_dim", type=int, default=None)
    parser.add_argument("--num-sub-heads", "--num_sub_heads", dest="num_sub_heads", type=int, default=None)
    parser.add_argument("--local-radius", "--local_radius", dest="local_radius", type=int, default=None)

    # Данные и кэширование латентов
    parser.add_argument("--data-path", "--data_path", dest="data_path", type=str, default="auto")
    parser.add_argument("--val-data-path", "--val_data_path", dest="val_data_path", type=str, default="auto")
    parser.add_argument("--max-train-rows", "--max_train_rows", dest="max_train_rows", type=int, default=0, help="Ограничить число строк train (0 = все)")
    parser.add_argument("--cache-latents", "--cache_latents", dest="cache_latents", type=int, default=0, help="Предкодировать N примеров энкодером в RAM/файл для ускорения на CPU")
    parser.add_argument("--cache-file", "--cache_file", dest="cache_file", type=str, default="", help="Путь к .pt кэшу латентов")

    # Гиперпараметры обучения
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", "--grad_accum_steps", dest="grad_accum_steps", type=int, default=4)
    parser.add_argument("--max-steps", "--max_steps", dest="max_steps", type=int, default=50000)
    parser.add_argument("--muon-lr", "--muon_lr", dest="muon_lr", type=float, default=1e-3)
    parser.add_argument("--adamw-lr", "--adamw_lr", dest="adamw_lr", type=float, default=3e-4)
    parser.add_argument("--scale-1d-lr", "--scale_1d_lr", dest="scale_1d_lr", type=float, default=1.2e-3)
    parser.add_argument("--lr-warmup-steps", "--lr_warmup_steps", dest="lr_warmup_steps", type=int, default=500)
    parser.add_argument("--ema-decay", "--ema_decay", dest="ema_decay", type=float, default=0.999)
    parser.add_argument("--no-ema", "--no_ema", dest="no_ema", action="store_true")

    # Опции скорости и лоссов
    parser.add_argument("--no-radar", "--no_radar", dest="no_radar", action="store_true", help="Отключить NoiseRadarHead на train-шаге (ускоряет CPU, s7=0)")
    parser.add_argument("--no-eval-decoder", "--no_eval_decoder", dest="no_eval_decoder", action="store_true", help="Пропустить декодер на валидации (только латентные метрики)")
    parser.add_argument("--no-gen-check", "--no_gen_check", dest="no_gen_check", action="store_true", help="Отключить сэмплинг генерации во время eval")
    parser.add_argument("--elastic-drop-prob", "--elastic_drop_prob", dest="elastic_drop_prob", type=float, default=0.0)
    parser.add_argument("--rollout-prob", "--rollout_prob", dest="rollout_prob", type=float, default=0.35)
    parser.add_argument("--rollout-start-step", "--rollout_start_step", dest="rollout_start_step", type=int, default=1000)
    parser.add_argument("--ce-weight", "--ce_weight", dest="ce_weight", type=float, default=0.05)
    parser.add_argument("--ce-warmup-steps", "--ce_warmup_steps", dest="ce_warmup_steps", type=int, default=300)
    parser.add_argument("--ce-max-tokens", "--ce_max_tokens", dest="ce_max_tokens", type=int, default=512)
    parser.add_argument("--ce-t-max", "--ce_t_max", dest="ce_t_max", type=float, default=0.70)
    parser.add_argument("--ce-stopword-weight", "--ce_stopword_weight", dest="ce_stopword_weight", type=float, default=0.25)
    parser.add_argument(
        "--no-fast-vocab", "--no_fast_vocab", dest="no_fast_vocab", action="store_true",
        help="Отключить TPU/GPU-оптимизацию словаря (откат к базовому FP32 радару и полному 131K CE)",
    )
    parser.add_argument(
        "--ce-mode", "--ce_mode", dest="ce_mode", type=str, default="exact_chunked",
        choices=["exact_chunked", "sampled"],
        help="Режим CE-лосса в FastVocabOptimizer: exact_chunked (1-проходный точный online CE, по умолчанию) или sampled (SVD-пул).",
    )
    parser.add_argument("--ce-vocab-chunk-size", "--ce_vocab_chunk_size", dest="ce_vocab_chunk_size", type=int, default=16384)
    parser.add_argument("--ce-pool-hard", "--ce_pool_hard", dest="ce_pool_hard", type=int, default=1536)
    parser.add_argument("--ce-pool-rand", "--ce_pool_rand", dest="ce_pool_rand", type=int, default=2048)
    parser.add_argument("--ce-batch-fraction", "--ce_batch_fraction", dest="ce_batch_fraction", type=float, default=0.5)
    parser.add_argument(
        "--ce-batch-select", "--ce_batch_select", dest="ce_batch_select", type=str, default="random",
        choices=["random", "max_active"],
        help="Отбор B_ce примеров для CE: random (несмещённо, по умолчанию) или max_active (legacy, перекос к длинным ответам).",
    )
    parser.add_argument(
        "--cfg-dsim-batches", "--cfg_dsim_batches", dest="cfg_dsim_batches", type=int, default=4,
        help="Число val-батчей для метрики CFG-ΔSim (SimGr(cond)-SimGr(uncond)) при eval; 0 = выкл.",
    )
    parser.add_argument("--vocab-svd-rank", "--vocab_svd_rank", dest="vocab_svd_rank", type=int, default=64)

    # Логирование, чекпоинты, multi-GPU
    parser.add_argument("--log-every", "--log_every", dest="log_every", type=int, default=20)
    parser.add_argument("--eval-every", "--eval_every", dest="eval_every", type=int, default=500)
    parser.add_argument("--eval-batches", "--eval_batches", dest="eval_batches", type=int, default=4)
    parser.add_argument("--gen-steps", "--gen_steps", dest="gen_steps", type=int, default=20)
    parser.add_argument("--save-every", "--save_every", dest="save_every", type=int, default=500)
    parser.add_argument("--resume-ckpt", "--resume_ckpt", dest="resume_ckpt", type=str, default="auto")
    parser.add_argument("--checkpoint-dir", "--checkpoint_dir", dest="checkpoint_dir", type=str, default="")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default="")
    parser.add_argument("--gcs-dir", "--gcs_dir", dest="gcs_dir", type=str, default="")
    parser.add_argument("--no-gcs", "--no_gcs", dest="no_gcs", action="store_true")
    parser.add_argument("--wandb", type=str, default="auto", choices=["auto", "online", "disabled"])
    parser.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", type=str, default="")
    parser.add_argument("--use-dataparallel", "--use_dataparallel", dest="use_dataparallel", action="store_true", default=True)
    parser.add_argument("--no-dataparallel", "--no_dataparallel", dest="no_dataparallel", action="store_true")
    parser.add_argument("--config-json", "--config_json", dest="config_json", type=str, default="")
    args, _ = parser.parse_known_args()
    return args


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = parse_args()

    cfg_json = getattr(args, "config_json", "")
    if cfg_json and os.path.exists(cfg_json):
        with open(cfg_json, "r", encoding="utf-8") as _f:
            _cfg_data = json.load(_f)
        for _k, _v in _cfg_data.items():
            if hasattr(args, _k):
                setattr(args, _k, _v)

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    is_ddp = ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ) and torch.cuda.is_available()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1")) if is_ddp else 1
    rank = int(os.environ.get("RANK", "0")) if is_ddp else 0
    is_main_process = rank == 0

    if is_ddp:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))

    # Применяем пресет быстрого CPU-тестирования (не перезаписывая явные значения, если это smoke-test)
    if args.cpu_fast and not args.smoke_test:
        args.device = "cpu"
        if args.batch_size == 16:
            args.batch_size = 4
        if args.grad_accum_steps == 4:
            args.grad_accum_steps = 2
        args.no_radar = True
        args.ce_weight = 0.0
        args.rollout_prob = 0.0
        if args.cache_latents == 0:
            args.cache_latents = 256
        if args.log_every == 20:
            args.log_every = 5
        if args.eval_every == 500:
            args.eval_every = 50
        if args.eval_batches == 4:
            args.eval_batches = 2
        if args.gen_steps == 20:
            args.gen_steps = 8
        if args.save_every == 500:
            args.save_every = 100
        args.wandb = "disabled"
        args.no_gcs = True

    env = AltEnvironmentConfig.detect(
        override_device=args.device,
        local_rank=local_rank,
        override_output_dir=args.output_dir,
        override_checkpoint_dir=args.checkpoint_dir,
        cpu_bf16=args.cpu_bf16,
        amp_dtype_str=args.amp_dtype,
    )
    if args.gcs_dir:
        env.gcs_checkpoints_dir = args.gcs_dir
    sync_gcs_flag = (env.is_kaggle and not args.no_gcs) or (not args.no_gcs and args.gcs_dir != "")

    if args.smoke_test:
        if is_main_process:
            print("=== РЕЖИМ SMOKE TEST (PHASE 3 ALT) АКТИВИРОВАН ===", flush=True)
        args.max_steps = 2
        args.batch_size = 2
        args.grad_accum_steps = 1
        args.log_every = 1
        args.eval_every = 2
        args.eval_batches = 1
        args.save_every = 2
        args.lr_warmup_steps = 1
        args.gen_steps = 2
        args.init_samples = min(args.init_samples, 8)
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
                run_name = (
                    args.wandb_name.strip()
                    if getattr(args, "wandb_name", "").strip()
                    else f"phase3-alt-{args.arch_preset}-{args.matrix_mode}-{args.init_mode}-{env.device.type}-{int(time.time())}"
                )
                wandb.init(
                    project="BEBLaDII-Phase3-Alt-PlanC",
                    name=run_name,
                    config=vars(args),
                )
                use_wandb = True
            except Exception as e:
                print(f"[WandB Warning] {e}", flush=True)

    # 1. Замороженная Система 1 (read-only) и опорные концепты
    tokenizer = get_tokenizer()
    sys1_amp = env.amp_dtype if env.use_amp else None
    encoder = get_encoder(
        weights_path=env.encoder_weights_path,
        device=env.device,
        eval_mode=True,
        quant_mode=args.sys1_quant,
        mode=args.sys1_mode,
        amp_dtype=sys1_amp,
    )
    decoder = get_decoder(
        weights_path=env.decoder_weights_path,
        encoder=encoder,
        device=env.device,
        eval_mode=True,
        quant_mode=args.sys1_quant,
        mode=args.sys1_mode,
        amp_dtype=sys1_amp,
    )
    for p in list(encoder.parameters()) + list(decoder.parameters()):
        p.requires_grad_(False)
    if is_main_process:
        q_pol = getattr(decoder, "quant_policy", None)
        enc_mb = estimate_model_memory_mb(encoder)
        dec_mb = estimate_model_memory_mb(decoder)
        if q_pol is not None:
            print(
                f"[System1 Quant] mode={q_pol.mode} | dev={q_pol.device_type} | quant={q_pol.quant_mode} "
                f"(enc={q_pol.encoder_preset}, dec={q_pol.decoder_preset}, layer={q_pol.layer_dtype}, "
                f"radar={q_pol.radar_dtype}, onednn={q_pol.use_cpu_onednn_int8}) | "
                f"mem: enc={enc_mb:.1f} MB, dec={dec_mb:.1f} MB | {q_pol.reason}",
                flush=True,
            )

    fast_vocab: Optional[FastVocabOptimizer] = None
    if not args.no_fast_vocab:
        fv_t0 = time.time()
        q_pol = getattr(decoder, "quant_policy", None)
        fv_dtype = q_pol.radar_dtype if q_pol is not None else (env.amp_dtype if env.use_amp else torch.float32)
        fast_vocab = FastVocabOptimizer.from_decoder(
            decoder=decoder,
            gemm_dtype=fv_dtype,
            ce_mode=args.ce_mode,
            ce_vocab_chunk_size=args.ce_vocab_chunk_size,
            svd_rank=args.vocab_svd_rank,
            ce_pool_hard=args.ce_pool_hard,
            ce_pool_rand=args.ce_pool_rand,
            ce_batch_fraction=args.ce_batch_fraction,
            attach=True,
            ce_batch_select=args.ce_batch_select,
        )
        if is_main_process:
            ce_desc = (
                f"chunk={fast_vocab.ce_vocab_chunk_size}"
                if fast_vocab.ce_mode == "exact_chunked"
                else f"svd_rank={fast_vocab.svd_rank}, pool={args.ce_max_tokens}+{fast_vocab.ce_pool_hard}+{fast_vocab.ce_pool_rand}"
            )
            print(
                f"[FastVocab] Подключён за {time.time() - fv_t0:.2f}s "
                f"(vocab={'int8' if fast_vocab.is_int8_vocab else 'float'}, radar_dtype={fv_dtype}, "
                f"ce_mode={fast_vocab.ce_mode}[{ce_desc}], "
                f"ce_batch_frac={fast_vocab.ce_batch_fraction:.2f}, ce_batch_select={fast_vocab.ce_batch_select})",
                flush=True,
            )

    sep_concept = torch.load(env.sep_concept_path, map_location="cpu", weights_only=False)
    void_concept = torch.load(env.void_concept_path, map_location="cpu", weights_only=False)
    if os.path.exists(env.stopwords_path):
        sw_obj = torch.load(env.stopwords_path, map_location="cpu", weights_only=False)
        stopword_mask = (sw_obj["mask"] if isinstance(sw_obj, dict) else sw_obj).bool()
    else:
        stopword_mask = torch.zeros(131072, dtype=torch.bool)

    # 2. Датасеты (Phase 6 parquet по умолчанию) — готовим до умной инициализации
    train_path = args.data_path if args.data_path != "auto" else env.train_data_path
    val_path = args.val_data_path if args.val_data_path != "auto" else env.val_data_path
    max_rows = 32 if args.smoke_test else (args.max_train_rows if args.max_train_rows > 0 else None)
    if args.cache_latents > 0 and max_rows is None:
        max_rows = args.cache_latents

    raw_train_dataset = PromptAnswerDataset(
        parquet_path=train_path,
        tokenizer=tokenizer,
        stopword_mask=stopword_mask,
        max_rows=max_rows,
        is_validation=False,
        elastic_drop_prob=args.elastic_drop_prob,
    )

    if args.cache_latents > 0:
        cache_file = args.cache_file or os.path.join(
            env.output_dir, f"latent_cache_{args.cache_latents}.pt"
        )
        train_dataset: Dataset = build_or_load_latent_cache(
            raw_dataset=raw_train_dataset,
            encoder=encoder,
            stopword_mask=stopword_mask,
            device=env.device,
            use_amp=env.use_amp,
            amp_dtype=env.amp_dtype,
            num_samples=args.cache_latents,
            cache_path=cache_file,
        )
    else:
        train_dataset = raw_train_dataset

    # 3. Создание гибридного Грассманианового ядра диффузии (с матричными оптимизациями)
    from matrix_layers import build_matrix_alt_core
    from init_strategies import apply_init_strategy, verify_init_invariants

    matrix_variant_map = {
        "dense": "ref_dense",
        "safe": "all_matrix_safe",
        "ultra": "all_matrix_ultra",
    }
    model = build_matrix_alt_core(
        void_ref=void_concept,
        variant=matrix_variant_map[args.matrix_mode],
        preset=args.arch_preset,
        phm_rank=args.phm_rank,
        cross_alpha=args.phm_alpha,
        full_ffn_dim=args.full_ffn_dim,
        grass_ffn_dim=args.grass_ffn_dim,
        num_sub_heads=args.num_sub_heads,
        local_radius=args.local_radius,
        arch_version=args.arch_version,
        axis_shift_delta=args.axis_shift_delta,
        adaln_bottleneck_r=args.adaln_bottleneck_r,
        output_head_phm_rank=args.output_head_phm_rank,
    ).to(env.device)

    resume_path = None
    has_resume_ckpt = False
    if is_main_process:
        if args.resume_ckpt and args.resume_ckpt.lower() != "none":
            if args.resume_ckpt == "auto":
                resume_path = find_latest_alt_checkpoint(
                    env.checkpoint_dir, env.gcs_checkpoints_dir if sync_gcs_flag else None
                )
            else:
                resume_path = args.resume_ckpt
        has_resume_ckpt = bool(resume_path and os.path.exists(resume_path))
    if is_ddp:
        resume_info = [resume_path, has_resume_ckpt]
        dist.broadcast_object_list(resume_info, src=0)
        resume_path, has_resume_ckpt = resume_info[0], bool(resume_info[1])

    # При старте с нуля применяем геометрическую/data-driven инициализацию ДО создания EMA
    if (not has_resume_ckpt) and args.init_mode != "none":
        if is_main_process:
            t_init0 = time.time()
            init_items = None
            if args.init_mode in ("compat", "all_5", "idea4_wiener_head", "idea5_kmeans_ffn"):
                if isinstance(train_dataset, PrecomputedLatentDataset):
                    init_items = train_dataset.items[: args.init_samples]
                else:
                    init_cache = build_or_load_latent_cache(
                        raw_dataset=raw_train_dataset,
                        encoder=encoder,
                        stopword_mask=stopword_mask,
                        device=env.device,
                        use_amp=env.use_amp,
                        amp_dtype=env.amp_dtype,
                        num_samples=min(args.init_samples, len(raw_train_dataset)),
                        cache_path=None,
                    )
                    init_items = init_cache.items

            init_info = apply_init_strategy(
                model=model,
                strategy_name=args.init_mode,
                void_ref=void_concept,
                sep_concept=sep_concept,
                train_items=init_items,
                wiener_damp=args.wiener_damp,
                base_seed=args.init_seed,
            )
            inv = verify_init_invariants(model, sep_concept=sep_concept, device=env.device)
            print(
                f"[Init] mode={args.init_mode} (damp={args.wiener_damp}) применена за {time.time() - t_init0:.2f}s | "
                f"sph_err={inv['max_sph_err']:.2e} | cfg_leak={inv['cfg_leak_diff']:.2e} | info={init_info}",
                flush=True,
            )
        if is_ddp:
            for p in model.parameters():
                dist.broadcast(p.data, src=0)
            for b in model.buffers():
                dist.broadcast(b.data, src=0)

    ema = EMAModel(model, decay=args.ema_decay, device=env.device, warmup=True) if not args.no_ema else None
    opt_muon, opt_adamw = create_alt_optimizers(
        model,
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
        scale_1d_lr=args.scale_1d_lr,
        orthmuon_sphere_scale=True,
        orthmuon_c_base=float(getattr(args, "orthmuon_c_base", 2.26)),
    )

    scaler = (
        torch.amp.GradScaler("cuda", init_scale=16384.0, growth_interval=1000)
        if env.use_grad_scaler
        else None
    )
    start_step = 0
    if has_resume_ckpt and resume_path is not None:
        start_step = load_alt_checkpoint(
            resume_path, model, opt_muon, opt_adamw, env.device, ema=ema, scaler=scaler
        )

    sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if is_ddp
        else None
    )
    use_bucket = (
        getattr(args, "gpu_dynamic_crop", False)
        and (not is_ddp)
        and env.device.type != "xla"
        and len(train_dataset) >= args.batch_size
    )
    if use_bucket:
        if isinstance(train_dataset, PrecomputedLatentDataset):
            ds_lengths = [int(it["i_last"].item()) for it in train_dataset.items]
        else:
            ds_lengths = [len(a) for _, a in getattr(train_dataset, "pairs", [])]
        if len(ds_lengths) == len(train_dataset):
            bucket_sampler = LengthBucketBatchSampler(
                ds_lengths, batch_size=args.batch_size, drop_last=True
            )
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=bucket_sampler,
                num_workers=env.num_workers,
                pin_memory=env.pin_memory,
            )
            sampler = bucket_sampler  # type: ignore[assignment]
        else:
            use_bucket = False
    if not use_bucket:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=env.num_workers,
            pin_memory=env.pin_memory,
            drop_last=(len(train_dataset) >= args.batch_size),
        )

    val_loader = None
    if is_main_process and val_path and os.path.exists(val_path):
        val_dataset = PromptAnswerDataset(
            parquet_path=val_path,
            tokenizer=tokenizer,
            stopword_mask=stopword_mask,
            max_rows=16 if args.smoke_test else 256,
            is_validation=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(args.batch_size, 8),
            shuffle=False,
            num_workers=env.num_workers,
            pin_memory=env.pin_memory,
        )

    # 4. Multi-GPU (DDP основной режим, DataParallel fallback)
    if is_ddp:
        # В DDP каждый процесс прогревает cuSOLVER только на своём GPU
        _eye = torch.eye(8, device=env.device, dtype=torch.float32)
        _ = torch.linalg.eigh(_eye)
    elif env.device.type == "cuda" and torch.cuda.device_count() > 1:
        warmup_linalg_backends()

    wrapped_model: nn.Module = model
    if is_ddp:
        wrapped_model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    elif (
        env.device.type == "cuda"
        and torch.cuda.device_count() > 1
        and args.use_dataparallel
        and not args.no_dataparallel
    ):
        if is_main_process:
            print(f"[DataParallel] Обнаружено {torch.cuda.device_count()} GPU! Включение DataParallel...", flush=True)
        wrapped_model = nn.DataParallel(model)

    try:
        run_alt_training_loop(
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
            fast_vocab=fast_vocab,
            scaler=scaler,
        )
    finally:
        if is_ddp and dist.is_initialized():
            dist.destroy_process_group()


# %% 9. Конфигурация запуска (Kaggle / Notebook)
TRAINING_CONFIG: Dict[str, Any] = {
    "smoke_test": False,
    "device": "auto",
    "sys1_quant": "auto",
    "sys1_mode": "train",
    "use_ddp": True,
    "use_dataparallel": True,
    "arch_preset": "standard",
    "arch_version": "v2",
    "axis_shift_delta": 0.35,
    "adaln_bottleneck_r": 64,
    "output_head_phm_rank": 8,
    "spec_diag_weight": 0.05,
    "orth_offdiag_weight": 0.05,
    "anchor_weight": 0.15,
    "basis_high_t_mode": "sign_only",
    "adamw_grad_clip": 1.0,
    "sphere_grad_clip": 10.0,
    "orthmuon_c_base": 2.26,
    "gpu_dynamic_crop": True,
    "gpu_dynamic_crop_full_prob": 0.25,
    "matrix_mode": "safe",
    "phm_rank": 4,
    "phm_alpha": 0.25,
    "init_mode": "all_5",
    "wiener_damp": 0.7,
    "init_samples": 128,
    "data_path": "auto",
    "val_data_path": "auto",
    "batch_size": 16,
    "grad_accum_steps": 4,
    "max_steps": 20000,
    "muon_lr": 1e-3,
    "adamw_lr": 3e-4,
    "scale_1d_lr": 1.2e-3,
    "lr_warmup_steps": 500,
    "ema_decay": 0.999,
    "log_every": 20,
    "eval_every": 200,
    "eval_batches": 4,
    "cfg_dsim_batches": 4,
    "gen_steps": 20,
    "save_every": 500,
    "resume_ckpt": "auto",
    "gcs_dir": "gs://bebladii-weigths-us/planC/Phase3_Alt_Diffusion/v2_safe_all5/checkpoints/",
    "no_gcs": False,
    "wandb": "auto",
    "wandb_name": "",
    "elastic_drop_prob": 0.0,
    "rollout_prob": 0.35,
    "rollout_start_step": 1000,
    "ce_weight": 0.05,
    "ce_warmup_steps": 300,
    "ce_max_tokens": 512,
    "ce_t_max": 0.70,
    "ce_stopword_weight": 0.25,
    "ce_mode": "exact_chunked",
    "ce_vocab_chunk_size": 16384,
    "ce_batch_fraction": 0.5,
    "ce_batch_select": "random",
}

if __name__ == "__main__":
    cli_args = parse_args()
    is_notebook = any("ipykernel" in m for m in sys.modules)
    if is_notebook:
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        use_ddp_auto = (
            bool(TRAINING_CONFIG.get("use_ddp", True))
            and str(TRAINING_CONFIG.get("device", "auto")) != "cpu"
            and n_gpus > 1
            and ("RANK" not in os.environ)
        )
        script_path = ALT_DIR / "train_phase3_alt.py"
        if use_ddp_auto and script_path.exists():
            ddp_cfg = dict(TRAINING_CONFIG)
            orig_ga = int(ddp_cfg.get("grad_accum_steps", 4))
            ddp_cfg["grad_accum_steps"] = max(1, orig_ga // n_gpus)
            eff_b = int(ddp_cfg.get("batch_size", 16)) * ddp_cfg["grad_accum_steps"] * n_gpus
            cfg_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else str(ALT_DIR)
            cfg_path = os.path.join(cfg_dir, "ddp_runtime_config.json")
            with open(cfg_path, "w", encoding="utf-8") as _f:
                json.dump(ddp_cfg, _f, indent=2, ensure_ascii=False)
            print(
                f"[DDP Launcher] Обнаружено {n_gpus} GPU! Запуск torchrun (DDP) вместо DataParallel | "
                f"grad_accum: {orig_ga} -> {ddp_cfg['grad_accum_steps']} (eff_batch={eff_b})",
                flush=True,
            )
            proc_env = os.environ.copy()
            proc_env["PYTHONUNBUFFERED"] = "1"
            proc_env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={n_gpus}",
                str(script_path),
                "--config-json",
                cfg_path,
            ]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=proc_env,
            )
            assert proc.stdout is not None
            for _line in proc.stdout:
                print(_line, end="", flush=True)
            ret = proc.wait()
            if ret != 0:
                raise RuntimeError(f"DDP-процесс (torchrun) завершился с кодом {ret}")
        else:
            for k, v in TRAINING_CONFIG.items():
                if hasattr(cli_args, k):
                    setattr(cli_args, k, v)
            main(cli_args)
    else:
        main(cli_args)
