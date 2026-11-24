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
import glob
import math
import os
import re
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
ALT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ALT_DIR.parents[1]
PHASE3_DIR = PROJECT_ROOT / "experiments" / "phase 3"
for _p in (str(PROJECT_ROOT), str(PHASE3_DIR), str(ALT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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

from train_phase3 import (
    ConditionEmbedder,
    EMAModel,
    GrassmannianOutputHead,
    N_ANS,
    N_CANVAS,
    N_PROMPT,
    OrthMuon,
    PAD_TOKEN_ID,
    PerHeadRMSNorm,
    PromptAnswerDataset,
    SIGMA_DATA,
    _clamp_fp16_safe,
    _clamp_swiglu,
    _compute_adaln_stats,
    _init_soft_gate_adaln,
    apply_prompt_dropout,
    block_ar_sample,
    build_rollout_noisy_canvas,
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
                    os.path.join(project_root, "data", "plan C", "tekken_stopwords_mask.pt"),
                ],
                search_roots,
            )
            encoder_weights_path = resolve_file_path(
                "encoder/weights.pt",
                [
                    os.path.join(project_root, "storage", "components", "encoder", "weights.pt"),
                    "/kaggle/input/datasets/bogdanbuliakov/bebladii-planc-phase1-data/encoder/weights.pt",
                ],
                search_roots,
            )
            decoder_weights_path = resolve_file_path(
                "phase2_decoder_step_25000.pt",
                [
                    "/kaggle/input/datasets/bogdanbuliakov/bebladii-planc-phase2-data/decoder/phase2_decoder_step_25000.pt",
                    "/kaggle/input/datasets/bogdanbuliakov/bebladii-planc-phase2-data/decoder/weights.pt",
                    os.path.join(project_root, "storage", "components", "decoder", "weights.pt"),
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
            stopwords_path = os.path.join(project_root, "data", "plan C", "tekken_stopwords_mask.pt")
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


def _apply_cell_rope_to_vectors(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Применяет RoPE с ОДИНАКОВЫМ позиционным углом клетки i ко всем K=8 векторам этой клетки.
    x: (B, H, N, K, d_h), cos/sin: (N, d_h).
    Внутри одной клетки относительный угол равен 0 -> порядок осей k=0..7 не искажается.
    """
    cos_b = cos.to(dtype=x.dtype).view(1, 1, x.shape[2], 1, x.shape[4])
    sin_b = sin.to(dtype=x.dtype).view(1, 1, x.shape[2], 1, x.shape[4])
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rot = torch.cat((-x2, x1), dim=-1)
    return (x * cos_b) + (rot * sin_b)


# %% 3. Блоки Грассманианового внимания в R^128 и Full-1024 внимания
class GrassmannianPromptAdapterLayer(nn.Module):
    """
    Один слой легкого Грассманианового PromptAdapter в R^128:
    Full векторное внимание по всем 8*N_p=512 базисным векторам промпта (общие 128x128 веса)
    + Shared Axis-wise SwiGLU FFN + LayerScale.
    """

    def __init__(
        self,
        axis_dim: int = AXIS_DIM,
        ffn_dim: int = 512,
        num_sub_heads: int = 4,
        init_layer_scale: float = 0.02,
    ):
        super().__init__()
        self.axis_dim = axis_dim
        self.num_sub_heads = num_sub_heads
        self.sub_head_dim = axis_dim // num_sub_heads

        self.attn_norm = RMSNorm(axis_dim)
        self.norm_emb = nn.Linear(1, axis_dim)
        nn.init.zeros_(self.norm_emb.weight)
        nn.init.zeros_(self.norm_emb.bias)

        self.q_proj = SphereLinear(axis_dim, axis_dim, bias=False)
        self.kv_proj = SphereLinear(axis_dim, axis_dim, bias=False)
        self.o_proj = SphereLinear(axis_dim, axis_dim, bias=False)
        self.q_norm = PerHeadRMSNorm(num_sub_heads, self.sub_head_dim)
        self.k_norm = PerHeadRMSNorm(num_sub_heads, self.sub_head_dim)
        self.v_gate = nn.Parameter(torch.ones(1, num_sub_heads, 1, self.sub_head_dim))

        self.mlp_norm = RMSNorm(axis_dim)
        self.gate_proj = SphereLinear(axis_dim, ffn_dim, bias=False)
        self.up_proj = SphereLinear(axis_dim, ffn_dim, bias=False)
        self.down_proj = SphereLinear(ffn_dim, axis_dim, bias=False)

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

        # Сохраняем информацию об энергии оси ||v_k|| перед RMSNorm (O(8)-эквивариантно)
        log_norm = torch.log1p(x.float().norm(p=2, dim=-1, keepdim=True)).to(x.dtype)
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
        gated = _clamp_swiglu(self.gate_proj(h_mlp), self.up_proj(h_mlp))
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
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GrassmannianPromptAdapterLayer(
                    axis_dim=axis_dim,
                    ffn_dim=ffn_dim,
                    num_sub_heads=num_sub_heads,
                    init_layer_scale=init_layer_scale,
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
        x = (V_prompt.float() / SIGMA_DATA).to(V_prompt.dtype)
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
    - Каждый базисный вектор v_{i,k} in R^128 видит ВСЕ 8 векторов соседних клеток v_{j,0..7}
      (включая 7 остальных векторов своей клетки при j == i) через общие проекции 128x128!
    - При `local_radius is not None`: быстрое явное оконное внимание (W = 2*r + 1 клеток +
      ровно 1 копия якоря SEP без дублирования), что в ~9-12x быстрее плотной 1024x1024 матрицы.
    - При `local_radius is None`: полное векторное внимание по всем 8*N=1024 векторам холста.
    - Shared Axis-wise SwiGLU FFN (128 -> ffn_dim -> 128) + O(8)-эквивариантный Soft-Gate AdaLN.
    """

    def __init__(
        self,
        axis_dim: int = AXIS_DIM,
        ffn_dim: int = 512,
        num_sub_heads: int = 4,
        local_radius: Optional[int] = 6,
        has_ca: bool = True,
        cond_dim: int = 256,
    ):
        super().__init__()
        self.axis_dim = axis_dim
        self.num_sub_heads = num_sub_heads
        self.sub_head_dim = axis_dim // num_sub_heads
        self.local_radius = local_radius

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
        self.q_norm = PerHeadRMSNorm(num_sub_heads, self.sub_head_dim)
        self.k_norm = PerHeadRMSNorm(num_sub_heads, self.sub_head_dim)
        self.v_gate = nn.Parameter(torch.ones(1, num_sub_heads, 1, self.sub_head_dim))

        self.mlp_norm = RMSNorm(axis_dim)
        self.gate_proj = SphereLinear(axis_dim, ffn_dim, bias=False)
        self.up_proj = SphereLinear(axis_dim, ffn_dim, bias=False)
        self.down_proj = SphereLinear(ffn_dim, axis_dim, bias=False)

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

        mods = self.adaln_proj(c.to(self.adaln_proj.weight.dtype)).to(x_prev.dtype)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods.chunk(6, dim=-1)
        M = torch.cat([shift_msa, scale_msa, shift_mlp, scale_mlp], dim=-1)
        stats = _repeat_adaln_stats_to_4096(_compute_adaln_stats(M, m_active))

        # Broadcasting AdaLN по 8 осям (O(8)-эквивариантно)
        s_msa = shift_msa.unsqueeze(2)
        sc_msa = scale_msa.unsqueeze(2)
        g_msa = gate_msa.unsqueeze(2)
        s_mlp = shift_mlp.unsqueeze(2)
        sc_mlp = scale_mlp.unsqueeze(2)
        g_mlp = gate_mlp.unsqueeze(2)

        # 3-слотовый поосевой DCA: отдельные композиции для Q и Shared KV
        x_q = self.dca_q(x_in, h_ca, x_prev)
        x_kv = self.dca_kv(x_in, h_ca, x_prev)

        ln_q = torch.log1p(x_q.float().norm(p=2, dim=-1, keepdim=True)).to(x_q.dtype)
        ln_kv = torch.log1p(x_kv.float().norm(p=2, dim=-1, keepdim=True)).to(x_kv.dtype)
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
        gated = _clamp_swiglu(self.gate_proj(h_mlp), self.up_proj(h_mlp))
        mlp_out = _clamp_fp16_safe(self.down_proj(gated))
        x = x + g_mlp * mlp_out
        return x, stats


class GrassmannianCABlock(nn.Module):
    """
    Грассманианов блок Cross-Attention на промпт в R^128:
    - Каждый из 8 векторов ответа v_{i,k} in R^128 делает CA ко всем 8*N_p=512 базисным
      векторам промпта p_{j,m} in R^128 + Zero-Vector Sink `k_sink` (v_sink == 0).
    - Использует 1 экземпляр `PerAxisGRNv3` (`dca_q`) на входном стеке `[x_in, h_ca, x_prev]`.
    - Shared KV + Split K/V Channel Gating + Token-Centered Contrast строго на K_p (ADR 135).
    - Однократный float32 softmax (ADR 136) + Cross-Gated SwiGLU FFN в R^128 (ADR 110).
    - Возвращает чистую дельту `ca_delta` для обновления слота `h_ca` в DCA-пробросе.
    """

    def __init__(
        self,
        axis_dim: int = AXIS_DIM,
        ffn_dim: int = 512,
        num_sub_heads: int = 4,
        rope_offset: int = N_PROMPT,
        has_ca: bool = False,
        cond_dim: int = 256,
    ):
        super().__init__()
        self.axis_dim = axis_dim
        self.num_sub_heads = num_sub_heads
        self.sub_head_dim = axis_dim // num_sub_heads
        self.rope_offset = rope_offset

        self.dca_q = PerAxisGRNv3(NUM_AXES, axis_dim, has_ca=has_ca)

        self.ca_norm = RMSNorm(axis_dim)
        self.prompt_norm = RMSNorm(axis_dim)
        self.norm_emb_q = nn.Linear(1, axis_dim)
        nn.init.zeros_(self.norm_emb_q.weight)
        nn.init.zeros_(self.norm_emb_q.bias)

        self.q_proj = SphereLinear(axis_dim, axis_dim, bias=False)
        self.kv_proj = SphereLinear(axis_dim, axis_dim, bias=False)
        self.o_proj = SphereLinear(axis_dim, axis_dim, bias=False)
        self.q_norm = PerHeadRMSNorm(num_sub_heads, self.sub_head_dim)
        self.k_norm = PerHeadRMSNorm(num_sub_heads, self.sub_head_dim)
        self.v_gate = nn.Parameter(torch.ones(1, num_sub_heads, 1, self.sub_head_dim))

        # Zero-Vector Attention Sink (без RoPE, v_sink == 0)
        self.k_sink = nn.Parameter(torch.randn(1, num_sub_heads, 1, self.sub_head_dim) * 0.02)

        self.mlp_norm = RMSNorm(axis_dim)
        self.gate_proj = SphereLinear(axis_dim, ffn_dim, bias=False)
        self.w_canvas_gate = SphereLinear(axis_dim, ffn_dim, bias=False)
        self.up_proj = SphereLinear(axis_dim, ffn_dim, bias=False)
        self.down_proj = SphereLinear(ffn_dim, axis_dim, bias=False)

        self.adaln_proj = nn.Linear(cond_dim, 6 * axis_dim)
        _init_soft_gate_adaln(self.adaln_proj, axis_dim, gate_bias=0.02)

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
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        B, N_a, K, D = x_prev.shape
        N_p = x_prompt.shape[1]
        H, dh = self.num_sub_heads, self.sub_head_dim

        mods = self.adaln_proj(c.to(self.adaln_proj.weight.dtype)).to(x_prev.dtype)
        shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = mods.chunk(6, dim=-1)
        M = torch.cat([shift_ca, scale_ca, shift_mlp, scale_mlp], dim=-1)
        stats = _repeat_adaln_stats_to_4096(_compute_adaln_stats(M, m_active))

        s_ca = shift_ca.unsqueeze(2)
        sc_ca = scale_ca.unsqueeze(2)
        g_ca = gate_ca.unsqueeze(2)
        s_mlp = shift_mlp.unsqueeze(2)
        sc_mlp = scale_mlp.unsqueeze(2)
        g_mlp = gate_mlp.unsqueeze(2)

        x_q = self.dca_q(x_in, h_ca_prev, x_prev)
        ln_q = torch.log1p(x_q.float().norm(p=2, dim=-1, keepdim=True)).to(x_q.dtype)
        h_ans_in = (self.ca_norm(x_q) + self.norm_emb_q(ln_q)) * (1.0 + sc_ca) + s_ca
        h_prompt_in = self.prompt_norm(x_prompt)

        q = self.q_proj(h_ans_in).view(B, N_a, K, H, dh).permute(0, 3, 1, 2, 4)
        u_p = self.kv_proj(h_prompt_in).view(B, N_p, K, H, dh).permute(0, 3, 1, 2, 4)
        u_p_flat = u_p.reshape(B, H, N_p * K, dh)

        # Маска активных (не-VOID) векторов промпта: (B, N_p * K)
        p_vec_mask = prompt_attn_mask.unsqueeze(-1).expand(B, N_p, K).reshape(B, N_p * K)
        mask_p_f = p_vec_mask.float().view(B, 1, N_p * K, 1)
        denom = mask_p_f.sum(dim=2, keepdim=True).clamp(min=1.0)
        # Token-Centered Contrast строго на ветке ключей K_p (ADR 135)
        u_p_mean = (u_p_flat.float() * mask_p_f).sum(dim=2, keepdim=True) / denom
        u_p_centered = (u_p_flat.float() - 0.5 * u_p_mean).to(u_p_flat.dtype)

        q = self.q_norm(q.reshape(B, H, N_a * K, dh)).view(B, H, N_a, K, dh)
        k_p = self.k_norm(u_p_centered).view(B, H, N_p, K, dh)
        v_p = u_p_flat * self.v_gate.to(u_p_flat.dtype)  # V никогда не проходит через RoPE

        cos_all, sin_all = rope_grass(x_prev, self.rope_offset + N_a)
        cos_p, sin_p = cos_all[:N_p], sin_all[:N_p]
        cos_a = cos_all[self.rope_offset : self.rope_offset + N_a]
        sin_a = sin_all[self.rope_offset : self.rope_offset + N_a]

        q_rot = _apply_cell_rope_to_vectors(q, cos_a, sin_a).reshape(B, H, N_a * K, dh)
        k_p_rot = _apply_cell_rope_to_vectors(k_p, cos_p, sin_p).reshape(B, H, N_p * K, dh)

        k_sink_exp = self.k_norm(self.k_sink.to(k_p_rot.dtype)).expand(B, -1, -1, -1)
        k_full = torch.cat([k_p_rot, k_sink_exp], dim=2)  # (B, H, N_p*K + 1, dh)

        attn_mask = torch.zeros(B, 1, 1, N_p * K + 1, device=x_prev.device, dtype=torch.float32)
        attn_mask[:, :, :, : N_p * K] = attn_mask[:, :, :, : N_p * K].masked_fill(
            ~p_vec_mask[:, None, None, :], -10000.0
        )

        # Однократный расчёт softmax в float32 (ADR 136)
        scores = torch.matmul(q_rot.float(), k_full.float().transpose(-2, -1)) / math.sqrt(dh)
        scores = scores + attn_mask
        probs = F.softmax(scores, dim=-1)

        sink_weights = None
        if return_sink_weights:
            # Средняя доля внимания в k_sink по головам H и 8 базисным векторам клетки -> (B, N_a)
            sink_weights = probs[:, :, :, -1].view(B, H, N_a, K).mean(dim=(1, 3))

        # v_sink == 0 -> учитываем только первые N_p*K ключей промпта
        attn_out = torch.matmul(probs[:, :, :, : N_p * K].to(v_p.dtype), v_p)
        attn_out = attn_out.view(B, H, N_a, K, dh).permute(0, 2, 3, 1, 4).reshape(B, N_a, K, D)
        h_ca = self.o_proj(attn_out)
        delta_attn = g_ca * h_ca
        x_mid = x_prev + delta_attn

        # Cross-Gated SwiGLU FFN в R^128: val строго из h_ca (физическая изоляция при CFG dropout)
        x_tilde = self.mlp_norm(x_mid) * (1.0 + sc_mlp) + s_mlp
        gate_pre = self.gate_proj(h_ca).float() + self.w_canvas_gate(x_tilde).float()
        prod = F.silu(gate_pre) * self.up_proj(h_ca).float()
        gated = torch.clamp(prod, -65000.0, 65000.0).to(x_prev.dtype)
        ffn_out = _clamp_fp16_safe(self.down_proj(gated))
        delta_mlp = g_mlp * ffn_out

        ca_delta = delta_attn + delta_mlp
        x_out = x_prev + ca_delta
        return x_out, ca_delta, stats, sink_weights


class FullSABlock(nn.Module):
    """
    Полный интеграторский слой (1024d):
    - Получает стек `[x_in, h_ca, x_prev]` формы `(B, N, 8, 128)` -> 2x `PerAxisGRNv3` (`dca_q`, `dca_kv`).
    - Переходит в полное 1024-мерное пространство клетки: Full SA (8 голов x 128d, Shared KV,
      PerHeadRMSNorm, RoPE) + полноценный SwiGLU FFN (`1024 -> full_ffn_dim -> 1024`) + Soft-Gate AdaLN.
    - Возвращает обновленный тензор в форме `(B, N, 8, 128)`.
    """

    def __init__(
        self,
        hidden_size: int = HIDDEN_SIZE,
        intermediate_size: int = 2048,
        num_heads: int = 8,
        has_ca: bool = True,
        cond_dim: int = 256,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

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
    Альтернативное условное ядро диффузии Фазы 3 Plan C:
    - Гибридный стек из слоев Грассманианового векторного внимания (R^128),
      2x Грассманианового CA_Prompt (R^128) и слоев Full Attention (1024d);
    - Поосевой 3-слотовый DCA-проброс `[x_in, h_ca, x_prev]` в каждом блоке;
    - 100% совместимость интерфейса (`encode_prompt`, `forward_with_cached_prompt`, `forward`)
      с лоссами, сэмплерами и Evaluation Gate из `train_phase3.py`.
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
    ):
        super().__init__()
        if preset not in ARCH_PRESETS:
            raise ValueError(f"Неизвестный пресет архитектуры: {preset}. Доступны: {list(ARCH_PRESETS)}")
        cfg = dict(ARCH_PRESETS[preset])
        if full_ffn_dim is not None:
            cfg["full_ffn_dim"] = int(full_ffn_dim)
        if grass_ffn_dim is not None:
            cfg["grass_ffn_dim"] = int(grass_ffn_dim)
        if num_sub_heads is not None:
            cfg["num_sub_heads"] = int(num_sub_heads)
        if local_radius is not None:
            cfg["local_radius"] = int(local_radius)

        self.preset = preset
        self.arch_cfg = cfg
        self.hidden_size = HIDDEN_SIZE
        self.num_sub_heads = cfg["num_sub_heads"]
        self.sub_head_dim = AXIS_DIM // self.num_sub_heads
        self.stack_spec: List[str] = list(cfg["stack"])

        self.rope_full = RotaryEmbedding128(dim=AXIS_DIM, max_seq_len=512)
        self.rope_grass = RotaryEmbedding128(dim=self.sub_head_dim, max_seq_len=512)
        self.cond_embedder = ConditionEmbedder(void_ref=void_ref, cond_dim=cond_dim)
        self.prompt_adapter = GrassmannianPromptAdapter(
            num_layers=cfg["prompt_layers"],
            axis_dim=AXIS_DIM,
            ffn_dim=cfg["grass_ffn_dim"],
            num_sub_heads=self.num_sub_heads,
            init_layer_scale=0.02,
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
                    )
                )
            elif kind == "grass_ca":
                layers.append(
                    GrassmannianCABlock(
                        axis_dim=AXIS_DIM,
                        ffn_dim=cfg["grass_ffn_dim"],
                        num_sub_heads=self.num_sub_heads,
                        rope_offset=N_PROMPT,
                        has_ca=seen_ca,
                        cond_dim=cond_dim,
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
                    )
                )
            else:
                raise ValueError(f"Неизвестный тип слоя: {kind}")

        self.layers = nn.ModuleList(layers)
        # Финальный DCA-миксер перед OutputHead: видит [x_in, последний h_ca, x_last]
        self.final_dca = PerAxisGRNv3(NUM_AXES, AXIS_DIM, has_ca=seen_ca)
        self.output_head = GrassmannianOutputHead(hidden_size=HIDDEN_SIZE, cond_dim=cond_dim)

    def encode_prompt(
        self,
        V_prompt: torch.Tensor,
        prompt_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Однократное кодирование промпта в (B, N_p, 8, 128) через GrassmannianPromptAdapter."""
        return self.prompt_adapter(V_prompt, self.rope_grass, prompt_valid_mask=prompt_valid_mask)

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
        B, N_a, K, D = V_t.shape
        m_active = t_mat > 1e-3

        c = self.cond_embedder(V_t, t_mat, m_prompt, radar_sigma=radar_sigma)
        x_in = (V_t.float() / SIGMA_DATA).to(V_t.dtype)  # (B, N_a, 8, 128)
        x_cur = x_in
        h_ca: Optional[torch.Tensor] = None

        cos_full, sin_full = self.rope_full(x_in, N_a)
        cos_grass, sin_grass = self.rope_grass(x_in, N_a)

        sum_m_list, sum_m2_list, count_list = [], [], []
        sink_list = []

        for layer in self.layers:
            if isinstance(layer, GrassmannianCABlock):
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
                )
                # Обновляем слот ближайшего CA чистой дельтой сигнала промпта
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

        x_out = self.final_dca(x_in, h_ca, x_cur).reshape(B, N_a, K * D)
        V_0_hat = self.output_head(x_out, V_t, t_mat, c)
        sink_weights = torch.stack(sink_list, dim=1) if sink_list else None  # (B, 2, N_a)

        num_layers = len(self.layers)
        sum_m_pack = torch.stack(sum_m_list, dim=1)                       # (1, L, 4096)
        sum_m2_pack = torch.stack(sum_m2_list, dim=1)                     # (1, L, 4096)
        count_pack = torch.stack(count_list, dim=0).view(1, num_layers)   # (1, L)
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
) -> Tuple[Optional[OrthMuon], torch.optim.AdamW]:
    """
    Распределение параметров:
    1. OrthMuon (sphere=True, wd=0.0): все *.q_weight (SphereLinear в Grass/Full/CA/PromptAdapter).
    2. AdamW (scale_1d_lr, wd=0.0): row_scale, v_gate, RMSNorm/QK-norm, ls_attn/ls_mlp.
    3. AdamW (adamw_lr, wd=0.0): все параметры DCA (dca_q, dca_kv, final_dca — чтобы WD не гасил
       identity-вес b_prev=1.0), bias, LayerNorm, norm_emb.
    4. AdamW (adamw_lr, wd=weight_decay): матрицы AdaLN, ConditionEmbedder, OutputHead, k_sink.
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
            or ("_norm" in name and "norm_emb" not in name)
            or ("q_norm." in name)
            or ("k_norm." in name)
        ):
            scale_1d_params.append(param)
        elif (
            "dca_q." in name
            or "dca_kv." in name
            or "final_dca." in name
            or "norm_emb" in name
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
            [{"params": sphere_params, "lr": muon_lr, "weight_decay": 0.0, "sphere": True}],
            lr=muon_lr,
            momentum=0.95,
            ns_steps=5,
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
            items.append(
                {
                    "V_prompt": vp_cpu[i],
                    "V_ans": va_cpu[i],
                    "ans_targets": batch["ans_targets"][i],
                    "n_real_prompt": batch["n_real_prompt"][i],
                    "i_last": batch["i_last"][i],
                    "w_cell": batch["w_cell"][i],
                }
            )
        if (b_idx + 1) % 10 == 0 or len(items) == n:
            print(f"  [Latent Cache] {len(items)}/{n} ({time.time() - t0:.1f}s)", flush=True)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(items, cache_path)
        print(f"[Latent Cache] Сохранено в {cache_path}.", flush=True)
    return PrecomputedLatentDataset(items)


# %% 7. Чекпоинты Phase 3 Alt и основной цикл обучения
def find_latest_alt_checkpoint(checkpoint_dir: str) -> Optional[str]:
    local_files = glob.glob(os.path.join(checkpoint_dir, "phase3_alt_step_*.pt"))
    if not local_files:
        return None
    local_files.sort(key=lambda f: int(re.findall(r"\d+", os.path.basename(f))[-1]))
    return local_files[-1]


def load_alt_checkpoint(
    path: str,
    model: nn.Module,
    opt_muon: Optional[OrthMuon],
    opt_adamw: torch.optim.AdamW,
    device: torch.device,
    ema: Optional[EMAModel] = None,
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

    step = int(ckpt.get("step", 0))
    print(f"[Resume] Восстановлено состояние Phase 3 Alt с шага {step} ({path}).", flush=True)
    return step


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
) -> None:
    raw_model = getattr(model, "module", model)
    scaler = (
        torch.amp.GradScaler("cuda", init_scale=16384.0, growth_interval=1000)
        if env.use_grad_scaler
        else None
    )

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
            f"=== Старт обучения Phase 3 Alt (preset={args.arch_preset}, device={env.device}, "
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
            B = n_real_prompt.shape[0]

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
                    t_mat = sample_mixed_noise_schedule(B, env.device, i_last=i_last)
                    noise = torch.randn_like(V_0, dtype=torch.float32) * SIGMA_DATA
                    V_t = (1.0 - t_mat.unsqueeze(-1).unsqueeze(-1)) * V_0.float() + t_mat.unsqueeze(-1).unsqueeze(-1) * noise
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
                    world_size=world_size,
                )

                # Сквозной токенный CE через замороженный Декодер (если включен)
                w_ce = float(args.ce_weight) * min(
                    1.0, float(max(1, step - start_step)) / float(max(1, args.ce_warmup_steps))
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

        # Шаг оптимизаторов
        if scaler is not None:
            if opt_muon is not None:
                scaler.unscale_(opt_muon)
            scaler.unscale_(opt_adamw)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
            if opt_muon is not None:
                scaler.step(opt_muon)
            scaler.step(opt_adamw)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
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

        # Валидация на EMA-весах
        if is_main_process and val_dataloader is not None and (step % args.eval_every == 0):
            if ema is not None:
                ema.apply_shadow(raw_model)
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
            if ema is not None:
                ema.restore(raw_model)

            gen_samples = val_res.pop("gen_samples", [])
            print(
                f"  [Alt Eval @ {step}] Sim_Gr={val_res['val/sim_gr_denoising']:.4f} (hi={val_res['val/sim_gr_hi']:.4f}) | "
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

        # Регулярный чекпоинт
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

    if is_main_process:
        print("=== Обучение Phase 3 Alt завершено ===", flush=True)


# %% 8. CLI и точка входа
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BEBLaDII Phase 3 Alt: Grassmannian 2D Attention (R^128) + Full SA + 2x GrassCA + 3-slot DCA"
    )
    parser.add_argument("--smoke-test", "--smoke_test", dest="smoke_test", action="store_true", help="Быстрый smoke-test на 2 шага")
    parser.add_argument("--cpu-fast", "--cpu_fast", dest="cpu_fast", action="store_true", help="Пресет быстрого тестирования на CPU (без тяжелых прогонов 131k-словаря на каждом шаге)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Устройство исполнения")
    parser.add_argument("--amp-dtype", "--amp_dtype", dest="amp_dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--cpu-bf16", "--cpu_bf16", dest="cpu_bf16", action="store_true", help="Включить bfloat16 autocast на CPU")
    parser.add_argument("--num-threads", "--num_threads", dest="num_threads", type=int, default=0, help="Ограничить число CPU-потоков PyTorch (0 = по умолчанию)")

    # Архитектура
    parser.add_argument("--arch-preset", "--arch_preset", dest="arch_preset", type=str, default="standard", choices=list(ARCH_PRESETS.keys()))
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
    parser.add_argument("--ce-t-max", "--ce_t_max", dest="ce_t_max", type=float, default=0.50)
    parser.add_argument("--ce-stopword-weight", "--ce_stopword_weight", dest="ce_stopword_weight", type=float, default=0.25)
    parser.add_argument(
        "--no-fast-vocab", "--no_fast_vocab", dest="no_fast_vocab", action="store_true",
        help="Отключить TPU/GPU-оптимизацию словаря (откат к базовому FP32 радару и полному 131K CE)",
    )
    parser.add_argument("--ce-pool-hard", "--ce_pool_hard", dest="ce_pool_hard", type=int, default=1536)
    parser.add_argument("--ce-pool-rand", "--ce_pool_rand", dest="ce_pool_rand", type=int, default=2048)
    parser.add_argument("--ce-batch-fraction", "--ce_batch_fraction", dest="ce_batch_fraction", type=float, default=0.5)
    parser.add_argument("--vocab-svd-rank", "--vocab_svd_rank", dest="vocab_svd_rank", type=int, default=64)

    # Логирование, чекпоинты, multi-GPU
    parser.add_argument("--log-every", "--log_every", dest="log_every", type=int, default=20)
    parser.add_argument("--eval-every", "--eval_every", dest="eval_every", type=int, default=500)
    parser.add_argument("--eval-batches", "--eval_batches", dest="eval_batches", type=int, default=16)
    parser.add_argument("--gen-steps", "--gen_steps", dest="gen_steps", type=int, default=20)
    parser.add_argument("--save-every", "--save_every", dest="save_every", type=int, default=500)
    parser.add_argument("--resume-ckpt", "--resume_ckpt", dest="resume_ckpt", type=str, default="auto")
    parser.add_argument("--checkpoint-dir", "--checkpoint_dir", dest="checkpoint_dir", type=str, default="")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default="")
    parser.add_argument("--gcs-dir", "--gcs_dir", dest="gcs_dir", type=str, default="")
    parser.add_argument("--no-gcs", "--no_gcs", dest="no_gcs", action="store_true")
    parser.add_argument("--wandb", type=str, default="auto", choices=["auto", "online", "disabled"])
    parser.add_argument("--use-dataparallel", "--use_dataparallel", dest="use_dataparallel", action="store_true", default=True)
    parser.add_argument("--no-dataparallel", "--no_dataparallel", dest="no_dataparallel", action="store_true")
    args, _ = parser.parse_known_args()
    return args


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = parse_args()

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    is_ddp = ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ) and torch.cuda.is_available()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1")) if is_ddp else 1
    rank = int(os.environ.get("RANK", "0")) if is_ddp else 0
    is_main_process = rank == 0

    if is_ddp:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

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
        if args.eval_batches == 16:
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
                    project="BEBLaDII-Phase3-Alt-PlanC",
                    name=f"phase3-alt-{args.arch_preset}-{env.device.type}-{int(time.time())}",
                    config=vars(args),
                )
                use_wandb = True
            except Exception as e:
                print(f"[WandB Warning] {e}", flush=True)

    # 1. Замороженная Система 1 (read-only) и опорные концепты
    tokenizer = get_tokenizer()
    encoder = get_encoder(weights_path=env.encoder_weights_path, device=env.device, eval_mode=True)
    decoder = get_decoder(weights_path=env.decoder_weights_path, encoder=encoder, device=env.device, eval_mode=True)
    for p in list(encoder.parameters()) + list(decoder.parameters()):
        p.requires_grad_(False)

    fast_vocab: Optional[FastVocabOptimizer] = None
    if not args.no_fast_vocab:
        fv_t0 = time.time()
        fv_dtype = env.amp_dtype if env.use_amp else torch.float32
        fast_vocab = FastVocabOptimizer.from_decoder(
            decoder=decoder,
            gemm_dtype=fv_dtype,
            svd_rank=args.vocab_svd_rank,
            ce_pool_hard=args.ce_pool_hard,
            ce_pool_rand=args.ce_pool_rand,
            ce_batch_fraction=args.ce_batch_fraction,
            attach=True,
        )
        if is_main_process:
            print(
                f"[FastVocab] Подключён за {time.time() - fv_t0:.2f}s "
                f"(dtype={fv_dtype}, svd_rank={fast_vocab.svd_rank}, "
                f"ce_pool={args.ce_max_tokens}+{fast_vocab.ce_pool_hard}+{fast_vocab.ce_pool_rand}, "
                f"ce_batch_frac={fast_vocab.ce_batch_fraction:.2f})",
                flush=True,
            )

    sep_concept = torch.load(env.sep_concept_path, map_location="cpu", weights_only=False)
    void_concept = torch.load(env.void_concept_path, map_location="cpu", weights_only=False)
    if os.path.exists(env.stopwords_path):
        sw_obj = torch.load(env.stopwords_path, map_location="cpu", weights_only=False)
        stopword_mask = (sw_obj["mask"] if isinstance(sw_obj, dict) else sw_obj).bool()
    else:
        stopword_mask = torch.zeros(131072, dtype=torch.bool)

    # 2. Создание гибридного Грассманианового ядра диффузии
    model = GrassmannianAltDiffusionCore(
        void_ref=void_concept,
        preset=args.arch_preset,
        full_ffn_dim=args.full_ffn_dim,
        grass_ffn_dim=args.grass_ffn_dim,
        num_sub_heads=args.num_sub_heads,
        local_radius=args.local_radius,
    ).to(env.device)

    ema = EMAModel(model, decay=args.ema_decay, device=env.device) if not args.no_ema else None
    opt_muon, opt_adamw = create_alt_optimizers(
        model, muon_lr=args.muon_lr, adamw_lr=args.adamw_lr, scale_1d_lr=args.scale_1d_lr
    )

    start_step = 0
    resume_path = None
    if args.resume_ckpt:
        if args.resume_ckpt == "auto":
            resume_path = find_latest_alt_checkpoint(env.checkpoint_dir)
        else:
            resume_path = args.resume_ckpt
    if resume_path and os.path.exists(resume_path):
        start_step = load_alt_checkpoint(resume_path, model, opt_muon, opt_adamw, env.device, ema=ema)

    # 3. Датасеты (Phase 6 parquet по умолчанию)
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

    sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if is_ddp
        else None
    )
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
    if env.device.type == "cuda" and torch.cuda.device_count() > 1:
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
        )
    finally:
        if is_ddp and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
