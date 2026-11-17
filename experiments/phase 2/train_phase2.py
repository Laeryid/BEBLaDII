# %% [markdown]
# # BEBLaDII Phase 2 (Plan C) — Генеративный Декодер-Распаковщик и Радар Шума
#
# **Архитектура:**
# - Замороженный `GrassmannianEncoder` Фазы 1 (`storage/components/encoder/`, `eval()`, `embeddings` зафиксированы).
# - Детерминированная $\mathcal{O}(8)$-канонизация базиса (`canonicalize_cell_basis`: `eigh(V V^T)` + сортировка $\lambda_1 \ge \dots \ge \lambda_8$ + SignFix 128D).
# - Словно-ориентированный эластичный чанкинг `WordAwareElasticChunker` (`1..4` BPE-токена на клетку без разрыва слов).
# - Двухвыходной `GrassmannianDecoder`:
#   - **Выход А (`NoiseRadarHead`):** поклеточная и потокенная оценка шума $(\hat{\sigma}_{\text{cell}}, \hat{\sigma}_{\text{tok}})$ с дистанцией до словаря.
#   - **Выход Б (Генератор текста):** `dec_context_layer` + `SetCrossAttentionUnpacker` + инъекция $\hat{\sigma}_{\text{tok}}$ + `4` сглаживающих/генерирующих слоя `ModernBERTLayer` -> `Tied LM Head (FP32)`.
# - Аугментация латента: `Gaussian + Ball Clamp`, `MixOther`, частичная кристаллизация и обязательные случайные повороты $R \in \mathcal{O}(8)$.

# %%
import argparse
import dataclasses
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

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# %% [markdown]
# ## 1. Инициализация окружения и путей

# %%
def setup_kaggle_secrets(is_kaggle: bool) -> None:
    """Извлечение секретов WANDB_API_KEY и GCP_SA_JSON через Kaggle Secrets API."""
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
                key_path = "/tmp/gcp_sa_key.json"
                with open(key_path, "w", encoding="utf-8") as f:
                    f.write(gcp_json.strip())
                res = subprocess.run(
                    ["gcloud", "auth", "activate-service-account", f"--key-file={key_path}"],
                    capture_output=True,
                    text=True,
                )
                if res.returncode == 0:
                    print("[Kaggle Secrets] Авторизация GCP активирована.", flush=True)
                else:
                    print(f"[Kaggle Secrets Warning] Ошибка gcloud auth: {res.stderr.strip()}", flush=True)
        except Exception:
            pass
    except ImportError:
        pass
    except Exception as e:
        print(f"[Kaggle Secrets Warning] Ошибка UserSecretsClient: {e}", flush=True)


def warmup_linalg_backends() -> None:
    """Последовательный прогрев cuSOLVER на всех GPU до запуска DataParallel."""
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        print(f"[Linalg Warmup] Прогрев cuSOLVER на {num_devices} GPU...", flush=True)
        for dev_idx in range(num_devices):
            dev = torch.device(f"cuda:{dev_idx}")
            dummy = torch.eye(32, device=dev, dtype=torch.float32)
            try:
                _ = torch.linalg.eigh(dummy)
            except Exception as e:
                print(f"[Linalg Warmup Warning] GPU {dev_idx}: {e}", flush=True)


def resolve_project_root() -> str:
    """Определяет корень репозитория BEBLaDII для локальной среды и Kaggle."""
    if os.path.exists("/kaggle") or ("KAGGLE_KERNEL_RUN_TYPE" in os.environ):
        kaggle_root = "/kaggle/working/BEBLaDII"
        if os.path.exists(kaggle_root):
            return kaggle_root
    return str(Path(__file__).resolve().parents[2])


def resolve_file_path(filename: str, candidates: List[str], search_roots: List[str]) -> str:
    """Находит путь к файлу среди кандидатов или рекурсивно в корневых директориях."""
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
    """Конфигурация путей, устройств и режимов обучения Фазы 2 (Decoder & Noise Radar)."""

    is_kaggle: bool
    project_root: str
    output_dir: str
    checkpoint_dir: str
    train_data_paths: List[str]
    val_data_paths: List[str]
    stopwords_path: str
    encoder_weights_path: str
    decoder_weights_path: str
    gcs_checkpoints_dir: str
    wandb_mode: str
    device: torch.device
    dtype: torch.dtype
    num_workers: int
    pin_memory: bool

    @classmethod
    def detect(cls, override_device: str = "auto") -> "EnvironmentConfig":
        is_kaggle = os.path.exists("/kaggle") or ("KAGGLE_KERNEL_RUN_TYPE" in os.environ)
        setup_kaggle_secrets(is_kaggle)
        project_root = resolve_project_root()
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # Отдельный бакет под Фазу 2 (Decoder & Radar), чтобы не пересекаться с диффузией Фазы 3
        gcs_checkpoints_dir = "gs://bebladii-weigths-us/planC/Phase2_Decoder/checkpoints/"

        if is_kaggle:
            output_dir = "/kaggle/working/output_phase2_decoder"
            checkpoint_dir = "/kaggle/working/checkpoints/phase2_decoder"
            search_roots = ["/kaggle/input", project_root]
            train_text = resolve_file_path(
                "train.parquet",
                ["/kaggle/input/bebladii-planb-phase3-data/phase 3/train_data/data/train.parquet"],
                search_roots,
            )
            train_qa = resolve_file_path(
                "train_phase2.parquet",
                ["/kaggle/input/datasets/bogdanbuliakov/bebladii-planc-phase2-data/data/train_phase2.parquet"],
                search_roots,
            )
            val_text = resolve_file_path(
                "val.parquet",
                ["/kaggle/input/bebladii-planb-phase3-data/phase 3/train_data/data/val.parquet"],
                search_roots,
            )
            val_qa = resolve_file_path(
                "val_phase2.parquet",
                ["/kaggle/input/datasets/bogdanbuliakov/bebladii-planc-phase2-data/data/val_phase2.parquet"],
                search_roots,
            )
            stopwords_path = resolve_file_path(
                "tekken_stopwords_mask.pt",
                [
                    "/kaggle/input/bebladii-planc-phase1-data/data/tekken_stopwords_mask.pt",
                    os.path.join(project_root, "data", "plan C", "tekken_stopwords_mask.pt"),
                ],
                search_roots,
            )
            encoder_weights_path = resolve_file_path(
                "encoder/weights.pt",
                [
                    os.path.join(project_root, "storage", "components", "encoder", "weights.pt"),
                    "/kaggle/input/bebladii-planc-phase1-data/encoder/weights.pt",
                ],
                search_roots,
            )
            decoder_weights_path = resolve_file_path(
                "decoder/weights.pt",
                [
                    os.path.join(project_root, "storage", "components", "decoder", "weights.pt"),
                    "/kaggle/input/bebladii-planc-phase1-data/decoder/weights.pt",
                ],
                search_roots,
            )
            wandb_mode = "online" if "WANDB_API_KEY" in os.environ else "disabled"
            num_workers = 2
            pin_memory = True
        else:
            output_dir = os.path.join(project_root, "experiments", "phase 2", "output")
            checkpoint_dir = os.path.join(project_root, "experiments", "phase 2", "checkpoints")
            train_text = os.path.join(
                project_root, "BEBLaDII-planB-Phase3-Data", "phase 3", "train_data", "data", "train.parquet"
            )
            train_qa = os.path.join(project_root, "BEBLaDII-plan\u0421-Phase2-Data", "data", "train_phase2.parquet")
            val_text = os.path.join(
                project_root, "BEBLaDII-planB-Phase3-Data", "phase 3", "train_data", "data", "val.parquet"
            )
            val_qa = os.path.join(project_root, "BEBLaDII-plan\u0421-Phase2-Data", "data", "val_phase2.parquet")
            stopwords_path = resolve_file_path(
                "tekken_stopwords_mask.pt",
                [os.path.join(project_root, "data", "plan C", "tekken_stopwords_mask.pt")],
                [project_root],
            )
            encoder_weights_path = os.path.join(project_root, "storage", "components", "encoder", "weights.pt")
            decoder_weights_path = os.path.join(project_root, "storage", "components", "decoder", "weights.pt")
            wandb_mode = "disabled"
            num_workers = 0
            pin_memory = False

        train_data_paths = [p for p in (train_text, train_qa) if p and os.path.exists(p)]
        val_data_paths = [p for p in (val_text, val_qa) if p and os.path.exists(p)]

        if override_device and override_device != "auto":
            device = torch.device(override_device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            warmup_linalg_backends()
        else:
            dtype = torch.float32

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        return cls(
            is_kaggle=is_kaggle,
            project_root=project_root,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            train_data_paths=train_data_paths,
            val_data_paths=val_data_paths,
            stopwords_path=stopwords_path,
            encoder_weights_path=encoder_weights_path,
            decoder_weights_path=decoder_weights_path,
            gcs_checkpoints_dir=gcs_checkpoints_dir,
            wandb_mode=wandb_mode,
            device=device,
            dtype=dtype,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )


# %% [markdown]
# ## 2. Импорт общих компонент Plan C

# %%
_CURR_ROOT = resolve_project_root()
if _CURR_ROOT not in sys.path:
    sys.path.insert(0, _CURR_ROOT)

from storage.components.plan_c_layers import (
    WordAwareElasticChunker,
    canonicalize_cell_basis,
    clamp_to_unit_ball,
)
from storage.components.encoder import GrassmannianEncoder, get_encoder
from storage.components.decoder import GrassmannianDecoder, get_decoder
from storage.components.tokenizer import get_tokenizer


# %% [markdown]
# ## 3. Датасет с умным эластичным чанкингом (`1..4` токена без разрыва слов)

# %%
class Phase2DecoderDataset(Dataset):
    """
    Датасет Фазы 2 (Generator-Decoder & Noise Radar):
    - Читает связные тексты ('text') и Q/A-ответы ('A' / 'Q').
    - Нарезает последовательности через WordAwareElasticChunker (1..4 токена без разрыва слов).
    - В 10% обучающих шагов добавляет словарный спектр (Regime B) для удержания редких токенов.
    """

    def __init__(
        self,
        parquet_paths: List[str],
        chunker: WordAwareElasticChunker,
        tokenizer: Any = None,
        num_cells: int = 64,
        vocab_size: int = 131072,
        elastic_drop_prob: float = 0.20,
        regime_b_ratio: float = 0.10,
        pad_token_id: int = 10,
        virtual_length: int = 50000,
        is_validation: bool = False,
    ):
        self.chunker = chunker
        self.tokenizer = tokenizer
        self.num_cells = num_cells
        self.chunk_size = chunker.chunk_size
        self.seq_len = num_cells * self.chunk_size
        self.vocab_size = vocab_size
        self.elastic_drop_prob = 0.0 if is_validation else elastic_drop_prob
        self.regime_b_ratio = 0.0 if is_validation else regime_b_ratio
        self.pad_token_id = pad_token_id
        self.virtual_length = virtual_length
        self.is_validation = is_validation

        self.texts: List[str] = []
        for path in parquet_paths:
            if not path or not os.path.exists(path):
                continue
            try:
                df = pd.read_parquet(path)
                added = 0
                if "text" in df.columns:
                    col_vals = [str(x).strip() for x in df["text"].dropna().tolist() if str(x).strip()]
                    self.texts.extend(col_vals)
                    added += len(col_vals)
                elif "A" in df.columns:
                    col_vals = [str(x).strip() for x in df["A"].dropna().tolist() if str(x).strip()]
                    self.texts.extend(col_vals)
                    added += len(col_vals)
                prefix = "[Val Dataset]" if is_validation else "[Dataset]"
                print(f"{prefix} Загружено {added} текстов из {os.path.basename(path)}", flush=True)
            except Exception as e:
                print(f"[Dataset Warning] Ошибка чтения {path}: {e}", flush=True)

        self.has_real_data = len(self.texts) > 0
        self.templates = [[1, 2, 3, 4], [10, 11, 12, 13], [20, 21, 22, 23]]

    def __len__(self) -> int:
        return len(self.texts) if self.has_real_data else self.virtual_length

    def _sample_regime_b(self) -> Dict[str, torch.Tensor]:
        input_ids = torch.zeros(self.seq_len, dtype=torch.long)
        weights = torch.zeros(self.seq_len, dtype=torch.float32)
        for c in range(self.num_cells):
            start = c * self.chunk_size
            d_pos = int(torch.randint(0, self.chunk_size, (1,)).item())
            d_tok = int(torch.randint(min(1000, self.vocab_size - 1), self.vocab_size, (1,)).item())
            tmpl = self.templates[int(torch.randint(0, len(self.templates), (1,)).item())]
            input_ids[start + d_pos] = d_tok
            weights[start + d_pos] = 0.50
            k = 0
            for m in range(self.chunk_size):
                if m != d_pos:
                    input_ids[start + m] = tmpl[k] % self.vocab_size
                    weights[start + m] = 0.50 / max(1, self.chunk_size - 1)
                    k += 1
        return {
            "input_ids": input_ids,
            "targets": input_ids.clone(),
            "weights": weights,
            "content_cell_mask": torch.ones(self.num_cells, dtype=torch.bool),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if torch.rand(1).item() < self.regime_b_ratio:
            return self._sample_regime_b()

        if self.has_real_data and self.tokenizer is not None:
            raw_text = self.texts[idx % len(self.texts)]
            token_ids = self.tokenizer.encode(raw_text, add_special_tokens=False)
            # Случайный сдвиг старта на длинных текстах при обучении (разнообразие фаз и слов)
            if not self.is_validation and len(token_ids) > self.seq_len:
                max_offset = min(len(token_ids) - self.seq_len // 2, 256)
                if max_offset > 0:
                    offset = int(torch.randint(0, max_offset + 1, (1,)).item())
                    token_ids = token_ids[offset:]
            token_ids = token_ids[: self.seq_len]
        else:
            max_tok = min(self.vocab_size, 2000)
            min_tok = min(1000 if max_tok > 1020 else 20, max_tok - 1)
            eff_len = int(torch.randint(max(8, self.seq_len // 2), self.seq_len + 1, (1,)).item())
            token_ids = torch.randint(min_tok, max_tok, (eff_len,)).tolist()

        packed = self.chunker.chunk_tokens(
            token_ids=token_ids,
            max_cells=self.num_cells,
            elastic_drop_prob=self.elastic_drop_prob,
        )
        return {
            "input_ids": packed["input_ids"],
            "targets": packed["targets"],
            "weights": packed["weights"],
            "content_cell_mask": packed["content_cell_mask"],
        }


# %% [markdown]
# ## 4. Геометрия Грассманиана, генератор искажений латента и функции потерь

# %%
def grassmann_cell_similarity(V_a: torch.Tensor, V_b: torch.Tensor) -> torch.Tensor:
    """
    O(8)-инвариантное сходство операторов плотности rho = V^T V (Теорема 7):
    Sim_Gr(V_a, V_b) = Tr(rho_a rho_b) / sqrt(Tr(rho_a^2) Tr(rho_b^2)) in [0, 1].
    Возвращает (B, N).
    """
    a_f = V_a.float()
    b_f = V_b.float()
    p_ab = torch.matmul(a_f, b_f.transpose(-2, -1))
    p_aa = torch.matmul(a_f, a_f.transpose(-2, -1))
    p_bb = torch.matmul(b_f, b_f.transpose(-2, -1))
    tr_ab = (p_ab ** 2).sum(dim=(-2, -1))
    tr_aa = (p_aa ** 2).sum(dim=(-2, -1))
    tr_bb = (p_bb ** 2).sum(dim=(-2, -1))
    return (tr_ab / torch.sqrt(tr_aa * tr_bb).clamp(min=1e-8)).clamp(0.0, 1.0)


def random_orthogonal_matrices(batch_size: int, num_cells: int, k: int, device: torch.device) -> torch.Tensor:
    """Генерирует случайные ортогональные матрицы R in O(k) формы (B, N, k, k) через QR."""
    raw = torch.randn(batch_size, num_cells, k, k, device=device, dtype=torch.float32)
    q, r = torch.linalg.qr(raw)
    diag_sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    diag_sign = torch.where(diag_sign == 0.0, torch.ones_like(diag_sign), diag_sign)
    return q * diag_sign.unsqueeze(-2)


@torch.no_grad()
def corrupt_latent_cells(
    V_0: torch.Tensor,
    content_cell_mask: torch.Tensor,
    mode_override: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """
    Реалистичная симуляция искажений диффузии для обучения Декодера-Генератора и Радара Шума:
    - clean_jitter (20%): микро-шум sigma in [0.0, 0.015]
    - gaussian (30%): сильный гауссов шум + clamp_to_unit_ball (Sim_Gr ~ 0.35..0.85)
    - mix_other (25%): барицентрическая смесь (1-alpha)*V_0 + alpha*V_other + clamp_to_unit_ball
    - crystal_mask (25%): 30-70% чистых опорных кристаллов на холсте + зашумлённые соседи
    Всегда применяет случайный ортогональный поворот R in O(8) для проверки и гарантии инвариантности.
    """
    B, N, K, D = V_0.shape
    device = V_0.device
    V_f = V_0.float()

    if mode_override is None:
        r = torch.rand(1).item()
        if r < 0.20:
            mode = "clean_jitter"
        elif r < 0.50:
            mode = "gaussian"
        elif r < 0.75:
            mode = "mix_other"
        else:
            mode = "crystal_mask"
    else:
        mode = mode_override

    if mode == "clean":
        V_tilde = V_f.clone()
    elif mode == "clean_jitter":
        sigma = torch.rand(B, N, 1, 1, device=device) * 0.015
        V_tilde = clamp_to_unit_ball(V_f + torch.randn_like(V_f) * sigma)
    elif mode == "gaussian":
        # Масштаб 0.01..0.065 даёт Sim_Gr от ~0.95 до ~0.35 (как в диагностике Probe 3)
        sigma = 0.010 + torch.rand(B, N, 1, 1, device=device) * 0.055
        V_tilde = clamp_to_unit_ball(V_f + torch.randn_like(V_f) * sigma)
    elif mode == "mix_other":
        V_other = torch.roll(V_f, shifts=1, dims=0) if B > 1 else torch.roll(V_f, shifts=max(1, N // 3), dims=1)
        if B > 1:
            V_other = torch.roll(V_other, shifts=max(1, N // 5), dims=1)
        alpha = 0.15 + torch.rand(B, N, 1, 1, device=device) * 0.40
        V_tilde = clamp_to_unit_ball((1.0 - alpha) * V_f + alpha * V_other)
    elif mode == "crystal_mask":
        keep_prob = 0.30 + torch.rand(B, 1, 1, 1, device=device) * 0.40
        is_crystal = (torch.rand(B, N, 1, 1, device=device) < keep_prob)
        sigma = 0.015 + torch.rand(B, N, 1, 1, device=device) * 0.050
        noise_v = clamp_to_unit_ball(V_f + torch.randn_like(V_f) * sigma)
        V_tilde = torch.where(is_crystal, V_f, noise_v)
    elif mode == "random_o8":
        V_tilde = V_f.clone()
    else:
        raise ValueError(f"Unknown corruption mode: {mode}")

    # Вычисляем истинное O(8)-инвариантное искажение до поворота
    sim_gr = grassmann_cell_similarity(V_tilde, V_f)
    true_sigma_cell = (1.0 - sim_gr).clamp(0.0, 1.0)

    # Применяем случайное вращение R in O(8) (кроме чистого контрольного режима 'clean')
    if mode != "clean":
        R = random_orthogonal_matrices(B, N, K, device=device)
        V_tilde = torch.matmul(R, V_tilde)

    return V_tilde.to(V_0.dtype), true_sigma_cell, mode


def compute_phase2_decoder_loss(
    logits: torch.Tensor,
    radar: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    true_sigma_cell: torch.Tensor,
    pad_token_id: int = 10,
    slot_pad_weight: float = 0.25,
    radar_cell_weight: float = 0.50,
    radar_tok_weight: float = 0.30,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Гибридная функция потерь Фазы 2 (строго в float32 снаружи DataParallel, ADR 135):
    L = L_ce_words + 0.25 * L_ce_slot_pad + 0.50 * L_radar_cell + 0.30 * L_radar_tok
    """
    B, L, V_size = logits.shape
    N = true_sigma_cell.shape[1]
    chunk_size = L // N

    logits_f32 = logits.float()
    targets_flat = targets.view(-1)
    ce_per_tok = F.cross_entropy(logits_f32.view(-1, V_size), targets_flat, ignore_index=-100, reduction="none")

    valid_mask = targets_flat != -100
    is_slot_pad = (targets_flat == pad_token_id) & valid_mask
    is_word = valid_mask & (~is_slot_pad)

    loss_ce_words = ce_per_tok[is_word].mean() if is_word.any() else logits_f32.sum() * 0.0
    loss_ce_pad = ce_per_tok[is_slot_pad].mean() if is_slot_pad.any() else logits_f32.sum() * 0.0

    # Поклеточный радар шума
    sigma_cell_hat = radar["sigma_cell_hat"].float()
    true_cell_f32 = true_sigma_cell.float()
    loss_radar_cell = F.smooth_l1_loss(sigma_cell_hat, true_cell_f32, beta=0.05)

    # Потокенный радар неуверенности: комбинация клеточного шума и фактической ошибки декодирования
    with torch.no_grad():
        pred_ids = torch.argmax(logits_f32, dim=-1)
        wrong_tok = (pred_ids != targets).float()
        cell_exp = true_cell_f32.unsqueeze(-1).expand(B, N, chunk_size).reshape(B, L)
        target_sigma_tok = (0.5 * cell_exp + 0.5 * wrong_tok).clamp(0.0, 1.0)

    tok_logits = radar["sigma_tok_logits"].float().view(-1)
    tok_tgt_flat = target_sigma_tok.view(-1)
    if valid_mask.any():
        loss_radar_tok = F.binary_cross_entropy_with_logits(tok_logits[valid_mask], tok_tgt_flat[valid_mask])
    else:
        loss_radar_tok = logits_f32.sum() * 0.0

    total_loss = (
        loss_ce_words
        + slot_pad_weight * loss_ce_pad
        + radar_cell_weight * loss_radar_cell
        + radar_tok_weight * loss_radar_tok
    )

    with torch.no_grad():
        word_acc = (
            (pred_ids.view(-1)[is_word] == targets_flat[is_word]).float().mean().item()
            if is_word.any()
            else 0.0
        )
        slot_pad_acc = (
            (pred_ids.view(-1)[is_slot_pad] == pad_token_id).float().mean().item()
            if is_slot_pad.any()
            else 1.0
        )
        radar_mae = torch.abs(sigma_cell_hat - true_cell_f32).mean().item()

    metrics = {
        "loss/total": float(total_loss.item()),
        "loss/ce_words": float(loss_ce_words.item()),
        "loss/ce_slot_pad": float(loss_ce_pad.item()),
        "loss/radar_cell": float(loss_radar_cell.item()),
        "loss/radar_tok": float(loss_radar_tok.item()),
        "acc/words": word_acc,
        "acc/slot_pad": slot_pad_acc,
        "radar/mae_cell": radar_mae,
    }
    return total_loss, metrics


# %% [markdown]
# ## 5. Оптимизатор OrthMuon (d=1) + AdamW и EMA с проекцией на сферу (ADR 132)

# %%
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    orig_dtype = G.dtype
    X = G.bfloat16() if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else G.float()
    M, N = X.shape
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = X / (X.norm() + eps)
    if M > N:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    if M > N:
        X = X.T
    return X.to(orig_dtype)


class OrthMuon(torch.optim.Optimizer):
    """Сферический OrthMuon (d=1, ADR 131) + плотный Muon для линейных слоёв декодера."""

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
            lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay, sphere=sphere
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
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
                    q_next = F.normalize(q_f32 - (lr * scale) * u_tan, p=2, dim=-1, eps=1e-8)
                    p.data.copy_(q_next.to(p.dtype))
                else:
                    if wd > 0:
                        p.data.mul_(1.0 - lr * wd)
                    p.data.add_(update_ortho, alpha=-lr * scale)


class EMAModel:
    """EMA весов модели с обязательной ренормализацией *.q_weight на единичную сферу (ADR 132) и PACE pullback."""

    def __init__(self, model: nn.Module, decay: float = 0.999, device: Optional[torch.device] = None):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        raw = getattr(model, "module", model)
        for name, param in raw.named_parameters():
            if param.requires_grad:
                t = param.detach().clone()
                if device is not None:
                    t = t.to(device)
                self.shadow[name] = t

    @torch.no_grad()
    def update(self, model: nn.Module, pullback_muon: float = 0.005, pullback_adamw: float = 0.002) -> None:
        raw = getattr(model, "module", model)
        for name, param in raw.named_parameters():
            if name not in self.shadow:
                continue
            p_f = param.detach().float()
            s_f = self.shadow[name].float()
            new_s = self.decay * s_f + (1.0 - self.decay) * p_f
            if name.endswith(".q_weight"):
                new_s = F.normalize(new_s, p=2, dim=-1, eps=1e-8)
                self.shadow[name].copy_(new_s.to(self.shadow[name].dtype))
                if pullback_muon > 0.0:
                    p_pulled = F.normalize((1.0 - pullback_muon) * p_f + pullback_muon * new_s, p=2, dim=-1, eps=1e-8)
                    param.data.copy_(p_pulled.to(param.dtype))
            else:
                self.shadow[name].copy_(new_s.to(self.shadow[name].dtype))
                if pullback_adamw > 0.0:
                    param.data.copy_(((1.0 - pullback_adamw) * p_f + pullback_adamw * new_s).to(param.dtype))

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module) -> None:
        raw = getattr(model, "module", model)
        self.backup = {}
        for name, param in raw.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.detach().clone()
                s = self.shadow[name]
                if name.endswith(".q_weight"):
                    s = F.normalize(s.float(), p=2, dim=-1, eps=1e-8).to(param.dtype)
                param.data.copy_(s.to(param.device))

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        raw = getattr(model, "module", model)
        for name, param in raw.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "shadow": {k: v.cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, state: Dict[str, Any], device: Optional[torch.device] = None) -> None:
        shadow = state.get("shadow", state) if isinstance(state, dict) else {}
        for k, v in shadow.items():
            if k in self.shadow:
                self.shadow[k] = v.to(device or self.shadow[k].device)

    def to(self, device: torch.device) -> None:
        for k in self.shadow:
            self.shadow[k] = self.shadow[k].to(device)


def create_decoder_optimizers(
    decoder: nn.Module,
    muon_lr: float = 1e-3,
    adamw_lr: float = 2e-4,
    weight_decay: float = 0.01,
    row_scale_lr_mult: float = 4.0,
) -> Tuple[Optional[OrthMuon], torch.optim.AdamW]:
    """Разделяет параметры декодера по схеме ADR 131/132."""
    raw = getattr(decoder, "module", decoder)
    sphere_params = []
    muon_dense_params = []
    row_scale_params = []
    adamw_no_decay = []
    adamw_decay = []

    for name, param in raw.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".q_weight"):
            sphere_params.append(param)
        elif name.endswith("out_proj.weight") and param.ndim == 2:
            muon_dense_params.append(param)
        elif name.endswith(".row_scale"):
            row_scale_params.append(param)
        elif "norm" in name or name.endswith(".bias") or param.ndim < 2:
            adamw_no_decay.append(param)
        else:
            adamw_decay.append(param)

    muon_groups = []
    if sphere_params:
        muon_groups.append({"params": sphere_params, "lr": muon_lr, "weight_decay": 0.0, "sphere": True})
    if muon_dense_params:
        muon_groups.append({"params": muon_dense_params, "lr": muon_lr, "weight_decay": weight_decay, "sphere": False})
    opt_muon = OrthMuon(muon_groups, lr=muon_lr) if muon_groups else None

    adamw_groups = []
    if adamw_decay:
        adamw_groups.append({"params": adamw_decay, "lr": adamw_lr, "weight_decay": weight_decay, "lr_mult": 1.0})
    if adamw_no_decay:
        adamw_groups.append({"params": adamw_no_decay, "lr": adamw_lr, "weight_decay": 0.0, "lr_mult": 1.0})
    if row_scale_params:
        adamw_groups.append(
            {
                "params": row_scale_params,
                "lr": adamw_lr * row_scale_lr_mult,
                "weight_decay": 0.0,
                "lr_mult": row_scale_lr_mult,
            }
        )
    opt_adamw = torch.optim.AdamW(adamw_groups, lr=adamw_lr)
    return opt_muon, opt_adamw


# %% [markdown]
# ## 6. Валидация по корзинам шума и сохранение/экспорт чекпоинтов

# %%
@torch.no_grad()
def evaluate_phase2_decoder(
    encoder: GrassmannianEncoder,
    decoder: nn.Module,
    val_dataloader: DataLoader,
    env: EnvironmentConfig,
    tokenizer: Any,
    eval_batches: int = 8,
    ema: Optional[EMAModel] = None,
    pad_token_id: int = 10,
) -> Dict[str, float]:
    """Оценка декодера и радара на фиксированных режимах: clean, random_o8, gaussian, mix_other."""
    if ema is not None:
        ema.apply_shadow(decoder)
    raw_dec = getattr(decoder, "module", decoder)
    raw_dec.eval()
    encoder.eval()

    modes = ["clean", "random_o8", "gaussian", "mix_other"]
    sums: Dict[str, float] = {f"{m}_{k}": 0.0 for m in modes for k in ("acc", "ce", "mae", "sim")}
    count = 0
    sample_printed = False

    use_amp = env.device.type == "cuda"
    for i, batch in enumerate(val_dataloader):
        if i >= eval_batches:
            break
        input_ids = batch["input_ids"].to(env.device)
        targets = batch["targets"].to(env.device)
        weights = batch["weights"].to(env.device)
        content_mask = batch["content_cell_mask"].to(env.device)

        with torch.autocast(device_type=env.device.type, dtype=env.dtype, enabled=use_amp):
            V_0 = encoder(input_ids, weights, canonicalize=False)

        for mode in modes:
            V_in, true_sigma, _ = corrupt_latent_cells(V_0, content_mask, mode_override=mode)
            with torch.autocast(device_type=env.device.type, dtype=env.dtype, enabled=use_amp):
                logits, radar = decoder(V_in, return_radar=True)
            _, metrics = compute_phase2_decoder_loss(
                logits=logits,
                radar=radar,
                targets=targets,
                true_sigma_cell=true_sigma,
                pad_token_id=pad_token_id,
            )
            sums[f"{mode}_acc"] += metrics["acc/words"]
            sums[f"{mode}_ce"] += metrics["loss/ce_words"]
            sums[f"{mode}_mae"] += metrics["radar/mae_cell"]
            sums[f"{mode}_sim"] += float((1.0 - true_sigma).mean().item())

            if not sample_printed and mode == "mix_other" and tokenizer is not None:
                pred_row = torch.argmax(logits[0], dim=-1)
                tgt_row = targets[0][targets[0] != -100]
                ref_txt = WordAwareElasticChunker.detokenize_non_pad(tgt_row, tokenizer, pad_token_id)[:180]
                gen_txt = WordAwareElasticChunker.detokenize_non_pad(pred_row, tokenizer, pad_token_id)[:180]
                print(
                    f"  [Sample MixOther | Sim={1.0 - true_sigma[0].mean().item():.3f} | "
                    f"Radar={radar['sigma_cell_hat'][0].mean().item():.3f}]\n"
                    f"    REF: {ref_txt!r}\n    GEN: {gen_txt!r}",
                    flush=True,
                )
        sample_printed = True
        count += 1

    if ema is not None:
        ema.restore(decoder)
    raw_dec.train()

    denom = max(1, count)
    out = {f"val/{k}": v / denom for k, v in sums.items()}
    out["val/loss"] = out["val/mix_other_ce"] + out["val/gaussian_ce"]
    return out


def sync_to_gcs(local_path: str, gcs_dir: str) -> bool:
    if not os.path.exists(local_path) or not gcs_dir:
        return False
    target = gcs_dir.rstrip("/") + "/" + os.path.basename(local_path)
    for cmd in (["gsutil", "cp", local_path, target], ["gcloud", "storage", "cp", local_path, target]):
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                print(f"[GCS] Выгружено: {target}", flush=True)
                return True
        except Exception:
            pass
    print(f"[GCS Warning] Не удалось выгрузить {local_path} в {target}", flush=True)
    return False


def save_checkpoint(
    decoder: nn.Module,
    opt_muon: Optional[OrthMuon],
    opt_adamw: torch.optim.AdamW,
    step: int,
    metrics: Dict[str, float],
    path: str,
    metrics_history: List[Dict[str, Any]],
    ema: Optional[EMAModel] = None,
    sync_gcs: bool = False,
    gcs_dir: str = "",
    keep_local: bool = False,
) -> None:
    raw = getattr(decoder, "module", decoder)
    # Исключаем не-параметрические буферы tied lm_head (ADR 134)
    sd = {k: v.cpu() for k, v in raw.state_dict().items() if k not in ("lm_head_weight", "lm_head_inv_norms")}
    ckpt = {
        "step": step,
        "decoder_state_dict": sd,
        "opt_adamw_state_dict": opt_adamw.state_dict(),
        "opt_muon_state_dict": opt_muon.state_dict() if opt_muon is not None else None,
        "metrics": metrics,
        "metrics_history": metrics_history,
        "ema_state_dict": ema.state_dict() if ema is not None else None,
        "config": {
            "vocab_size": raw.vocab_size,
            "hidden_size": raw.hidden_size,
            "intermediate_size": raw.intermediate_size,
            "num_axes": raw.num_axes,
            "axis_dim": raw.axis_dim,
            "chunk_size": raw.chunk_size,
            "num_smooth_layers": raw.num_smooth_layers,
            "canonicalize_basis": raw.canonicalize_basis,
            "enable_noise_radar": raw.enable_noise_radar,
        },
    }
    torch.save(ckpt, path)
    print(f"[Checkpoint] Шаг {step} сохранен: {path}", flush=True)
    if sync_gcs and gcs_dir:
        ok = sync_to_gcs(path, gcs_dir)
        if ok and not keep_local:
            try:
                os.remove(path)
            except Exception:
                pass


def find_latest_checkpoint(checkpoint_dir: str, gcs_dir: Optional[str] = None) -> Optional[str]:
    local_files = glob.glob(os.path.join(checkpoint_dir, "phase2_decoder_step_*.pt"))
    if local_files:
        local_files.sort(key=lambda f: int(re.findall(r"\d+", os.path.basename(f))[-1]))
        return local_files[-1]
    if gcs_dir:
        try:
            res = subprocess.run(
                ["gsutil", "ls", gcs_dir.rstrip("/") + "/phase2_decoder_step_*.pt"],
                capture_output=True,
                text=True,
            )
            if res.returncode == 0 and res.stdout.strip():
                lines = [x.strip() for x in res.stdout.strip().split("\n") if x.strip()]
                lines.sort(key=lambda f: int(re.findall(r"\d+", os.path.basename(f))[-1]))
                latest = lines[-1]
                dest = os.path.join(checkpoint_dir, os.path.basename(latest))
                dl = subprocess.run(["gsutil", "cp", latest, dest], capture_output=True, text=True)
                if dl.returncode == 0 and os.path.exists(dest):
                    return dest
        except Exception:
            pass
    return None


def get_cosine_lr(initial_lr: float, min_lr: float, step: int, max_steps: int, warmup_steps: int = 0) -> float:
    if step >= max_steps:
        return min_lr
    if warmup_steps > 0 and step < warmup_steps:
        return initial_lr * (0.05 + 0.95 * (step / float(warmup_steps)))
    decay_steps = max(1, max_steps - max(0, warmup_steps))
    progress = min(1.0, max(0, step - max(0, warmup_steps)) / float(decay_steps))
    return min_lr + 0.5 * (initial_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# %% [markdown]
# ## 7. Основной цикл обучения

# %%
def train(
    encoder: GrassmannianEncoder,
    decoder: nn.Module,
    dataloader: DataLoader,
    env: EnvironmentConfig,
    tokenizer: Any,
    opt_muon: Optional[OrthMuon],
    opt_adamw: torch.optim.AdamW,
    start_step: int = 0,
    metrics_history: Optional[List[Dict[str, Any]]] = None,
    max_steps: int = 30000,
    muon_lr: float = 1e-3,
    adamw_lr: float = 2e-4,
    muon_min_lr: float = 1e-4,
    adamw_min_lr: float = 2e-5,
    lr_warmup_steps: int = 500,
    pullback_muon: float = 0.005,
    pullback_adamw: float = 0.002,
    pad_token_id: int = 10,
    log_every: int = 20,
    save_every: int = 500,
    eval_every: int = 200,
    eval_batches: int = 8,
    val_dataloader: Optional[DataLoader] = None,
    ema: Optional[EMAModel] = None,
    sync_gcs: bool = False,
    use_wandb: bool = False,
) -> None:
    encoder.to(env.device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    decoder.to(env.device)
    if ema is not None:
        ema.to(env.device)

    raw_dec = getattr(decoder, "module", decoder)
    raw_dec.train()

    if metrics_history is None:
        metrics_history = []

    best_val_loss = min(
        [e["val/loss"] for e in metrics_history if "val/loss" in e],
        default=float("inf"),
    )

    use_amp = env.device.type == "cuda"
    use_scaler = use_amp and env.dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    print(
        f"=== Обучение Phase 2 (Generator-Decoder & Noise Radar) | device={env.device} | dtype={env.dtype} ===",
        flush=True,
    )
    print(
        f"    Обучаемых параметров: OrthMuon={sum(p.numel() for g in opt_muon.param_groups for p in g['params']) if opt_muon else 0:,} | "
        f"AdamW={sum(p.numel() for g in opt_adamw.param_groups for p in g['params']):,}",
        flush=True,
    )

    step = start_step
    data_iter = iter(dataloader)
    last_t = time.time()
    last_step = start_step

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(env.device)
        targets = batch["targets"].to(env.device)
        weights = batch["weights"].to(env.device)
        content_mask = batch["content_cell_mask"].to(env.device)

        if opt_muon is not None:
            opt_muon.zero_grad()
        opt_adamw.zero_grad()

        with torch.no_grad():
            with torch.autocast(device_type=env.device.type, dtype=env.dtype, enabled=use_amp):
                V_0 = encoder(input_ids, weights, canonicalize=False)

        V_in, true_sigma_cell, _ = corrupt_latent_cells(V_0, content_mask)

        with torch.autocast(device_type=env.device.type, dtype=env.dtype, enabled=use_amp):
            logits, radar = decoder(V_in, return_radar=True)

        # Вычисление функции потерь строго в float32 снаружи DataParallel (ADR 135)
        loss, metrics = compute_phase2_decoder_loss(
            logits=logits,
            radar=radar,
            targets=targets,
            true_sigma_cell=true_sigma_cell,
            pad_token_id=pad_token_id,
        )

        if not torch.isfinite(loss):
            print(f"[Warning Step {step+1}] Нечисловой loss={loss.item()}. Пропуск шага.", flush=True)
            continue

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt_adamw)
            grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)

        grad_norm_val = float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
        if not math.isfinite(grad_norm_val):
            print(f"[Warning Step {step+1}] Нечисловой GradNorm={grad_norm_val}. Пропуск.", flush=True)
            if use_scaler:
                scaler.update()
            continue

        curr_muon_lr = get_cosine_lr(muon_lr, muon_min_lr, step, max_steps, lr_warmup_steps)
        curr_adamw_lr = get_cosine_lr(adamw_lr, adamw_min_lr, step, max_steps, lr_warmup_steps)
        if opt_muon is not None:
            for g in opt_muon.param_groups:
                g["lr"] = curr_muon_lr
            opt_muon.step()
        for g in opt_adamw.param_groups:
            g["lr"] = curr_adamw_lr * g.get("lr_mult", 1.0)

        if use_scaler:
            scaler.step(opt_adamw)
            scaler.update()
        else:
            opt_adamw.step()

        if ema is not None:
            ema.update(decoder, pullback_muon=pullback_muon, pullback_adamw=pullback_adamw)

        step += 1

        if step % log_every == 0 or step == 1 or step == max_steps:
            now = time.time()
            sps = (step - last_step) / max(1e-5, now - last_t)
            last_t, last_step = now, step
            print(
                f"Step {step:05d}/{max_steps:05d} | Loss: {metrics['loss/total']:.4f} | "
                f"CE_w: {metrics['loss/ce_words']:.4f} | CE_p: {metrics['loss/ce_slot_pad']:.4f} | "
                f"AccW: {metrics['acc/words']:.3f} | AccP: {metrics['acc/slot_pad']:.3f} | "
                f"RadarMAE: {metrics['radar/mae_cell']:.3f} | GN: {grad_norm_val:.2f} | {sps:.2f} step/s",
                flush=True,
            )
            metrics_history.append({"step": step, "type": "train", **metrics})
            if use_wandb:
                import wandb

                wandb.log({"step": step, **metrics}, step=step)

        if val_dataloader is not None and (step % eval_every == 0 or step == max_steps):
            val_m = evaluate_phase2_decoder(
                encoder=encoder,
                decoder=decoder,
                val_dataloader=val_dataloader,
                env=env,
                tokenizer=tokenizer,
                eval_batches=eval_batches,
                ema=ema,
                pad_token_id=pad_token_id,
            )
            print(
                f"\n{'='*88}\n"
                f"  [Val Step {step:05d}] Clean={val_m['val/clean_acc']:.3f} | "
                f"RandO8={val_m['val/random_o8_acc']:.3f} | Gauss={val_m['val/gaussian_acc']:.3f} | "
                f"MixOther={val_m['val/mix_other_acc']:.3f} | RadarMAE={val_m['val/mix_other_mae']:.3f}\n"
                f"{'='*88}\n",
                flush=True,
            )
            metrics_history.append({"step": step, "type": "val", **val_m})
            if use_wandb:
                import wandb

                wandb.log(val_m, step=step)

            if val_m["val/loss"] < best_val_loss:
                best_val_loss = val_m["val/loss"]
                best_path = os.path.join(env.checkpoint_dir, "phase2_decoder_best.pt")
                save_checkpoint(
                    decoder,
                    opt_muon,
                    opt_adamw,
                    step,
                    val_m,
                    best_path,
                    metrics_history,
                    ema=ema,
                    sync_gcs=sync_gcs,
                    gcs_dir=env.gcs_checkpoints_dir,
                    keep_local=True,
                )

        if step % save_every == 0 or step == max_steps:
            ckpt_path = os.path.join(env.checkpoint_dir, f"phase2_decoder_step_{step}.pt")
            save_checkpoint(
                decoder,
                opt_muon,
                opt_adamw,
                step,
                metrics,
                ckpt_path,
                metrics_history,
                ema=ema,
                sync_gcs=sync_gcs,
                gcs_dir=env.gcs_checkpoints_dir,
            )

    print("=== Обучение Phase 2 (Decoder & Noise Radar) завершено ===", flush=True)


# %% [markdown]
# ## 8. Точка входа CLI

# %%
def parse_args():
    parser = argparse.ArgumentParser(description="BEBLaDII Phase 2 Plan C (Generator-Decoder & Noise Radar)")
    parser.add_argument("--smoke_test", action="store_true", help="Быстрый CPU sanity-тест")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_cells", type=int, default=64, help="Число концепт-клеток холста (64 * 4 = 256 слотов)")
    parser.add_argument("--num_smooth_layers", type=int, default=4, help="Число сглаживающих слоев генератора")
    parser.add_argument("--elastic_drop_prob", type=float, default=0.20)
    parser.add_argument("--regime_b_ratio", type=float, default=0.10)
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--muon_lr", type=float, default=1e-3)
    parser.add_argument("--adamw_lr", type=float, default=2e-4)
    parser.add_argument("--muon_min_lr", type=float, default=1e-4)
    parser.add_argument("--adamw_min_lr", type=float, default=2e-5)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--eval_batches", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--resume_ckpt", type=str, default="auto")
    parser.add_argument("--gcs_dir", type=str, default="")
    parser.add_argument("--no_gcs", action="store_true")
    parser.add_argument("--wandb", type=str, default="auto", choices=["auto", "online", "disabled"])
    args, _ = parser.parse_known_args()
    return args


def main(args=None):
    if args is None:
        args = parse_args()

    env = EnvironmentConfig.detect(override_device=args.device)
    if args.gcs_dir:
        env.gcs_checkpoints_dir = args.gcs_dir
    sync_gcs = (env.is_kaggle and not args.no_gcs) or (not args.no_gcs and args.gcs_dir != "")

    vocab_size = 131072
    if args.smoke_test:
        import tempfile

        print("=== РЕЖИМ SMOKE TEST (Phase 2 Generator-Decoder & Noise Radar) ===", flush=True)
        smoke_tmp = tempfile.mkdtemp(prefix="bebladii_p2_smoke_")
        env.output_dir = smoke_tmp
        env.checkpoint_dir = smoke_tmp
        args.max_steps = 3
        args.batch_size = 2
        args.num_cells = 4
        args.num_smooth_layers = 3
        args.log_every = 1
        args.eval_every = 2
        args.eval_batches = 1
        args.save_every = 3
        args.lr_warmup_steps = 1
        args.wandb = "disabled"
        sync_gcs = False
        vocab_size = 2048

    use_wandb = False
    wandb_mode = args.wandb if args.wandb != "auto" else env.wandb_mode
    if wandb_mode == "online" and os.environ.get("WANDB_API_KEY", "").strip():
        try:
            import wandb

            wandb.login(key=os.environ["WANDB_API_KEY"].strip())
            wandb.init(project="BEBLaDII-Phase2-Decoder-PlanC", name=f"run-{int(time.time())}", config=vars(args))
            use_wandb = True
        except Exception as e:
            print(f"[WandB Warning] {e}", flush=True)

    tokenizer = None
    if not args.smoke_test:
        try:
            tokenizer = get_tokenizer()
        except Exception as e:
            print(f"[Tokenizer Warning] Не удалось загрузить Tekken ({e}). Используется fallback.", flush=True)

    if args.smoke_test:
        encoder = GrassmannianEncoder(vocab_size=vocab_size, hidden_size=1024, intermediate_size=2048, num_axes=8, axis_dim=128)
        decoder = GrassmannianDecoder(
            vocab_size=vocab_size,
            hidden_size=1024,
            intermediate_size=2048,
            num_axes=8,
            axis_dim=128,
            num_smooth_layers=args.num_smooth_layers,
            canonicalize_basis=True,
            enable_noise_radar=True,
        )
        decoder.tie_embeddings(encoder.embeddings.weight)
    else:
        print(f"[System 1] Загрузка замороженного Энкодера: {env.encoder_weights_path}", flush=True)
        encoder = get_encoder(weights_path=env.encoder_weights_path, device="cpu", eval_mode=True)
        print(f"[System 1] Инициализация Декодера-Генератора из: {env.decoder_weights_path}", flush=True)
        decoder = get_decoder(
            weights_path=env.decoder_weights_path,
            encoder=encoder,
            device="cpu",
            eval_mode=False,
            canonicalize_basis=True,
            enable_noise_radar=True,
            num_smooth_layers=args.num_smooth_layers,
        )

    ema = EMAModel(decoder, decay=args.ema_decay, device=env.device) if not args.no_ema else None
    opt_muon, opt_adamw = create_decoder_optimizers(decoder, muon_lr=args.muon_lr, adamw_lr=args.adamw_lr)

    start_step = 0
    metrics_history: List[Dict[str, Any]] = []
    if not args.smoke_test and args.resume_ckpt:
        resume_path = (
            find_latest_checkpoint(env.checkpoint_dir, env.gcs_checkpoints_dir if sync_gcs else None)
            if args.resume_ckpt == "auto"
            else args.resume_ckpt
        )
        if resume_path and os.path.exists(resume_path):
            print(f"[Resume] Восстановление из {resume_path}", flush=True)
            ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
            decoder.load_state_dict(ckpt["decoder_state_dict"], strict=False)
            if ckpt.get("opt_adamw_state_dict"):
                opt_adamw.load_state_dict(ckpt["opt_adamw_state_dict"])
            if opt_muon is not None and ckpt.get("opt_muon_state_dict"):
                opt_muon.load_state_dict(ckpt["opt_muon_state_dict"])
            if ema is not None and ckpt.get("ema_state_dict"):
                ema.load_state_dict(ckpt["ema_state_dict"], device=env.device)
            start_step = ckpt.get("step", 0)
            metrics_history = ckpt.get("metrics_history", [])

    stopword_mask = None
    if os.path.exists(env.stopwords_path):
        loaded = torch.load(env.stopwords_path, map_location="cpu", weights_only=False)
        stopword_mask = loaded["mask"] if isinstance(loaded, dict) and "mask" in loaded else loaded

    chunker = WordAwareElasticChunker(
        tokenizer=None if args.smoke_test else tokenizer,
        vocab_size=vocab_size,
        chunk_size=4,
        pad_token_id=10,
        stopword_mask=stopword_mask,
    )

    train_paths = [] if args.smoke_test else env.train_data_paths
    val_paths = [] if args.smoke_test else env.val_data_paths

    dataset = Phase2DecoderDataset(
        parquet_paths=train_paths,
        chunker=chunker,
        tokenizer=None if args.smoke_test else tokenizer,
        num_cells=args.num_cells,
        vocab_size=vocab_size,
        elastic_drop_prob=args.elastic_drop_prob,
        regime_b_ratio=0.0 if args.smoke_test else args.regime_b_ratio,
        virtual_length=16 if args.smoke_test else 50000,
        is_validation=False,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=env.num_workers, pin_memory=env.pin_memory
    )

    val_ds = Phase2DecoderDataset(
        parquet_paths=val_paths,
        chunker=chunker,
        tokenizer=None if args.smoke_test else tokenizer,
        num_cells=args.num_cells,
        vocab_size=vocab_size,
        elastic_drop_prob=0.0,
        regime_b_ratio=0.0,
        virtual_length=4 if args.smoke_test else 500,
        is_validation=True,
    )
    val_dataloader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=env.num_workers, pin_memory=env.pin_memory
    )

    if env.device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"[DataParallel] Включение DataParallel на {torch.cuda.device_count()} GPU.", flush=True)
        decoder = nn.DataParallel(decoder)

    train(
        encoder=encoder,
        decoder=decoder,
        dataloader=dataloader,
        env=env,
        tokenizer=None if args.smoke_test else tokenizer,
        opt_muon=opt_muon,
        opt_adamw=opt_adamw,
        start_step=start_step,
        metrics_history=metrics_history,
        max_steps=args.max_steps,
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
        muon_min_lr=args.muon_min_lr,
        adamw_min_lr=args.adamw_min_lr,
        lr_warmup_steps=args.lr_warmup_steps,
        log_every=args.log_every,
        save_every=args.save_every,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        val_dataloader=val_dataloader,
        ema=ema,
        sync_gcs=sync_gcs,
        use_wandb=use_wandb,
    )


# %%
TRAINING_CONFIG = {
    "smoke_test": False,
    "device": "auto",
    "batch_size": 16,
    "num_cells": 64,
    "num_smooth_layers": 4,
    "elastic_drop_prob": 0.20,
    "regime_b_ratio": 0.10,
    "max_steps": 30000,
    "muon_lr": 1e-3,
    "adamw_lr": 2e-4,
    "muon_min_lr": 1e-4,
    "adamw_min_lr": 2e-5,
    "lr_warmup_steps": 500,
    "ema_decay": 0.999,
    "eval_every": 200,
    "eval_batches": 8,
    "log_every": 20,
    "save_every": 500,
    "resume_ckpt": "auto",
    "gcs_dir": "",
    "no_gcs": False,
    "wandb": "auto",
}

if __name__ == "__main__":
    cli_args = parse_args()
    is_notebook = any("ipykernel" in m for m in sys.modules) or not sys.argv[1:]
    if is_notebook:
        for k, v in TRAINING_CONFIG.items():
            if hasattr(cli_args, k):
                setattr(cli_args, k, v)
    main(cli_args)
