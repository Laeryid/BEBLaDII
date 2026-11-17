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
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Отключаем TorchDynamo для исключения конфликтов с компилятором XLA
try:
    torch._dynamo.disable()
except Exception:
    pass

# Условный импорт PyTorch/XLA для запуска на Kaggle/Cloud TPU (или CPU-моке)
HAS_XLA = False
_XLA_SPMD_READY = False
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr

    try:
        import torch_xla.experimental.xla_sharding as xs
    except ImportError:
        import torch_xla.distributed.spmd as xs
        if not hasattr(xs, "mark_sharding"):
            import torch_xla.distributed.spmd.xla_sharding as xs

    HAS_XLA = True
except ImportError:
    xm = None
    xr = None
    xs = None

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# %% [markdown]
# ## 1. Инициализация окружения, SPMD (TPU) и путей

# %%
GLOBAL_SPMD_MESH = None


def ensure_spmd_mesh():
    """Лениво включает режим SPMD (single-process TPU) и создаёт глобальный меш ('fsdp', 'model')."""
    global GLOBAL_SPMD_MESH, _XLA_SPMD_READY
    if not HAS_XLA:
        return None
    if not _XLA_SPMD_READY:
        try:
            xr.use_spmd()
        except Exception:
            pass
        _XLA_SPMD_READY = True
    if GLOBAL_SPMD_MESH is None:
        num_devices = max(1, int(xr.global_runtime_device_count()))
        mesh_shape = (num_devices, 1)
        device_ids = np.array(range(num_devices))
        mesh = xs.Mesh(device_ids, mesh_shape, ("fsdp", "model"))
        if hasattr(xs, "set_global_mesh"):
            xs.set_global_mesh(mesh)
        GLOBAL_SPMD_MESH = mesh
    return GLOBAL_SPMD_MESH


def mark_batch_sharding(tensor: Optional[torch.Tensor], mesh: Any) -> Optional[torch.Tensor]:
    """Шардирует батч по оси 'fsdp' (0-я размерность) в режиме SPMD без разрезания весов модели."""
    if not HAS_XLA or mesh is None or tensor is None or not torch.is_tensor(tensor):
        return tensor
    if hasattr(xs, "mark_sharding") and tensor.dim() >= 1:
        try:
            spec = ("fsdp",) + (None,) * (tensor.dim() - 1)
            xs.mark_sharding(tensor, mesh, spec)
        except Exception:
            pass
    return tensor

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
    is_tpu: bool
    is_master: bool
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
    mesh: Any = None

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
            val_text = resolve_file_path(
                "val.parquet",
                ["/kaggle/input/bebladii-planb-phase3-data/phase 3/train_data/data/val.parquet"],
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
            num_workers = 4
            pin_memory = True
        else:
            output_dir = os.path.join(project_root, "experiments", "phase 2", "output")
            checkpoint_dir = os.path.join(project_root, "experiments", "phase 2", "checkpoints")
            train_text = os.path.join(
                project_root, "BEBLaDII-planB-Phase3-Data", "phase 3", "train_data", "data", "train.parquet"
            )
            val_text = os.path.join(
                project_root, "BEBLaDII-planB-Phase3-Data", "phase 3", "train_data", "data", "val.parquet"
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
            pin_memory = False

        train_data_paths = [train_text] if (train_text and os.path.exists(train_text)) else []
        val_data_paths = [val_text] if (val_text and os.path.exists(val_text)) else []

        want_tpu = override_device in ("tpu", "xla") or (
            override_device == "auto"
            and HAS_XLA
            and (
                not torch.cuda.is_available()
                or bool(os.environ.get("PJRT_DEVICE"))
                or bool(os.environ.get("COLAB_TPU_ADDR"))
                or bool(os.environ.get("TPU_NAME"))
                or os.environ.get("SMOKE_TEST") == "1"
            )
        )

        mesh = None
        is_tpu = False
        is_master = True
        if want_tpu:
            if not HAS_XLA:
                raise RuntimeError("Запрошен --device tpu, но библиотека torch_xla не установлена.")
            mesh = ensure_spmd_mesh()
            device = xm.xla_device()
            is_tpu = True
            is_master = bool(xm.is_master_ordinal())
            dtype = torch.bfloat16
            pin_memory = False
        elif override_device and override_device not in ("auto", "tpu", "xla"):
            device = torch.device(override_device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not is_tpu:
            if device.type == "cuda":
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                warmup_linalg_backends()
            else:
                dtype = torch.float32
                pin_memory = False

        if is_master:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(checkpoint_dir, exist_ok=True)

        return cls(
            is_kaggle=is_kaggle,
            is_tpu=is_tpu,
            is_master=is_master,
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
            mesh=mesh,
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
def build_punctuation_id_set(tokenizer: Any) -> set:
    """Множество token_id, содержащих знаки завершения предложений (. ! ? \\n)."""
    punct_ids = {1046, 2247, 1063, 1047, 13, 28723, 28725, 120, 26, 29}
    if tokenizer is not None:
        try:
            vocab = tokenizer.get_vocab()
            for text, tid in vocab.items():
                if any(p in text for p in (".", "!", "?", "\n", "Ċ")):
                    punct_ids.add(int(tid))
        except Exception:
            pass
    return punct_ids


# %%
class Phase2DecoderDataset(Dataset):
    """
    Датасет Фазы 2 (Generator-Decoder & Noise Radar):
    - Читает связные тексты ('text') только из датасета Фазы 1 (train.parquet / val.parquet).
    - Обрезает последовательность до последнего знака завершения предложения (. ! ? \\n).
    - Нарезает последовательности через WordAwareElasticChunker (1..4 токена без разрыва слов).
    - В 10% обучающих шагов добавляет словарный спектр (Regime B) для удержания редких токенов.
    """

    def __init__(
        self,
        parquet_paths: List[str],
        chunker: WordAwareElasticChunker,
        tokenizer: Any = None,
        num_cells: int = 128,
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
        self.punct_ids = build_punctuation_id_set(tokenizer)

        self.texts: List[str] = []
        for path in parquet_paths:
            if not path or not os.path.exists(path):
                continue
            try:
                df = pd.read_parquet(path, columns=["text"])
                added = 0
                if "text" in df.columns:
                    col_vals = [str(x).strip() for x in df["text"].dropna().tolist() if str(x).strip()]
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

    def _truncate_tokens_to_sentence_end(self, token_ids: List[int]) -> List[int]:
        """Обрезает срез токенов до последнего знака завершения предложения (. ! ? \\n)."""
        for i in range(len(token_ids) - 1, -1, -1):
            if int(token_ids[i]) in self.punct_ids:
                if i >= 4:
                    return token_ids[: i + 1]
                break
        return token_ids

    def _trim_packed_to_sentence_end(self, packed: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """При переполнении эластичного холста обрезает упакованную последовательность по последнему знаку препинания."""
        input_ids = packed["input_ids"]
        targets = packed["targets"]
        weights = packed["weights"]
        content_cell_mask = packed["content_cell_mask"]

        last_punct = -1
        for pos in range(input_ids.numel() - 1, -1, -1):
            tid = int(input_ids[pos].item())
            if tid != self.pad_token_id and tid in self.punct_ids:
                last_punct = pos
                break
        if last_punct < 4:
            return packed

        cs = self.chunk_size
        c_last = last_punct // cs
        cell_start = c_last * cs
        cell_end = (c_last + 1) * cs

        if last_punct + 1 < cell_end:
            input_ids[last_punct + 1 : cell_end] = self.pad_token_id
            targets[last_punct + 1 : cell_end] = self.pad_token_id

        cell_ids = input_ids[cell_start:cell_end]
        is_pad = cell_ids == self.pad_token_id
        safe_ids = torch.clamp(cell_ids, 0, self.chunker.stopword_mask.numel() - 1)
        is_stop = self.chunker.stopword_mask[safe_ids] & (~is_pad)
        raw_w = torch.where(
            is_pad,
            torch.tensor(self.chunker.eps_pad, dtype=torch.float32),
            torch.where(
                is_stop,
                torch.tensor(self.chunker.eps_stop, dtype=torch.float32),
                torch.tensor(1.0, dtype=torch.float32),
            ),
        )
        weights[cell_start:cell_end] = raw_w / raw_w.sum().clamp(min=1e-8)

        if cell_end < input_ids.numel():
            input_ids[cell_end:] = self.pad_token_id
            targets[cell_end:] = -100
            weights[cell_end:] = 1.0 / cs
            content_cell_mask[c_last + 1 :] = False

        return packed

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
        if random.random() < self.regime_b_ratio:
            return self._sample_regime_b()

        if self.has_real_data and self.tokenizer is not None:
            raw_text = self.texts[idx % len(self.texts)]
            token_ids = self.tokenizer.encode(raw_text, add_special_tokens=False)
            # Случайный сдвиг старта на длинных текстах при обучении (разнообразие фаз и слов)
            if not self.is_validation and len(token_ids) > self.seq_len:
                max_offset = min(len(token_ids) - self.seq_len // 2, 256)
                if max_offset > 0:
                    offset = random.randint(0, max_offset)
                    token_ids = token_ids[offset:]
            token_ids = self._truncate_tokens_to_sentence_end(token_ids[: self.seq_len])
        else:
            max_tok = min(self.vocab_size, 2000)
            min_tok = min(1000 if max_tok > 1020 else 20, max_tok - 1)
            eff_len = random.randint(max(8, self.seq_len // 2), self.seq_len)
            token_ids = torch.randint(min_tok, max_tok, (eff_len,)).tolist()

        packed = self.chunker.chunk_tokens(
            token_ids=token_ids,
            max_cells=self.num_cells,
            elastic_drop_prob=self.elastic_drop_prob,
        )
        if self.has_real_data and self.tokenizer is not None:
            n_packed = int((packed["input_ids"] != self.pad_token_id).sum().item())
            if n_packed < len(token_ids):
                packed = self._trim_packed_to_sentence_end(packed)
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
    """Генерирует случайные ортогональные матрицы R in O(k) формы (B, N, k, k) через QR на CPU (безопасно для XLA)."""
    raw = torch.randn(batch_size, num_cells, k, k, device=torch.device("cpu"), dtype=torch.float32)
    q, r = torch.linalg.qr(raw)
    diag_sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    diag_sign = torch.where(diag_sign == 0.0, torch.ones_like(diag_sign), diag_sign)
    out = q * diag_sign.unsqueeze(-2)
    return out if device.type == "cpu" else out.to(device)


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
    При mode_override=None смешивание выполняется по-сэмплово внутри батча через torch.where (единый граф XLA).
    Случайные величины и QR-повороты R in O(8) генерируются на CPU (ADR 097).
    """
    B, N, K, D = V_0.shape
    device = V_0.device
    V_f = V_0.float()
    cpu = torch.device("cpu")

    if mode_override == "clean":
        V_tilde = V_f.clone()
        mode = "clean"
    elif mode_override == "random_o8":
        V_tilde = V_f.clone()
        mode = "random_o8"
    elif mode_override == "clean_jitter":
        sigma = (torch.rand(B, N, 1, 1, device=cpu) * 0.015).to(device)
        noise = torch.randn(B, N, K, D, device=cpu, dtype=torch.float32).to(device)
        V_tilde = clamp_to_unit_ball(V_f + noise * sigma)
        mode = "clean_jitter"
    elif mode_override == "gaussian":
        sigma = (0.010 + torch.rand(B, N, 1, 1, device=cpu) * 0.055).to(device)
        noise = torch.randn(B, N, K, D, device=cpu, dtype=torch.float32).to(device)
        V_tilde = clamp_to_unit_ball(V_f + noise * sigma)
        mode = "gaussian"
    elif mode_override == "mix_other":
        V_other = torch.roll(V_f, shifts=1, dims=0) if B > 1 else torch.roll(V_f, shifts=max(1, N // 3), dims=1)
        if B > 1:
            V_other = torch.roll(V_other, shifts=max(1, N // 5), dims=1)
        alpha = (0.15 + torch.rand(B, N, 1, 1, device=cpu) * 0.40).to(device)
        V_tilde = clamp_to_unit_ball((1.0 - alpha) * V_f + alpha * V_other)
        mode = "mix_other"
    elif mode_override == "crystal_mask":
        keep_prob = torch.rand(B, 1, 1, 1, device=cpu) * 0.40 + 0.30
        is_crystal = (torch.rand(B, N, 1, 1, device=cpu) < keep_prob).to(device)
        sigma = (0.015 + torch.rand(B, N, 1, 1, device=cpu) * 0.050).to(device)
        noise = torch.randn(B, N, K, D, device=cpu, dtype=torch.float32).to(device)
        noise_v = clamp_to_unit_ball(V_f + noise * sigma)
        V_tilde = torch.where(is_crystal, V_f, noise_v)
        mode = "crystal_mask"
    elif mode_override is None:
        # Детерминированный статический граф для XLA: все 4 режима объединяются по-сэмплово в батче
        mode_u = torch.rand(B, 1, 1, 1, device=cpu, dtype=torch.float32).to(device)
        noise_cj = torch.randn(B, N, K, D, device=cpu, dtype=torch.float32).to(device)
        noise_g = torch.randn(B, N, K, D, device=cpu, dtype=torch.float32).to(device)
        noise_cm = torch.randn(B, N, K, D, device=cpu, dtype=torch.float32).to(device)
        sigma_cj = (torch.rand(B, N, 1, 1, device=cpu, dtype=torch.float32) * 0.015).to(device)
        sigma_g = (0.010 + torch.rand(B, N, 1, 1, device=cpu, dtype=torch.float32) * 0.055).to(device)
        alpha_mix = (0.15 + torch.rand(B, N, 1, 1, device=cpu, dtype=torch.float32) * 0.40).to(device)
        keep_prob = torch.rand(B, 1, 1, 1, device=cpu, dtype=torch.float32) * 0.40 + 0.30
        is_crystal = (torch.rand(B, N, 1, 1, device=cpu, dtype=torch.float32) < keep_prob).to(device)
        sigma_cm = (0.015 + torch.rand(B, N, 1, 1, device=cpu, dtype=torch.float32) * 0.050).to(device)

        v_cj = clamp_to_unit_ball(V_f + noise_cj * sigma_cj)
        v_g = clamp_to_unit_ball(V_f + noise_g * sigma_g)

        V_other = torch.roll(V_f, shifts=1, dims=0) if B > 1 else torch.roll(V_f, shifts=max(1, N // 3), dims=1)
        if B > 1:
            V_other = torch.roll(V_other, shifts=max(1, N // 5), dims=1)
        v_mix = clamp_to_unit_ball((1.0 - alpha_mix) * V_f + alpha_mix * V_other)
        v_cm = torch.where(is_crystal, V_f, clamp_to_unit_ball(V_f + noise_cm * sigma_cm))

        V_tilde = torch.where(
            mode_u < 0.20,
            v_cj,
            torch.where(mode_u < 0.50, v_g, torch.where(mode_u < 0.75, v_mix, v_cm)),
        )
        mode = "mixed_batch"
    else:
        raise ValueError(f"Unknown corruption mode: {mode_override}")

    # Вычисляем истинное O(8)-инвариантное искажение до поворота
    sim_gr = grassmann_cell_similarity(V_tilde, V_f)
    true_sigma_cell = (1.0 - sim_gr).clamp(0.0, 1.0)

    # Применяем случайное вращение R in O(8) (кроме чистого контрольного режима 'clean')
    if mode != "clean":
        R = random_orthogonal_matrices(B, N, K, device=device)
        V_tilde = torch.matmul(R, V_tilde)

    return V_tilde.to(V_0.dtype), true_sigma_cell, mode


def _finalize_phase2_loss_and_metrics(
    ce_words_sum: torch.Tensor,
    words_count: torch.Tensor,
    ce_pad_sum: torch.Tensor,
    pad_count: torch.Tensor,
    word_correct_sum: torch.Tensor,
    pad_correct_sum: torch.Tensor,
    pred_ids: torch.Tensor,
    radar: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    true_sigma_cell: torch.Tensor,
    slot_pad_weight: float,
    radar_cell_weight: float,
    radar_tok_weight: float,
    return_tensors: bool,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Общий XLA-совместимый расчёт итогового лосса и метрик без динамических форм."""
    B, L = targets.shape
    N = true_sigma_cell.shape[1]
    chunk_size = L // N

    loss_ce_words = ce_words_sum / words_count.clamp(min=1.0)
    loss_ce_pad = ce_pad_sum / pad_count.clamp(min=1.0)

    sigma_cell_hat = radar["sigma_cell_hat"].float()
    true_cell_f32 = true_sigma_cell.float()
    loss_radar_cell = F.smooth_l1_loss(sigma_cell_hat, true_cell_f32, beta=0.05)

    with torch.no_grad():
        wrong_tok = (pred_ids != targets).float()
        cell_exp = true_cell_f32.unsqueeze(-1).expand(B, N, chunk_size).reshape(B, L)
        target_sigma_tok = (0.5 * cell_exp + 0.5 * wrong_tok).clamp(0.0, 1.0)

    valid_w = (targets.view(-1) != -100).float()
    tok_logits = radar["sigma_tok_logits"].float().view(-1)
    tok_tgt_flat = target_sigma_tok.view(-1)
    bce_tok = F.binary_cross_entropy_with_logits(tok_logits, tok_tgt_flat, reduction="none")
    loss_radar_tok = (bce_tok * valid_w).sum() / valid_w.sum().clamp(min=1.0)

    total_loss = (
        loss_ce_words
        + slot_pad_weight * loss_ce_pad
        + radar_cell_weight * loss_radar_cell
        + radar_tok_weight * loss_radar_tok
    )

    with torch.no_grad():
        word_acc = word_correct_sum / words_count.clamp(min=1.0)
        slot_pad_acc = torch.where(
            pad_count > 0.0,
            pad_correct_sum / pad_count.clamp(min=1.0),
            torch.ones_like(pad_count),
        )
        radar_mae = torch.abs(sigma_cell_hat - true_cell_f32).mean()

    if return_tensors:
        metrics: Dict[str, Any] = {
            "loss/total": total_loss.detach(),
            "loss/ce_words": loss_ce_words.detach(),
            "loss/ce_slot_pad": loss_ce_pad.detach(),
            "loss/radar_cell": loss_radar_cell.detach(),
            "loss/radar_tok": loss_radar_tok.detach(),
            "acc/words": word_acc.detach(),
            "acc/slot_pad": slot_pad_acc.detach(),
            "radar/mae_cell": radar_mae.detach(),
        }
    else:
        metrics = {
            "loss/total": float(total_loss.item()),
            "loss/ce_words": float(loss_ce_words.item()),
            "loss/ce_slot_pad": float(loss_ce_pad.item()),
            "loss/radar_cell": float(loss_radar_cell.item()),
            "loss/radar_tok": float(loss_radar_tok.item()),
            "acc/words": float(word_acc.item()),
            "acc/slot_pad": float(slot_pad_acc.item()),
            "radar/mae_cell": float(radar_mae.item()),
        }
    return total_loss, metrics


def compute_phase2_decoder_loss(
    logits: torch.Tensor,
    radar: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    true_sigma_cell: torch.Tensor,
    pad_token_id: int = 10,
    slot_pad_weight: float = 0.25,
    radar_cell_weight: float = 0.50,
    radar_tok_weight: float = 0.30,
    return_tensors: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Гибридная функция потерь Фазы 2 (строго в float32 снаружи DataParallel, ADR 135):
    L = L_ce_words + 0.25 * L_ce_slot_pad + 0.50 * L_radar_cell + 0.30 * L_radar_tok
    """
    V_size = logits.shape[-1]
    logits_f32 = logits.float()
    targets_flat = targets.view(-1)
    ce_per_tok = F.cross_entropy(logits_f32.view(-1, V_size), targets_flat, ignore_index=-100, reduction="none")

    valid_mask = targets_flat != -100
    is_slot_pad = (targets_flat == pad_token_id) & valid_mask
    is_word = valid_mask & (~is_slot_pad)
    w_word = is_word.float()
    w_pad = is_slot_pad.float()

    ce_words_sum = (ce_per_tok * w_word).sum()
    ce_pad_sum = (ce_per_tok * w_pad).sum()
    with torch.no_grad():
        pred_ids = torch.argmax(logits_f32, dim=-1)
        pred_flat = pred_ids.view(-1)
        word_correct_sum = ((pred_flat == targets_flat).float() * w_word).sum()
        pad_correct_sum = ((pred_flat == pad_token_id).float() * w_pad).sum()

    return _finalize_phase2_loss_and_metrics(
        ce_words_sum=ce_words_sum,
        words_count=w_word.sum(),
        ce_pad_sum=ce_pad_sum,
        pad_count=w_pad.sum(),
        word_correct_sum=word_correct_sum,
        pad_correct_sum=pad_correct_sum,
        pred_ids=pred_ids,
        radar=radar,
        targets=targets,
        true_sigma_cell=true_sigma_cell,
        slot_pad_weight=slot_pad_weight,
        radar_cell_weight=radar_cell_weight,
        radar_tok_weight=radar_tok_weight,
        return_tensors=return_tensors,
    )


def compute_phase2_decoder_loss_from_hidden(
    hidden_states: torch.Tensor,
    decoder: GrassmannianDecoder,
    radar: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    true_sigma_cell: torch.Tensor,
    pad_token_id: int = 10,
    slot_pad_weight: float = 0.25,
    radar_cell_weight: float = 0.50,
    radar_tok_weight: float = 0.30,
    seq_chunks: int = 4,
    return_tensors: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any], torch.Tensor]:
    """
    Чанкованный расчёт логитов и Cross-Entropy по последовательности (для TPU HBM <= 15.75 GB):
    не материализует полный тензор (B, L, V_size=131072) сразу.
    Возвращает (total_loss, metrics, pred_ids).
    """
    L = hidden_states.shape[1]
    n_chunks = max(1, min(seq_chunks, L))
    h_chunks = hidden_states.chunk(n_chunks, dim=1)
    t_chunks = targets.chunk(n_chunks, dim=1)

    device = hidden_states.device
    ce_words_sum = torch.zeros((), device=device, dtype=torch.float32)
    words_count = torch.zeros((), device=device, dtype=torch.float32)
    ce_pad_sum = torch.zeros((), device=device, dtype=torch.float32)
    pad_count = torch.zeros((), device=device, dtype=torch.float32)
    word_correct_sum = torch.zeros((), device=device, dtype=torch.float32)
    pad_correct_sum = torch.zeros((), device=device, dtype=torch.float32)
    pred_chunks: List[torch.Tensor] = []

    for h_c, t_c in zip(h_chunks, t_chunks):
        logits_c = decoder.get_tied_head_logits(h_c)
        V_size = logits_c.shape[-1]
        t_c_flat = t_c.reshape(-1)
        ce_c = F.cross_entropy(logits_c.reshape(-1, V_size), t_c_flat, ignore_index=-100, reduction="none")

        valid_c = t_c_flat != -100
        is_pad_c = (t_c_flat == pad_token_id) & valid_c
        is_word_c = valid_c & (~is_pad_c)
        w_word_c = is_word_c.float()
        w_pad_c = is_pad_c.float()

        ce_words_sum = ce_words_sum + (ce_c * w_word_c).sum()
        words_count = words_count + w_word_c.sum()
        ce_pad_sum = ce_pad_sum + (ce_c * w_pad_c).sum()
        pad_count = pad_count + w_pad_c.sum()

        with torch.no_grad():
            pred_c = torch.argmax(logits_c, dim=-1)
            pred_c_flat = pred_c.reshape(-1)
            word_correct_sum = word_correct_sum + ((pred_c_flat == t_c_flat).float() * w_word_c).sum()
            pad_correct_sum = pad_correct_sum + ((pred_c_flat == pad_token_id).float() * w_pad_c).sum()
            pred_chunks.append(pred_c)

    pred_ids = torch.cat(pred_chunks, dim=1)
    total_loss, metrics = _finalize_phase2_loss_and_metrics(
        ce_words_sum=ce_words_sum,
        words_count=words_count,
        ce_pad_sum=ce_pad_sum,
        pad_count=pad_count,
        word_correct_sum=word_correct_sum,
        pad_correct_sum=pad_correct_sum,
        pred_ids=pred_ids,
        radar=radar,
        targets=targets,
        true_sigma_cell=true_sigma_cell,
        slot_pad_weight=slot_pad_weight,
        radar_cell_weight=radar_cell_weight,
        radar_tok_weight=radar_tok_weight,
        return_tensors=return_tensors,
    )
    return total_loss, metrics, pred_ids


# %% [markdown]
# ## 5. Оптимизатор OrthMuon (d=1) + AdamW и EMA с проекцией на сферу (ADR 132)

# %%
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    orig_dtype = G.dtype
    use_bf16 = (G.device.type == "xla") or (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    X = G.bfloat16() if use_bf16 else G.float()
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
    def step(self, is_safe: Optional[torch.Tensor] = None):
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
                next_buf = buf * momentum + g
                if is_safe is not None:
                    buf.copy_(torch.where(is_safe, next_buf, buf))
                else:
                    buf.copy_(next_buf)

                update = g.add(buf, alpha=momentum) if nesterov else buf
                update_ortho = zeropower_via_newtonschulz5(update, steps=ns_steps)
                scale = 0.2 * math.sqrt(max(p.size(0), p.size(1)))

                if sphere:
                    q_f32 = p.data.float()
                    u_f32 = update_ortho.float()
                    u_tan = u_f32 - (u_f32 * q_f32).sum(dim=-1, keepdim=True) * q_f32
                    p_next = F.normalize(q_f32 - (lr * scale) * u_tan, p=2, dim=-1, eps=1e-8).to(p.dtype)
                else:
                    p_next = p.data.float()
                    if wd > 0:
                        p_next = p_next * (1.0 - lr * wd)
                    p_next = (p_next - (lr * scale) * update_ortho.float()).to(p.dtype)

                if is_safe is not None:
                    p.data.copy_(torch.where(is_safe, p_next, p.data))
                else:
                    p.data.copy_(p_next)


class EMAModel:
    """EMA весов модели с обязательной ренормализацией *.q_weight на единичную сферу (ADR 132) и PACE pullback."""

    def __init__(self, model: nn.Module, decay: float = 0.999, device: Optional[torch.device] = None):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        raw = getattr(model, "module", model)
        for name, param in raw.named_parameters():
            if param.requires_grad:
                t = param.detach().clone().float()
                if device is not None:
                    t = t.to(device)
                self.shadow[name] = t

    @torch.no_grad()
    def update(
        self,
        model: nn.Module,
        pullback_muon: float = 0.005,
        pullback_adamw: float = 0.002,
        is_safe: Optional[torch.Tensor] = None,
    ) -> None:
        raw = getattr(model, "module", model)
        for name, param in raw.named_parameters():
            if name not in self.shadow:
                continue
            p_f = param.detach().float()
            s_f = self.shadow[name].float()
            new_s = self.decay * s_f + (1.0 - self.decay) * p_f
            if name.endswith(".q_weight"):
                new_s = F.normalize(new_s, p=2, dim=-1, eps=1e-8)
                target_s = new_s.to(self.shadow[name].dtype)
                if is_safe is not None:
                    self.shadow[name].copy_(torch.where(is_safe, target_s, self.shadow[name]))
                else:
                    self.shadow[name].copy_(target_s)
                if pullback_muon > 0.0:
                    p_pulled = F.normalize((1.0 - pullback_muon) * p_f + pullback_muon * new_s, p=2, dim=-1, eps=1e-8).to(param.dtype)
                    if is_safe is not None:
                        param.data.copy_(torch.where(is_safe, p_pulled, param.data))
                    else:
                        param.data.copy_(p_pulled)
            else:
                target_s = new_s.to(self.shadow[name].dtype)
                if is_safe is not None:
                    self.shadow[name].copy_(torch.where(is_safe, target_s, self.shadow[name]))
                else:
                    self.shadow[name].copy_(target_s)
                if pullback_adamw > 0.0:
                    p_pulled = ((1.0 - pullback_adamw) * p_f + pullback_adamw * new_s).to(param.dtype)
                    if is_safe is not None:
                        param.data.copy_(torch.where(is_safe, p_pulled, param.data))
                    else:
                        param.data.copy_(p_pulled)

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module) -> None:
        raw = getattr(model, "module", model)
        self.backup = {}
        dev_type = "cpu"
        for name, param in raw.named_parameters():
            dev_type = param.device.type
            if name in self.shadow:
                # Бэкап в CPU RAM для экономии TPU HBM при валидации и чекпоинтинге
                self.backup[name] = param.data.detach().cpu().clone()
                s = self.shadow[name]
                if name.endswith(".q_weight"):
                    s = F.normalize(s.float(), p=2, dim=-1, eps=1e-8).to(param.dtype)
                param.data.copy_(s.to(param.device))
        if HAS_XLA and dev_type == "xla":
            xm.mark_step()

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        raw = getattr(model, "module", model)
        dev_type = "cpu"
        for name, param in raw.named_parameters():
            dev_type = param.device.type
            if name in self.backup:
                param.data.copy_(self.backup[name].to(device=param.device, dtype=param.dtype))
        self.backup = {}
        if HAS_XLA and dev_type == "xla":
            xm.mark_step()

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "shadow": {k: v.cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, state: Dict[str, Any], device: Optional[torch.device] = None) -> None:
        shadow = state.get("shadow", state) if isinstance(state, dict) else {}
        for k, v in shadow.items():
            if k in self.shadow:
                target_dev = device or self.shadow[k].device
                self.shadow[k].copy_(v.to(device=target_dev, dtype=self.shadow[k].dtype))

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
def get_amp_params(env: EnvironmentConfig) -> Tuple[str, torch.dtype, bool]:
    """Возвращает (device_type, autocast_dtype, enabled) для CUDA, TPU (XLA) и CPU."""
    if env.is_tpu:
        dev_type = env.device.type
        return dev_type, torch.bfloat16, (dev_type == "xla")
    if env.device.type == "cuda":
        return "cuda", env.dtype, True
    return "cpu", torch.float32, False


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
    raw_dec = getattr(decoder, "module", decoder)
    is_dp = isinstance(decoder, nn.DataParallel)
    modes = ["clean", "random_o8", "gaussian", "mix_other"]
    metric_keys = [f"{m}_{k}" for m in modes for k in ("acc", "ce", "mae", "sim")]
    sums_t: Dict[str, torch.Tensor] = {
        k: torch.zeros((), device=env.device, dtype=torch.float32) for k in metric_keys
    }
    count = 0
    sample_tensors: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None
    amp_dev_type, amp_dtype, use_amp = get_amp_params(env)

    if ema is not None:
        ema.apply_shadow(decoder)
    raw_dec.eval()
    encoder.eval()
    try:
        for i, batch in enumerate(val_dataloader):
            if i >= eval_batches:
                break
            input_ids = batch["input_ids"].to(env.device)
            targets = batch["targets"].to(env.device)
            weights = batch["weights"].to(env.device)
            content_mask = batch["content_cell_mask"].to(env.device)
            if env.is_tpu:
                mark_batch_sharding(input_ids, env.mesh)
                mark_batch_sharding(targets, env.mesh)
                mark_batch_sharding(weights, env.mesh)
                mark_batch_sharding(content_mask, env.mesh)

            enc_w_dtype = encoder.embeddings.weight.dtype
            with torch.autocast(device_type=amp_dev_type, dtype=amp_dtype, enabled=use_amp):
                V_0 = encoder(input_ids, weights.to(enc_w_dtype), canonicalize=False).float()

            for mode in modes:
                V_in, true_sigma, _ = corrupt_latent_cells(V_0, content_mask, mode_override=mode)
                if not is_dp:
                    with torch.autocast(device_type=amp_dev_type, dtype=amp_dtype, enabled=use_amp):
                        hidden_dec, radar = raw_dec.decode_hidden(V_in, return_radar=True)
                    _, metrics_t, pred_ids = compute_phase2_decoder_loss_from_hidden(
                        hidden_states=hidden_dec,
                        decoder=raw_dec,
                        radar=radar,
                        targets=targets,
                        true_sigma_cell=true_sigma,
                        pad_token_id=pad_token_id,
                        seq_chunks=4,
                        return_tensors=True,
                    )
                    del hidden_dec
                else:
                    with torch.autocast(device_type=amp_dev_type, dtype=amp_dtype, enabled=use_amp):
                        logits, radar = decoder(V_in, return_radar=True)
                    _, metrics_t = compute_phase2_decoder_loss(
                        logits=logits,
                        radar=radar,
                        targets=targets,
                        true_sigma_cell=true_sigma,
                        pad_token_id=pad_token_id,
                        return_tensors=True,
                    )
                    pred_ids = torch.argmax(logits.float(), dim=-1)
                    del logits

                sums_t[f"{mode}_acc"] = sums_t[f"{mode}_acc"] + metrics_t["acc/words"]
                sums_t[f"{mode}_ce"] = sums_t[f"{mode}_ce"] + metrics_t["loss/ce_words"]
                sums_t[f"{mode}_mae"] = sums_t[f"{mode}_mae"] + metrics_t["radar/mae_cell"]
                sums_t[f"{mode}_sim"] = sums_t[f"{mode}_sim"] + (1.0 - true_sigma).mean()

                if sample_tensors is None and mode == "mix_other" and tokenizer is not None:
                    sample_tensors = (
                        pred_ids[0].detach(),
                        targets[0].detach(),
                        true_sigma[0].mean().detach(),
                        radar["sigma_cell_hat"][0].mean().detach(),
                    )
                del V_in, true_sigma, radar, metrics_t, pred_ids

            del input_ids, targets, weights, content_mask, V_0
            if env.is_tpu:
                xm.mark_step()
            count += 1
    finally:
        if ema is not None:
            ema.restore(decoder)
        raw_dec.train()

    if sample_tensors is not None and env.is_master:
        pred_row_t, tgt_row_t, sim_t, rad_t = sample_tensors
        pred_row = pred_row_t.cpu()
        tgt_cpu = tgt_row_t.cpu()
        tgt_row = tgt_cpu[tgt_cpu != -100]
        sim_val = float(sim_t.cpu().item())
        rad_val = float(rad_t.cpu().item())
        ref_txt = WordAwareElasticChunker.detokenize_non_pad(tgt_row, tokenizer, pad_token_id)[:180]
        gen_txt = WordAwareElasticChunker.detokenize_non_pad(pred_row, tokenizer, pad_token_id)[:180]
        print(
            f"  [Sample MixOther | Sim={sim_val:.3f} | Radar={rad_val:.3f}]\n"
            f"    REF: {ref_txt!r}\n    GEN: {gen_txt!r}",
            flush=True,
        )

    denom = max(1, count)
    packed_vals = torch.stack([sums_t[k] for k in metric_keys]).cpu().tolist()
    out = {f"val/{k}": float(v) / denom for k, v in zip(metric_keys, packed_vals)}
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
    is_tpu: bool = False,
    is_master: bool = True,
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
    if is_tpu and HAS_XLA:
        xm.save(ckpt, path, master_only=True)
    elif is_master:
        torch.save(ckpt, path)

    if is_master:
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
    enc_dtype = torch.bfloat16 if env.is_tpu else torch.float32
    encoder.to(device=env.device, dtype=enc_dtype).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # Мастер-веса обучаемого декодера и буферы оптимизатора строго в float32 (ADR 082).
    # Гарантируем, что lm_head_weight — буфер (не Parameter), чтобы .to(float32) не мутировал Parameter энкодера.
    raw_dec = getattr(decoder, "module", decoder)
    if "lm_head_weight" in raw_dec._parameters:
        del raw_dec._parameters["lm_head_weight"]
        raw_dec.register_buffer("lm_head_weight", None, persistent=False)
    else:
        raw_dec.lm_head_weight = None
    decoder.to(device=env.device, dtype=torch.float32)
    raw_dec.tie_embeddings(encoder.embeddings.weight.detach())

    if ema is not None:
        ema.to(env.device)

    # Перенос загруженных состояний оптимизаторов на целевое устройство (если был resume на CPU)
    for opt in (opt_adamw, opt_muon):
        if opt is None:
            continue
        for st in opt.state.values():
            for k, v in list(st.items()):
                if torch.is_tensor(v):
                    st[k] = v.to(env.device)

    raw_dec.train()
    trainable_params = [p for p in raw_dec.parameters() if p.requires_grad]
    is_dp = isinstance(decoder, nn.DataParallel)

    if metrics_history is None:
        metrics_history = []

    best_val_loss = min(
        [e["val/loss"] for e in metrics_history if "val/loss" in e],
        default=float("inf"),
    )

    amp_dev_type, amp_dtype, use_amp = get_amp_params(env)
    use_scaler = (env.device.type == "cuda") and (env.dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    if env.is_master:
        mode_str = "TPU-SPMD" if env.is_tpu else env.device.type.upper()
        print(
            f"=== Обучение Phase 2 (Generator-Decoder & Noise Radar) | mode={mode_str} | "
            f"device={env.device} | amp_dtype={amp_dtype} ===",
            flush=True,
        )
        print(
            f"    Обучаемых параметров: OrthMuon={sum(p.numel() for g in opt_muon.param_groups for p in g['params']) if opt_muon else 0:,} | "
            f"AdamW={sum(p.numel() for g in opt_adamw.param_groups for p in g['params']):,}",
            flush=True,
        )

    metric_names = [
        "loss/total",
        "loss/ce_words",
        "loss/ce_slot_pad",
        "loss/radar_cell",
        "loss/radar_tok",
        "acc/words",
        "acc/slot_pad",
        "radar/mae_cell",
    ]
    last_metrics_scalars: Dict[str, float] = {k: 0.0 for k in metric_names}

    global_step = start_step + 1
    data_iter = iter(dataloader)
    last_t = time.time()
    last_step = start_step

    while global_step <= max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # На TPU обновляем LR раз в 50 шагов, чтобы не провоцировать перекомпиляцию графа XLA
        if (not env.is_tpu) or (global_step == start_step + 1) or (global_step % 50 == 0):
            lr_step_ref = ((global_step - 1) // 50) * 50 if env.is_tpu else (global_step - 1)
            curr_muon_lr = get_cosine_lr(muon_lr, muon_min_lr, lr_step_ref, max_steps, lr_warmup_steps)
            curr_adamw_lr = get_cosine_lr(adamw_lr, adamw_min_lr, lr_step_ref, max_steps, lr_warmup_steps)
            if opt_muon is not None:
                for g in opt_muon.param_groups:
                    g["lr"] = curr_muon_lr
            for g in opt_adamw.param_groups:
                g["lr"] = curr_adamw_lr * g.get("lr_mult", 1.0)

        input_ids = batch["input_ids"].to(env.device)
        targets = batch["targets"].to(env.device)
        weights = batch["weights"].to(env.device)
        content_mask = batch["content_cell_mask"].to(env.device)
        if env.is_tpu:
            mark_batch_sharding(input_ids, env.mesh)
            mark_batch_sharding(targets, env.mesh)
            mark_batch_sharding(weights, env.mesh)
            mark_batch_sharding(content_mask, env.mesh)

        if opt_muon is not None:
            opt_muon.zero_grad()
        opt_adamw.zero_grad()

        with torch.no_grad():
            enc_w_dtype = encoder.embeddings.weight.dtype
            with torch.autocast(device_type=amp_dev_type, dtype=amp_dtype, enabled=use_amp):
                V_0 = encoder(input_ids, weights.to(enc_w_dtype), canonicalize=False).float()

        V_in, true_sigma_cell, _ = corrupt_latent_cells(V_0, content_mask)

        if not is_dp:
            with torch.autocast(device_type=amp_dev_type, dtype=amp_dtype, enabled=use_amp):
                hidden_dec, radar = raw_dec.decode_hidden(V_in, return_radar=True)
            loss, metrics_t, _ = compute_phase2_decoder_loss_from_hidden(
                hidden_states=hidden_dec,
                decoder=raw_dec,
                radar=radar,
                targets=targets,
                true_sigma_cell=true_sigma_cell,
                pad_token_id=pad_token_id,
                seq_chunks=4,
                return_tensors=True,
            )
            del hidden_dec
        else:
            with torch.autocast(device_type=amp_dev_type, dtype=amp_dtype, enabled=use_amp):
                logits, radar = decoder(V_in, return_radar=True)
            # Вычисление функции потерь строго в float32 снаружи DataParallel (ADR 135)
            loss, metrics_t = compute_phase2_decoder_loss(
                logits=logits,
                radar=radar,
                targets=targets,
                true_sigma_cell=true_sigma_cell,
                pad_token_id=pad_token_id,
                return_tensors=True,
            )
            del logits

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt_adamw)
        else:
            loss.backward()

        if env.is_tpu:
            xm.reduce_gradients(opt_adamw)
            if opt_muon is not None:
                xm.reduce_gradients(opt_muon)

        # In-graph защита от не-конечного loss и градиентов без разрыва графа XLA
        is_loss_finite = torch.isfinite(loss)
        loss_mask_f = is_loss_finite.to(torch.float32)
        for p in trainable_params:
            if p.grad is not None:
                p.grad.nan_to_num_(nan=0.0, posinf=1.0, neginf=-1.0)
                p.grad.mul_(loss_mask_f.to(p.grad.dtype))

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        grad_norm_t = (
            grad_norm
            if torch.is_tensor(grad_norm)
            else torch.tensor(float(grad_norm), device=env.device, dtype=torch.float32)
        )
        is_safe = is_loss_finite & torch.isfinite(grad_norm_t)
        safe_mask_f = is_safe.to(torch.float32)
        for p in trainable_params:
            if p.grad is not None:
                p.grad.mul_(safe_mask_f.to(p.grad.dtype))

        if opt_muon is not None:
            opt_muon.step(is_safe=is_safe)

        if use_scaler:
            scaler.step(opt_adamw)
            scaler.update()
        else:
            opt_adamw.step()

        if ema is not None:
            ema.update(decoder, pullback_muon=pullback_muon, pullback_adamw=pullback_adamw, is_safe=is_safe)

        should_log = (global_step % log_every == 0) or (global_step == start_step + 1) or (global_step == max_steps)
        if should_log:
            stacked_metrics = torch.stack([metrics_t[k] for k in metric_names] + [grad_norm_t.detach(), is_safe.float()])

            def _emit_train_log(step_val: int, vec_t: torch.Tensor) -> None:
                nonlocal last_t, last_step
                vals = vec_t.cpu().tolist()
                m_dict = {name: float(vals[idx]) for idx, name in enumerate(metric_names)}
                gn_val = float(vals[len(metric_names)])
                safe_flag = float(vals[len(metric_names) + 1]) > 0.5
                last_metrics_scalars.update(m_dict)
                if not env.is_master:
                    return
                if not safe_flag:
                    print(f"[Warning Step {step_val}] Нечисловой loss/grad (GN={gn_val}). Шаг изолирован.", flush=True)
                    return
                now = time.time()
                sps = (step_val - last_step) / max(1e-5, now - last_t)
                last_t, last_step = now, step_val
                print(
                    f"Step {step_val:05d}/{max_steps:05d} | Loss: {m_dict['loss/total']:.4f} | "
                    f"CE_w: {m_dict['loss/ce_words']:.4f} | CE_p: {m_dict['loss/ce_slot_pad']:.4f} | "
                    f"AccW: {m_dict['acc/words']:.3f} | AccP: {m_dict['acc/slot_pad']:.3f} | "
                    f"RadarMAE: {m_dict['radar/mae_cell']:.3f} | GN: {gn_val:.2f} | {sps:.2f} step/s",
                    flush=True,
                )
                metrics_history.append({"step": step_val, "type": "train", **m_dict})
                if use_wandb:
                    import wandb

                    wandb.log({"step": step_val, **m_dict}, step=step_val)

            if env.is_tpu:
                xm.add_step_closure(_emit_train_log, args=(global_step, stacked_metrics))
            else:
                _emit_train_log(global_step, stacked_metrics)
            del stacked_metrics

        # Удаляем промежуточные тензоры шага ДО mark_step, чтобы не удерживать буферы в HBM
        del input_ids, targets, weights, content_mask, V_0, V_in, true_sigma_cell, radar
        del loss, metrics_t, grad_norm, grad_norm_t, is_loss_finite, loss_mask_f, is_safe, safe_mask_f
        if env.is_tpu:
            xm.mark_step()

        if val_dataloader is not None and (global_step % eval_every == 0 or global_step == max_steps):
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
            if env.is_master:
                print(
                    f"\n{'='*88}\n"
                    f"  [Val Step {global_step:05d}] Clean={val_m['val/clean_acc']:.3f} | "
                    f"RandO8={val_m['val/random_o8_acc']:.3f} | Gauss={val_m['val/gaussian_acc']:.3f} | "
                    f"MixOther={val_m['val/mix_other_acc']:.3f} | RadarMAE={val_m['val/mix_other_mae']:.3f}\n"
                    f"{'='*88}\n",
                    flush=True,
                )
                metrics_history.append({"step": global_step, "type": "val", **val_m})
                if use_wandb:
                    import wandb

                    wandb.log(val_m, step=global_step)

            if val_m["val/loss"] < best_val_loss:
                best_val_loss = val_m["val/loss"]
                best_path = os.path.join(env.checkpoint_dir, "phase2_decoder_best.pt")
                save_checkpoint(
                    decoder,
                    opt_muon,
                    opt_adamw,
                    global_step,
                    val_m,
                    best_path,
                    metrics_history,
                    ema=ema,
                    sync_gcs=sync_gcs,
                    gcs_dir=env.gcs_checkpoints_dir,
                    keep_local=True,
                    is_tpu=env.is_tpu,
                    is_master=env.is_master,
                )

        if global_step % save_every == 0 or global_step == max_steps:
            ckpt_path = os.path.join(env.checkpoint_dir, f"phase2_decoder_step_{global_step}.pt")
            save_checkpoint(
                decoder,
                opt_muon,
                opt_adamw,
                global_step,
                dict(last_metrics_scalars),
                ckpt_path,
                metrics_history,
                ema=ema,
                sync_gcs=sync_gcs,
                gcs_dir=env.gcs_checkpoints_dir,
                is_tpu=env.is_tpu,
                is_master=env.is_master,
            )

        global_step += 1

    if env.is_master:
        print("=== Обучение Phase 2 (Decoder & Noise Radar) завершено ===", flush=True)


# %% [markdown]
# ## 8. Точка входа CLI

# %%
def parse_args():
    parser = argparse.ArgumentParser(description="BEBLaDII Phase 2 Plan C (Generator-Decoder & Noise Radar)")
    parser.add_argument("--smoke_test", action="store_true", help="Быстрый CPU sanity-тест")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "tpu", "xla", "cpu"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_cells", type=int, default=128, help="Число концепт-клеток холста (128 * 4 = 512 слотов)")
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

    if os.environ.get("SMOKE_TEST") == "1":
        args.smoke_test = True

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
        env.num_workers = 0
        args.max_steps = int(os.environ.get("MAX_STEPS", 3))
        args.batch_size = 2
        args.num_cells = 4
        args.num_smooth_layers = 3
        args.log_every = 1
        args.eval_every = int(os.environ.get("VAL_STEPS", 2))
        args.eval_batches = 1
        args.save_every = args.max_steps
        args.lr_warmup_steps = 1
        args.wandb = "disabled"
        sync_gcs = False
        vocab_size = 2048

    use_wandb = False
    wandb_mode = args.wandb if args.wandb != "auto" else env.wandb_mode
    if env.is_master and wandb_mode == "online" and os.environ.get("WANDB_API_KEY", "").strip():
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

    ema = EMAModel(decoder, decay=args.ema_decay, device=torch.device("cpu")) if not args.no_ema else None
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
                ema.load_state_dict(ckpt["ema_state_dict"], device=torch.device("cpu"))
            start_step = int(ckpt.get("step", 0))
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
    dl_kwargs: Dict[str, Any] = {}
    if env.num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 4
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=env.num_workers,
        pin_memory=env.pin_memory,
        drop_last=True,
        **dl_kwargs,
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
    val_dl_kwargs: Dict[str, Any] = {}
    if env.num_workers > 0:
        val_dl_kwargs["persistent_workers"] = True
        val_dl_kwargs["prefetch_factor"] = 4
    val_dataloader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=env.num_workers,
        pin_memory=env.pin_memory,
        drop_last=True,
        **val_dl_kwargs,
    )

    if (not env.is_tpu) and env.device.type == "cuda" and torch.cuda.device_count() > 1:
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
    "num_cells": 128,
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
