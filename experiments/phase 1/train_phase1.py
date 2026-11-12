# %% [markdown]
# # BEBLaDII Phase 1 (Plan C) — Grassmannian Burer–Monteiro VAE & Hierarchical Decompressor
# 
# **Архитектура:** Plan C (Burer–Monteiro $V \in \mathbb{R}^{8 \times 128}$, $\rho = V^T V$, размер клетки $d=1024$)
# - Токенизатор Tekken (Mistral AI, $V = 131\,072$)
# - SVD-инициализированные Tied Word Embeddings ($131\,072 \times 1024$)
# - 2-слойный ModernBERT Encoder ($d=1024$, RoPE-128)
# - Grassmannian Token Normalizer (сжатие в шар $\mathbb{B}^{128}$, $\text{rk}_{\text{eff}} \ge 1.5$, Axis Dropout)
# - Grassmannian Spectral Pooler $4 \to 1$ (Partial RoPE 112:128, Lorentzian Autograd SVD, Sign Fixing)
# - $\mathcal{O}(8)$-инвариантный Decompressor Decoder (1 контекстный слой + Set-Cross-Attention $1 \to 4$ + 2 сглаживающих слоя + Tied LM Head)
# - Гибридный оптимизатор: Muon (2D-веса трансформера) + AdamW (эмбеддинги, 1D, RMSNorm)
# - Двухрежимный датасет: 85% связный текст + 15% словарный спектр `randperm(131k)`
# - Единый запуск: CPU sanity-тест (`--smoke_test`) и полномасштабное обучение на GPU (Kaggle T4/P100/A100)

# %% [markdown]
# ## 1. Автоопределение среды (Kaggle vs Local) и конфигурация

# %%
import os
import sys
import re
import math
import time
import glob
import subprocess
import argparse
import dataclasses
from typing import Dict, Any, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Вывод UTF-8 для Windows консоли
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def setup_kaggle_secrets(is_kaggle: bool):
    """Извлечение секретов WANDB_API_KEY и GCP_SA_JSON через Kaggle Secrets API."""
    if not is_kaggle:
        return
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        
        # 1. Извлечение ключа WandB
        try:
            wandb_key = user_secrets.get_secret("WANDB_API_KEY")
            if wandb_key:
                os.environ["WANDB_API_KEY"] = wandb_key.strip()
                print("[Kaggle Secrets] Секрет WANDB_API_KEY успешно получен из Kaggle Secrets.", flush=True)
        except Exception as e:
            print(f"[Kaggle Secrets Info] Секрет WANDB_API_KEY не найден в Kaggle Secrets ({e}).", flush=True)

        # 2. Извлечение сервисного аккаунта GCP для выгрузки чекпоинтов
        try:
            gcp_json = user_secrets.get_secret("GCP_SA_JSON")
            if gcp_json:
                key_path = "/tmp/gcp_sa_key.json"
                with open(key_path, "w", encoding="utf-8") as f:
                    f.write(gcp_json.strip())
                res = subprocess.run(
                    ["gcloud", "auth", "activate-service-account", f"--key-file={key_path}"],
                    capture_output=True,
                    text=True,
                )
                if res.returncode == 0:
                    print("[Kaggle Secrets] Авторизация GCP через GCP_SA_JSON успешно активирована.", flush=True)
                else:
                    print(f"[Kaggle Secrets Warning] Ошибка авторизации gcloud: {res.stderr.strip()}", flush=True)
        except Exception as e:
            print(f"[Kaggle Secrets Info] Секрет GCP_SA_JSON не найден или пропущен ({e}).", flush=True)

    except ImportError:
        pass
    except Exception as e:
        print(f"[Kaggle Secrets Warning] Ошибка при работе с UserSecretsClient: {e}", flush=True)


def warmup_linalg_backends() -> None:
    """
    Последовательный прогрев диспетчеров линейной алгебры (torch.linalg.eigh)
    на всех доступных GPU в главном потоке до запуска DataParallel.
    Устраняет race condition: 'RuntimeError: lazy wrapper should be called at most once'.
    """
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
        print("[Linalg Warmup] cuSOLVER успешно инициализирован на всех GPU.", flush=True)


@dataclasses.dataclass
class EnvironmentConfig:
    """Единая точка конфигурации путей и режимов для Kaggle и локальной среды."""
    is_kaggle: bool
    project_root: str
    data_dir: str
    output_dir: str
    checkpoint_dir: str
    embeddings_path: str
    stopwords_path: str
    data_path: str
    val_data_path: str
    gcs_checkpoints_dir: str
    wandb_mode: str
    device: torch.device
    dtype: torch.dtype
    num_workers: int
    pin_memory: bool

    @classmethod
    def detect(cls, override_device: Optional[str] = None) -> "EnvironmentConfig":
        is_kaggle = os.path.exists("/kaggle") or ("KAGGLE_KERNEL_RUN_TYPE" in os.environ)
        
        # Подключаем секреты Kaggle при их наличии
        setup_kaggle_secrets(is_kaggle)

        # Целевая папка в Google Cloud Storage для весов Plan C Phase 1
        gcs_checkpoints_dir = "gs://bebladii-weigths-us/planC/Phase 1/checkpoints/"

        if is_kaggle:
            project_root = "/kaggle/working/BEBLaDII"
            repo_url = "https://github.com/Laeryid/BEBLaDII.git"
            try:
                if not os.path.exists(project_root):
                    subprocess.run(["git", "clone", repo_url, project_root], check=False)
                else:
                    subprocess.run(["git", "-C", project_root, "pull"], check=False)
            except Exception as e:
                print(f"[Repo Sync Info] Не удалось выполнить git clone/pull: {e}", flush=True)

            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            data_dir = "/kaggle/input"
            output_dir = "/kaggle/working/output"
            checkpoint_dir = "/kaggle/working/checkpoints"
            embeddings_path = "/kaggle/input/bebladii-planc-phase1-data/data/pretrained_embeddings_1024.pt"
            stopwords_path = "/kaggle/input/bebladii-planc-phase1-data/data/tekken_stopwords_mask.pt"
            data_path = "/kaggle/input/bebladii-planb-phase3-data/phase 3/train_data/data/train.parquet"
            val_data_path = "/kaggle/input/bebladii-planb-phase3-data/phase 3/train_data/data/val.parquet"
            # Динамический автопоиск в /kaggle/input на случай изменения структуры датасета
            if os.path.exists(data_dir):
                for root, _, files in os.walk(data_dir):
                    if "pretrained_embeddings_1024.pt" in files:
                        embeddings_path = os.path.join(root, "pretrained_embeddings_1024.pt")
                    if "tekken_stopwords_mask.pt" in files:
                        stopwords_path = os.path.join(root, "tekken_stopwords_mask.pt")
                    if "train.parquet" in files and ("phase3" in root.lower() or "phase 3" in root.lower() or not os.path.exists(data_path)):
                        data_path = os.path.join(root, "train.parquet")
                    if "val.parquet" in files and ("phase3" in root.lower() or "phase 3" in root.lower() or not os.path.exists(val_data_path)):
                        val_data_path = os.path.join(root, "val.parquet")
            if not os.path.exists(val_data_path) and os.path.exists(data_path):
                alt_val = data_path.replace("train.parquet", "val.parquet")
                if os.path.exists(alt_val):
                    val_data_path = alt_val
            # Wandb в Kaggle: онлайн, если задан ключ WANDB_API_KEY, иначе offline/disabled
            wandb_mode = "online" if "WANDB_API_KEY" in os.environ else "disabled"
            num_workers = 2
            pin_memory = True
        else:
            # Локальная среда (Windows / Linux)
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            data_dir = os.path.join(project_root, "data")
            output_dir = os.path.join(project_root, "experiments", "phase 1", "output")
            checkpoint_dir = os.path.join(project_root, "experiments", "phase 1", "checkpoints")
            embeddings_path = os.path.join(data_dir, "plan C", "pretrained_embeddings_1024.pt")
            stopwords_path = os.path.join(data_dir, "plan C", "tekken_stopwords_mask.pt")
            data_path = os.path.join(project_root, "BEBLaDII-planB-Phase3-Data", "phase 3", "train_data", "data", "train.parquet")
            val_data_path = os.path.join(project_root, "BEBLaDII-planB-Phase3-Data", "phase 3", "train_data", "data", "val.parquet")
            wandb_mode = "disabled"
            num_workers = 0  # 0 безопаснее для Windows
            pin_memory = False

        if override_device and override_device != "auto":
            device = torch.device(override_device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # На GPU используем bfloat16 (для Ampere CC>=8.0) или float16 (для Turing T4 CC=7.5), на CPU - float32
        if device.type == "cuda":
            major, _ = torch.cuda.get_device_capability()
            dtype = torch.bfloat16 if (major >= 8 and torch.cuda.is_bf16_supported()) else torch.float16
            warmup_linalg_backends()
        else:
            dtype = torch.float32

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        return cls(
            is_kaggle=is_kaggle,
            project_root=project_root,
            data_dir=data_dir,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            embeddings_path=embeddings_path,
            stopwords_path=stopwords_path,
            data_path=data_path,
            val_data_path=val_data_path,
            gcs_checkpoints_dir=gcs_checkpoints_dir,
            wandb_mode=wandb_mode,
            device=device,
            dtype=dtype,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )


# %% [markdown]
# ## 2. Математические примитивы Burer–Monteiro и Lorentzian SVD

# %%
class LorentzianEigh(torch.autograd.Function):
    """
    Спектральное разложение симметричной матрицы с анизотропным сдвигом спектра
    и лоренцевой регуляризацией градиентов. Гарантирует отсутствие кратных корней в cuSOLVER на GPU.
    """
    @staticmethod
    def forward(ctx, G: torch.Tensor, eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = G.dtype
        G_f32 = 0.5 * (G.float() + G.float().transpose(-2, -1))
        K = G_f32.size(-1)

        # Строго монотонный диагональный сдвиг спектра: снимает любое вырождение (даже при G = 0)
        # и гарантирует отсутствие кратных корней для GPU cuSOLVER без CPU fallback
        diag_shift = torch.arange(1, K + 1, device=G_f32.device, dtype=torch.float32) * 1e-6
        G_reg = G_f32 + torch.diag(diag_shift)

        try:
            L_reg, V = torch.linalg.eigh(G_reg)
            is_valid = torch.isfinite(L_reg).all() and torch.isfinite(V).all()
        except Exception:
            is_valid = False

        if not is_valid:
            # Страховка на GPU: cuSOLVER в float64 (с машинной точностью 10^-16)
            try:
                L_64, V_64 = torch.linalg.eigh(G_reg.double())
                L_reg = L_64.float()
                V = V_64.float()
            except Exception:
                pass
            L_reg = torch.nan_to_num(L_reg, nan=1e-5, posinf=1.0, neginf=1e-5)
            V = torch.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)

        # Сохраняем L_reg (со сдвигом!) для backward, чтобы разности L_i - L_j никогда не были строго нулевыми
        ctx.save_for_backward(L_reg, V)
        ctx.eps = eps
        ctx.orig_dtype = orig_dtype

        # Для прямого прохода возвращаем очищенные неотрицательные собственные числа
        L_clean = torch.clamp(L_reg - diag_shift, min=1e-8)
        return L_clean.to(orig_dtype), V.to(orig_dtype)

    @staticmethod
    def backward(ctx, grad_L: torch.Tensor, grad_V: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        L_reg, V = ctx.saved_tensors
        eps = ctx.eps

        diff = L_reg.unsqueeze(-2) - L_reg.unsqueeze(-1)
        lorentz = diff / (diff ** 2 + eps ** 2)
        lorentz.diagonal(dim1=-2, dim2=-1).zero_()

        vt_gv = torch.matmul(V.transpose(-2, -1), grad_V.float())
        vt_gv_skew = 0.5 * (vt_gv - vt_gv.transpose(-2, -1))
        mid = torch.diag_embed(grad_L.float()) + lorentz * vt_gv_skew

        grad_G = torch.matmul(torch.matmul(V, mid), V.transpose(-2, -1))
        grad_G = 0.5 * (grad_G + grad_G.transpose(-2, -1))
        grad_G = torch.nan_to_num(grad_G, nan=0.0, posinf=0.0, neginf=0.0)
        return grad_G.to(ctx.orig_dtype), None


def safe_lorentzian_eigh(G: torch.Tensor, eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    return LorentzianEigh.apply(G, eps)


def apply_rope_16d(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """
    Partial RoPE для последних 16 измерений (8 пар координат) по плотной гармонической сетке:
    theta_k in linspace(pi/8, pi/2, 8).
    x: (..., L, K, 16)
    positions: (L,) целые числа позиций [0, 1, ..., L-1]
    """
    L = positions.shape[0]
    # 8 частот
    thetas = torch.linspace(math.pi / 8.0, math.pi / 2.0, steps=8, device=x.device, dtype=x.dtype)
    # углы: (L, 8)
    angles = positions.unsqueeze(-1).to(x.dtype) * thetas.unsqueeze(0)  # (L, 8)
    # Форма (L, 1, 8) для корректного бродкастинга по K осям
    sin = torch.sin(angles).unsqueeze(-2)  # (L, 1, 8)
    cos = torch.cos(angles).unsqueeze(-2)  # (L, 1, 8)

    # Решейпим x в 8 пар: (..., L, K, 8, 2)
    shape_prefix = x.shape[:-1]
    x_pairs = x.view(*shape_prefix, 8, 2)
    x0 = x_pairs[..., 0]  # (..., L, K, 8)
    x1 = x_pairs[..., 1]  # (..., L, K, 8)

    # Двумерный поворот каждой пары
    x0_rot = x0 * cos - x1 * sin
    x1_rot = x0 * sin + x1 * cos

    x_rot = torch.stack([x0_rot, x1_rot], dim=-1).view(*shape_prefix, 16)
    return x_rot


# %% [markdown]
# ## 3. Архитектурные блоки Burer–Monteiro (Plan C)

# %%
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x.float() ** 2, dim=-1, keepdim=True)
        normed = x * torch.rsqrt(var + self.eps).to(x.dtype)
        return normed * self.weight


class RotaryEmbedding128(nn.Module):
    """Стандартный RoPE для ModernBERT attention (head_dim=128)."""
    def __init__(self, dim: int = 128, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len].to(x.dtype), self.sin_cached[:seq_len].to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # q, k: (B, H, T, D)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class ModernBERTLayer(nn.Module):
    """
    Легкий слой трансформера архитектуры ModernBERT:
    - Pre-RMSNorm
    - Multi-Head Attention (d=1024, 8 heads, head_dim=128) с RoPE
    - SwiGLU MLP (1024 -> 2048 -> 1024)
    """
    def __init__(self, hidden_size: int = 1024, intermediate_size: int = 2048, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.attn_norm = RMSNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.mlp_norm = RMSNorm(hidden_size)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        # Pre-Norm Attention
        normed_x = self.attn_norm(x)
        q = self.q_proj(normed_x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed_x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed_x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # PyTorch SDPA (Scaled Dot-Product Attention)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.o_proj(attn_out)

        # SwiGLU MLP
        normed_mlp = self.mlp_norm(x)
        gate = F.silu(self.gate_proj(normed_mlp))
        up = self.up_proj(normed_mlp)
        mlp_out = self.down_proj(gate * up)
        x = x + mlp_out
        return x


class LexicalCoreMLP(nn.Module):
    """
    Ветка 1: Точечный (Token-wise) 2-слойный SwiGLU MLP.
    Работает строго по изолированному статическому эмбеддингу токена E[tok] (БЕЗ внимания к соседям).
    Нелинейно раскладывает 1D-вектор d=1024 на 8 ортогональных смысловых осей:
    - Первые 96 координат на ось (0:96) -> чистое семантическое ядро (v_sem).
    - Последние 16 координат на ось (112:128) -> сцепленный хвост под RoPE (v_pos).
    Всего выходов: 8 осей * (96 + 16) = 8 * 112 = 896.
    """
    def __init__(self, hidden_size: int = 1024, intermediate_size: int = 2048, num_axes: int = 8):
        super().__init__()
        self.num_axes = num_axes
        self.out_dim = num_axes * (96 + 16)  # 8 * 112 = 896
        
        self.norm = RMSNorm(hidden_size)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, self.out_dim, bias=False)
        self.out_norm = RMSNorm(self.out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, 1024)
        normed = self.norm(x)
        gate = F.silu(self.gate_proj(normed))
        up = self.up_proj(normed)
        h = self.out_norm(self.down_proj(gate * up))  # (B, T, 896)
        
        B, T, _ = h.shape
        h = h.view(B, T, self.num_axes, 112)
        h_sem = h[..., :96]  # (B, T, 8, 96) — лексическая семантика
        h_pos = h[..., 96:]  # (B, T, 8, 16) — сцепленный хвост для RoPE
        return h_sem, h_pos


class ContextSyntaxEncoder(nn.Module):
    """
    Ветка 2: Контекстный синтаксический энкодер (2 слоя ModernBERT с RoPE).
    Окно внимания видит соседние слова, предлоги и падежные согласования фразы.
    Выдает 16 координат контекстного синтаксиса на ось: v_syn in R^{8 x 16}.
    """
    def __init__(self, hidden_size: int = 1024, intermediate_size: int = 2048, num_axes: int = 8):
        super().__init__()
        self.num_axes = num_axes
        self.layers = nn.ModuleList([
            ModernBERTLayer(hidden_size, intermediate_size) for _ in range(2)
        ])
        self.final_norm = RMSNorm(hidden_size)
        self.syntax_proj = nn.Linear(hidden_size, num_axes * 16, bias=False)  # 1024 -> 128
        self.syntax_norm = RMSNorm(num_axes * 16)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1024)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.final_norm(x)
        h_syn = self.syntax_norm(self.syntax_proj(x))  # (B, T, 128)
        B, T, _ = h_syn.shape
        return h_syn.view(B, T, self.num_axes, 16)  # (B, T, 8, 16)


class BlockDiagonalVAEHead(nn.Module):
    """
    Блочно-диагональный mu-VAE проектор, гарантирующий абсолютную изолированность
    сегментов осей (0:96 семантика, 96:112 синтаксис, 112:128 RoPE-хвост) от перекрестной утечки:
    - 96 каналов семантики проецируются независимо (Linear(96, 96))
    - 16 каналов синтаксиса проецируются независимо (Linear(16, 16))
    - 16 каналов позиции проецируются независимо (Linear(16, 16))
    """
    def __init__(self, num_axes: int = 8):
        super().__init__()
        self.num_axes = num_axes
        
        # mu проекторы
        self.mu_sem = nn.Linear(96, 96)
        self.mu_syn = nn.Linear(16, 16)
        self.mu_pos = nn.Linear(16, 16)

        # logvar проекторы
        self.logvar_sem = nn.Linear(96, 96)
        self.logvar_syn = nn.Linear(16, 16)
        self.logvar_pos = nn.Linear(16, 16)

    def forward(self, h_sem: torch.Tensor, h_syn: torch.Tensor, h_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h_sem: (B, T, 8, 96), h_syn: (B, T, 8, 16), h_pos: (B, T, 8, 16)
        mu = torch.cat([self.mu_sem(h_sem), self.mu_syn(h_syn), self.mu_pos(h_pos)], dim=-1)  # (B, T, 8, 128)
        logvar = torch.cat([self.logvar_sem(h_sem), self.logvar_syn(h_syn), self.logvar_pos(h_pos)], dim=-1)  # (B, T, 8, 128)
        return mu, logvar


class GrassmannianTokenNormalizer(nn.Module):
    """
    Нормализатор токенов Бурера–Монтейро:
    1. Сжатие в единичный шар B^{128}: v_i = safe_squash(z_i), ||v_i||_2 in [0, 1).
    2. Расчет эффективного ранга: rk_eff(V) = ||V||_F^4 / ||V V^T||_F^2 >= 1.5.
    3. Axis Dropout: в training mode с вероятностью 20% зануляется главная собственная мода Gram(V).
    """
    def __init__(self, axis_dim: int = 128, num_axes: int = 8, rank_floor: float = 1.5, dropout_prob: float = 0.20):
        super().__init__()
        self.axis_dim = axis_dim
        self.num_axes = num_axes
        self.rank_floor = rank_floor
        self.dropout_prob = dropout_prob
        self.alpha = 1.0 / math.sqrt(axis_dim)

    def safe_squash(self, z: torch.Tensor) -> torch.Tensor:
        # z: (..., num_axes, axis_dim)
        norms = torch.norm(z, p=2, dim=-1, keepdim=True)
        # v_i = (z_i / (||z_i|| + eps)) * tanh(alpha * ||z_i||)
        direction = z / (norms + 1e-7)
        scale = torch.tanh(self.alpha * norms)
        return direction * scale

    def compute_rank_eff(self, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        V: (..., 8, 128)
        rk_eff = ||V||_F^4 / ||V V^T||_F^2
        """
        # Матрица Грама по 8 осям: G = V @ V^T in R^{8 x 8}
        G = torch.matmul(V, V.transpose(-2, -1))  # (..., 8, 8)
        # ||V||_F^2 = Tr(G)
        frob_sq = torch.diagonal(G, dim1=-2, dim2=-1).sum(dim=-1)  # (...)
        frob_4 = frob_sq ** 2
        # ||V V^T||_F^2 = Tr(G @ G) = sum(G_ij^2)
        gram_frob_sq = (G ** 2).sum(dim=[-2, -1])  # (...)
        rk_eff = frob_4 / (gram_frob_sq + 1e-7)
        return rk_eff, G

    def forward(
        self, z: torch.Tensor, non_pad_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z: (B, T, 8, 128)
        V = self.safe_squash(z)
        rk_eff, G = self.compute_rank_eff(V)
        
        # Функция штрафа за ранг: ReLU(1.5 - rk_eff)^2 (только для реальных не-PAD токенов)
        rank_loss_elem = F.relu(self.rank_floor - rk_eff) ** 2
        if non_pad_mask is not None:
            mask_f = non_pad_mask.float()
            rank_loss = (rank_loss_elem * mask_f).sum() / mask_f.sum().clamp(min=1.0)
        else:
            rank_loss = rank_loss_elem.mean()

        # Axis Dropout: в 20% батчей зануляем проекцию на главную моду Грама
        if self.training and self.dropout_prob > 0.0:
            if torch.rand(1).item() < self.dropout_prob:
                with torch.no_grad():
                    # Быстрый и 100% стабильный степенной метод (Power Iteration, 5 шагов)
                    # Исключает вызовы cuSOLVER/eigh на 2048 матрицах 8x8, гарантируя отсутствие NaN
                    q = G.new_ones(*G.shape[:-1], 1)
                    q = q / (torch.norm(q, dim=-2, keepdim=True) + 1e-7)
                    for _ in range(5):
                        q = torch.matmul(G, q)
                        q = q / (torch.norm(q, dim=-2, keepdim=True) + 1e-7)
                    q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
                    q_top = q.to(V.dtype)
                proj = torch.matmul(q_top.transpose(-2, -1), V)
                V = V - torch.matmul(q_top, proj)

        return V, rk_eff, rank_loss


class GrassmannianSpectralPooler(nn.Module):
    """
    Адаптивный спектральный пулинг окна 4 -> 1 по теореме Эккарта–Янга:
    - Partial RoPE в хвосте 112:128 (8 пар координат)
    - Взвешенное окно W_tilde in R^{32 x 128} (sum w_m = 1)
    - Матрица Грама окна G_win in R^{32 x 32} в float32 с Lorentzian Autograd
    - Топ-8 осей: V_cell = U_{[:, :8]}^T W_tilde in R^{8 x 128} (d=1024)
    - Sign Fixing: u_{k, argmax(|u_k|)} > 0
    """
    def __init__(self, chunk_size: int = 4, num_axes_per_tok: int = 8, out_axes: int = 8, axis_dim: int = 128):
        super().__init__()
        self.chunk_size = chunk_size
        self.num_axes_per_tok = num_axes_per_tok
        self.out_axes = out_axes
        self.axis_dim = axis_dim
        # Позиции токенов внутри окна: [0, 1, 2, 3]
        self.register_buffer("positions", torch.arange(chunk_size), persistent=False)

    def forward(self, V: torch.Tensor, weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        V: (B, N_cells, 4, 8, 128)
        weights: (B, N_cells, 4) неотрицательные веса аттенюации (по умолчанию равномерные 1.0)
        Возвращает:
            V_cell: (B, N_cells, 8, 128)
            singular_vals: (B, N_cells, 8)
        """
        B, N_cells, L, K, D = V.shape
        assert L == self.chunk_size and K == self.num_axes_per_tok and D == self.axis_dim

        # 1. Partial RoPE в хвосте 112:128
        v_sem = V[..., :112]  # (B, N_cells, 4, 8, 112)
        v_pos = V[..., 112:]  # (B, N_cells, 4, 8, 16)
        
        # Применяем RoPE к v_pos по позициям m in {0, 1, 2, 3}
        v_pos_rot = apply_rope_16d(v_pos, self.positions)
        V_tagged = torch.cat([v_sem, v_pos_rot], dim=-1)  # (B, N_cells, 4, 8, 128)

        # 2. Взвешивание окна
        if weights is None:
            # Равномерные веса 1/4
            w_norm = torch.full((B, N_cells, L, 1, 1), 1.0 / math.sqrt(L), device=V.device, dtype=V.dtype)
        else:
            # weights: (B, N_cells, 4)
            w_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-5)
            w_norm = torch.sqrt(weights / w_sum).unsqueeze(-1).unsqueeze(-1)  # (B, N_cells, 4, 1, 1)

        V_weighted = V_tagged * w_norm  # (B, N_cells, 4, 8, 128)
        # Разворачиваем окно 4 токена * 8 осей = 32 вектора:
        W_tilde = V_weighted.view(B, N_cells, L * K, D)  # (B, N_cells, 32, 128)

        # 3. Матрица Грама окна 32 x 32 в float32
        # G_win = W_tilde @ W_tilde^T in R^{32 x 32}
        G_win = torch.matmul(W_tilde.float(), W_tilde.float().transpose(-2, -1))  # (B, N_cells, 32, 32)
        # Симметризация для численной устойчивости
        G_win = 0.5 * (G_win + G_win.transpose(-2, -1))

        # 4. Lorentzian Eigh (дифференцируемый без NaN)
        eigvals, eigvecs = safe_lorentzian_eigh(G_win, eps=1e-4)  # eigvals: (..., 32), eigvecs: (..., 32, 32)

        # eigh возвращает собственные значения по возрастанию: берем топ-8 наибольших (хвост [-8:])
        top_eigvals = torch.flip(eigvals[..., -self.out_axes:], dims=[-1])  # (B, N_cells, 8)
        top_eigvecs = torch.flip(eigvecs[..., -self.out_axes:], dims=[-1])  # (B, N_cells, 32, 8)
        
        # Защита от отрицательных собственных чисел из-за численной погрешности
        top_eigvals = F.relu(top_eigvals)

        # 5. Sign Fixing Convention: u_{k, argmax(|u_k|)} > 0
        # top_eigvecs: (B, N_cells, 32, 8) -> транспонируем в (B, N_cells, 8, 32)
        u = top_eigvecs.transpose(-2, -1)  # (B, N_cells, 8, 32)
        max_idx = torch.argmax(torch.abs(u), dim=-1, keepdim=True)  # (B, N_cells, 8, 1)
        max_vals = torch.gather(u, dim=-1, index=max_idx)  # (B, N_cells, 8, 1)
        signs = torch.sign(max_vals)
        signs = torch.where(signs == 0.0, torch.ones_like(signs), signs)
        u_fixed = u * signs  # (B, N_cells, 8, 32)

        # 6. Базис клетки Эккарта–Янга: V_cell = u_fixed @ W_tilde in R^{8 x 128}
        # Математически v_k = u_k^T W_tilde уже взвешен собственным числом sqrt(lambda_k)!
        V_cell = torch.matmul(u_fixed.to(V.dtype), W_tilde)  # (B, N_cells, 8, 128)

        singular_vals = torch.sqrt(top_eigvals.clamp(min=1e-8)).to(V.dtype)
        return V_cell, singular_vals


class SetCrossAttentionUnpacker(nn.Module):
    """
    Архитектурно O(8)-инвариантный узел распаковки клетки 1 -> 4:
    - 4 обучаемых вектора-запроса Q in R^{4 x 128}
    - 8 осей клетки V in R^{8 x 128} как неупорядоченное множество
    - Cross-Attention: Softmax(Q V^T / sqrt(128)) V in R^{4 x 128}
    - Инъекция матрицы Грама G = V V^T in R^{8 x 8} (квадратичные спектральные инварианты клетки)
    - Проекция (128 + 128) -> 1024 + RMSNorm
    """
    def __init__(self, axis_dim: int = 128, num_slots: int = 4, hidden_size: int = 1024):
        super().__init__()
        self.axis_dim = axis_dim
        self.num_slots = num_slots
        self.hidden_size = hidden_size

        # 4 обучаемых слота распаковки
        self.query_slots = nn.Parameter(torch.randn(num_slots, axis_dim) * 0.02)
        self.q_norm = RMSNorm(axis_dim)
        self.k_norm = RMSNorm(axis_dim)

        # MLP для инъекции матрицы Грама 8 x 8 = 64
        self.gram_mlp = nn.Sequential(
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            RMSNorm(128),
        )

        # Проектор в скрытое состояние трансформера (128 + 128 -> 1024)
        self.out_proj = nn.Linear(axis_dim + 128, hidden_size)
        self.out_norm = RMSNorm(hidden_size)

    def forward(self, V_cell: torch.Tensor) -> torch.Tensor:
        """
        V_cell: (B, N_cells, 8, 128)
        Возвращает:
            H_unpacked: (B, N_cells * 4, 1024)
        """
        B, N, K, D = V_cell.shape
        # Q: (1, 1, 4, 128)
        Q = self.q_norm(self.query_slots).unsqueeze(0).unsqueeze(0).expand(B, N, self.num_slots, D).to(V_cell.dtype)
        # K, V: (B, N, 8, 128)
        K_mat = self.k_norm(V_cell)
        V_mat = V_cell

        # Внимание: (B, N, 4, 8)
        scores = torch.matmul(Q, K_mat.transpose(-2, -1)) / math.sqrt(D)
        attn = F.softmax(scores.float(), dim=-1).to(V_mat.dtype)
        h_attn = torch.matmul(attn, V_mat)  # (B, N, 4, 128)

        # Матрица Грама клетки: расчет G и gram_mlp строго в float32 с явным отключением autocast!
        # Без enabled=False PyTorch autocast принудительно даункастит torch.matmul и F.linear обратно в FP16,
        # вызывая переполнение G > 65504 (+inf) и inf - inf = NaN.
        device_type = V_cell.device.type
        target_dtype = V_cell.dtype
        with torch.autocast(device_type=device_type, enabled=False):
            V_f32 = V_cell.float()
            G_f32 = torch.matmul(V_f32, V_f32.transpose(-2, -1))  # (B, N, 8, 8)
            G_flat_f32 = G_f32.view(B, N, 64)

            # Выполняем проекцию gram_mlp строго в float32 (независимо от autocast/половинной точности весов)
            mlp_w0 = self.gram_mlp[0].weight.float()
            mlp_b0 = self.gram_mlp[0].bias.float() if self.gram_mlp[0].bias is not None else None
            h1 = F.silu(F.linear(G_flat_f32, mlp_w0, mlp_b0))
            mlp_w2 = self.gram_mlp[2].weight.float()
            mlp_b2 = self.gram_mlp[2].bias.float() if self.gram_mlp[2].bias is not None else None
            h2 = F.linear(h1, mlp_w2, mlp_b2)
            norm_w = self.gram_mlp[3].weight.float()
            variance = h2.pow(2).mean(-1, keepdim=True)
            h_normed = h2 * torch.rsqrt(variance + 1e-6) * norm_w
            h_gram = h_normed.to(target_dtype).unsqueeze(2).expand(B, N, self.num_slots, 128)

        # Конкатенация внимания и квадратичных инвариантов
        h_comb = torch.cat([h_attn, h_gram], dim=-1)  # (B, N, 4, 256)
        h_out = self.out_norm(self.out_proj(h_comb))  # (B, N, 4, 1024)

        # Разворачиваем в плоскую последовательность длины 4N: (B, 4N, 1024)
        return h_out.view(B, N * self.num_slots, self.hidden_size)


# %% [markdown]
# ## 4. Полная модель GrassmannianPhase1Model

# %%
class GrassmannianPhase1Model(nn.Module):
    """
    Полная архитектура автоэнкодера Фазы 1 Plan C:
    - Shared Word Embeddings E in R^{131072 x 1024} (tied с головой)
    - 2-слойный ModernBERT Encoder (d=1024)
    - mu-VAE Head -> сэмплирование z_raw in R^{8 x 128}
    - Grassmannian Token Normalizer (safe_squash, rank_floor >= 1.5, axis dropout)
    - Grassmannian Spectral Pooler (Partial RoPE, Lorentzian eigh 4 -> 1)
    - Декомпрессор:
      - Этап 1: 1 контекстный слой ModernBERT по N клеткам
      - Узел распаковки: SetCrossAttentionUnpacker 1 -> 4
      - Этап 2: 2 сглаживающих слоя ModernBERT по 4N токенам
    - Tied Output Head E^T -> logits (B, 4N, 131072)
    """
    def __init__(
        self,
        vocab_size: int = 131072,
        hidden_size: int = 1024,
        intermediate_size: int = 2048,
        num_axes: int = 8,
        axis_dim: int = 128,
        chunk_size: int = 4,
        pad_token_id: int = 10,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_axes = num_axes
        self.axis_dim = axis_dim
        self.chunk_size = chunk_size
        self.pad_token_id = pad_token_id

        # 1. Tied Word Embeddings
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.embed_norm = RMSNorm(hidden_size)

        # RoPE генератор для слоев трансформера
        self.rope = RotaryEmbedding128(dim=128, max_seq_len=4096)

        # 2. Split-Subspace Residual Encoder (Разделение семантики и синтаксиса)
        # Ветка 1: Точечный SwiGLU MLP (Лексическое ядро 0:96 + RoPE хвост 112:128)
        self.lexical_mlp = LexicalCoreMLP(hidden_size, intermediate_size, num_axes)
        # Ветка 2: Контекстный синтаксический энкодер (2 слоя ModernBERT для 96:112)
        self.context_encoder = ContextSyntaxEncoder(hidden_size, intermediate_size, num_axes)

        # 3. Блочно-диагональный mu-VAE Head (изолированные проекторы для 96, 16 и 16 каналов)
        self.vae_head = BlockDiagonalVAEHead(num_axes=num_axes)

        # 4. Нормализатор и Пулер
        self.normalizer = GrassmannianTokenNormalizer(axis_dim=axis_dim, num_axes=num_axes)
        self.pooler = GrassmannianSpectralPooler(chunk_size=chunk_size, num_axes_per_tok=num_axes)

        # 5. Декомпрессор Декодер:
        # Проектор клетки V_cell (8x128 = 1024) в контекстный слой холста
        self.cell_in_norm = RMSNorm(hidden_size)
        # Этап 1: 1 контекстный слой трансформера по холсту N
        self.dec_context_layer = ModernBERTLayer(hidden_size, intermediate_size)
        # Узел распаковки Set-Cross-Attention 1 -> 4
        self.unpacker = SetCrossAttentionUnpacker(axis_dim=axis_dim, num_slots=chunk_size, hidden_size=hidden_size)
        # Этап 2: 2 синтаксических сглаживающих слоя трансформера по полной длине 4N
        self.dec_smooth_layers = nn.ModuleList([
            ModernBERTLayer(hidden_size, intermediate_size) for _ in range(2)
        ])
        self.dec_final_norm = RMSNorm(hidden_size)

    def get_tied_head_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Линейная проекция через транспонированную матрицу эмбеддингов.
        Вычисляется строго в float32 (с временным отключением autocast) для исключения
        переполнения FP16 (>65504) при проекции на 131 072 токена.
        """
        device_type = hidden_states.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            h_f32 = hidden_states.float()
            w_f32 = self.embeddings.weight.float()
            return F.linear(h_f32, w_f32)

    def encode_and_pool(
        self, input_ids: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Прямой проход энкодера и пулинга.
        input_ids: (B, 4N)
        Возвращает:
            V_cell: (B, N, 8, 128)
            kl_loss: скаляр
            rank_loss: скаляр
            rk_eff: (B, 4N)
        """
        B, seq_len = input_ids.shape
        assert seq_len % self.chunk_size == 0, f"Seq len {seq_len} must be divisible by {self.chunk_size}"
        num_cells = seq_len // self.chunk_size

        # 1. Эмбеддинги
        x_embed = self.embed_norm(self.embeddings(input_ids))
        cos, sin = self.rope(x_embed, seq_len)

        # 2. Split-Subspace Residual Encoder:
        # ВЕТКА 1: Лексическое ядро и RoPE-хвост (Token-wise, БЕЗ внимания к соседям)
        h_sem, h_pos = self.lexical_mlp(x_embed)  # (B, 4N, 8, 96), (B, 4N, 8, 16)

        # ВЕТКА 2: Контекстный синтаксис (2 слоя ModernBERT с окном внимания)
        h_syn = self.context_encoder(x_embed, cos, sin)  # (B, 4N, 8, 16)

        # 3. Блочно-диагональный mu-VAE Head (полная изоляция подпространств)
        mu, logvar = self.vae_head(h_sem, h_syn, h_pos)  # (B, 4N, 8, 128)
        
        non_pad_mask = (input_ids != self.pad_token_id)  # (B, 4N)
        mask_4d = non_pad_mask.unsqueeze(-1).unsqueeze(-1).float()

        # Variance-Only KL считаем только по реальным (не-PAD) токенам
        var = torch.exp(torch.clamp(logvar, min=-10.0, max=10.0))
        kl_raw = -0.5 * (1.0 + logvar - var)
        kl_elem = torch.clamp(kl_raw, min=0.2) * mask_4d
        kl_loss = kl_elem.sum() / (mask_4d.sum() * self.num_axes * self.axis_dim).clamp(min=1.0)

        if self.training:
            eps = torch.randn_like(mu)
            z_raw = mu + eps * torch.sqrt(var)
        else:
            z_raw = mu

        # 4. Нормализатор токенов (сжатие в шар, rank floor с маскированием PAD)
        V_tok, rk_eff, rank_loss = self.normalizer(z_raw, non_pad_mask=non_pad_mask)  # (B, 4N, 8, 128)

        # Канонический невырожденный репер VOID для PAD токенов:
        # 8 взаимно ортогональных осей в синтаксическом/позиционном хвосте [104:120]
        # со строго убывающим спектром lambda_k = 0.01 * (0.6^k) -> нет кратных нулей в eigh!
        is_pad = ~non_pad_mask  # (B, 4N)
        if is_pad.any():
            V_pad = torch.zeros(self.num_axes, self.axis_dim, device=V_tok.device, dtype=V_tok.dtype)
            for k in range(self.num_axes):
                amp = math.sqrt(0.01 * (0.6 ** k))
                V_pad[k, 104 + k] = amp      # синтаксический карман [96:112]
                V_pad[k, 112 + k] = amp      # RoPE хвост [112:128] (снимает вырождение между 4 позициями окна!)
            pad_expanded = is_pad.unsqueeze(-1).unsqueeze(-1)  # (B, 4N, 1, 1)
            V_tok = torch.where(pad_expanded, V_pad, V_tok)
            rk_eff = torch.where(is_pad, torch.full_like(rk_eff, 2.0), rk_eff)

        # 5. Спектральный пулинг 4 -> 1
        V_tok_windows = V_tok.view(B, num_cells, self.chunk_size, self.num_axes, self.axis_dim)
        if weights is not None:
            weights_windows = weights.view(B, num_cells, self.chunk_size)
        else:
            weights_windows = None

        V_cell, _ = self.pooler(V_tok_windows, weights_windows)  # (B, N, 8, 128)

        # Stage-wise диагностика
        if self.training:
            if not torch.isfinite(z_raw).all():
                print("[STAGE-WISE NaN] 'z_raw' в VAE Head содержит нечисловые значения!", flush=True)
            if not torch.isfinite(V_tok).all():
                print("[STAGE-WISE NaN] 'V_tok' после normalizer содержит нечисловые значения!", flush=True)
            if not torch.isfinite(V_cell).all():
                print("[STAGE-WISE NaN] 'V_cell' после pooler содержит нечисловые значения!", flush=True)

        return V_cell, kl_loss, rank_loss, rk_eff

    def decode(self, V_cell: torch.Tensor) -> torch.Tensor:
        """
        Декомпрессия латентных концепт-клеток в токены текста:
        V_cell: (B, N, 8, 128)
        Возвращает:
            logits: (B, 4N, 131072)
        """
        B, N, K, D = V_cell.shape
        # Превращаем клетку 8x128 в вектор d=1024 для контекстного слоя
        cell_vec = self.cell_in_norm(V_cell.view(B, N, K * D))
        cos_N, sin_N = self.rope(cell_vec, N)

        # Этап 1: 1 контекстный слой ModernBERT по холсту концептов N
        h_cell = self.dec_context_layer(cell_vec, cos_N, sin_N)
        # Модулируем клетку контекстом холста
        V_cell_mod = V_cell + h_cell.view(B, N, K, D)

        # Узел распаковки Set-Cross-Attention 1 -> 4
        h_unpacked = self.unpacker(V_cell_mod)  # (B, 4N, 1024)

        # Этап 2: 2 сглаживающих слоя трансформера по полной длине 4N
        full_len = N * self.chunk_size
        cos_full, sin_full = self.rope(h_unpacked, full_len)
        x = h_unpacked
        for layer in self.dec_smooth_layers:
            x = layer(x, cos_full, sin_full)
        x = self.dec_final_norm(x)

        # Stage-wise диагностика декомпрессора
        if self.training:
            if not torch.isfinite(h_cell).all():
                print("[STAGE-WISE NaN] 'h_cell' в decode содержит нечисловые значения!", flush=True)
            if not torch.isfinite(h_unpacked).all():
                print("[STAGE-WISE NaN] 'h_unpacked' после unpacker содержит нечисловые значения!", flush=True)
            if not torch.isfinite(x).all():
                print("[STAGE-WISE NaN] 'x' после dec_smooth_layers содержит нечисловые значения!", flush=True)

        # Tied Output Head
        logits = self.get_tied_head_logits(x)  # (B, 4N, 131072)
        if self.training and not torch.isfinite(logits).all():
            print("[STAGE-WISE NaN] 'logits' в decode содержит нечисловые значения!", flush=True)
        return logits

    def forward(
        self, input_ids: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        V_cell, kl_loss, rank_loss, rk_eff = self.encode_and_pool(input_ids, weights)
        logits = self.decode(V_cell)
        return logits, V_cell, kl_loss, rank_loss, rk_eff


# %% [markdown]
# ## 5. Функции потерь Фазы 1 Plan C

# %%
def compute_prior_loss(V_cell: torch.Tensor) -> torch.Tensor:
    """
    Взвешенный Prior Loss (Изотропия 128 каналов):
    bar_rho_batch = (1 / (sum ||V||_F^2 + eps)) * sum V^T V in R^{128 x 128}.
    Штраф Huber на недиагональные элементы:
    L_prior = (2 / 128) * sum_{i < j} Huber(bar_rho_batch_{ij}, delta=0.01).
    """
    # V_cell: (B, N, 8, 128) -> объединяем строки осей: (B * N * 8, 128)
    axes = V_cell.view(-1, 128).float()
    frob_total = (axes ** 2).sum() + 1e-6
    # bar_rho: (128, 128)
    bar_rho = torch.matmul(axes.transpose(0, 1), axes) / frob_total

    # Верхний треугольник без диагонали
    triu_indices = torch.triu_indices(128, 128, offset=1, device=V_cell.device)
    off_diag = bar_rho[triu_indices[0], triu_indices[1]]
    
    # Huber loss с delta=0.01
    prior_loss = F.huber_loss(off_diag, torch.zeros_like(off_diag), delta=0.01, reduction="sum")
    prior_loss = (2.0 / 128.0) * prior_loss
    return prior_loss


def compute_grassmann_uniformity_loss(V_cell: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """
    Grassmannian Uniformity Loss (Расталкивание концепт-клеток):
    Tr(rho_A rho_B) = ||V_A V_B^T||_F^2.
    L_uniformity = log E_{A != B} [ exp(2t * ||V_A V_B^T||_F^2) ].
    """
    # V_cell: (B, N, 8, 128) -> собираем все клетки батча: (M, 8, 128)
    cells = V_cell.view(-1, 8, 128).float()
    M = cells.shape[0]
    if M <= 1:
        return torch.tensor(0.0, device=V_cell.device, dtype=V_cell.dtype)

    # Ограничиваем случайной подвыборкой для экономии памяти, если батч слишком велик
    max_m = 64
    if M > max_m:
        idx = torch.randperm(M, device=V_cell.device)[:max_m]
        cells = cells[idx]
        M = max_m

    # Взаимные скалярные произведения: M_AB = V_A @ V_B^T in R^{8 x 8}
    # cells: (M, 8, 128)
    # Tr(rho_A rho_B) = sum_{i,j} <v_{A,i}, v_{B,j}>^2
    # Вычисляем через матричное умножение:
    # dot_prods: (M, M, 8, 8)
    dot_prods = torch.einsum("mik,njk->mnij", cells, cells)
    frob_sq = (dot_prods ** 2).sum(dim=[-2, -1])  # (M, M)

    # Исключаем диагональ A == B
    mask = ~torch.eye(M, dtype=torch.bool, device=V_cell.device)
    off_diag_frob = frob_sq[mask]

    # LogSumExp для устойчивости
    uniformity_loss = torch.logsumexp(2.0 * t * off_diag_frob, dim=0) - math.log(off_diag_frob.numel())
    return uniformity_loss.to(V_cell.dtype)


def compute_phase1_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    V_cell: torch.Tensor,
    kl_loss: torch.Tensor,
    rank_loss: torch.Tensor,
    beta: float = 0.005,
    gamma: float = 0.05,
    delta: float = 0.05,
    eta: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    L_Phase1 = L_recon + beta * L_var_kl + gamma * L_rank_floor + delta * L_prior + eta * L_uniformity
    """
    # Гарантируем скалярность для сбора выходов DataParallel (тензоры размерности [num_gpus] -> скаляр)
    if kl_loss.ndim > 0:
        kl_loss = kl_loss.mean()
    if rank_loss.ndim > 0:
        rank_loss = rank_loss.mean()

    # 1. Recon Loss (CrossEntropy)
    valid_targets = (targets != -100).sum()
    if valid_targets > 0:
        recon_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
    else:
        recon_loss = logits.sum() * 0.0

    # 2. Prior Loss
    prior_loss = compute_prior_loss(V_cell)

    # 3. Uniformity Loss
    uniformity_loss = compute_grassmann_uniformity_loss(V_cell)

    # Суммарный лосс
    total_loss = (
        recon_loss
        + beta * kl_loss
        + gamma * rank_loss
        + delta * prior_loss
        + eta * uniformity_loss
    )

    metrics = {
        "loss/total": total_loss.item(),
        "loss/recon": recon_loss.item(),
        "loss/kl": kl_loss.item(),
        "loss/rank_floor": rank_loss.item(),
        "loss/prior": prior_loss.item(),
        "loss/uniformity": uniformity_loss.item(),
    }
    return total_loss, metrics


# %% [markdown]
# ## 6. Оптимизатор Muon + AdamW

# %%
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Ортогонализация градиента алгоритмом Ньютона–Шульца 5-го порядка (ядро оптимизатора Muon).
    G: 2D тензор весов (M, N).
    """
    orig_dtype = G.dtype
    X = G.bfloat16() if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else G.float()
    M, N = X.shape
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # Нормировка спектральной нормы
    norm = X.norm() + eps
    X = X / norm

    if M > N:
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if M > N:
        X = X.T

    return X.to(orig_dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon (Momentum Orthogonalized by Newton-Schulz) для 2D-параметров трансформера.
    """
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim != 2:
                    raise ValueError(f"Muon optimizer accepts only 2D parameters! Got shape {p.shape}")

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                elif state["momentum_buffer"].device != g.device:
                    state["momentum_buffer"] = state["momentum_buffer"].to(g.device)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                update = g.add(buf, alpha=momentum) if nesterov else buf
                update_ortho = zeropower_via_newtonschulz5(update, steps=ns_steps)
                
                # Масштабирование обновления с учетом геометрии параметров
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.data.add_(update_ortho, alpha=-lr * scale)


def create_hybrid_optimizers(
    model: GrassmannianPhase1Model,
    muon_lr: float = 0.02,
    adamw_lr: float = 3e-4,
    weight_decay: float = 0.01,
) -> Tuple[Muon, torch.optim.AdamW]:
    """
    Разделение параметров по ADR и плану эксперимента:
    - 2D-веса Attention и MLP -> Muon
    - Эмбеддинги, RMSNorm, bias, 1D-параметры, mu-VAE Head -> AdamW
    """
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Эмбеддинги всегда в AdamW
        if "embeddings" in name:
            adamw_params.append(param)
        elif "vae_" in name:
            adamw_params.append(param)
        elif "norm" in name or param.ndim < 2:
            adamw_params.append(param)
        elif param.ndim == 2:
            # 2D веса слоев трансформера
            muon_params.append(param)
        else:
            adamw_params.append(param)

    opt_muon = Muon(muon_params, lr=muon_lr) if muon_params else None
    opt_adamw = torch.optim.AdamW(adamw_params, lr=adamw_lr, weight_decay=weight_decay)

    return opt_muon, opt_adamw


class EMAModel:
    """
    Экспоненциальное скользящее среднее (EMA) параметров модели.
    Хранит теневые копии обучаемых параметров строго в FP32 (ADR 045, ADR 057)
    для предотвращения накопления ошибок округления при decay=0.999.
    Поддерживает как единичные модели, так и обертки DataParallel (через unwrap model.module).
    """
    def __init__(self, model: nn.Module, decay: float = 0.999, device: Optional[torch.device] = None):
        self.decay = decay
        self.device = device
        self.shadow = {}
        raw_model = getattr(model, "module", model)
        for name, param in raw_model.named_parameters():
            if param.requires_grad:
                t = param.detach().clone().float()
                if device is not None:
                    t = t.to(device)
                self.shadow[name] = t
        self.backup = {}

    def to(self, device: torch.device):
        """Перемещение всех теневых весов на целевое устройство."""
        self.device = device
        for k in self.shadow:
            self.shadow[k] = self.shadow[k].to(device)
        return self

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Обновление теневых весов: shadow = decay * shadow + (1 - decay) * param."""
        one_minus_decay = 1.0 - self.decay
        raw_model = getattr(model, "module", model)
        for name, param in raw_model.named_parameters():
            if param.requires_grad and name in self.shadow:
                s = self.shadow[name]
                if s.device != param.device:
                    s = s.to(param.device)
                    self.shadow[name] = s
                s.lerp_(param.detach().float(), one_minus_decay)

    def apply_shadow(self, model: nn.Module):
        """Подменяет веса модели на теневые EMA перед валидацией или сохранением."""
        self.backup = {}
        raw_model = getattr(model, "module", model)
        for name, param in raw_model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name].to(device=param.device, dtype=param.dtype))

    def restore(self, model: nn.Module):
        """Восстанавливает горячие обучаемые веса модели."""
        raw_model = getattr(model, "module", model)
        for name, param in raw_model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        for k, v in state_dict.items():
            if k in self.shadow:
                target_dev = self.shadow[k].device
                self.shadow[k].copy_(v.to(target_dev))


# %% [markdown]
# ## 7. Двухрежимный датасет (Dual-Regime Dataset) с шаблонами Режима Б и взвешиванием стоп-слов

# %%
REGIME_B_TEMPLATES = [
    # 1. Русские словарные связки:
    [29, 28705, 1205],     # ":", " ", "это"
    [3412, 1102, 1205],    # "слово", "—", "это"
    [5123, 3412, 29],      # "значение", "слова", ":"
    [1050, 412, 3412],     # "термин", "в", "контексте"
    [28725, 1205, 1102],   # ",", "это", "—"
    # 2. Кавычки и знаки препинания:
    [171, 187, 28723],     # "«", "»", "."
    [187, 28723, 13],      # "»", ".", "\n"
    [28725, 171, 187],     # ",", "«", "»"
    [28723, 13, 13],       # ".", "\n", "\n"
    # 3. Английский словарный контекст:
    [415, 264, 1820],      # "is", "a", "term"
    [456, 304, 415],       # "The", "meaning", "is"
    [395, 412, 456],       # "defined", "as", "the"
    [264, 3412, 29],       # "a", "word", ":"
    # 4. Код / форматирование:
    [28, 120, 26],         # "=", "value", ";"
    [105, 29, 28725],      # "id", ":", ","
    [512, 28723, 15],      # "item", ".", "\n"
]


def load_tekken_tokenizer(env: EnvironmentConfig) -> Optional[Any]:
    """Загрузка токенизатора Tekken из локального файла, кэша или Hugging Face Hub."""
    candidates = [
        os.path.join(env.project_root, "data", "plan C", "tokenizer.json"),
        os.path.join(env.data_dir, "bebladii-planc-phase1-data", "tokenizer.json"),
        os.path.join(env.data_dir, "tokenizer.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                from tokenizers import Tokenizer
                print(f"[Tokenizer] Загрузка токенизатора из локального пути: {c}", flush=True)
                return Tokenizer.from_file(c)
            except Exception:
                pass

    if os.path.exists(env.data_dir):
        for root, _, files in os.walk(env.data_dir):
            if "tokenizer.json" in files:
                try:
                    from tokenizers import Tokenizer
                    p = os.path.join(root, "tokenizer.json")
                    print(f"[Tokenizer] Найден токенизатор в датасете: {p}", flush=True)
                    return Tokenizer.from_file(p)
                except Exception:
                    pass

    try:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        p = hf_hub_download(repo_id="mistralai/Mistral-Nemo-Base-2407", filename="tokenizer.json")
        print(f"[Tokenizer] Загружен токенизатор из кэша HF: {p}", flush=True)
        return Tokenizer.from_file(p)
    except Exception as e:
        print(f"[Tokenizer Warning] Не удалось загрузить Tekken tokenizer.json: {e}", flush=True)
        return None


def build_punctuation_id_set(tokenizer: Optional[Any]) -> set:
    """Множество token_id, содержащих знаки завершения предложений (. ! ? \\n)."""
    punct_ids = set()
    if tokenizer is not None:
        try:
            vocab = tokenizer.get_vocab()
            for text, tid in vocab.items():
                if any(p in text for p in [".", "!", "?", "\n"]):
                    punct_ids.add(tid)
        except Exception:
            pass
    # Базовые ID знаков препинания Tekken по умолчанию
    default_puncts = {1046, 2247, 1063, 1047, 13, 28723, 28725, 120, 26, 29}
    punct_ids.update(default_puncts)
    return punct_ids


class DualRegimeDataset(Dataset):
    """
    Двухрежимный датасет Фазы 1 Plan C:
    - 85% Режим А: Связный текст из train.parquet (BEBLaDII-planB-Phase3-Data).
      - Обрезка по последней точке/знаку препинания до границы seq_len (256 токенов).
      - Дозаполнение до seq_len токенами PAD (pad_token_id = 10).
      - Двухуровневое взвешивание:
        - Уровень 1 (смешанное окно): w_content = 1.0, w_stop = 0.02, w_pad = 0.001 -> вклад PAD < 0.1% энергии.
        - Уровень 2 (окно чистого PAD): веса по 0.25 каждому -> чистая VOID-клетка без деления на ноль.
      - Цели: targets[is_pad] = -100 (маскирование в CrossEntropy).
    - 15% Режим Б: Словарный спектр (16 разнообразных шаблонов, 50% вес на редкий токен, по 16.7% на связки).
    """
    def __init__(
        self,
        parquet_path: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        seq_len: int = 256,
        vocab_size: int = 131072,
        chunk_size: int = 4,
        regime_b_ratio: float = 0.15,
        eps_stop: float = 0.02,
        eps_pad: float = 0.001,
        pad_token_id: int = 10,
        stopword_mask: Optional[torch.Tensor] = None,
        virtual_length: int = 50000,
        is_validation: bool = False,
    ):
        assert seq_len % chunk_size == 0, f"seq_len {seq_len} must be divisible by chunk_size {chunk_size}"
        self.seq_len = seq_len
        self.chunk_size = chunk_size
        self.num_chunks = seq_len // chunk_size
        self.vocab_size = vocab_size
        self.regime_b_ratio = regime_b_ratio
        self.eps_stop = eps_stop
        self.eps_pad = eps_pad
        self.pad_token_id = pad_token_id
        self.virtual_length = virtual_length
        self.tokenizer = tokenizer
        self.is_validation = is_validation

        # Загрузка текстов из Parquet
        self.texts = []
        if parquet_path and os.path.exists(parquet_path):
            try:
                import pandas as pd
                prefix = "[Val Dataset]" if is_validation else "[Dataset]"
                print(f"{prefix} Загрузка текстов из: {parquet_path} ...", flush=True)
                df = pd.read_parquet(parquet_path, columns=["text"])
                self.texts = df["text"].dropna().tolist()
                print(f"{prefix} Загружено {len(self.texts)} строк текста.", flush=True)
            except Exception as e:
                print(f"[Dataset Warning] Ошибка чтения {parquet_path}: {e}", flush=True)

        self.has_real_data = len(self.texts) > 0
        self.punct_ids = build_punctuation_id_set(self.tokenizer)

        # Маска стоп-слов / связок
        if stopword_mask is not None:
            self.stopword_mask = stopword_mask.bool()
        else:
            self.stopword_mask = torch.zeros(vocab_size, dtype=torch.bool)
            safe_limit = min(64, vocab_size)
            self.stopword_mask[:safe_limit] = True

        self.templates = REGIME_B_TEMPLATES

        # Инспекция первого сэмпла при инициализации обучающего датасета
        if not is_validation:
            self.inspect_first_sample()

    def inspect_first_sample(self):
        """Инспекция и подробное логирование первого сэмпла датасета для контроля качества данных."""
        if not self.has_real_data:
            print("[Dataset Inspection] Используются синтетические данные (Режим А/Б).", flush=True)
            return

        print("\n" + "=" * 80, flush=True)
        print("=== ИНСПЕКЦИЯ ПЕРВОГО СЭМПЛА ДАТАСЕТА (SAMPLE #0) ===", flush=True)
        raw_text = self.texts[0]
        preview = (raw_text[:250] + "...") if len(raw_text) > 250 else raw_text
        print(f"1. Исходный текст ({len(raw_text)} символов):\n   \"{preview}\"", flush=True)

        if self.tokenizer is not None:
            enc = self.tokenizer.encode(raw_text)
            total_tokens = len(enc.ids)
            sample_tokens = enc.ids[: self.seq_len]
            print(f"2. Токенизация: всего токенов в тексте = {total_tokens}, окно для анализа = {len(sample_tokens)} токенов.", flush=True)

            last_dot = -1
            for i in range(len(sample_tokens) - 1, -1, -1):
                if sample_tokens[i] in self.punct_ids:
                    last_dot = i
                    break

            cut_len = (last_dot + 1) if last_dot >= 4 else len(sample_tokens)
            pad_len = self.seq_len - cut_len
            print(f"3. Обрезка по последней точке: позиция точки = {last_dot}, длина сэмпла L_cut = {cut_len}, PAD токенов в хвосте = {pad_len}.", flush=True)

            first_8_decoded = [self.tokenizer.decode([t]) for t in sample_tokens[:8]]
            print(f"4. Первые 8 токенов текста: {first_8_decoded}", flush=True)

            # Получаем реальный сэмпл через __getitem__ в Режиме А
            saved_ratio = self.regime_b_ratio
            self.regime_b_ratio = 0.0  # принудительно Режим А для инспекции
            sample_item = self.__getitem__(0)
            self.regime_b_ratio = saved_ratio

            inp = sample_item["input_ids"]
            w = sample_item["weights"]
            tgt = sample_item["targets"]

            if cut_len >= 4:
                last_win_idx = max(0, (cut_len // 4 - 1) * 4)
                last_win_tokens = inp[last_win_idx : last_win_idx + 4].tolist()
                last_win_w = [round(x, 4) for x in w[last_win_idx : last_win_idx + 4].tolist()]
                print(f"5. Окно текста перед PAD (токены {last_win_idx}..{last_win_idx+3}): id={last_win_tokens}, веса={last_win_w}, сумма={sum(last_win_w):.4f}", flush=True)

            if pad_len >= 4:
                pad_win_idx = self.seq_len - 4
                pad_win_tokens = inp[pad_win_idx : pad_win_idx + 4].tolist()
                pad_win_w = [round(x, 4) for x in w[pad_win_idx : pad_win_idx + 4].tolist()]
                print(f"6. Окно чистого PAD (токены {pad_win_idx}..{pad_win_idx+3}): id={pad_win_tokens}, веса={pad_win_w}, сумма={sum(pad_win_w):.4f} (цель={tgt[pad_win_idx].item()})", flush=True)

        print("=" * 80 + "\n", flush=True)

    def __len__(self) -> int:
        if self.has_real_data:
            return len(self.texts)
        return self.virtual_length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        is_regime_b = torch.rand(1).item() < self.regime_b_ratio

        if is_regime_b:
            # === РЕЖИМ Б: Словарный спектр (шаблоны, веса 50% / 16.7%) ===
            input_ids = torch.zeros(self.seq_len, dtype=torch.long)
            weights = torch.zeros(self.seq_len, dtype=torch.float32)

            for chunk_idx in range(self.num_chunks):
                start = chunk_idx * self.chunk_size
                dict_pos = torch.randint(0, self.chunk_size, (1,)).item()
                dict_token = torch.randint(0, self.vocab_size, (1,)).item()
                tmpl_idx = torch.randint(0, len(self.templates), (1,)).item()
                tmpl = self.templates[tmpl_idx]

                input_ids[start + dict_pos] = dict_token
                weights[start + dict_pos] = 0.50

                tmpl_t_idx = 0
                for offset in range(self.chunk_size):
                    if offset != dict_pos:
                        t_val = tmpl[tmpl_t_idx] % self.vocab_size
                        input_ids[start + offset] = t_val
                        weights[start + offset] = 0.50 / (self.chunk_size - 1)
                        tmpl_t_idx += 1

            targets = input_ids.clone()

        else:
            # === РЕЖИМ А: Связный текст с обрезкой по последней точке и двухуровневой PAD-маской ===
            if self.has_real_data and self.tokenizer is not None:
                text = self.texts[idx % len(self.texts)]
                enc = self.tokenizer.encode(text)
                raw_token_ids = enc.ids
            else:
                max_tok = min(self.vocab_size, 5000)
                min_tok = min(10, max_tok - 1)
                raw_token_ids = torch.randint(min_tok, max_tok, (self.seq_len,)).tolist()

            # 1. Берем срез до границы seq_len
            sample_tokens = raw_token_ids[: self.seq_len]

            # 2. Ищем позицию последней точки/знака завершения
            last_dot = -1
            for i in range(len(sample_tokens) - 1, -1, -1):
                if sample_tokens[i] in self.punct_ids:
                    last_dot = i
                    break

            if last_dot >= 4:
                cut_tokens = sample_tokens[: last_dot + 1]
            else:
                cut_tokens = sample_tokens

            # 3. Дополняем до seq_len токенами PAD
            n_real = len(cut_tokens)
            input_ids = torch.full((self.seq_len,), self.pad_token_id, dtype=torch.long)
            if n_real > 0:
                input_ids[:n_real] = torch.tensor(cut_tokens, dtype=torch.long)

            # Цели: для PAD выставляем -100 (маскирование в CrossEntropy)
            targets = input_ids.clone()
            targets[n_real:] = -100

            # 4. Двухуровневое взвешивание окон по 4 токена
            weights = torch.zeros(self.seq_len, dtype=torch.float32)
            for chunk_idx in range(self.num_chunks):
                start = chunk_idx * self.chunk_size
                window_tokens = input_ids[start : start + self.chunk_size]

                is_pad = (window_tokens == self.pad_token_id)
                valid_mask_idx = torch.clamp(window_tokens, 0, len(self.stopword_mask) - 1)
                is_stop = self.stopword_mask[valid_mask_idx] & (~is_pad)

                # Базовые сырые веса: 1.0 смысловые, 0.02 стоп-слова, 0.001 PAD
                raw_w = torch.where(
                    is_pad,
                    torch.tensor(self.eps_pad, dtype=torch.float32),
                    torch.where(
                        is_stop,
                        torch.tensor(self.eps_stop, dtype=torch.float32),
                        torch.tensor(1.0, dtype=torch.float32),
                    ),
                )
                # Нормировка окна (sum(w) = 1.0)
                weights[start : start + self.chunk_size] = raw_w / raw_w.sum()

        return {
            "input_ids": input_ids,
            "targets": targets,
            "weights": weights,
            "is_regime_b": torch.tensor(1.0 if is_regime_b else 0.0),
        }


# %% [markdown]
# ## 8. Главный цикл обучения (Engine)

# %%
def init_embeddings(model: GrassmannianPhase1Model, env: EnvironmentConfig):
    """Инициализация эмбеддингов весами Truncated SVD (5120 -> 1024) из Mistral NeMo."""
    if os.path.exists(env.embeddings_path):
        print(f"[Init] Загрузка SVD-эмбеддингов из: {env.embeddings_path}")
        emb_weights = torch.load(env.embeddings_path, map_location="cpu", weights_only=True)
        if isinstance(emb_weights, dict) and "weight" in emb_weights:
            emb_weights = emb_weights["weight"]
        with torch.no_grad():
            model.embeddings.weight.copy_(emb_weights)
        print("   -> SVD-эмбеддинги успешно применены.")
    else:
        print(f"[Init] Предобученный файл эмбеддингов не найден по пути: {env.embeddings_path}")
        print("   -> Инициализация эмбеддингов с дисперсией 1/sqrt(d)...")
        nn.init.normal_(model.embeddings.weight, mean=0.0, std=1.0 / math.sqrt(model.hidden_size))


def sync_to_gcs(local_path: str, gcs_dir: str) -> bool:
    """
    Синхронизация чекпоинта с Google Cloud Storage (gsutil / gcloud storage).
    Возвращает True только при успешном завершении передачи.
    """
    if not os.path.exists(local_path):
        return False
    filename = os.path.basename(local_path)
    # Нормализуем путь в GCS (с учетом завершающего слэша)
    target_gcs = gcs_dir.rstrip("/") + "/" + filename

    # 1. Попытка через gsutil cp
    try:
        res = subprocess.run(["gsutil", "cp", local_path, target_gcs], capture_output=True, text=True)
        if res.returncode == 0:
            print(f"[GCS] Успешно синхронизирован чекпоинт в: {target_gcs}", flush=True)
            return True
    except Exception:
        pass

    # 2. Fallback через gcloud storage cp
    try:
        res = subprocess.run(["gcloud", "storage", "cp", local_path, target_gcs], capture_output=True, text=True)
        if res.returncode == 0:
            print(f"[GCS] Успешно синхронизирован через gcloud в: {target_gcs}", flush=True)
            return True
    except Exception:
        pass

    print(f"[GCS Warning] Не удалось скопировать {filename} в {target_gcs}. Локальный файл сохранен.", flush=True)
    return False


def find_latest_checkpoint(checkpoint_dir: str, gcs_dir: Optional[str] = None) -> Optional[str]:
    """Поиск последнего доступного чекпоинта локально или в GCS."""
    # 1. Локальный поиск
    local_files = glob.glob(os.path.join(checkpoint_dir, "phase1_step_*.pt"))
    if local_files:
        def extract_step(f):
            nums = [int(s) for s in re.findall(r"\d+", os.path.basename(f))]
            return nums[-1] if nums else 0
        local_files.sort(key=extract_step)
        return local_files[-1]

    # 2. Поиск в GCS, если локально не найдено
    if gcs_dir:
        try:
            target_pattern = gcs_dir.rstrip("/") + "/phase1_step_*.pt"
            res = subprocess.run(["gsutil", "ls", target_pattern], capture_output=True, text=True)
            if res.returncode == 0 and res.stdout.strip():
                gcs_lines = [line.strip() for line in res.stdout.strip().split("\n") if line.strip()]
                if gcs_lines:
                    def extract_step_gcs(f):
                        nums = [int(s) for s in re.findall(r"\d+", os.path.basename(f))]
                        return nums[-1] if nums else 0
                    gcs_lines.sort(key=extract_step_gcs)
                    latest_gcs = gcs_lines[-1]
                    local_dest = os.path.join(checkpoint_dir, os.path.basename(latest_gcs))
                    print(f"[GCS] Скачивание последнего чекпоинта из {latest_gcs} в {local_dest}...", flush=True)
                    dl_res = subprocess.run(["gsutil", "cp", latest_gcs, local_dest], capture_output=True, text=True)
                    if dl_res.returncode == 0 and os.path.exists(local_dest):
                        return local_dest
        except Exception as e:
            print(f"[GCS] Ошибка при проверке удаленных чекпоинтов: {e}", flush=True)

    return None


def save_checkpoint(
    model: nn.Module,
    opt_muon: Optional[Muon],
    opt_adamw: torch.optim.AdamW,
    step: int,
    metrics: Dict[str, float],
    path: str,
    metrics_history: Optional[List[Dict[str, Any]]] = None,
    ema: Optional[EMAModel] = None,
    sync_gcs: bool = False,
    gcs_dir: str = "",
    keep_local: bool = False,
):
    raw_model = getattr(model, "module", model)
    checkpoint = {
        "step": step,
        "model_state_dict": raw_model.state_dict(),
        "opt_adamw_state_dict": opt_adamw.state_dict(),
        "metrics": metrics,
        "metrics_history": metrics_history or [],
        "ema_state_dict": ema.state_dict() if ema is not None else None,
    }
    if opt_muon is not None:
        checkpoint["opt_muon_state_dict"] = opt_muon.state_dict()
    torch.save(checkpoint, path)
    hist_len = len(checkpoint["metrics_history"])
    print(f"[Checkpoint] Сохранено состояние на шаге {step} в: {path} (история: {hist_len} записей, EMA: {'Да' if ema is not None else 'Нет'})", flush=True)

    if sync_gcs and gcs_dir:
        success = sync_to_gcs(path, gcs_dir)
        if success and not keep_local:
            try:
                os.remove(path)
                print(f"[Storage Cleanup] Локальный файл {os.path.basename(path)} удален после выгрузки на GCP для экономии диска Kaggle.", flush=True)
            except Exception as e:
                print(f"[Storage Warning] Не удалось удалить локальный файл {path}: {e}", flush=True)


def load_checkpoint(
    path: str,
    model: nn.Module,
    opt_muon: Optional[Muon],
    opt_adamw: torch.optim.AdamW,
    device: torch.device,
    ema: Optional[EMAModel] = None,
) -> Tuple[int, List[Dict[str, Any]]]:
    """Восстановление состояния модели, оптимизаторов и истории метрик из чекпоинта."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"[Resume] Файл чекпоинта не найден: {path}")

    print(f"[Resume] Загрузка чекпоинта из: {path} ...", flush=True)
    checkpoint = torch.load(path, map_location=device)

    raw_model = getattr(model, "module", model)
    # Проверка совпадения словаря эмбеддингов
    ckpt_sd = checkpoint["model_state_dict"]
    ckpt_embed = ckpt_sd.get("embeddings.weight", None)
    if ckpt_embed is not None and ckpt_embed.shape != raw_model.embeddings.weight.shape:
        print(f"[Resume Warning] Несовпадение словаря эмбеддингов: чекпоинт {ckpt_embed.shape} != модель {raw_model.embeddings.weight.shape}. Пропускаем embeddings.", flush=True)
        del ckpt_sd["embeddings.weight"]
        raw_model.load_state_dict(ckpt_sd, strict=False)
    else:
        raw_model.load_state_dict(ckpt_sd)

    try:
        opt_adamw.load_state_dict(checkpoint["opt_adamw_state_dict"])
        for state in opt_adamw.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    except Exception as e:
        print(f"[Resume Warning] Не удалось восстановить состояние AdamW: {e}", flush=True)

    if opt_muon is not None and "opt_muon_state_dict" in checkpoint:
        try:
            opt_muon.load_state_dict(checkpoint["opt_muon_state_dict"])
            for state in opt_muon.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        except Exception as e:
            print(f"[Resume Warning] Не удалось полностью восстановить состояние Muon: {e}", flush=True)

    if ema is not None and "ema_state_dict" in checkpoint and checkpoint["ema_state_dict"] is not None:
        try:
            ema.load_state_dict(checkpoint["ema_state_dict"])
            print("[Resume] Успешно восстановлены теневые веса EMA!", flush=True)
        except Exception as e:
            print(f"[Resume Warning] Не удалось восстановить EMA веса: {e}", flush=True)

    start_step = checkpoint.get("step", 0)
    metrics_history = checkpoint.get("metrics_history", [])
    print(f"[Resume] Успешно восстановлено состояние с шага {start_step} (история: {len(metrics_history)} записей)!", flush=True)
    return start_step, metrics_history


def evaluate(
    model: nn.Module,
    val_dataloader: DataLoader,
    env: EnvironmentConfig,
    eval_batches: int = 20,
    ema: Optional[EMAModel] = None,
) -> Dict[str, float]:
    """
    Прогон валидации на отложенном датасете (val.parquet) в режиме eval() и no_grad().
    При наличии EMA валидация выполняется строго на теневых весах EMA (ADR 045).
    """
    if ema is not None:
        ema.apply_shadow(model)
    model.eval()

    total_loss_accum = 0.0
    recon_accum = 0.0
    kl_accum = 0.0
    prior_accum = 0.0
    rank_floor_accum = 0.0
    uniformity_accum = 0.0
    rk_eff_accum = 0.0
    batches_counted = 0

    use_amp = (env.device.type == "cuda")
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            if i >= eval_batches:
                break
            input_ids = batch["input_ids"].to(env.device)
            targets = batch["targets"].to(env.device)
            weights = batch["weights"].to(env.device)

            with torch.autocast(device_type=env.device.type, dtype=env.dtype, enabled=use_amp):
                logits, V_cell, kl_loss, rank_loss, rk_eff = model(input_ids, weights)
                loss, metrics = compute_phase1_loss(
                    logits=logits,
                    targets=targets,
                    V_cell=V_cell,
                    kl_loss=kl_loss,
                    rank_loss=rank_loss,
                )

            if not torch.isfinite(loss):
                print(f"[Val Warning] Пропуск сэмпла с нечисловым val_loss: {loss.item()}", flush=True)
                continue

            total_loss_accum += loss.item()
            recon_accum += metrics.get("loss/recon", 0.0)
            kl_accum += metrics.get("loss/kl", 0.0)
            prior_accum += metrics.get("loss/prior", 0.0)
            rank_floor_accum += metrics.get("loss/rank_floor", 0.0)
            uniformity_accum += metrics.get("loss/uniformity", 0.0)
            rk_eff_accum += rk_eff.mean().item()
            batches_counted += 1

    if ema is not None:
        ema.restore(model)
    model.train()

    count = max(1, batches_counted)
    return {
        "val/loss": total_loss_accum / count,
        "val/recon": recon_accum / count,
        "val/kl": kl_accum / count,
        "val/prior": prior_accum / count,
        "val/rank_floor": rank_floor_accum / count,
        "val/uniformity": uniformity_accum / count,
        "val/rk_eff": rk_eff_accum / count,
    }


def train(
    model: nn.Module,
    dataloader: DataLoader,
    env: EnvironmentConfig,
    opt_muon: Optional[Muon] = None,
    opt_adamw: Optional[torch.optim.AdamW] = None,
    start_step: int = 0,
    metrics_history: Optional[List[Dict[str, Any]]] = None,
    max_steps: int = 5000,
    muon_lr: float = 0.02,
    adamw_lr: float = 3e-4,
    log_every: int = 20,
    save_every: int = 500,
    eval_every: int = 200,
    eval_batches: int = 20,
    val_dataloader: Optional[DataLoader] = None,
    ema: Optional[EMAModel] = None,
    sync_gcs: bool = False,
    use_wandb: bool = False,
):
    model.to(env.device)
    if ema is not None:
        ema.to(env.device)
    for opt in [opt_muon, opt_adamw]:
        if opt is not None:
            for state in opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(env.device)
    model.train()

    if metrics_history is None:
        metrics_history = []

    # Восстанавливаем лучший val_loss из истории, если он есть
    best_val_loss = float("inf")
    for entry in metrics_history:
        if "val/loss" in entry and entry["val/loss"] < best_val_loss:
            best_val_loss = entry["val/loss"]

    if opt_adamw is None:
        opt_muon, opt_adamw = create_hybrid_optimizers(
            model, muon_lr=muon_lr, adamw_lr=adamw_lr
        )

    print(f"=== Запуск обучения Phase 1 Plan C на устройстве: {env.device} (Dtype: {env.dtype}) ===", flush=True)
    print(f"    - Muon параметров: {sum(p.numel() for g in opt_muon.param_groups for p in g['params']) if opt_muon else 0:,}", flush=True)
    print(f"    - AdamW параметров: {sum(p.numel() for g in opt_adamw.param_groups for p in g['params']):,}", flush=True)
    print(f"    - EMA: {'Включен (decay=' + str(ema.decay) + ')' if ema else 'Отключен'}", flush=True)
    print(f"    - Валидация: {'Каждые ' + str(eval_every) + ' шагов (' + str(eval_batches) + ' батчей)' if val_dataloader else 'Отключена'}", flush=True)
    print(f"    - Начальный шаг: {start_step} | Целевой шаг: {max_steps}", flush=True)
    if sync_gcs:
        print(f"    - Синхронизация с GCS: {env.gcs_checkpoints_dir}", flush=True)

    step = start_step
    data_iter = iter(dataloader)
    start_time = time.time()

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(env.device)
        targets = batch["targets"].to(env.device)
        weights = batch["weights"].to(env.device)

        if opt_muon is not None:
            opt_muon.zero_grad()
        opt_adamw.zero_grad()

        # Mixed Precision Autocast для GPU
        use_amp = (env.device.type == "cuda")
        with torch.autocast(device_type=env.device.type, dtype=env.dtype, enabled=use_amp):
            logits, V_cell, kl_loss, rank_loss, rk_eff = model(input_ids, weights)
            total_loss, metrics = compute_phase1_loss(
                logits=logits,
                targets=targets,
                V_cell=V_cell,
                kl_loss=kl_loss,
                rank_loss=rank_loss,
            )

        if not torch.isfinite(total_loss):
            diag_losses = ", ".join([f"{k}={round(v, 6) if isinstance(v, float) else v}" for k, v in metrics.items()])
            t_diag = []
            for t_name, t_val in [("V_cell", V_cell), ("logits", logits), ("rk_eff", rk_eff)]:
                if t_val is not None:
                    t_diag.append(f"{t_name}_finite={torch.isfinite(t_val).all().item()}")
            print(f"[Warning Step {step+1}] Нечисловой loss: {total_loss.item()}! [{diag_losses}] | [{', '.join(t_diag)}]. Пропуск шага оптимизатора.", flush=True)
            if opt_muon is not None:
                opt_muon.zero_grad()
            opt_adamw.zero_grad()
            continue

        total_loss.backward()

        # Градиентный клиппинг
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(grad_norm):
            print(f"[Warning Step {step+1}] Нечисловая норма градиента ({grad_norm}). Пропуск шага оптимизатора.", flush=True)
            if opt_muon is not None:
                opt_muon.zero_grad()
            opt_adamw.zero_grad()
            continue

        if opt_muon is not None:
            opt_muon.step()
        opt_adamw.step()

        if ema is not None:
            ema.update(model)

        step += 1

        # Логирование
        if step % log_every == 0 or step == 1 or step == max_steps:
            elapsed = time.time() - start_time
            steps_per_sec = step / max(1e-5, elapsed)
            mean_rk = rk_eff.mean().item()
            
            log_str = (
                f"Step {step:05d}/{max_steps:05d} | "
                f"Loss: {total_loss.item():.4f} | "
                f"Recon: {metrics['loss/recon']:.4f} | "
                f"RankFloor: {metrics['loss/rank_floor']:.4f} | "
                f"rk_eff: {mean_rk:.2f} | "
                f"Speed: {steps_per_sec:.2f} step/s"
            )
            print(log_str, flush=True)

            # Сохранение записи в историю метрик
            history_entry = {
                "step": step,
                "type": "train",
                "elapsed_sec": round(elapsed, 2),
                "speed_step_per_sec": round(steps_per_sec, 2),
                "mean_rk_eff": round(mean_rk, 4),
                **{k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()}
            }
            metrics_history.append(history_entry)

            if use_wandb:
                import wandb
                metrics.update({
                    "step": step,
                    "metrics/rk_eff": mean_rk,
                    "train/speed_step_per_sec": steps_per_sec,
                })
                wandb.log(metrics, step=step)

        # Валидация модели
        if val_dataloader is not None and (step % eval_every == 0 or step == max_steps):
            val_metrics = evaluate(
                model=model,
                val_dataloader=val_dataloader,
                env=env,
                eval_batches=eval_batches,
                ema=ema,
            )
            v_loss = val_metrics["val/loss"]
            v_recon = val_metrics["val/recon"]
            v_rk = val_metrics["val/rk_eff"]
            v_kl = val_metrics["val/kl"]

            print(
                f"\n{'='*75}\n"
                f"  [Validation Step {step:05d}] Val Loss: {v_loss:.4f} | Recon: {v_recon:.4f} | rk_eff: {v_rk:.2f} | KL: {v_kl:.4f}"
                f"\n{'='*75}\n",
                flush=True,
            )

            val_entry = {
                "step": step,
                "type": "val",
                **{k: round(v, 6) if isinstance(v, float) else v for k, v in val_metrics.items()}
            }
            metrics_history.append(val_entry)

            if use_wandb:
                import wandb
                wandb.log(val_metrics, step=step)

            # Сохранение лучшей модели (Best Checkpoint)
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_ckpt_path = os.path.join(env.checkpoint_dir, "phase1_best_model.pt")
                save_checkpoint(
                    model=model,
                    opt_muon=opt_muon,
                    opt_adamw=opt_adamw,
                    step=step,
                    metrics=val_metrics,
                    metrics_history=metrics_history,
                    path=best_ckpt_path,
                    ema=ema,
                    sync_gcs=sync_gcs,
                    gcs_dir=env.gcs_checkpoints_dir,
                    keep_local=True,
                )
                print(f"[Best Model] Зафиксирован новый рекорд val_loss = {best_val_loss:.4f} на шаге {step}!", flush=True)

        # Сохранение периодического чекпоинта
        if step % save_every == 0 or step == max_steps:
            ckpt_path = os.path.join(env.checkpoint_dir, f"phase1_step_{step}.pt")
            save_checkpoint(
                model=model,
                opt_muon=opt_muon,
                opt_adamw=opt_adamw,
                step=step,
                metrics=metrics,
                metrics_history=metrics_history,
                path=ckpt_path,
                ema=ema,
                sync_gcs=sync_gcs,
                gcs_dir=env.gcs_checkpoints_dir,
            )

    print("=== Обучение успешно завершено ===", flush=True)


# %% [markdown]
# ## 9. Точка входа CLI

# %%
def parse_args():
    parser = argparse.ArgumentParser(description="BEBLaDII Phase 1 Plan C Training Script")
    parser.add_argument("--smoke_test", action="store_true", help="Быстрый запуск на микро-масштабе для проверки целостности")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Устройство исполнения")
    parser.add_argument("--data_path", type=str, default="auto", help="Путь к train.parquet (BEBLaDII-planB-Phase3-Data) или 'auto'")
    parser.add_argument("--val_data_path", type=str, default="auto", help="Путь к val.parquet (BEBLaDII-planB-Phase3-Data) или 'auto'")
    parser.add_argument("--use_ema", action="store_true", default=True, help="Использовать EMA весов")
    parser.add_argument("--no_ema", action="store_true", help="Отключить EMA весов")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="Коэффициент сглаживания EMA")
    parser.add_argument("--eval_every", type=int, default=200, help="Периодичность валидации (шагов)")
    parser.add_argument("--eval_batches", type=int, default=20, help="Количество батчей для валидации")
    parser.add_argument("--use_dataparallel", action="store_true", default=True, help="Использовать DataParallel при нескольких GPU")
    parser.add_argument("--no_dataparallel", action="store_true", help="Отключить DataParallel")
    parser.add_argument("--batch_size", type=int, default=4, help="Размер батча")
    parser.add_argument("--seq_len", type=int, default=256, help="Длина последовательности (кратно 4)")
    parser.add_argument("--max_steps", type=int, default=5000, help="Количество шагов обучения")
    parser.add_argument("--muon_lr", type=float, default=0.02, help="Learning rate для Muon")
    parser.add_argument("--adamw_lr", type=float, default=3e-4, help="Learning rate для AdamW")
    parser.add_argument("--log_every", type=int, default=20, help="Периодичность логирования")
    parser.add_argument("--save_every", type=int, default=500, help="Периодичность сохранения чекпоинтов")
    parser.add_argument("--resume_ckpt", type=str, default="auto", help="Путь к чекпоинту для возобновления или 'auto' для поиска последнего")
    parser.add_argument("--gcs_dir", type=str, default="", help="Переопределение пути директории в Google Cloud Storage")
    parser.add_argument("--no_gcs", action="store_true", help="Отключить выгрузку чекпоинтов на Google Cloud Storage")
    parser.add_argument("--wandb", type=str, default="auto", choices=["auto", "online", "disabled"], help="Режим WandB")
    args, _ = parser.parse_known_args()
    return args


def main(args=None):
    if args is None:
        args = parse_args()

    # 1. Автоопределение окружения по единому признаку (IS_KAGGLE)
    env = EnvironmentConfig.detect(override_device=args.device)
    if args.gcs_dir:
        env.gcs_checkpoints_dir = args.gcs_dir

    # Датасеты train.parquet и val.parquet
    data_path = args.data_path if args.data_path != "auto" else env.data_path
    val_data_path = args.val_data_path if args.val_data_path != "auto" else env.val_data_path

    # Синхронизация с GCS активна по умолчанию на Kaggle, если не отключена явно
    sync_gcs = (env.is_kaggle and not args.no_gcs) or (not args.no_gcs and args.gcs_dir != "")

    # EMA флаг
    use_ema = args.use_ema and not getattr(args, "no_ema", False)

    # При smoke_test масштабируем параметры для мгновенной проверки без нагрузки на CPU
    vocab_size = 131072
    if args.smoke_test:
        print("=== РЕЖИМ SMOKE TEST АКТИВИРОВАН ===", flush=True)
        args.max_steps = 3
        args.batch_size = 2
        args.seq_len = 16  # 4 клетки по 4 токена
        args.log_every = 1
        args.save_every = 3
        args.eval_every = 2
        args.eval_batches = 2
        args.wandb = "disabled"
        sync_gcs = False
        # Если локально нет весов Mistral-Nemo, уменьшаем словарь для мгновенного выполнения
        if not os.path.exists(env.embeddings_path):
            vocab_size = 2048
            print(f"[SmokeTest] Использован микро-словарь {vocab_size} токенов для экономии CPU.", flush=True)

    # WandB инициализация
    use_wandb = False
    wandb_mode = args.wandb if args.wandb != "auto" else env.wandb_mode
    if wandb_mode == "online":
        api_key = os.environ.get("WANDB_API_KEY", "").strip()
        if api_key:
            try:
                import wandb
                wandb.login(key=api_key)
                cfg_dict = vars(args) if hasattr(args, "__dict__") else (args if isinstance(args, dict) else {})
                run = wandb.init(
                    project="BEBLaDII-Phase1-PlanC",
                    name=f"run-{'kaggle' if env.is_kaggle else 'local'}-{int(time.time())}",
                    config=cfg_dict,
                )
                use_wandb = True
                run_url = getattr(run, "url", None) or (run.get_url() if hasattr(run, "get_url") else "active")
                print(f"[WandB] Логирование активно! Run URL: {run_url}", flush=True)
            except Exception as e:
                print(f"[WandB Warning] Ошибка инициализации wandb.init(): {e}", flush=True)
        else:
            print("[WandB Info] Ключ WANDB_API_KEY не обнаружен (ни в os.environ, ни в Kaggle Secrets). WandB отключен.", flush=True)
    else:
        print(f"[WandB Info] Режим WandB: {wandb_mode}", flush=True)

    # 2. Инициализация токенизатора и модели
    tokenizer = load_tekken_tokenizer(env)

    print(f"Создание модели GrassmannianPhase1Model (vocab_size={vocab_size}, pad_token_id=10)...", flush=True)
    model = GrassmannianPhase1Model(
        vocab_size=vocab_size,
        hidden_size=1024,
        intermediate_size=2048,
        num_axes=8,
        axis_dim=128,
        chunk_size=4,
        pad_token_id=10,
    )
    init_embeddings(model, env)

    # Инициализация EMA на целевом устройстве
    ema = EMAModel(model, decay=args.ema_decay, device=env.device) if use_ema else None

    # 3. Восстановление из чекпоинта (Resume)
    start_step = 0
    metrics_history = []
    resume_path = None
    if args.resume_ckpt:
        if args.resume_ckpt == "auto":
            resume_path = find_latest_checkpoint(env.checkpoint_dir, env.gcs_checkpoints_dir if sync_gcs else None)
            if resume_path:
                print(f"[Resume] Найден последний доступный чекпоинт: {resume_path}", flush=True)
            else:
                print("[Resume Warning] Чекпоинтов для возобновления не найдено. Обучение начнется с нуля.", flush=True)
        else:
            resume_path = args.resume_ckpt

    # Создаем оптимизаторы заранее для возможности восстановления их состояния
    opt_muon, opt_adamw = create_hybrid_optimizers(
        model, muon_lr=args.muon_lr, adamw_lr=args.adamw_lr
    )

    if resume_path and os.path.exists(resume_path):
        start_step, metrics_history = load_checkpoint(
            path=resume_path,
            model=model,
            opt_muon=opt_muon,
            opt_adamw=opt_adamw,
            device=env.device,
            ema=ema,
        )

    # 4. Загрузка маски стоп-слов и подготовка DualRegimeDataset
    stopword_mask = None
    if os.path.exists(env.stopwords_path):
        try:
            print(f"[Init] Загрузка маски стоп-слов из: {env.stopwords_path}", flush=True)
            loaded_mask = torch.load(env.stopwords_path, map_location="cpu", weights_only=False)
            if isinstance(loaded_mask, dict) and "mask" in loaded_mask:
                loaded_mask = loaded_mask["mask"]
            if loaded_mask.shape[0] < vocab_size:
                padded = torch.zeros(vocab_size, dtype=torch.bool)
                padded[:loaded_mask.shape[0]] = loaded_mask.bool()
                stopword_mask = padded
            else:
                stopword_mask = loaded_mask[:vocab_size].bool()
            print("   -> Маска стоп-слов успешно применена.", flush=True)
        except Exception as e:
            print(f"[Init Warning] Не удалось загрузить маску стоп-слов: {e}", flush=True)

    print(f"Подготовка DualRegimeDataset (seq_len={args.seq_len}, Режим А: обрезка предложений + двухуровневый PAD, Режим Б: шаблоны)...", flush=True)
    dataset = DualRegimeDataset(
        parquet_path=data_path if not args.smoke_test else None,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        vocab_size=vocab_size,
        chunk_size=4,
        regime_b_ratio=0.15,
        eps_stop=0.02,
        eps_pad=0.001,
        pad_token_id=10,
        stopword_mask=stopword_mask,
        virtual_length=100 if args.smoke_test else 50000,
        is_validation=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=env.num_workers,
        pin_memory=env.pin_memory,
    )

    # Подготовка валидационного датасета из val.parquet
    val_dataloader = None
    if val_data_path and os.path.exists(val_data_path):
        print(f"Подготовка валидационного датасета из: {val_data_path}...", flush=True)
        val_dataset = DualRegimeDataset(
            parquet_path=val_data_path if not args.smoke_test else None,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            vocab_size=vocab_size,
            chunk_size=4,
            regime_b_ratio=0.0,  # 100% чистый текст без шаблонов Режима Б
            eps_stop=0.02,
            eps_pad=0.001,
            pad_token_id=10,
            stopword_mask=stopword_mask,
            virtual_length=100 if args.smoke_test else 1000,
            is_validation=True,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=env.num_workers,
            pin_memory=env.pin_memory,
        )
    else:
        print(f"[Validation Info] Валидационный файл не найден ({val_data_path}). Валидация отключена.", flush=True)

    # 4.5. Подключение DataParallel при наличии нескольких GPU
    use_dataparallel = getattr(args, "use_dataparallel", True) and not getattr(args, "no_dataparallel", False)
    if env.device.type == "cuda" and torch.cuda.device_count() > 1 and use_dataparallel:
        warmup_linalg_backends()
        print(f"[DataParallel] Обнаружено {torch.cuda.device_count()} GPU (акселератор T4 x2)! Включение torch.nn.DataParallel...", flush=True)
        model = nn.DataParallel(model)

    # 5. Запуск тренировочного конвейера с перехватом прерывания (Emergency Save)
    try:
        train(
            model=model,
            dataloader=dataloader,
            env=env,
            opt_muon=opt_muon,
            opt_adamw=opt_adamw,
            start_step=start_step,
            metrics_history=metrics_history,
            max_steps=args.max_steps,
            muon_lr=args.muon_lr,
            adamw_lr=args.adamw_lr,
            log_every=args.log_every,
            save_every=args.save_every,
            eval_every=args.eval_every,
            eval_batches=args.eval_batches,
            val_dataloader=val_dataloader,
            ema=ema,
            sync_gcs=sync_gcs,
            use_wandb=use_wandb,
        )
    except KeyboardInterrupt:
        print("\n[Emergency] Получен сигнал прерывания (KeyboardInterrupt)!", flush=True)
        emergency_path = os.path.join(env.checkpoint_dir, f"phase1_emergency_step_{start_step}.pt")
        save_checkpoint(
            model=model,
            opt_muon=opt_muon,
            opt_adamw=opt_adamw,
            step=start_step,
            metrics={},
            metrics_history=metrics_history,
            path=emergency_path,
            ema=ema,
            sync_gcs=sync_gcs,
            gcs_dir=env.gcs_checkpoints_dir,
            keep_local=True,
        )
        print(f"[Emergency] Аварийный чекпоинт успешно сохранен в: {emergency_path}", flush=True)


# %% [markdown]
# ## 10. Запуск обучения (User Training Config)
# Настройте параметры ниже и запустите ячейку.

# %%
# ==============================================================================
# ПАРАМЕТРЫ ЗАПУСКА ОБУЧЕНИЯ (CONFIG ДЛЯ KAGGLE / ЛОКАЛЬНО)
# ==============================================================================
TRAINING_CONFIG = {
    "smoke_test": False,       # True: быстрый микро-тест на 3 шага для проверки
    "device": "auto",          # "auto" (GPU при наличии, иначе CPU), "cuda", "cpu"
    "use_dataparallel": True,  # Использовать оба GPU (DataParallel) при наличии T4 x2
    "data_path": "auto",       # "auto": автопоиск train.parquet в BEBLaDII-planB-Phase3-Data
    "val_data_path": "auto",   # "auto": автопоиск val.parquet в BEBLaDII-planB-Phase3-Data
    "use_ema": True,           # Использовать экспоненциальное скользящее среднее (EMA) весов
    "ema_decay": 0.999,        # Decay фактор EMA (0.999 для сглаживания градиентного шума)
    "eval_every": 200,         # Проводить валидацию каждые N шагов
    "eval_batches": 20,        # Количество батчей для валидации (20 батчей * batch_size)
    "batch_size": 16,          # Общий размер батча (по 8 на каждый из 2 GPU при DataParallel)
    "seq_len": 256,            # Длина последовательности (256 токенов = 64 клетки по 4)
    "max_steps": 5000,         # Общее число шагов обучения
    "muon_lr": 0.02,           # Learning Rate для Muon (2D веса Attention и MLP)
    "adamw_lr": 3e-4,          # Learning Rate для AdamW (эмбеддинги, RMSNorm, VAE Head)
    "log_every": 20,           # Логирование каждые N шагов
    "save_every": 500,         # Сохранение чекпоинта каждые N шагов
    "resume_ckpt": "auto",     # "auto": продолжить с последнего чекпоинта из GCS, "" - с нуля
    "gcs_dir": "",             # Переопределение gs://bebladii-weigths-us/planC/Phase 1/checkpoints/
    "no_gcs": False,           # False: выгружать чекпоинты на GCS и удалять с диска Kaggle
    "wandb": "auto",           # "auto": online если есть WANDB_API_KEY в Secrets, иначе disabled
}

if __name__ == "__main__":
    # Парсим системные CLI аргументы, если они переданы
    cli_args = parse_args()
    
    # Применяем значения из TRAINING_CONFIG, если они не переопределены явно через CLI
    for key, val in TRAINING_CONFIG.items():
        if hasattr(cli_args, key):
            # Если аргумент остался дефолтным или запуск идет из блокнота Jupyter
            is_notebook = any("ipykernel" in m for m in sys.modules) or not sys.argv[1:]
            if is_notebook:
                setattr(cli_args, key, val)

    main(cli_args)
