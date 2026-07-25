# %% [markdown]
# # BEBLaDII Phase 3 Training — Canonical Spherical Diffusion (Kaggle T4 x2)
# *Архитектура: ADR 060 (каноническая диффузия на сфере)*
# *Ключевые изменения:*
# *- Полный диапазон t∈[0,1] с косинусным расписанием κ(t)*
# *- Зашумление ВСЕХ токенов (не частичное)*
# *- AdaLN вместо c_embed (нейтральная инициализация)*
# *- x0-prediction: Loss = 1 - cos(DUS_out, z_clean)*
# *- Разделённые чекпоинты: модель и оптимайзер по очереди → GCS → удаление*

# %% [markdown]
# ## 1. Setup Environment

# %%
# !pip install -q einops wandb indexed_parquet_dataset

# %%
import math
import os
import re
import subprocess
import sys

# --- Отладочный вывод структуры Kaggle ---
if os.path.exists("/kaggle/input"):
    print("=== Kaggle Input Structure ===")
    for root, dirs, files in os.walk("/kaggle/input"):
        level = root.replace("/kaggle/input", "").count(os.sep)
        if level < 3:
            indent = " " * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 4 * (level + 1)
            for f in files:
                if f.endswith(".json") or f.endswith(".pt") or f.endswith(".pth"):
                    print(f"{subindent}{f}")
    print("==============================")
# -----------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = "/kaggle/working/BEBLaDII"
REPO_URL = "https://github.com/Laeryid/BEBLaDII.git"

if not os.path.exists(PROJECT_ROOT):
    print(f"Клонирование репозитория из {REPO_URL}...")
    subprocess.run(["git", "clone", REPO_URL, PROJECT_ROOT], check=True)
else:
    print("Репозиторий уже существует. Выполняю git pull...")
    subprocess.run(["git", "-C", PROJECT_ROOT, "pull"], check=True)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.beb_la_dii.model.dus import DUSModel
    from src.beb_la_dii.model.vae import LatentEncoder
    from src.beb_la_dii.utils.data import get_dataloader, DistillationDataset
    from src.beb_la_dii.utils.loss import safe_normalize
except ImportError as e:
    print(f"Warning: Не удалось импортировать модули проекта. Ошибка: {e}")

# === Monkey-patching для ADR 061 (Clean Text Diffusion) ===
def _clean_apply_mapper(self, item, dtype):
    if item is None: return ""
    text = ""
    if dtype == 'raw':
        text = item.get('text', '') or ""
    elif dtype == 'magpie':
        text = item.get('response', '') or ""
    elif dtype == 'sharegpt':
        convs = item.get('conversations') or item.get('messages') or []
        if isinstance(convs, (list, tuple)):
            for msg in convs:
                if not isinstance(msg, dict): continue
                role_val = msg.get('from', msg.get('role'))
                role = "user" if role_val in ['human', 'user'] else "assistant"
                content = msg.get('value', msg.get('content', '')) or ""
                if role == "assistant":
                    text += content + "\n"
    if not text.strip() and isinstance(item, dict):
        vals = [str(v) for k, v in item.items() if isinstance(v, str) and len(str(v)) > 10]
        text = "\n".join(vals) if vals else str(item)
    return text.strip()

if 'DistillationDataset' in globals():
    DistillationDataset._apply_mapper = _clean_apply_mapper
    print("[Init] DistillationDataset monkey-patched for Clean Text (ADR 061)")
# ==========================================================


def resolve_model_path(base_path: str) -> str:
    """
    Находит директорию с config.json начиная с base_path.
    Если не находит — делает глобальный fallback-поиск по /kaggle/input/.
    """
    import pathlib
    p = pathlib.Path(base_path)

    def check_dir(dir_path):
        return (dir_path / "config.json").exists()

    if check_dir(p):
        print(f"[resolve_model_path] Found config.json at: {p}")
        return str(p)
    for parent in list(p.parents)[:4]:
        if check_dir(parent):
            print(f"[resolve_model_path] Found config.json in parent: {parent}")
            return str(parent)
    if p.exists():
        for config_file in sorted(p.rglob("config.json")):
            print(f"[resolve_model_path] Found config.json recursively: {config_file.parent}")
            return str(config_file.parent)
    print(f"[resolve_model_path] WARNING: config.json not found under {base_path}. Searching globally...")
    keyword = ""
    if "qwen" in base_path.lower(): keyword = "qwen"
    elif "modernbert" in base_path.lower(): keyword = "modernbert"
    if keyword:
        kaggle_input = pathlib.Path("/kaggle/input")
        if kaggle_input.exists():
            for config_file in kaggle_input.rglob("config.json"):
                if keyword in str(config_file).lower():
                    print(f"[resolve_model_path] Found fallback config for '{keyword}': {config_file.parent}")
                    return str(config_file.parent)
    print(f"[resolve_model_path] FAILED to resolve {base_path}, using as-is")
    return base_path


# %% [markdown]
# ## 2. Configuration

# %%
class Config:
    # Пути к базовым моделям
    embedding_model_path = resolve_model_path("/kaggle/input/datasets/ragnar123/qwen2-5-1-5b")
    modernbert_path      = resolve_model_path("/kaggle/input/models/answer-ai/modernbert/transformers/large/2")

    # Пути к данным
    dataset_path = "/kaggle/input/datasets/bogdanbuliakov/bebladii-planb-phase3-data/phase 3/train_data/data"

    # Пути к весам
    local_encoder_weights = "/kaggle/input/datasets/bogdanbuliakov/bebladii-planb-phase3-data/planB_phase1_checkpoints_phase1_vae_step_20000.pth"
    local_dus_weights     = "/kaggle/input/datasets/bogdanbuliakov/bebladii-phase1-awakaned-weights/AWAKENED_WEIGHTS_FINAL.pt"
    local_sep_token       = "/kaggle/working/BEBLaDII/storage/components/sep_token.pt"

    # GCS (для resume и сохранения чекпоинтов)
    resume_from_checkpoint = False
    gcs_checkpoint_dir = "gs://bebladii-weigths-us/planB/phase3/checkpoints/"

    # Директория вывода
    output_dir = "/kaggle/working/checkpoints/phase3"

    # Гиперпараметры
    batch_size    = 8
    max_length    = 512
    # LR для разных групп параметров (ADR-060 + рекомендации)
    dus_learning_rate    = 2e-5   # Пиковый LR для тела BERT (ModernBERT)
    new_layers_lr        = 1e-4   # Пиковый LR для новых слоев (AdaLN, t_proj)
    epochs               = 100
    max_steps            = 400000
    log_steps            = 10
    val_steps            = 200
    save_steps           = 1000
    
    # Параметры расписания LR: Linear Warmup + Cosine Decay
    warmup_steps_ratio = 0.05    # 5% от max_steps для warmup
    min_lr_ratio       = 0.01    # Минимальный LR относительно base_lr в конце

    # ADR-060: Диффузия — параметры расписания шума
    t_min = 0.02          # минимальный уровень шума (не 0 — избегаем κ→∞)
    t_max = 1.00          # максимальный уровень шума
    t_emb_dim = 256       # размерность синусоидального t-эмбеддинга

    # Вес геометрического лосса
    w_prior = 0.05

    # Смещённая выборка t (DiffuSeq-v2 / LD4LG): >1.0 → больше сэмплов при высоких t
    # 1.0 = равномерная (текущее поведение), 2.0 = квадратичное смещение к t_max
    t_sample_alpha = 2.0

    wandb_project = "BEBLaDII-Phase3-Kaggle"


args = Config()


# %% [markdown]
# ## 3. Utilities

# %%
class EMA:
    """
    Тени хранятся на CPU (float32) для экономии VRAM.
    Копирование CPU<->GPU только при apply/restore (раз в save_steps).
    """
    def __init__(self, model, decay=0.998):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach().float().cpu()

    def step(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    param_cpu = param.data.float().cpu()
                    self.shadow[name].mul_(self.decay).add_(param_cpu, alpha=1.0 - self.decay)

    def apply(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.backup[name] = param.data.clone().detach().cpu()
                    param.data.copy_(self.shadow[name].to(param.device, dtype=param.dtype))

    def restore(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.backup:
                    param.data.copy_(self.backup[name].to(param.device, dtype=param.dtype))
        self.backup = {}


def get_latest_gcs_checkpoint(gcs_dir: str, suffix: str = ".pth"):
    """
    Возвращает путь к последнему чекпоинту модели в GCS.
    suffix: ".pth" (модель) или "_opt.pth" (оптимайзер).
    """
    try:
        result = subprocess.run(["gsutil", "ls", gcs_dir], capture_output=True, text=True, check=True)
        files = result.stdout.splitlines()
        ckpt_files = [f for f in files if "phase3_step_" in f and f.endswith(suffix)]
        # Исключаем optimizer-файлы при поиске модели
        if suffix == ".pth":
            ckpt_files = [f for f in ckpt_files if "_opt.pth" not in f]
        if not ckpt_files:
            return None
        def extract_step(filename):
            try:
                base = filename.split("_step_")[-1].replace(suffix, "")
                return int(base)
            except ValueError:
                return -1
        ckpt_files.sort(key=extract_step)
        return ckpt_files[-1]
    except Exception as e:
        print(f"Failed to list GCS checkpoints: {e}")
        return None


def sync_to_gcs_and_delete(local_path: str, gcs_dir: str):
    """Копирует файл в GCS и удаляет локально для освобождения дискового пространства."""
    gcs_path = gcs_dir + os.path.basename(local_path)
    try:
        subprocess.run(["gsutil", "-q", "cp", local_path, gcs_path], check=True)
        os.remove(local_path)
        print(f"[GCS] Synced and deleted: {local_path} → {gcs_path}")
    except Exception as e:
        print(f"[GCS] Error syncing {local_path}: {e}")


def compute_layer_divergence(model_module):
    """
    Нормализованная L2-дистанция между парными слоями DUS (Block 1: 8-19 и Block 2: 20-31).
    Диагностика специализации клонированных слоёв.
    """
    layer_params = {}
    for name, param in model_module.named_parameters():
        m = re.search(r'layers\.([0-9]+)\.', name)
        if m and param.requires_grad:
            idx = int(m.group(1))
            suffix = name[m.end():]
            if idx not in layer_params:
                layer_params[idx] = {}
            layer_params[idx][suffix] = param.data

    divergences = []
    for k in range(8, 20):
        pair_k = k + 12
        if k in layer_params and pair_k in layer_params:
            diff_sq = 0.0
            norm_sq = 0.0
            keys_intersect = set(layer_params[k].keys()).intersection(layer_params[pair_k].keys())
            for suffix in keys_intersect:
                w1 = layer_params[k][suffix]
                w2 = layer_params[pair_k][suffix]
                diff_sq += (w1 - w2).pow(2).sum().item()
                norm_sq += w1.pow(2).sum().item()
            if norm_sq > 0:
                divergences.append(math.sqrt(diff_sq) / math.sqrt(norm_sq))
    if not divergences:
        return 0.0
    return sum(divergences) / len(divergences)


# %% [markdown]
# ## 4. Noise Schedule & Diffusion Utilities (ADR-060)
#
# Косинусное расписание параметра концентрации κ(t) vMF:
#   μ(t) = cos(t · π/2)   — средний косинус угла между x_t и x_0
#   Зашумление через slerp(x_0, ε_uniform, t)

# %%
class SinusoidalEmbedding(nn.Module):
    """Синусоидальное позиционное кодирование для t ∈ [0, 1]."""
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] float, values in [0, 1]
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )  # [half]
        args = t.unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim]


def cosine_noise_schedule(t: torch.Tensor) -> torch.Tensor:
    """
    μ(t) = cos(t · π/2) — средний косинус угла x_t с x_0.
    t=0 → μ=1 (чистый вектор), t=1 → μ=0 (равномерно на сфере).
    """
    return torch.cos(t * (math.pi / 2))


def spherical_noise(x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Зашумление через slerp по схеме vMF:
      x_t = slerp(x_0, ε, μ(t))
    где ε ~ Uniform(S^{d-1}).

    x0: [B, T, D] нормализованный
    t:  [B] float ∈ [0, 1]
    returns: x_t [B, T, D] нормализованный
    """
    B, T, D = x0.shape
    # Сэмплируем случайные точки на сфере
    eps = safe_normalize(torch.randn_like(x0), dim=-1)  # [B, T, D]
    # Параметр смешения μ(t): близко к 1 = чисто, близко к 0 = шум
    mu = cosine_noise_schedule(t).view(B, 1, 1)  # [B, 1, 1]
    # Slerp: x_t = normalize(μ · x_0 + (1-μ) · ε)
    # (точный slerp требует arccos, но при нормализации результата
    #  это эквивалентно для целей обучения Score Field)
    x_t = mu * x0 + (1.0 - mu) * eps
    return safe_normalize(x_t, dim=-1)


# %% [markdown]
# ## 5. AdaLN Module (ADR-060)
#
# Adaptive Layer Normalization: кондиционирование DUS по уровню шума t.
# Инициализация: scale=1, shift=0 → нулевое влияние на старте.

# %%
class AdaLNModulation(nn.Module):
    """
    Тонкий модуль AdaLN для одного блока DUS.
    Применяется как forward-hook к каждому слою DUS.
    """
    def __init__(self, t_emb_dim: int, hidden_dim: int):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, 2 * hidden_dim, bias=True),
        )
        # Нейтральная инициализация: scale→1, shift→0
        nn.init.zeros_(self.modulation[-1].weight)
        bias = torch.zeros(2 * hidden_dim)
        bias[hidden_dim:] = 1.0  # scale часть = 1.0
        self.modulation[-1].bias = nn.Parameter(bias)

    def forward(self, t_emb: torch.Tensor) -> tuple:
        """Возвращает (shift, scale) — оба [B, 1, D]."""
        out = self.modulation(t_emb)  # [B, 2*D]
        shift, scale = out.chunk(2, dim=-1)
        return shift.unsqueeze(1), scale.unsqueeze(1)  # [B, 1, D]


# %% [markdown]
# ## 6. Model Definition (ADR-060)
#
# Удалено: confidence_proj, c_embed_alphas, hooks c_embed.
# Добавлено: SinusoidalEmbedding + MLP (t_proj) + AdaLNModulation per layer.

# %%
class BEBLaDIIPhase3(nn.Module):
    def __init__(
        self,
        embedding_model_path: str,
        modernbert_path: str,
        dus_weights: str | None,
        encoder_weights: str | None,
        sep_token_path: str | None,
        t_emb_dim: int = 256,
    ):
        super().__init__()

        # 1. Qwen Embeddings (заморожены)
        _qwen_base = AutoModel.from_pretrained(
            embedding_model_path, torch_dtype=torch.bfloat16, local_files_only=True
        )
        self.qwen_embeddings = _qwen_base.get_input_embeddings()
        del _qwen_base
        for p in self.qwen_embeddings.parameters():
            p.requires_grad = False

        # 2. LatentEncoder (заморожен)
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

        # 3. DUS Backbone (обучаемый, float32 — ADR 057)
        dus_wrapper = DUSModel.from_scratch(
            config={"base_model_id": modernbert_path}, weights_path=None, local_files_only=True
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
                # Пропускаем устаревшие ключи c_embed
                if "confidence_proj" in k_clean or "c_embed_alphas" in k_clean:
                    continue
                clean_state[k_clean] = v
            dus_wrapper.model.load_state_dict(clean_state, strict=False)
            print(f"[Init] DUS weights loaded from {dus_weights}", flush=True)
        else:
            raise FileNotFoundError(f"CRITICAL: dus_weights not found at '{dus_weights}'.")

        self.dus = dus_wrapper.model
        if hasattr(self.dus, "gradient_checkpointing_enable"):
            self.dus.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if hasattr(self.dus, "_maybe_set_compile"):
            self.dus._maybe_set_compile = lambda *a, **kw: None
        type(self.dus).device = property(lambda self: torch.device("cuda"))
        type(self.dus).dtype  = property(lambda self: torch.float32)

        # 4. Time Embedding (ADR-060)
        hidden_dim = 1024  # размерность DUS/ModernBERT-large
        self.t_sin_embed = SinusoidalEmbedding(t_emb_dim)
        self.t_proj = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(t_emb_dim * 4, t_emb_dim),
        )

        # 5. AdaLN per DUS layer (ADR-060 & ADR-062)
        # Два отдельных AdaLNModulation на каждый блок DUS (attn и mlp)
        num_layers = len(self.dus.layers)
        self.adaLN_attn = nn.ModuleList([
            AdaLNModulation(t_emb_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.adaLN_mlp = nn.ModuleList([
            AdaLNModulation(t_emb_dim, hidden_dim) for _ in range(num_layers)
        ])
        # Регистрируем forward hooks на attn_norm и mlp_norm (ADR-062)
        for i, layer in enumerate(self.dus.layers):
            layer.attn_norm.register_forward_hook(self._make_adaLN_hook(i, target="attn"))
            layer.mlp_norm.register_forward_hook(self._make_adaLN_hook(i, target="mlp"))

        # 6. Токен-разделитель (ADR 058)
        if sep_token_path and os.path.exists(sep_token_path):
            sep_tensor = torch.load(sep_token_path, map_location="cpu").float()
            self.register_buffer("sep_embed", sep_tensor)
            print(f"[Init] Separator token loaded from {sep_token_path}", flush=True)
        else:
            self.register_buffer("sep_embed", torch.zeros(hidden_dim, dtype=torch.float32))
            print(f"[WARN] Separator token NOT found at {sep_token_path}, initialized to zeros.", flush=True)

        # 7. Self-Conditioning projection (подготовка архитектуры — нулевая инициализация)
        # При self_cond=None эффект нулевой → поведение идентично текущему.
        # Двухпроходное обучение активируется на следующем этапе (после базовой сходимости).
        self.self_cond_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.self_cond_proj.weight)
        print("[Init] self_cond_proj initialized (zeros — neutral mode).", flush=True)

    def _make_adaLN_hook(self, layer_idx: int, target: str):
        """Forward hook: применяет AdaLN после LayerNorm, до Attention/MLP (ADR-062)."""
        def hook(module, input, output):
            if not hasattr(self, "_current_t_emb") or self._current_t_emb is None:
                return output
            
            if target == "attn":
                shift, scale = self.adaLN_attn[layer_idx](self._current_t_emb)
            else:
                shift, scale = self.adaLN_mlp[layer_idx](self._current_t_emb)
                
            shift = shift.to(output.dtype)
            scale = scale.to(output.dtype)
            # Модулируем уже нормализованный output
            return output * scale + shift
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
        t: torch.Tensor | None = None,            # [B] float ∈ [t_min, t_max]; None → сэмплируем
        t_min: float = 0.02,
        t_max: float = 1.0,
        t_sample_alpha: float = 1.0,              # >1.0 → смещённая выборка к высоким t
        self_cond: torch.Tensor | None = None,    # [B, T, D] предсказание x̂_0; None → нейтральный режим
    ) -> dict:
        with torch.no_grad():
            # --- Получаем чистые латентные векторы ---
            qwen_embeds = self.qwen_embeddings(input_ids)
            z_clean, _, _ = self.encoder(qwen_embeds)  # [B, T, 1024], нормализованы
            B, T, D = z_clean.shape

            # --- Сэмплируем t если не передан ---
            if t is None:
                u = torch.rand(B, device=z_clean.device)
                if t_sample_alpha != 1.0:
                    # Смещённая выборка: 1 - u^α при α>1 → bias к t_max (высокий шум)
                    # Обоснование: денойзинг при высоком t сложнее, модель недообучается
                    u = 1.0 - u ** t_sample_alpha
                t = u * (t_max - t_min) + t_min

            # --- Зашумляем ВСЕ токены (ADR-060) ---
            z_clean_f = z_clean.float()
            z_clean_f = safe_normalize(z_clean_f, dim=-1)  # страховка
            z_noisy = spherical_noise(z_clean_f, t)  # [B, T, D], float32

        # --- Time Embedding (обучаемые параметры) ---
        t_sin = self.t_sin_embed(t)           # [B, t_emb_dim]
        t_emb = self.t_proj(t_sin)            # [B, t_emb_dim]
        self._current_t_emb = t_emb           # сохраняем для hooks

        # --- Self-Conditioning injection (нейтрально при self_cond=None) ---
        x_in = z_noisy.float()
        if self_cond is not None:
            x_in = x_in + self.self_cond_proj(self_cond.float().to(x_in.device))

        # --- Конкатенируем sep_prefix (ADR 058) ---
        sep_prefix = self.sep_embed.unsqueeze(0).unsqueeze(0).expand(B, 1, -1).to(x_in.dtype)
        dus_input_extended = torch.cat([sep_prefix, x_in], dim=1)            # [B, T+1, D]
        attention_mask_extended = F.pad(attention_mask, (1, 0), value=1)    # [B, T+1]

        # --- DUS forward (float32, ADR 059) ---
        dus_outputs = self.dus(
            inputs_embeds=dus_input_extended,
            attention_mask=attention_mask_extended,
            output_hidden_states=True,
        )

        # --- Финальная нормализация (отрезаем sep) ---
        pre_norm = dus_outputs.hidden_states[-1][:, 1:, :].float()
        dus_final_raw = self.dus.final_norm(pre_norm.to(self.dus.dtype)).float()
        dus_final = safe_normalize(dus_final_raw, dim=-1)  # [B, T, D]

        # Очищаем сохранённый t_emb после forward
        self._current_t_emb = None

        return {
            "z_clean":       z_clean_f,
            "z_noisy":       z_noisy,
            "t":             t,
            "dus_final":     dus_final,
            "dus_final_raw": dus_final_raw,
            "attention_mask": attention_mask,
            "hidden_states": dus_outputs.hidden_states,
        }


# %% [markdown]
# ## 7. Loss Function (ADR-060)
#
# Unified Cosine Loss (x0-prediction):
#   L = mean(1 - cos(DUS_out, z_clean)) по всем активным токенам
# + Prior Loss (геометрия сферы)

# %%
def compute_phase3_loss(outputs: dict, w_prior: float = 0.05):
    z_clean   = outputs["z_clean"].float()
    dus_final = outputs["dus_final"].float()
    attn_f    = outputs["attention_mask"].float()
    t         = outputs["t"]
    hidden_states = outputs.get("hidden_states", [])

    metrics = {}
    B, T, D = z_clean.size()
    active_tokens = attn_f.sum().clamp(min=1.0)

    # --- x0-prediction Cosine Loss ---
    target  = safe_normalize(z_clean, dim=-1)
    cos_sim = (dus_final * target).sum(dim=-1)           # [B, T]
    loss_el = 1.0 - cos_sim
    main_loss = (loss_el * attn_f).sum() / active_tokens

    metrics["denoising_loss"] = main_loss.detach()
    metrics["cos_sim_all"]    = (cos_sim * attn_f).sum().detach() / active_tokens

    # Метрики по уровням шума (диагностика)
    t_mean = t.mean()
    metrics["t_mean"] = t_mean.detach()
    # Косинусное сходство на низком (t<0.3) и высоком (t>0.7) шуме
    lo_mask = (t < 0.3).float()
    hi_mask = (t > 0.7).float()
    if lo_mask.sum() > 0:
        cos_sim_lo = (cos_sim.mean(dim=1) * lo_mask).sum() / lo_mask.sum()
        metrics["cos_sim_t_low"] = cos_sim_lo.detach()
    if hi_mask.sum() > 0:
        cos_sim_hi = (cos_sim.mean(dim=1) * hi_mask).sum() / hi_mask.sum()
        metrics["cos_sim_t_high"] = cos_sim_hi.detach()
    # Метрика среднего диапазона шума (0.3 ≤ t ≤ 0.7)
    mid_mask = ((t >= 0.3) & (t <= 0.7)).float()
    if mid_mask.sum() > 0:
        cos_sim_mid = (cos_sim.mean(dim=1) * mid_mask).sum() / mid_mask.sum()
        metrics["cos_sim_t_mid"] = cos_sim_mid.detach()

    # --- Prior Loss (геометрия сферы) ---
    z_flat    = dus_final.view(-1, D)
    mask_flat = attn_f.view(-1, 1)
    m_state   = (z_flat * mask_flat).sum(dim=0) / active_tokens
    z_centered = z_flat - m_state
    v_state   = (z_centered.pow(2) * mask_flat).sum(dim=0) / active_tokens
    var_floor = 1.0 / (D * 2)
    var_loss  = F.relu(var_floor - v_state).mean()
    cov       = (z_centered.T @ (z_centered * mask_flat)) / active_tokens
    cov_off   = cov - torch.diag(torch.diag(cov))
    cov_loss  = cov_off.pow(2).sum() / D
    prior_loss = m_state.pow(2).mean() + var_loss + 0.1 * cov_loss

    metrics["prior_loss"] = prior_loss.detach()
    metrics["var_loss"]   = var_loss.detach()
    metrics["cov_loss"]   = cov_loss.detach()

    total_loss = main_loss + w_prior * prior_loss

    # --- Diagnostic Norms ---
    if hidden_states and len(hidden_states) > 1:
        l0_norm  = hidden_states[0][:, 1:, :].float().norm(dim=-1).mean()
        l40_norm = hidden_states[-1][:, 1:, :].float().norm(dim=-1).mean()
        metrics["norm_L0"]    = l0_norm.detach()
        metrics["norm_Llast"] = l40_norm.detach()
        total_delta = 0.0
        for i in range(len(hidden_states) - 1):
            delta = hidden_states[i+1][:, 1:, :].float() - hidden_states[i][:, 1:, :].float()
            total_delta += delta.norm(dim=-1).mean()
        metrics["delta_norm_avg"] = (total_delta / (len(hidden_states) - 1)).detach()

    return total_loss, metrics


# %% [markdown]
# ## 8. Checkpoint Helpers (разделённое сохранение)
#
# Стратегия: модель и оптимайзер сохраняются в отдельные файлы,
# синхронизируются в GCS и удаляются локально — обходим лимиты Kaggle ~5 ГБ.

# %%
def save_checkpoint_split(
    actual_model,
    optimizer,
    scheduler,
    ema,
    step: int,
    metrics_history: list,
    output_dir: str,
    gcs_checkpoint_dir: str,
):
    """
    Сохраняет чекпоинт двумя файлами:
      - phase3_step_{step}.pth       (веса DUS + AdaLN + t_proj + EMA)
      - phase3_step_{step}_opt.pth   (состояние оптимайзера + шедулера)
    После синхронизации каждый файл удаляется локально.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"phase3_step_{step}.pth")
    opt_path   = os.path.join(output_dir, f"phase3_step_{step}_opt.pth")

    # --- Веса модели ---
    dus_state            = {k: v.cpu() for k, v in actual_model.dus.state_dict().items()}
    adaLN_attn_state     = {k: v.cpu() for k, v in actual_model.adaLN_attn.state_dict().items()}
    adaLN_mlp_state      = {k: v.cpu() for k, v in actual_model.adaLN_mlp.state_dict().items()}
    t_proj_state         = {k: v.cpu() for k, v in actual_model.t_proj.state_dict().items()}
    self_cond_proj_state = {k: v.cpu() for k, v in actual_model.self_cond_proj.state_dict().items()}

    ema.apply(actual_model)
    dus_ema_state             = {k: v.cpu() for k, v in actual_model.dus.state_dict().items()}
    adaLN_attn_ema_state      = {k: v.cpu() for k, v in actual_model.adaLN_attn.state_dict().items()}
    adaLN_mlp_ema_state       = {k: v.cpu() for k, v in actual_model.adaLN_mlp.state_dict().items()}
    t_proj_ema_state          = {k: v.cpu() for k, v in actual_model.t_proj.state_dict().items()}
    self_cond_proj_ema_state  = {k: v.cpu() for k, v in actual_model.self_cond_proj.state_dict().items()}
    ema.restore(actual_model)

    torch.save({
        "dus":                  dus_state,
        "adaLN_attn":           adaLN_attn_state,
        "adaLN_mlp":            adaLN_mlp_state,
        "t_proj":               t_proj_state,
        "self_cond_proj":       self_cond_proj_state,
        "dus_ema":              dus_ema_state,
        "adaLN_attn_ema":       adaLN_attn_ema_state,
        "adaLN_mlp_ema":        adaLN_mlp_ema_state,
        "t_proj_ema":           t_proj_ema_state,
        "self_cond_proj_ema":   self_cond_proj_ema_state,
        "step":                 step,
        "metrics_history":      metrics_history,
    }, model_path)
    print(f"[SAVE] Model weights → {model_path}")
    sync_to_gcs_and_delete(model_path, gcs_checkpoint_dir)

    # --- Оптимайзер (сохраняем отдельно и тоже удаляем) ---
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step":      step,
    }, opt_path)
    print(f"[SAVE] Optimizer state → {opt_path}")
    sync_to_gcs_and_delete(opt_path, gcs_checkpoint_dir)


def load_checkpoint_split(
    actual_model,
    optimizer,
    scheduler,
    ema,
    gcs_checkpoint_dir: str,
    output_dir: str,
    device,
):
    """
    Загружает модель и (если доступен) оптимайзер из GCS.
    Возвращает start_step, metrics_history и флаг загрузки планировщика scheduler_loaded.
    """
    start_step = 0
    metrics_history = []
    scheduler_loaded = False

    latest_model_gs = get_latest_gcs_checkpoint(gcs_checkpoint_dir, suffix=".pth")
    if not latest_model_gs:
        print("[Resume] No checkpoints found on GCS. Starting from scratch.")
        return start_step, metrics_history, scheduler_loaded

    print(f"[Resume] Found model checkpoint: {latest_model_gs}")
    os.makedirs(output_dir, exist_ok=True)
    local_model = os.path.join(output_dir, "resume_model.pth")

    try:
        subprocess.run(["gsutil", "-q", "cp", latest_model_gs, local_model], check=True)
        ckpt = torch.load(local_model, map_location="cpu")

        if "dus" in ckpt:
            clean_dus = {k.replace("_orig_module.", ""): v for k, v in ckpt["dus"].items()}
            missing, unexpected = actual_model.dus.load_state_dict(clean_dus, strict=False)
            if missing:
                print(f"[Resume] DUS missing keys: {missing[:5]}")
            print(f"[Resume] DUS weights loaded.")

        if "adaLN_attn" in ckpt:
            actual_model.adaLN_attn.load_state_dict(ckpt["adaLN_attn"], strict=True)
            print(f"[Resume] AdaLN_attn weights loaded.")
        if "adaLN_mlp" in ckpt:
            actual_model.adaLN_mlp.load_state_dict(ckpt["adaLN_mlp"], strict=True)
            print(f"[Resume] AdaLN_mlp weights loaded.")

        if "t_proj" in ckpt:
            actual_model.t_proj.load_state_dict(ckpt["t_proj"], strict=True)
            print(f"[Resume] t_proj weights loaded.")
        if "self_cond_proj" in ckpt:
            actual_model.self_cond_proj.load_state_dict(ckpt["self_cond_proj"], strict=True)
            print(f"[Resume] self_cond_proj weights loaded.")

        # EMA shadows
        ema_update = {}
        if "dus_ema" in ckpt:
            for k, v in ckpt["dus_ema"].items():
                ema_update[f"dus.{k}"] = v
        if "adaLN_attn_ema" in ckpt:
            for k, v in ckpt["adaLN_attn_ema"].items():
                ema_update[f"adaLN_attn.{k}"] = v
        if "adaLN_mlp_ema" in ckpt:
            for k, v in ckpt["adaLN_mlp_ema"].items():
                ema_update[f"adaLN_mlp.{k}"] = v
        if "t_proj_ema" in ckpt:
            for k, v in ckpt["t_proj_ema"].items():
                ema_update[f"t_proj.{k}"] = v
        if "self_cond_proj_ema" in ckpt:
            for k, v in ckpt["self_cond_proj_ema"].items():
                ema_update[f"self_cond_proj.{k}" ] = v
        if ema_update:
            ema.shadow.update(ema_update)
            print(f"[Resume] EMA shadows updated ({len(ema_update)} tensors).")

        if "step" in ckpt:
            start_step = ckpt["step"]
        if "metrics_history" in ckpt:
            metrics_history = ckpt["metrics_history"]

        os.remove(local_model)
        print(f"[Resume] Model resumed from step {start_step}.")

    except Exception as e:
        print(f"[Resume] WARN: Failed to load model checkpoint: {e}")
        return 0, [], False

    # Пробуем загрузить оптимайзер
    step_num = int(latest_model_gs.split("_step_")[-1].replace(".pth", ""))
    opt_gcs = gcs_checkpoint_dir + f"phase3_step_{step_num}_opt.pth"
    local_opt = os.path.join(output_dir, "resume_opt.pth")
    try:
        subprocess.run(["gsutil", "-q", "cp", opt_gcs, local_opt], check=True, timeout=120)
        opt_ckpt = torch.load(local_opt, map_location="cpu")
        optimizer.load_state_dict(opt_ckpt["optimizer"])
        print(f"[Resume] Optimizer state loaded.")
        try:
            scheduler.load_state_dict(opt_ckpt["scheduler"])
            scheduler_loaded = True
            print(f"[Resume] Scheduler state loaded successfully.")
        except Exception as e:
            print(f"[Resume] Scheduler load skipped: {e}")
        os.remove(local_opt)
    except Exception as e:
        print(f"[Resume] WARN: Optimizer checkpoint not found or failed ({e}). Fresh optimizer.")

    return start_step, metrics_history, scheduler_loaded


# %% [markdown]
# ## 9. Training Loop

# %%
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] Using device: {device}")
    num_gpus = torch.cuda.device_count()
    print(f"[Init] Available GPUs: {num_gpus}")

    # WandB + GCP Auth
    if args.wandb_project:
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            wandb_api = user_secrets.get_secret("WANDB_API_KEY")
            wandb.login(key=wandb_api)
            print("[Init] W&B login successful via Kaggle Secrets.")
            try:
                gcp_sa = user_secrets.get_secret("GCP_SA_JSON")
                with open("gcp_sa.json", "w") as f:
                    f.write(gcp_sa)
                subprocess.run(
                    ["gcloud", "auth", "activate-service-account", "--key-file", "gcp_sa.json"],
                    check=True,
                )
                print("[Init] GCP Authentication successful.")
                with open("gcs_test.txt", "w") as f:
                    f.write("GCS Auth Test Successful")
                subprocess.run(
                    ["gsutil", "cp", "gcs_test.txt",
                     "gs://bebladii-weigths-us/planB/phase3/checkpoints/gcs_test.txt"],
                    check=True,
                )
                print("[Init] GCS write access verified.")
            except Exception as e_gcp:
                print(f"[Init] WARN: GCP auth failed: {e_gcp}")
        except Exception as e:
            print(f"[Init] WARN: Could not login to W&B: {e}")
        wandb.init(project=args.wandb_project, config=vars(args), resume="allow")

    # --- Модель ---
    print("[Init] Instantiating BEBLaDIIPhase3 model...", flush=True)
    model = BEBLaDIIPhase3(
        embedding_model_path=args.embedding_model_path,
        modernbert_path=args.modernbert_path,
        dus_weights=args.local_dus_weights,
        encoder_weights=args.local_encoder_weights,
        sep_token_path=args.local_sep_token,
        t_emb_dim=args.t_emb_dim,
    ).to(device)

    if num_gpus > 1:
        print(f"[Init] Wrapping model in DataParallel across {num_gpus} GPUs", flush=True)
        model = nn.DataParallel(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"[Init] Trainable parameters: {n_trainable:,}", flush=True)

    # Разделяем параметры на группы: тело BERT (DUS) и новые слои (AdaLN, t_proj)
    dus_params = []
    new_layers_params = []
    
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    
    # Параметры DUS (тело BERT) — низкий LR
    for name, param in actual_model.named_parameters():
        if param.requires_grad:
            if name.startswith('dus.'):
                dus_params.append(param)
            else:
                new_layers_params.append(param)
    
    print(f"[Init] DUS params: {sum(p.numel() for p in dus_params):,}", flush=True)
    print(f"[Init] New layers params: {sum(p.numel() for p in new_layers_params):,}", flush=True)
    
    # Создаем группы параметров с разными base LR
    param_groups = [
        {'params': dus_params, 'lr': args.dus_learning_rate},
        {'params': new_layers_params, 'lr': args.new_layers_lr},
    ]
    
    optimizer = torch.optim.AdamW(param_groups, lr=0.0, weight_decay=1e-2)
    
    # --- DataLoaders ---
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

    if len(dataloader) == 0:
        raise RuntimeError(
            f"[FATAL] Dataset is empty or not found at: '{args.dataset_path}'. Training cannot start."
        )

    total_steps = (
        min(args.max_steps, len(dataloader) * args.epochs)
        if len(dataloader) > 0 else args.max_steps
    )

    # Цикличное расписание LR: CosineAnnealingWarmRestarts с ручной warmup логикой
    # (как в коммите bdd4ec3)
    cosine_T0 = 2000  # Длина первого цикла
    cosine_T_mult = 1  # Множитель для следующих циклов (1 = одинаковая длина)
    cosine_eta_min = args.dus_learning_rate * 0.01  # Минимальный LR (1% от base_lr для DUS)
    
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=cosine_T0, 
        T_mult=cosine_T_mult, 
        eta_min=cosine_eta_min
    )
    
    # Параметры warmup
    warmup_steps = min(1000, int(total_steps * 0.1))  # 10% от total_steps или 1000
    restart_warmup_steps = 200  # Warmup внутри каждого цикла
    ema = EMA(actual_model, decay=0.998)

    # --- Resume ---
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    start_step = 0
    metrics_history = []
    scheduler_loaded = False
    if getattr(args, "resume_from_checkpoint", False):
        start_step, metrics_history, scheduler_loaded = load_checkpoint_split(
            actual_model, optimizer, scheduler, ema,
            args.gcs_checkpoint_dir, args.output_dir, device,
        )
        if start_step > 0 and not scheduler_loaded:
            # Восстанавливаем состояние шедулера (fast-forward) только если он не был загружен
            for _ in range(start_step):
                scheduler.step()

    # --- Training Loop ---
    model.train()
    step = start_step

    from tqdm.auto import tqdm
    pbar = tqdm(total=total_steps, initial=start_step, desc="Phase 3 Diffusion")
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        for batch in dataloader:
            if step >= total_steps:
                break

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            fwd_outputs = model(
                input_ids,
                attention_mask=attention_mask,
                t_min=args.t_min,
                t_max=args.t_max,
                t_sample_alpha=args.t_sample_alpha,
            )
            loss, metrics = compute_phase3_loss(fwd_outputs, w_prior=args.w_prior)

            if loss.dim() > 0:
                loss = loss.mean()

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0).item()
            optimizer.step()
            ema.step(actual_model)
            scheduler.step()

            # Цикличная warmup логика (как в коммите bdd4ec3)
            current_optim_step = step + 1
            
            if current_optim_step <= warmup_steps:
                # Основной warmup в начале обучения
                lr_warmup_factor = max(0.01, current_optim_step / warmup_steps)
                for idx_p, param_group in enumerate(optimizer.param_groups):
                    param_group["lr"] = scheduler.base_lrs[idx_p] * lr_warmup_factor
            else:
                # Warmup внутри каждого цикла CosineAnnealingWarmRestarts
                rel_step = current_optim_step % cosine_T0
                if rel_step < restart_warmup_steps:
                    lr_warmup_factor = max(0.01, rel_step / restart_warmup_steps)
                    for idx_p, param_group in enumerate(optimizer.param_groups):
                        param_group["lr"] = param_group["lr"] * lr_warmup_factor

            # Логирование
            if step % args.log_steps == 0:
                metrics_dict = {
                    k: (v.mean().item() if isinstance(v, torch.Tensor) else v)
                    for k, v in metrics.items()
                }
                metrics_dict["loss"]      = loss.item()
                metrics_dict["lr_dus"]   = optimizer.param_groups[0]["lr"]  # DUS LR
                metrics_dict["lr_new"]   = optimizer.param_groups[1]["lr"]  # New layers LR
                metrics_dict["step"]      = step
                metrics_dict["grad_norm"] = grad_norm
                metrics_history.append(metrics_dict)
                if args.wandb_project:
                    wandb.log(metrics_dict, step=step)
                pbar.set_postfix({
                    "loss":   f"{metrics_dict['loss']:.4f}",
                    "cos_lo": f"{metrics_dict.get('cos_sim_t_low', 0):.4f}",
                    "cos_hi": f"{metrics_dict.get('cos_sim_t_high', 0):.4f}",
                    "t_mean": f"{metrics_dict.get('t_mean', 0):.3f}",
                    "grad":   f"{grad_norm:.3f}",
                })

            # Validation
            if step > start_step and (step + 5) % args.val_steps == 0:
                model.eval()
                val_metrics_sum = {}
                val_batches = 0
                with torch.no_grad():
                    for val_batch in val_dataloader:
                        v_ids  = val_batch["input_ids"].to(device)
                        v_mask = val_batch["attention_mask"].to(device)
                        v_out  = model(v_ids, attention_mask=v_mask, t_min=args.t_min, t_max=args.t_max, t_sample_alpha=args.t_sample_alpha)
                        v_loss, v_metrics = compute_phase3_loss(v_out, w_prior=args.w_prior)
                        for k, v in v_metrics.items():
                            val_metrics_sum[f"val_{k}"] = (
                                val_metrics_sum.get(f"val_{k}", 0)
                                + (v.mean().item() if isinstance(v, torch.Tensor) else v)
                            )
                        val_metrics_sum["val_loss"] = (
                            val_metrics_sum.get("val_loss", 0)
                            + (v_loss.mean().item() if isinstance(v_loss, torch.Tensor) else v_loss)
                        )
                        val_batches += 1
                        if val_batches >= 50:
                            break
                if val_batches > 0:
                    val_avg = {k: v / val_batches for k, v in val_metrics_sum.items()}
                    layer_div = compute_layer_divergence(actual_model.dus)
                    val_avg["val_layer_divergence"] = layer_div
                    if args.wandb_project:
                        wandb.log(val_avg, step=step)
                    print(
                        f"\n[VAL] Step {step} | loss: {val_avg.get('val_loss', 0):.4f} "
                        f"| cos_all: {val_avg.get('val_cos_sim_all', 0):.4f} "
                        f"| cos_lo: {val_avg.get('val_cos_sim_t_low', 0):.4f} "
                        f"| cos_hi: {val_avg.get('val_cos_sim_t_high', 0):.4f}"
                    )
                model.train()

            # Checkpoint (разделённый: модель + оптимайзер → GCS → удалить)
            if step > start_step and (step + 5) % args.save_steps == 0:
                save_checkpoint_split(
                    actual_model, optimizer, scheduler, ema,
                    step, metrics_history, args.output_dir, args.gcs_checkpoint_dir,
                )

            step += 1
            pbar.update(1)

        if step >= total_steps:
            break

    pbar.close()

    # Финальное сохранение
    print("[SAVE] Final checkpoint...")
    save_checkpoint_split(
        actual_model, optimizer, scheduler, ema,
        step, metrics_history, args.output_dir, args.gcs_checkpoint_dir,
    )

    if args.wandb_project:
        wandb.finish()


# %%
train()
