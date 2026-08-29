# %% [markdown]
# # BEBLaDII Phase 4a Training — Hierarchical Noise (Kaggle T4 x2)
# *Архитектура: ADR 060 + Phase 4 Plan (per-token diffusion)*
# *Ключевые изменения:*
# *- t_global и t_reported для каждого токена*
# *- Ложные уверенности*
# *- Убрано Min-SNR взвешивание*

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
    from src.beb_la_dii.model.modern_decoder import ModernLatentDecoder
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
    local_dus_weights     = "hf://bulyakovbr/bebladii-foundation-weights/phase3_diffusion_step_17995.pth"
    local_decoder_weights = "/kaggle/input/datasets/bogdanbuliakov/bebladii-planb-phase3-data/planB_phase2_phase2_decoder_step_8000.pth"
    local_sep_token       = "/kaggle/working/BEBLaDII/storage/components/sep_token.pt"

    # GCS (для resume и сохранения чекпоинтов)
    resume_from_checkpoint = True
    gcs_checkpoint_dir = "gs://bebladii-weigths-us/planB/phase4/checkpoints/"

    # Директория вывода
    output_dir = "/kaggle/working/checkpoints/phase4"

    # Гиперпараметры Phase 4a (~0.1x Phase 3 LR)
    batch_size    = 8
    max_length    = 512
    dus_learning_rate    = 2e-6   # Пиковый LR для тела BERT
    new_layers_lr        = 1e-5   # Пиковый LR для новых слоев
    epochs               = 100
    max_steps            = 4000
    log_steps            = 10
    val_steps            = 200
    save_steps           = 1000

    # Параметры расписания LR
    warmup_steps_ratio = 0.05
    min_lr_ratio       = 0.01

    t_min = 0.02
    t_max = 1.00
    t_emb_dim = 256

    w_prior = 0.05
    w_entropy = 1.0
    w_seq_rkd = 0.0

    t_sample_alpha = 2.0

    use_gradient_checkpointing = True

    wandb_project = "BEBLaDII-Phase4-Kaggle"
    
    # Optimizer options
    optimizer_mode = "cyclic"  # "cyclic" or "pace"
    pullback_alpha = 0.1       # Для режима pace


args = Config()


# %% [markdown]
# ## 3. Utilities

# %%
class EMA:
    """
    Тени хранятся на CPU (float32) для экономии VRAM.
    Копирование CPU<->GPU только при apply/restore (раз в save_steps).
    """
    def __init__(self, model, decay=0.998, pullback_alpha=0.0):
        self.decay = decay
        self.pullback_alpha = pullback_alpha
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
                    
                    if self.pullback_alpha > 0:
                        # Pullback: w_live = w_live - alpha * (w_live - w_ema)
                        # We push the difference to GPU to subtract from live weights
                        ema_gpu = self.shadow[name].to(param.device, dtype=param.dtype)
                        param.data.sub_(param.data - ema_gpu, alpha=self.pullback_alpha)

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
        ckpt_files = [f for f in files if "phase4_step_" in f and f.endswith(suffix)]
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
    """Синусоидальное позиционное кодирование для t ∈ [0, 1]. Поддерживает [B] и [B, T]."""
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )  # [half]
        args = t.unsqueeze(-1) * freqs  # [..., half]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [..., dim]


def cosine_noise_schedule(t: torch.Tensor) -> torch.Tensor:
    """
    μ(t) = cos(t · π/2) — средний косинус угла x_t с x_0.
    """
    return torch.cos(t * (math.pi / 2))


def spherical_noise(x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Зашумление через slerp по схеме vMF.
    x0: [B, T, D] нормализованный
    t:  [B] или [B, T] float ∈ [0, 1]
    returns: x_t [B, T, D] нормализованный
    """
    B, T, D = x0.shape
    eps = safe_normalize(torch.randn_like(x0), dim=-1)  # [B, T, D]

    if t.dim() == 1:
        t = t.view(B, 1, 1)
    elif t.dim() == 2:
        t = t.unsqueeze(-1)

    mu = cosine_noise_schedule(t)
    sigma = torch.sin(t * (math.pi / 2))
    x_t = mu * x0 + sigma * eps
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
        # Нейтральная инициализация (Zero Init): scale→1, shift→0. Стандарт для DiT. (ADR 073)
        nn.init.zeros_(self.modulation[-1].weight)
        bias = torch.zeros(2 * hidden_dim)
        bias[hidden_dim:] = 1.0  # scale часть = 1.0
        self.modulation[-1].bias = nn.Parameter(bias)

    def forward(self, t_emb: torch.Tensor) -> tuple:
        """Возвращает (shift, scale)."""
        out = self.modulation(t_emb)
        shift, scale = out.chunk(2, dim=-1)
        if shift.dim() == 2:
            return shift.unsqueeze(1), scale.unsqueeze(1)
        return shift, scale


# %% [markdown]
# ## 6. Model Definition (ADR-060)
#
# Удалено: confidence_proj, c_embed_alphas, hooks c_embed.
# Добавлено: SinusoidalEmbedding + MLP (t_proj) + AdaLNModulation per layer.

# %%
class AdaLNWrappedLayerNorm(nn.Module):
    """
    Обертка для LayerNorm, заменяющая register_forward_hook,
    чтобы избежать багов с DataParallel и замыканиями (ADR 072/075).
    """
    def __init__(self, original_norm, adaln_module):
        super().__init__()
        self.original_norm = original_norm
        self.adaln = adaln_module
        self._current_t_emb = None

    def forward(self, x):
        out = self.original_norm(x)
        if self._current_t_emb is None:
            return out
        shift, scale = self.adaln(self._current_t_emb)
        shift = shift.to(out.dtype)
        scale = scale.to(out.dtype)
        return out * scale + shift


class BEBLaDIIPhase4a(nn.Module):
    def __init__(
        self,
        embedding_model_path: str,
        modernbert_path: str,
        dus_weights: str | None,
        encoder_weights: str | None,
        decoder_weights: str | None,
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
            state = torch.load(encoder_weights, map_location="cpu", weights_only=False)
            if "encoder" in state:
                state = state["encoder"]
            self.encoder.load_state_dict(state, strict=False)
            print(f"[Init] LatentEncoder weights loaded from {encoder_weights}", flush=True)
        else:
            raise FileNotFoundError(f"CRITICAL: encoder_weights not found at '{encoder_weights}'.")
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.to(torch.bfloat16)

        # 2.5 ModernLatentDecoder
        self.decoder = ModernLatentDecoder(
            latent_dim=1024, qwen_dim=1536, num_layers=3, dus_weights_path=None
        )
        if decoder_weights and os.path.exists(decoder_weights):
            state = torch.load(decoder_weights, map_location="cpu", weights_only=False)
            if "decoder_state_dict" in state:
                state = state["decoder_state_dict"]
            elif "model" in state:
                state = state["model"]
            self.decoder.load_state_dict(state, strict=False)
            print(f"[Init] Decoder weights loaded from {decoder_weights}", flush=True)
        else:
            print(f"[Init] WARN: decoder_weights not found. Training without entropy loss.", flush=True)
        for p in self.decoder.parameters():
            p.requires_grad = False
        self.decoder.to(torch.bfloat16)

        # 3. DUS Backbone (обучаемый, float32 — ADR 057)
        dus_wrapper = DUSModel.from_scratch(
            config={"base_model_id": modernbert_path}, weights_path=None, local_files_only=True
        )
        if dus_weights and os.path.exists(dus_weights):
            state = torch.load(dus_weights, map_location="cpu", weights_only=False)
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
        if getattr(args, "use_gradient_checkpointing", False) and hasattr(self.dus, "gradient_checkpointing_enable"):
            self.dus.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print("[Init] Gradient Checkpointing enabled.", flush=True)
        else:
            if hasattr(self.dus, "gradient_checkpointing_disable"):
                self.dus.gradient_checkpointing_disable()
            print("[Init] Gradient Checkpointing disabled (prevents AdaLN hook conflict).", flush=True)
        if hasattr(self.dus, "_maybe_set_compile"):
            self.dus._maybe_set_compile = lambda *a, **kw: None
        type(self.dus).device = property(lambda self: torch.device("cuda"))
        type(self.dus).dtype  = property(lambda self: torch.float32)

        # 4. Time Embedding (Phase 4: Hierarchical)
        hidden_dim = 1024  # размерность DUS/ModernBERT-large
        self.t_sin_embed = SinusoidalEmbedding(t_emb_dim)

        self.t_proj_global = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(t_emb_dim * 4, t_emb_dim),
        )

        self.t_proj_token = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(t_emb_dim * 4, t_emb_dim),
        )

        self.t_joint_proj = nn.Linear(t_emb_dim * 2, t_emb_dim)
        # Zero-Init Compatibility Trick: cat([token, global]) -> global
        nn.init.zeros_(self.t_joint_proj.weight)
        nn.init.zeros_(self.t_joint_proj.bias)
        with torch.no_grad():
            self.t_joint_proj.weight[:, t_emb_dim:] = torch.eye(t_emb_dim)

        # 5. AdaLN per DUS layer (ADR-060 & ADR-062)
        # Два отдельных AdaLNModulation на каждый блок DUS (attn и mlp)
        num_layers = len(self.dus.layers)
        self.adaLN_attn = nn.ModuleList([
            AdaLNModulation(t_emb_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.adaLN_mlp = nn.ModuleList([
            AdaLNModulation(t_emb_dim, hidden_dim) for _ in range(num_layers)
        ])
        # Заворачиваем LayerNorm в обертку с поддержкой AdaLN и DataParallel (без хуков)
        for i, layer in enumerate(self.dus.layers):
            layer.attn_norm = AdaLNWrappedLayerNorm(layer.attn_norm, self.adaLN_attn[i])
            layer.mlp_norm = AdaLNWrappedLayerNorm(layer.mlp_norm, self.adaLN_mlp[i])

        # 6. Токен-разделитель (ADR 058)
        if sep_token_path and os.path.exists(sep_token_path):
            sep_tensor = torch.load(sep_token_path, map_location="cpu", weights_only=False).float()
            self.register_buffer("sep_embed", sep_tensor)
            print(f"[Init] Separator token loaded from {sep_token_path}", flush=True)
        else:
            raise FileNotFoundError(f"CRITICAL: sep_token.pt not found at {sep_token_path}. Укажите правильный путь на Kaggle!")

        # 7. Self-Conditioning projection (ACTIVE — xavier init)
        # SC даёт модели "второй взгляд": первый проход предсказывает x̂_0,
        # второй проход использует его как подсказку через эту проекцию.
        self.self_cond_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.self_cond_proj.weight)
        print("[Init] self_cond_proj initialized (xavier_uniform_ — SC ACTIVE).", flush=True)

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
        t_global: torch.Tensor | None = None,
        t_actual: torch.Tensor | None = None,
        t_reported: torch.Tensor | None = None,
        t_min: float = 0.02,
        t_max: float = 1.0,
        t_sample_alpha: float = 1.0,              # >1.0 → смещённая выборка к высоким t
        self_cond: torch.Tensor | None = None,    # [B, T, D] предсказание x̂_0; None → нейтральный режим
        z_noisy_input: torch.Tensor | None = None,
    ) -> dict:
        with torch.no_grad():
            # --- Получаем чистые латентные векторы ---
            qwen_embeds = self.qwen_embeddings(input_ids)
            z_clean, _, _ = self.encoder(qwen_embeds)  # [B, T, 1024], нормализованы
            B, T, D = z_clean.shape

            # --- Сэмплируем t если не передан (Phase 4 Hierarchical Noise) ---
            if t_global is None or t_actual is None or t_reported is None:
                u = torch.rand(B, device=z_clean.device)
                if t_sample_alpha != 1.0:
                    u = 1.0 - u ** t_sample_alpha
                t_global = u * (t_max - t_min) + t_min

                # Вероятность ложной уверенности p_false = t_global ** 1.5
                p_false = t_global ** 1.5
                is_false_confident = torch.rand(B, T, device=z_clean.device) < p_false.unsqueeze(-1)

                t_min_true = torch.clamp(t_global - 0.3, min=0.0)
                t_max_true = torch.clamp(t_global + 0.3, max=1.0)

                # Normal tokens
                t_actual_true = torch.rand(B, T, device=z_clean.device) * (t_max_true - t_min_true).unsqueeze(-1) + t_min_true.unsqueeze(-1)
                t_reported_true = t_actual_true

                # False confident tokens
                t_actual_false = torch.rand(B, T, device=z_clean.device) * (t_global - t_global * 0.6).unsqueeze(-1) + (t_global * 0.6).unsqueeze(-1)
                t_reported_false = torch.rand(B, T, device=z_clean.device) * (0.15 - 0.02) + 0.02

                t_actual = torch.where(is_false_confident, t_actual_false, t_actual_true)
                t_reported = torch.where(is_false_confident, t_reported_false, t_reported_true)

            z_clean_f = z_clean.float()
            z_clean_f = safe_normalize(z_clean_f, dim=-1)  # страховка

            if z_noisy_input is not None:
                z_noisy = z_noisy_input
            else:
                z_noisy = spherical_noise(z_clean_f, t_actual)  # [B, T, D], float32

        # --- Time Embedding (Hierarchical) ---
        t_sin_global = self.t_sin_embed(t_global)           # [B, t_emb_dim]
        t_emb_global = self.t_proj_global(t_sin_global)     # [B, t_emb_dim]

        t_sin_token = self.t_sin_embed(t_reported)          # [B, T, t_emb_dim]
        t_emb_token = self.t_proj_token(t_sin_token)        # [B, T, t_emb_dim]

        cond = torch.cat([t_emb_token, t_emb_global.unsqueeze(1).expand(-1, T, -1)], dim=-1)
        t_emb = self.t_joint_proj(cond)                     # [B, T, t_emb_dim]

        # Pad t_emb for the sep_prefix (zero vector -> neutral AdaLN modulation)
        sep_t_emb = torch.zeros(B, 1, t_emb.shape[-1], device=t_emb.device, dtype=t_emb.dtype)
        t_emb_extended = torch.cat([sep_t_emb, t_emb], dim=1) # [B, T+1, t_emb_dim]

        # --- Self-Conditioning injection (нейтрально при self_cond=None) ---
        x_in = z_noisy.float()
        if self_cond is not None:
            x_in = x_in + self.self_cond_proj(self_cond.float().to(x_in.device))

        # --- Конкатенируем sep_prefix (ADR 058) ---
        sep_prefix = self.sep_embed.unsqueeze(0).unsqueeze(0).expand(B, 1, -1).to(x_in.dtype)
        dus_input_extended = torch.cat([sep_prefix, x_in], dim=1)            # [B, T+1, D]
        attention_mask_extended = F.pad(attention_mask, (1, 0), value=1)    # [B, T+1]

        # --- Прокидываем t_emb в обертки AdaLN ---
        for layer in self.dus.layers:
            if hasattr(layer, "attn_norm") and isinstance(layer.attn_norm, AdaLNWrappedLayerNorm):
                layer.attn_norm._current_t_emb = t_emb_extended
            if hasattr(layer, "mlp_norm") and isinstance(layer.mlp_norm, AdaLNWrappedLayerNorm):
                layer.mlp_norm._current_t_emb = t_emb_extended

        # --- DUS forward (float32, ADR 059) ---
        dus_outputs = self.dus(
            inputs_embeds=dus_input_extended,
            attention_mask=attention_mask_extended,
            output_hidden_states=False,  # Отключено для экономии VRAM (OOM фикс)
        )

        # --- Финальная нормализация (отрезаем sep) ---
        pre_norm = dus_outputs.last_hidden_state[:, 1:, :].float()
        dus_final_raw = self.dus.final_norm(pre_norm.to(self.dus.dtype)).float()
        h_39 = safe_normalize(dus_final_raw, dim=-1)  # [B, T, D] — выход DUS ядра

        # --- ИНФЕРЕНС В ОТЛИЧИЕ ОТ ОБУЧЕНИЯ (ADR 072) ---
        # В обучении мы ТРЕБУЕМ, чтобы сеть предсказывала чистый x_0 напрямую.
        # Никаких skip-connection (блендинга с x_noisy) в forward быть не должно!
        # Блендинг происходит только в цикле сэмплирования на инференсе.
        dus_final = h_39

        return {
            "z_clean":       z_clean_f,
            "z_noisy":       z_noisy,
            "t_global":      t_global,
            "t_actual":      t_actual,
            "t_reported":    t_reported,
            "t_emb":         t_emb,
            "dus_final":     dus_final,
            "h_39":          h_39,
            "dus_final_raw": dus_final_raw,
            "attention_mask": attention_mask,
        }


# %% [markdown]
# ## 7. Loss Function (ADR-060)
#
# Unified Cosine Loss (x0-prediction):
#   L = mean(1 - cos(DUS_out, z_clean)) по всем активным токенам
# + Prior Loss (геометрия сферы)

# %%
def compute_phase4_loss(outputs: dict, w_prior: float = 0.05, w_seq_rkd: float = 0.0):
    z_clean   = outputs["z_clean"].float()
    dus_final = outputs["dus_final"].float()
    attn_f    = outputs["attention_mask"].float()
    t_global  = outputs["t_global"]
    t_actual  = outputs["t_actual"]

    metrics = {}
    B, T, D = z_clean.size()
    active_tokens = attn_f.sum().clamp(min=1.0)

    # --- x0-prediction Cosine Loss ---
    target  = safe_normalize(z_clean, dim=-1)
    cos_sim = (dus_final * target).sum(dim=-1)           # [B, T]
    loss_el = 1.0 - cos_sim

    # В Phase 4 Min-SNR убран, взвешивание uniform (w=1.0)
    main_loss = (loss_el * attn_f).sum() / active_tokens

    metrics["denoising_loss"] = main_loss.detach()
    metrics["cos_sim_all"]    = (cos_sim * attn_f).sum().detach() / active_tokens

    # Метрики по уровням шума (диагностика)
    metrics["t_global_mean"] = t_global.mean().detach()
    metrics["t_actual_mean"] = t_actual.mean().detach()

    # Маски по диапазонам t_actual — вычисляем один раз для обоих блоков метрик
    lo_mask_2d  = (t_actual < 0.3).float()   # [B, T]
    mid_mask_2d = ((t_actual >= 0.3) & (t_actual <= 0.7)).float()
    hi_mask_2d  = (t_actual > 0.7).float()

    # --- cos_sim_t_* : cos(dus_final, z_clean) — выход ПОСЛЕ gate ---
    if lo_mask_2d.sum() > 0:
        lo_tokens = (attn_f * lo_mask_2d).sum().clamp(min=1.0)
        metrics["cos_sim_t_low"]  = ((cos_sim * attn_f * lo_mask_2d).sum() / lo_tokens).detach()
    if hi_mask_2d.sum() > 0:
        hi_tokens = (attn_f * hi_mask_2d).sum().clamp(min=1.0)
        metrics["cos_sim_t_high"] = ((cos_sim * attn_f * hi_mask_2d).sum() / hi_tokens).detach()
    if mid_mask_2d.sum() > 0:
        mid_tokens = (attn_f * mid_mask_2d).sum().clamp(min=1.0)
        metrics["cos_sim_t_mid"]  = ((cos_sim * attn_f * mid_mask_2d).sum() / mid_tokens).detach()

    # --- cos_h39_t_* : cos(h_39, z_clean) — выход DUS ДО gate (истинный деноизинг) ---
    h39_low_loss = torch.tensor(0.0, device=z_clean.device)
    if "h_39" in outputs:
        h_39_norm   = safe_normalize(outputs["h_39"].float(), dim=-1)
        cos_h39     = (h_39_norm * target).sum(dim=-1)  # [B, T]
        metrics["cos_h39_all"] = (cos_h39 * attn_f).sum().detach() / active_tokens
        if lo_mask_2d.sum() > 0:
            lo_tokens = (attn_f * lo_mask_2d).sum().clamp(min=1.0)
            metrics["cos_h39_t_low"]  = ((cos_h39 * attn_f * lo_mask_2d).sum() / lo_tokens).detach()
        if hi_mask_2d.sum() > 0:
            hi_tokens = (attn_f * hi_mask_2d).sum().clamp(min=1.0)
            metrics["cos_h39_t_high"] = ((cos_h39 * attn_f * hi_mask_2d).sum() / hi_tokens).detach()
        if mid_mask_2d.sum() > 0:
            mid_tokens = (attn_f * mid_mask_2d).sum().clamp(min=1.0)
            metrics["cos_h39_t_mid"]  = ((cos_h39 * attn_f * mid_mask_2d).sum() / mid_tokens).detach()

        # h39 Identity Loss удален (ADR 072)

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

    # Variance Matching Loss удален (ADR 072)
    var_match_loss = torch.tensor(0.0, device=z_clean.device)

    # --- Token-to-Token RKD Loss (анти-коллапс внутри последовательности) ---
    seq_rkd_loss = torch.tensor(0.0, device=z_clean.device)
    if "h_39" in outputs and w_seq_rkd > 0:
        h_39_norm  = safe_normalize(outputs["h_39"].float(), dim=-1)
        sim_pred   = h_39_norm @ h_39_norm.transpose(1, 2)
        sim_target = target @ target.transpose(1, 2)
        mask_2d    = (attn_f.unsqueeze(2) * attn_f.unsqueeze(1))
        active_pairs = mask_2d.sum().clamp(min=1.0)
        seq_rkd_loss = ((sim_pred - sim_target).pow(2) * mask_2d).sum() / active_pairs
        metrics["seq_rkd_loss"] = seq_rkd_loss.detach()

    total_loss = main_loss + w_prior * prior_loss + w_seq_rkd * seq_rkd_loss

    # Decoder Entropy Loss удален (ADR 072)

    return total_loss, metrics


def compute_adaln_diagnostics(actual_model, t_emb: torch.Tensor) -> dict:
    """
    Пассивный сбор диагностических метрик AdaLN (torch.no_grad).
    Контролирует выход из Zero Init (ADR 074) без добавления лоссов и без влияния на градиенты.
    """
    with torch.no_grad():
        w_norm_attn = torch.stack([m.modulation[-1].weight.norm() for m in actual_model.adaLN_attn]).mean()
        w_norm_mlp  = torch.stack([m.modulation[-1].weight.norm() for m in actual_model.adaLN_mlp]).mean()

        all_outs_attn = torch.stack([m.modulation(t_emb) for m in actual_model.adaLN_attn], dim=0)  # [L, B, 2*D]
        all_outs_mlp  = torch.stack([m.modulation(t_emb) for m in actual_model.adaLN_mlp], dim=0)   # [L, B, 2*D]

        out_attn_mean = all_outs_attn.mean(dim=0)
        out_mlp_mean  = all_outs_mlp.mean(dim=0)

        shift_attn, scale_attn = out_attn_mean.chunk(2, dim=-1)
        shift_mlp, scale_mlp   = out_mlp_mean.chunk(2, dim=-1)

        shift_attn_norm = shift_attn.abs().mean()
        scale_attn_dev  = (scale_attn - 1.0).abs().mean()
        shift_mlp_norm  = shift_mlp.abs().mean()
        scale_mlp_dev   = (scale_mlp - 1.0).abs().mean()

        var_shift_attn = shift_attn.var(dim=-1).mean()
        var_shift_mlp  = shift_mlp.var(dim=-1).mean()

        sc_w_norm = actual_model.self_cond_proj.weight.norm() if hasattr(actual_model, "self_cond_proj") else torch.tensor(0.0)

        return {
            "adaln_w_norm":          0.5 * (w_norm_attn + w_norm_mlp).detach(),
            "adaln_attn_w_norm":     w_norm_attn.detach(),
            "adaln_mlp_w_norm":      w_norm_mlp.detach(),
            "sc_w_norm":             sc_w_norm.detach(),
            "adaln_attn_shift_norm": shift_attn_norm.detach(),
            "adaln_mlp_shift_norm":  shift_mlp_norm.detach(),
            "adaln_attn_scale_dev":  scale_attn_dev.detach(),
            "adaln_mlp_scale_dev":   scale_mlp_dev.detach(),
            "adaln_chan_var_attn":   var_shift_attn.detach(),
            "adaln_chan_var_mlp":    var_shift_mlp.detach(),
        }


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
      - phase4_step_{step}.pth       (веса DUS + AdaLN + t_proj + EMA)
      - phase4_step_{step}_opt.pth   (состояние оптимайзера + шедулера)
    После синхронизации каждый файл удаляется локально.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"phase4_step_{step}.pth")
    opt_path   = os.path.join(output_dir, f"phase4_step_{step}_opt.pth")

    # --- Веса модели ---
    dus_state            = {k: v.cpu() for k, v in actual_model.dus.state_dict().items()}
    adaLN_attn_state     = {k: v.cpu() for k, v in actual_model.adaLN_attn.state_dict().items()}
    adaLN_mlp_state      = {k: v.cpu() for k, v in actual_model.adaLN_mlp.state_dict().items()}
    t_proj_global_state  = {k: v.cpu() for k, v in actual_model.t_proj_global.state_dict().items()}
    t_proj_token_state   = {k: v.cpu() for k, v in actual_model.t_proj_token.state_dict().items()}
    t_joint_proj_state   = {k: v.cpu() for k, v in actual_model.t_joint_proj.state_dict().items()}
    self_cond_proj_state = {k: v.cpu() for k, v in actual_model.self_cond_proj.state_dict().items()}

    ema.apply(actual_model)
    dus_ema_state             = {k: v.cpu() for k, v in actual_model.dus.state_dict().items()}
    adaLN_attn_ema_state      = {k: v.cpu() for k, v in actual_model.adaLN_attn.state_dict().items()}
    adaLN_mlp_ema_state       = {k: v.cpu() for k, v in actual_model.adaLN_mlp.state_dict().items()}
    t_proj_global_ema_state   = {k: v.cpu() for k, v in actual_model.t_proj_global.state_dict().items()}
    t_proj_token_ema_state    = {k: v.cpu() for k, v in actual_model.t_proj_token.state_dict().items()}
    t_joint_proj_ema_state    = {k: v.cpu() for k, v in actual_model.t_joint_proj.state_dict().items()}
    self_cond_proj_ema_state  = {k: v.cpu() for k, v in actual_model.self_cond_proj.state_dict().items()}
    ema.restore(actual_model)

    torch.save({
        "dus":                  dus_state,
        "adaLN_attn":           adaLN_attn_state,
        "adaLN_mlp":            adaLN_mlp_state,
        "t_proj_global":        t_proj_global_state,
        "t_proj_token":         t_proj_token_state,
        "t_joint_proj":         t_joint_proj_state,
        "self_cond_proj":       self_cond_proj_state,
        "dus_ema":              dus_ema_state,
        "adaLN_attn_ema":       adaLN_attn_ema_state,
        "adaLN_mlp_ema":        adaLN_mlp_ema_state,
        "t_proj_global_ema":    t_proj_global_ema_state,
        "t_proj_token_ema":     t_proj_token_ema_state,
        "t_joint_proj_ema":     t_joint_proj_ema_state,
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
        ckpt = torch.load(local_model, map_location="cpu", weights_only=False)

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

        if "t_proj_global" in ckpt:
            actual_model.t_proj_global.load_state_dict(ckpt["t_proj_global"], strict=True)
            print(f"[Resume] t_proj_global weights loaded.")
        elif "t_proj" in ckpt:
            # Migration from Phase 3
            actual_model.t_proj_global.load_state_dict(ckpt["t_proj"], strict=True)
            print(f"[Resume] t_proj_global loaded from legacy t_proj (Phase 3).")

        if "t_proj_token" in ckpt:
            actual_model.t_proj_token.load_state_dict(ckpt["t_proj_token"], strict=True)
            print(f"[Resume] t_proj_token weights loaded.")
        if "t_joint_proj" in ckpt:
            actual_model.t_joint_proj.load_state_dict(ckpt["t_joint_proj"], strict=True)
            print(f"[Resume] t_joint_proj weights loaded.")

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

        if "t_proj_global_ema" in ckpt:
            for k, v in ckpt["t_proj_global_ema"].items():
                ema_update[f"t_proj_global.{k}"] = v
        elif "t_proj_ema" in ckpt:
            for k, v in ckpt["t_proj_ema"].items():
                ema_update[f"t_proj_global.{k}"] = v

        if "t_proj_token_ema" in ckpt:
            for k, v in ckpt["t_proj_token_ema"].items():
                ema_update[f"t_proj_token.{k}"] = v
        if "t_joint_proj_ema" in ckpt:
            for k, v in ckpt["t_joint_proj_ema"].items():
                ema_update[f"t_joint_proj.{k}"] = v

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
    opt_gcs = gcs_checkpoint_dir + f"phase4_step_{step_num}_opt.pth"
    local_opt = os.path.join(output_dir, "resume_opt.pth")
    try:
        subprocess.run(["gsutil", "-q", "cp", opt_gcs, local_opt], check=True, timeout=120)
        opt_ckpt = torch.load(local_opt, map_location="cpu", weights_only=False)
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
                     "gs://bebladii-weigths-us/planB/phase4/checkpoints/gcs_test.txt"],
                    check=True,
                )
                print("[Init] GCS write access verified.")
            except Exception as e_gcp:
                print(f"[Init] WARN: GCP auth failed: {e_gcp}")
        except Exception as e:
            print(f"[Init] WARN: Could not login to W&B: {e}")
        wandb.init(project=args.wandb_project, config=vars(args), resume="allow")

    # --- Download HF weights if necessary ---
    dus_weights_path = args.local_dus_weights
    if dus_weights_path.startswith("hf://"):
        from huggingface_hub import hf_hub_download
        parts = dus_weights_path.replace("hf://", "").split("/", 2)
        repo_id, filename = f"{parts[0]}/{parts[1]}", parts[2]
        print(f"[Init] Downloading DUS weights from HF: {repo_id}/{filename} ...", flush=True)
        dus_weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
        print(f"[Init] DUS weights downloaded to {dus_weights_path}", flush=True)

    # --- Модель ---
    print("[Init] Instantiating BEBLaDIIPhase4a model...", flush=True)
    model = BEBLaDIIPhase4a(
        embedding_model_path=args.embedding_model_path,
        modernbert_path=args.modernbert_path,
        dus_weights=dus_weights_path,
        encoder_weights=args.local_encoder_weights,
        decoder_weights=args.local_decoder_weights,
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
        if param.requires_grad and name.startswith('dus.') and 'adaln' not in name:
            dus_params.append(param)

    # Собираем параметры новых слоев
    new_layers_params.extend([p for p in actual_model.t_proj_global.parameters() if p.requires_grad])
    new_layers_params.extend([p for p in actual_model.t_proj_token.parameters() if p.requires_grad])
    new_layers_params.extend([p for p in actual_model.t_joint_proj.parameters() if p.requires_grad])
    new_layers_params.extend([p for p in actual_model.adaLN_attn.parameters() if p.requires_grad])
    new_layers_params.extend([p for p in actual_model.adaLN_mlp.parameters() if p.requires_grad])
    if hasattr(actual_model, "self_cond_proj"):
        new_layers_params.extend([p for p in actual_model.self_cond_proj.parameters() if p.requires_grad])

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

    # Защита от мусорных данных (ADR 073) — выводим один сэмпл в логи
    print("\n" + "="*50)
    print("[DATASET SAMPLE CHECK]")
    try:
        sample_batch = next(iter(dataloader))
        tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_path, local_files_only=True)
        sample_ids = sample_batch["input_ids"][0]
        sample_text = tokenizer.decode(sample_ids, skip_special_tokens=False)
        print("--- TEXT PREVIEW (first 500 chars) ---")
        print(sample_text[:500])
        print("--- TOKENS PREVIEW (first 20 ids) ---")
        print(sample_ids[:20].tolist())
    except Exception as e:
        print(f"Failed to print sample: {e}")
    print("="*50 + "\n")

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
    
    # Инициализация EMA и PACE
    if getattr(args, "optimizer_mode", "cyclic") == "pace":
        ema = EMA(actual_model, decay=0.998, pullback_alpha=getattr(args, "pullback_alpha", 0.1))
        print(f"[Init] Optimizer mode: PACE (pullback_alpha={getattr(args, 'pullback_alpha', 0.1)})", flush=True)
    else:
        ema = EMA(actual_model, decay=0.998, pullback_alpha=0.0)
        print("[Init] Optimizer mode: Cyclic (CosineAnnealingWarmRestarts)", flush=True)

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

        # --- Activate Self-Conditioning if weights are still zeros (from old checkpoint) ---
        if start_step > 0:
            sc_weight = actual_model.self_cond_proj.weight.data
            if sc_weight.abs().max().item() < 1e-8:
                nn.init.xavier_uniform_(actual_model.self_cond_proj.weight)
                # Обновляем EMA shadow для self_cond_proj
                for name, param in actual_model.named_parameters():
                    if 'self_cond_proj' in name and name in ema.shadow:
                        ema.shadow[name] = param.data.clone().detach().float().cpu()
                print("[SC] self_cond_proj was zero → RE-INITIALIZED with xavier_uniform_ (SC ACTIVATED).")
            else:
                print(f"[SC] self_cond_proj already active (max_w={sc_weight.abs().max().item():.6f}).")

    # --- Training Loop ---
    model.train()
    step = start_step

    from tqdm.auto import tqdm
    pbar = tqdm(total=total_steps, initial=start_step, desc="Phase 4 Hierarchical Noise")
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        for batch in dataloader:
            if step >= total_steps:
                break

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()

            # --- Self-Conditioning (SC) Injection ---
            # 50% probability to use self-conditioning to prevent mode collapse at high noise
            if torch.rand(1).item() < 0.5:
                # 1. No-grad first pass to get estimate
                with torch.no_grad():
                    out_sc = model(
                        input_ids,
                        attention_mask=attention_mask,
                        t_min=args.t_min,
                        t_max=args.t_max,
                        t_sample_alpha=args.t_sample_alpha,
                        self_cond=None
                    )
                    self_cond_est = out_sc["dus_final"].detach()
                    t_g_sampled = out_sc["t_global"].detach()
                    t_a_sampled = out_sc["t_actual"].detach()
                    t_r_sampled = out_sc["t_reported"].detach()
                    z_noisy_sampled = out_sc["z_noisy"].detach()

                # 2. Actual pass using the exact same t, exact same noise, and the self_cond estimate
                fwd_outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    t_global=t_g_sampled,
                    t_actual=t_a_sampled,
                    t_reported=t_r_sampled,
                    self_cond=self_cond_est,
                    z_noisy_input=z_noisy_sampled
                )
            else:
                fwd_outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    t_min=args.t_min,
                    t_max=args.t_max,
                    t_sample_alpha=args.t_sample_alpha,
                    self_cond=None
                )
            loss, metrics = compute_phase4_loss(fwd_outputs, w_prior=args.w_prior, w_seq_rkd=args.w_seq_rkd)



            if loss.dim() > 0:
                loss = loss.mean()

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0).item()
            optimizer.step()
            ema.step(actual_model)
            
            # Логика LR
            current_optim_step = step + 1
            is_pace = getattr(args, "optimizer_mode", "cyclic") == "pace"
            
            if is_pace:
                # PACE: Постоянный LR с первичным warmup
                if current_optim_step <= warmup_steps:
                    lr_warmup_factor = max(0.01, current_optim_step / warmup_steps)
                    for idx_p, param_group in enumerate(optimizer.param_groups):
                        param_group["lr"] = scheduler.base_lrs[idx_p] * lr_warmup_factor
                else:
                    for idx_p, param_group in enumerate(optimizer.param_groups):
                        param_group["lr"] = scheduler.base_lrs[idx_p]
            else:
                # Cyclic: CosineAnnealingWarmRestarts
                scheduler.step()
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
                adaln_diag = compute_adaln_diagnostics(actual_model, fwd_outputs["t_emb"])
                metrics.update(adaln_diag)
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
                    "loss":      f"{metrics_dict['loss']:.4f}",
                    "h39_hi":    f"{metrics_dict.get('cos_h39_t_high', 0):.4f}",
                    "h39_mid":   f"{metrics_dict.get('cos_h39_t_mid', 0):.4f}",
                    "cos_hi":    f"{metrics_dict.get('cos_sim_t_high', 0):.4f}",
                    "adaln_w":   f"{metrics_dict.get('adaln_w_norm', 0):.4f}",
                    "sc_w":      f"{metrics_dict.get('sc_w_norm', 0):.4f}",
                    "grad":      f"{grad_norm:.3f}",
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
                        v_loss, v_metrics = compute_phase4_loss(v_out, w_prior=args.w_prior, w_seq_rkd=args.w_seq_rkd)
                        v_adaln_diag = compute_adaln_diagnostics(actual_model, v_out["t_emb"])
                        v_metrics.update(v_adaln_diag)



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

                # --- Очистка памяти после валидации (OOM Fix) ---
                if 'v_out' in locals():
                    del v_out, v_loss, v_metrics
                torch.cuda.empty_cache()
                # ------------------------------------------------

                if val_batches > 0:
                    val_avg = {k: v / val_batches for k, v in val_metrics_sum.items()}
                    layer_div = compute_layer_divergence(actual_model.dus)
                    val_avg["val_layer_divergence"] = layer_div
                    if args.wandb_project:
                        wandb.log(val_avg, step=step)
                    print(
                        f"\n[VAL] Step {step} | loss: {val_avg.get('val_loss', 0):.4f} "
                        f"| h39_hi: {val_avg.get('val_cos_h39_t_high', 0):.4f} "
                        f"| h39_mid: {val_avg.get('val_cos_h39_t_mid', 0):.4f} "
                        f"| h39_lo: {val_avg.get('val_cos_h39_t_low', 0):.4f} "
                        f"| cos_hi(gated): {val_avg.get('val_cos_sim_t_high', 0):.4f}"
                    )
                model.train()

            # Checkpoint (разделённый: модель + оптимайзер → GCS → удалить)
            if step > start_step and (step + 5) % args.save_steps == 0:
                save_checkpoint_split(
                    actual_model, optimizer, scheduler, ema,
                    step, metrics_history, args.output_dir, args.gcs_checkpoint_dir,
                )

            # --- Explicit memory cleanup (OOM Fix) ---
            if 'fwd_outputs' in locals():
                del fwd_outputs, loss, metrics
            # -----------------------------------------

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
