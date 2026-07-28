import os
import subprocess
import sys
import gc
import json
import math
import pathlib
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.beb_la_dii.model.dus import DUSModel
    from src.beb_la_dii.model.vae import LatentEncoder
    from src.beb_la_dii.utils.data import get_dataloader, DistillationDataset
    from src.beb_la_dii.utils.loss import safe_normalize
except ImportError as e:
    print(f"Error: Failed to import project modules. Error: {e}")
    sys.exit(1)

# === Clean Text Mapper Monkey-Patching (ADR 061) ===
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


def resolve_path(base_path: str) -> str:
    """Helper to check if a local path exists or return base_path."""
    if os.path.exists(base_path):
        return base_path
    return base_path


# ==========================================
# 1. Configuration
# ==========================================
class Config:
    # Base model paths
    embedding_model_path = "Qwen/Qwen2.5-1.5B"
    modernbert_path      = "answerdotai/ModernBERT-large"

    # Data paths
    dataset_path = "./data/train_data/data"

    # Weight paths
    local_encoder_weights = "gs://bebladii-weigths-us/planB/phase1/checkpoints/phase1_vae_step_20000.pth"
    local_dus_weights     = "gs://bebladii-weigths-us/kaggle_upload_1_2/AWAKENED_WEIGHTS_FINAL.pt"
    local_sep_token       = "storage/components/sep_token.pt"

    # Checkpointing and GCS
    resume_from_checkpoint = False
    gcs_checkpoint_dir = "gs://bebladii-weigths-us/planB/phase3/checkpoints/"
    output_dir = "./checkpoints/phase3"

    # Hyperparameters
    batch_size    = 8
    max_length    = 512
    dus_learning_rate    = 2e-5   # Peak LR for ModernBERT body
    new_layers_lr        = 1e-4   # Peak LR for new layers (AdaLN, t_proj)
    epochs               = 100
    max_steps            = 400000
    log_steps            = 10
    val_steps            = 200
    save_steps           = 1000
    
    warmup_steps_ratio = 0.05    # 5% of max_steps for warmup
    min_lr_ratio       = 0.01    # Min LR ratio at the end

    # ADR 060: Noise schedule parameters
    t_min = 0.02
    t_max = 1.00
    t_emb_dim = 256

    # Loss Weights
    w_prior     = 0.05
    w_div       = 0.1
    w_var_match = 150.0  # Variance matching (ADR 065)
    w_seq_rkd   = 10.0   # Token-to-Token RKD Loss
    w_adaln_l2  = 1.0    # AdaLN Output L2 Penalty

    # Gradient Checkpointing (enabled - hook state preserved during backward)
    use_gradient_checkpointing = True

    # Biased t sampling (DiffuSeq-v2 / LD4LG): >1.0 -> bias towards t_max
    t_sample_alpha = 2.0

    wandb_project = "BEBLaDII-Phase3-Lightning"


args = Config()


# ==========================================
# 2. Utilities
# ==========================================
class EMA:
    """
    Exponential Moving Average for DUS, AdaLN, and t_proj parameters.
    Stored on CPU (float32) to save GPU VRAM.
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
                if param.requires_grad:
                    param_cpu = param.data.float().cpu()
                    self.shadow[name].mul_(self.decay).add_(param_cpu, alpha=1.0 - self.decay)

    def apply(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.backup[name] = param.data.clone().detach().cpu()
                    param.data.copy_(self.shadow[name].to(param.device, dtype=param.dtype))

    def restore(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(self.backup[name].to(param.device, dtype=param.dtype))
        self.backup = {}


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / (half - 1))
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def cosine_noise_schedule(t: torch.Tensor) -> torch.Tensor:
    """κ(t) = cos(t * π/2) ∈ [0, 1]. κ(0)=1 (clean), κ(1)=0 (pure noise)."""
    return torch.cos(t * (math.pi / 2))


def spherical_noise(z_clean: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Slerp interpolation on sphere (ADR 060):
    x_t = normalize(μ · z_clean + σ · ε), where μ = cos(t*π/2), σ = sin(t*π/2)
    """
    B = z_clean.size(0)
    mu = cosine_noise_schedule(t).view(B, 1, 1)          # [B, 1, 1]
    sigma = torch.sin(t * (math.pi / 2)).view(B, 1, 1)   # [B, 1, 1]

    eps = torch.randn_like(z_clean)
    eps = safe_normalize(eps, dim=-1)

    mixed = mu * z_clean + sigma * eps
    return safe_normalize(mixed, dim=-1)


class AdaLNModulation(nn.Module):
    """
    AdaLN Modulation module for each DUS block.
    """
    def __init__(self, t_emb_dim: int, hidden_dim: int):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, 2 * hidden_dim, bias=True),
        )
        # Neutral init: scale -> 1, shift -> 0 with micro-noise (ADR 064)
        nn.init.normal_(self.modulation[-1].weight, std=1e-3)
        bias = torch.zeros(2 * hidden_dim)
        bias[hidden_dim:] = 1.0  # scale part = 1.0
        self.modulation[-1].bias = nn.Parameter(bias)

    def forward(self, t_emb: torch.Tensor) -> tuple:
        out = self.modulation(t_emb)  # [B, 2*D]
        shift, scale = out.chunk(2, dim=-1)
        return shift.unsqueeze(1), scale.unsqueeze(1)


def compute_layer_divergence(model_module):
    """Computes average L2 distance between paired DUS layers."""
    layer_params = {}
    for name, param in model_module.named_parameters():
        m = re.search(r'layers\.([0-9]+)\.', name)
        if m:
            idx = int(m.group(1))
            suffix = name[m.end():]
            if idx not in layer_params:
                layer_params[idx] = {}
            layer_params[idx][suffix] = param.data.float().cpu()

    if not layer_params:
        return None

    pairs = [(8 + k, 20 + k) for k in range(12)]
    diffs = []
    for l1, l2 in pairs:
        if l1 not in layer_params or l2 not in layer_params:
            continue
        for suffix, p1 in layer_params[l1].items():
            if suffix in layer_params[l2]:
                p2 = layer_params[l2][suffix]
                if p1.shape == p2.shape and p1.numel() > 0:
                    diff = (p1 - p2).norm().item() / (p1.numel() ** 0.5)
                    diffs.append(diff)

    return float(sum(diffs) / len(diffs)) if diffs else None


def get_latest_gcs_checkpoint(gcs_dir):
    try:
        result = subprocess.run(["gsutil", "ls", gcs_dir], capture_output=True, text=True, check=True)
        files = result.stdout.splitlines()
        ckpt_files = [f for f in files if "phase3_step_" in f and f.endswith(".pth")]
        if not ckpt_files:
            return None

        def extract_step(filename):
            try:
                return int(filename.split("_step_")[-1].split(".pth")[0])
            except ValueError:
                return -1

        ckpt_files.sort(key=extract_step)
        return ckpt_files[-1]
    except Exception as e:
        print(f"Failed to list GCS checkpoints: {e}")
        return None


# ==========================================
# 3. Model Definition (ADR-060 & ADR-065)
# ==========================================
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

        # 1. Qwen Embeddings (frozen)
        _qwen_base = AutoModel.from_pretrained(
            embedding_model_path, torch_dtype=torch.bfloat16
        )
        self.qwen_embeddings = _qwen_base.get_input_embeddings()
        del _qwen_base
        for p in self.qwen_embeddings.parameters():
            p.requires_grad = False

        # 2. LatentEncoder (frozen)
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

        # 3. DUS Backbone (trainable, float32 - ADR 057)
        dus_wrapper = DUSModel.from_scratch(
            config={"base_model_id": modernbert_path}, weights_path=None
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

        # 4. Time Embedding (ADR-060)
        hidden_dim = 1024
        self.t_sin_embed = SinusoidalEmbedding(t_emb_dim)
        self.t_proj = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(t_emb_dim * 4, t_emb_dim),
        )

        # 5. AdaLN per DUS layer (ADR-060 & ADR-062)
        num_layers = len(self.dus.layers)
        self.adaLN_attn = nn.ModuleList([
            AdaLNModulation(t_emb_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.adaLN_mlp = nn.ModuleList([
            AdaLNModulation(t_emb_dim, hidden_dim) for _ in range(num_layers)
        ])
        for i, layer in enumerate(self.dus.layers):
            layer.attn_norm.register_forward_hook(self._make_adaLN_hook(i, target="attn"))
            layer.mlp_norm.register_forward_hook(self._make_adaLN_hook(i, target="mlp"))

        # 6. Separator token (ADR 058)
        if sep_token_path and os.path.exists(sep_token_path):
            sep_tensor = torch.load(sep_token_path, map_location="cpu").float()
            self.register_buffer("sep_embed", sep_tensor)
            print(f"[Init] Separator token loaded from {sep_token_path}", flush=True)
        else:
            self.register_buffer("sep_embed", torch.zeros(hidden_dim, dtype=torch.float32))
            print(f"[WARN] Separator token NOT found at {sep_token_path}, initialized to zeros.", flush=True)

        # 7. Self-Conditioning projection
        self.self_cond_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.self_cond_proj.weight)
        print("[Init] self_cond_proj initialized (zeros — neutral mode).", flush=True)

    def _make_adaLN_hook(self, layer_idx: int, target: str):
        def hook(module, input, output):
            if not hasattr(self, "_current_t_emb") or self._current_t_emb is None:
                return output
            if target == "attn":
                shift, scale = self.adaLN_attn[layer_idx](self._current_t_emb)
            else:
                shift, scale = self.adaLN_mlp[layer_idx](self._current_t_emb)
            shift = shift.to(output.dtype)
            scale = scale.to(output.dtype)
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
        t: torch.Tensor | None = None,
        t_min: float = 0.02,
        t_max: float = 1.0,
        t_sample_alpha: float = 1.0,
        self_cond: torch.Tensor | None = None,
    ) -> dict:
        with torch.no_grad():
            qwen_embeds = self.qwen_embeddings(input_ids)
            z_clean, _, _ = self.encoder(qwen_embeds)
            B, T, D = z_clean.shape

            if t is None:
                u = torch.rand(B, device=z_clean.device)
                if t_sample_alpha != 1.0:
                    u = 1.0 - u ** t_sample_alpha
                t = u * (t_max - t_min) + t_min

            z_clean_f = z_clean.float()
            z_clean_f = safe_normalize(z_clean_f, dim=-1)
            z_noisy = spherical_noise(z_clean_f, t)

        t_sin = self.t_sin_embed(t)
        t_emb = self.t_proj(t_sin)
        self._current_t_emb = t_emb

        x_in = z_noisy.float()
        if self_cond is not None:
            x_in = x_in + self.self_cond_proj(self_cond.float().to(x_in.device))

        sep_prefix = self.sep_embed.unsqueeze(0).unsqueeze(0).expand(B, 1, -1).to(x_in.dtype)
        dus_input_extended = torch.cat([sep_prefix, x_in], dim=1)
        attention_mask_extended = F.pad(attention_mask, (1, 0), value=1)

        dus_outputs = self.dus(
            inputs_embeds=dus_input_extended,
            attention_mask=attention_mask_extended,
            output_hidden_states=False,
        )

        pre_norm = dus_outputs.last_hidden_state[:, 1:, :].float()
        dus_final_raw = self.dus.final_norm(pre_norm.to(self.dus.dtype)).float()
        h_39 = safe_normalize(dus_final_raw, dim=-1)

        # Skip-Connection & Identity Gate(t) blending (ADR 065)
        gate_t = t.view(-1, 1, 1).expand(B, 1, 1).float()
        dus_final_blended = gate_t * h_39 + (1.0 - gate_t) * x_in
        dus_final = safe_normalize(dus_final_blended, dim=-1)

        return {
            "z_clean":       z_clean_f,
            "z_noisy":       z_noisy,
            "t":             t,
            "t_emb":         t_emb,
            "dus_final":     dus_final,
            "h_39":          h_39,
            "dus_final_raw": dus_final_raw,
            "attention_mask": attention_mask,
        }


# ==========================================
# 4. Loss Functions
# ==========================================
def compute_adaln_diversity_loss(adaLN_attn, adaLN_mlp, t_emb_batch, w_adaln_l2: float = 1.0):
    B = t_emb_batch.shape[0]
    mask = ~torch.eye(B, dtype=torch.bool, device=t_emb_batch.device)

    all_outs_attn = torch.stack([m.modulation(t_emb_batch) for m in adaLN_attn], dim=0)  # [L, B, 2*D]
    all_outs_mlp  = torch.stack([m.modulation(t_emb_batch) for m in adaLN_mlp], dim=0)   # [L, B, 2*D]

    out_attn_mean = all_outs_attn.mean(dim=0)
    out_mlp_mean  = all_outs_mlp.mean(dim=0)

    shift_attn, scale_attn = out_attn_mean.chunk(2, dim=-1)
    delta_attn = torch.cat([shift_attn, scale_attn - 1.0], dim=-1)
    m_norm_attn = safe_normalize(delta_attn, dim=-1)
    sim_attn = m_norm_attn @ m_norm_attn.T
    div_attn = sim_attn[mask].mean() if mask.sum() > 0 else torch.tensor(0.0, device=t_emb_batch.device)

    shift_mlp, scale_mlp = out_mlp_mean.chunk(2, dim=-1)
    delta_mlp = torch.cat([shift_mlp, scale_mlp - 1.0], dim=-1)
    m_norm_mlp = safe_normalize(delta_mlp, dim=-1)
    sim_mlp = m_norm_mlp @ m_norm_mlp.T
    div_mlp = sim_mlp[mask].mean() if mask.sum() > 0 else torch.tensor(0.0, device=t_emb_batch.device)

    batch_div_loss = 0.5 * (div_attn + div_mlp)

    hidden_dim = shift_attn.shape[-1]
    var_floor = 1.0 / hidden_dim

    var_shift_attn = shift_attn.var(dim=-1).mean()
    var_scale_attn = (scale_attn - 1.0).var(dim=-1).mean()
    var_shift_mlp  = shift_mlp.var(dim=-1).mean()
    var_scale_mlp  = (scale_mlp - 1.0).var(dim=-1).mean()

    channel_var_loss = (
        F.relu(var_floor - var_shift_attn) +
        F.relu(var_floor - var_scale_attn) +
        F.relu(var_floor - var_shift_mlp) +
        F.relu(var_floor - var_scale_mlp)
    )

    total_div_loss = batch_div_loss + 0.5 * channel_var_loss

    # L2 Regularization on AdaLN (prevents activation norm explosion)
    shift_all_attn, scale_all_attn = all_outs_attn.chunk(2, dim=-1)
    shift_all_mlp, scale_all_mlp   = all_outs_mlp.chunk(2, dim=-1)
    adaln_l2_loss = (
        (scale_all_attn - 1.0).pow(2).mean() +
        shift_all_attn.pow(2).mean() +
        (scale_all_mlp - 1.0).pow(2).mean() +
        shift_all_mlp.pow(2).mean()
    )
    total_div_loss = total_div_loss + w_adaln_l2 * adaln_l2_loss

    shift_attn_norm = shift_attn.abs().mean()
    scale_attn_dev  = (scale_attn - 1.0).abs().mean()
    shift_mlp_norm  = shift_mlp.abs().mean()
    scale_mlp_dev   = (scale_mlp - 1.0).abs().mean()

    w_norm_attn = torch.stack([m.modulation[-1].weight.norm() for m in adaLN_attn]).mean()
    w_norm_mlp  = torch.stack([m.modulation[-1].weight.norm() for m in adaLN_mlp]).mean()

    adaln_metrics = {
        "adaln_w_norm":          0.5 * (w_norm_attn + w_norm_mlp).detach(),
        "adaln_attn_w_norm":     w_norm_attn.detach(),
        "adaln_mlp_w_norm":      w_norm_mlp.detach(),
        "adaln_attn_shift_norm": shift_attn_norm.detach(),
        "adaln_mlp_shift_norm":  shift_mlp_norm.detach(),
        "adaln_attn_scale_dev":  scale_attn_dev.detach(),
        "adaln_mlp_scale_dev":   scale_mlp_dev.detach(),
        "adaln_chan_var_attn":   var_shift_attn.detach(),
        "adaln_chan_var_mlp":    var_shift_mlp.detach(),
        "adaln_l2_loss":         adaln_l2_loss.detach(),
    }
    return total_div_loss, adaln_metrics


def compute_phase3_loss(outputs: dict, w_prior: float = 0.05, w_var_match: float = 150.0, w_seq_rkd: float = 10.0):
    z_clean   = outputs["z_clean"].float()
    dus_final = outputs["dus_final"].float()
    attn_f    = outputs["attention_mask"].float()
    t         = outputs["t"]

    metrics = {}
    B, T, D = z_clean.size()
    active_tokens = attn_f.sum().clamp(min=1.0)

    # x0-prediction Cosine Loss
    target  = safe_normalize(z_clean, dim=-1)
    cos_sim = (dus_final * target).sum(dim=-1)
    loss_el = 1.0 - cos_sim
    main_loss = (loss_el * attn_f).sum() / active_tokens

    metrics["denoising_loss"] = main_loss.detach()
    metrics["cos_sim_all"]    = (cos_sim * attn_f).sum().detach() / active_tokens

    metrics["t_mean"] = t.mean().detach()

    lo_mask_2d  = (t < 0.3).float().unsqueeze(1)
    mid_mask_2d = ((t >= 0.3) & (t <= 0.7)).float().unsqueeze(1)
    hi_mask_2d  = (t > 0.7).float().unsqueeze(1)

    if lo_mask_2d.sum() > 0:
        lo_tokens = (attn_f * lo_mask_2d).sum().clamp(min=1.0)
        metrics["cos_sim_t_low"]  = ((cos_sim * attn_f * lo_mask_2d).sum() / lo_tokens).detach()
    if hi_mask_2d.sum() > 0:
        hi_tokens = (attn_f * hi_mask_2d).sum().clamp(min=1.0)
        metrics["cos_sim_t_high"] = ((cos_sim * attn_f * hi_mask_2d).sum() / hi_tokens).detach()
    if mid_mask_2d.sum() > 0:
        mid_tokens = (attn_f * mid_mask_2d).sum().clamp(min=1.0)
        metrics["cos_sim_t_mid"]  = ((cos_sim * attn_f * mid_mask_2d).sum() / mid_tokens).detach()

    if "h_39" in outputs:
        h_39_norm   = safe_normalize(outputs["h_39"].float(), dim=-1)
        cos_h39     = (h_39_norm * target).sum(dim=-1)
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

    # Prior Loss
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

    # Variance Matching Loss
    var_match_loss = torch.tensor(0.0, device=z_clean.device)
    if "h_39" in outputs and w_var_match > 0:
        h_39_vm = safe_normalize(outputs["h_39"].float(), dim=-1)
        mask_1d = attn_f.view(-1).bool()
        h39_flat_vm = h_39_vm.view(-1, D)[mask_1d]
        z_flat_vm   = target.view(-1, D)[mask_1d]
        if h39_flat_vm.shape[0] > 1:
            h39_var = h39_flat_vm.var(dim=0)
            z_var   = z_flat_vm.var(dim=0).detach()
            var_match_loss = F.relu(z_var - h39_var).mean()
        metrics["var_match_loss"] = var_match_loss.detach()
        metrics["h39_var_mean"]   = h39_flat_vm.var(dim=0).mean().detach() if h39_flat_vm.shape[0] > 1 else torch.tensor(0.0)
        metrics["z_var_mean"]     = z_flat_vm.var(dim=0).mean().detach()   if h39_flat_vm.shape[0] > 1 else torch.tensor(0.0)

    # Token-to-Token RKD Loss
    seq_rkd_loss = torch.tensor(0.0, device=z_clean.device)
    if "h_39" in outputs and w_seq_rkd > 0:
        h_39_norm  = safe_normalize(outputs["h_39"].float(), dim=-1)
        sim_pred   = h_39_norm @ h_39_norm.transpose(1, 2)
        sim_target = target @ target.transpose(1, 2)
        mask_2d    = (attn_f.unsqueeze(2) * attn_f.unsqueeze(1))
        active_pairs = mask_2d.sum().clamp(min=1.0)
        seq_rkd_loss = ((sim_pred - sim_target).pow(2) * mask_2d).sum() / active_pairs
        metrics["seq_rkd_loss"] = seq_rkd_loss.detach()

    total_loss = main_loss + w_prior * prior_loss + w_var_match * var_match_loss + w_seq_rkd * seq_rkd_loss

    return total_loss, metrics


# ==========================================
# 5. Training Loop
# ==========================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] Using device: {device}")
    num_gpus = torch.cuda.device_count()
    print(f"[Init] Available GPUs: {num_gpus}")

    # GCP Auth
    if os.path.exists("gcp_sa.json"):
        try:
            subprocess.run(["gcloud", "auth", "activate-service-account", "--key-file", "gcp_sa.json"], check=True)
            print("[Init] GCP Authentication successful.")
        except Exception as e_gcp:
            print(f"[Init] WARN: Could not authenticate GCP: {e_gcp}")

    # WandB
    if args.wandb_project:
        wandb_api = os.environ.get("WANDB")
        if wandb_api:
            wandb.login(key=wandb_api)
        wandb.init(project=args.wandb_project, config=vars(args), resume="allow")

    # Download Weights if necessary
    os.makedirs("./weights", exist_ok=True)

    enc_path = args.local_encoder_weights
    if enc_path.startswith("gs://"):
        local_enc = os.path.join("./weights", os.path.basename(enc_path))
        if not os.path.exists(local_enc):
            print(f"Downloading Encoder from {enc_path}...")
            subprocess.run(["gsutil", "-q", "cp", enc_path, local_enc], check=True)
        enc_path = local_enc

    dus_path = args.local_dus_weights
    if dus_path.startswith("gs://"):
        local_dus = os.path.join("./weights", os.path.basename(dus_path))
        if not os.path.exists(local_dus):
            print(f"Downloading DUS from {dus_path}...")
            subprocess.run(["gsutil", "-q", "cp", dus_path, local_dus], check=True)
        dus_path = local_dus

    sep_path = resolve_path(args.local_sep_token)

    print("[Init] Instantiating BEBLaDIIPhase3 model...", flush=True)
    model = BEBLaDIIPhase3(
        embedding_model_path=args.embedding_model_path,
        modernbert_path="answerdotai/ModernBERT-large",
        dus_weights=dus_path,
        encoder_weights=enc_path,
        sep_token_path=sep_path,
        t_emb_dim=args.t_emb_dim,
    ).to(device)

    actual_model = model

    # Param groups with separate LRs
    dus_params = list(model.dus.parameters())
    new_layers_params = (
        list(model.t_proj.parameters()) +
        list(model.adaLN_attn.parameters()) +
        list(model.adaLN_mlp.parameters()) +
        list(model.self_cond_proj.parameters())
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[Init] DUS params: {sum(p.numel() for p in dus_params):,}", flush=True)
    print(f"[Init] New layers params: {sum(p.numel() for p in new_layers_params):,}", flush=True)

    param_groups = [
        {'params': dus_params, 'lr': args.dus_learning_rate},
        {'params': new_layers_params, 'lr': args.new_layers_lr},
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=0.0, weight_decay=1e-2)

    # Цикличное расписание LR: CosineAnnealingWarmRestarts с ручной warmup логикой (как в ноутбуке)
    cosine_T0 = 2000
    cosine_T_mult = 1
    cosine_eta_min = args.dus_learning_rate * 0.01

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cosine_T0,
        T_mult=cosine_T_mult,
        eta_min=cosine_eta_min
    )
    ema = EMA(actual_model, decay=0.998)

    if num_gpus > 1:
        print(f"[Init] Wrapping model in DataParallel across {num_gpus} GPUs", flush=True)
        model = nn.DataParallel(model)

    # Dataloaders
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

    total_steps = (min(args.max_steps, len(dataloader) * args.epochs) if len(dataloader) > 0 else args.max_steps)

    warmup_steps = min(1000, int(total_steps * 0.1))
    restart_warmup_steps = 200

    # Resume from checkpoint
    start_step = 0
    metrics_history = []
    if getattr(args, "resume_from_checkpoint", False):
        latest_gs_ckpt = get_latest_gcs_checkpoint(args.gcs_checkpoint_dir)
        if latest_gs_ckpt:
            print(f"Found latest checkpoint on GCS: {latest_gs_ckpt}")
            os.makedirs(args.output_dir, exist_ok=True)
            local_ckpt_path = os.path.join(args.output_dir, "resume_checkpoint.pth")
            try:
                subprocess.run(["gsutil", "-q", "cp", latest_gs_ckpt, local_ckpt_path], check=True)
                if os.path.exists(local_ckpt_path):
                    print(f"Loading checkpoint {local_ckpt_path}...")
                    ckpt = torch.load(local_ckpt_path, map_location="cpu")

                    if "dus" in ckpt:
                        clean_dus = {k.replace("_orig_module.", ""): v for k, v in ckpt["dus"].items()}
                        actual_model.dus.load_state_dict(clean_dus, strict=False)
                    for name in ["adaLN_attn", "adaLN_mlp", "t_proj"]:
                        if name in ckpt:
                            getattr(actual_model, name).load_state_dict(ckpt[name], strict=False)

                    if "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
                    if "scheduler" in ckpt: scheduler.load_state_dict(ckpt["scheduler"])
                    if "step" in ckpt: start_step = ckpt["step"]
                    if "metrics_history" in ckpt: metrics_history = ckpt["metrics_history"]

                    print(f"Successfully resumed from step {start_step}!")
            except Exception as e:
                print(f"Warning: Failed to resume from GS checkpoint! Error: {e}")

    model.train()
    step = start_step

    from tqdm.auto import tqdm
    pbar = tqdm(total=total_steps, initial=start_step, desc="Training Phase 3")
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        for batch in dataloader:
            if step >= total_steps:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            fwd_outputs = model(
                input_ids,
                attention_mask=attention_mask,
                t_min=args.t_min,
                t_max=args.t_max,
                t_sample_alpha=args.t_sample_alpha,
            )
            loss, metrics = compute_phase3_loss(fwd_outputs, w_prior=args.w_prior, w_var_match=args.w_var_match, w_seq_rkd=args.w_seq_rkd)

            div_loss, adaln_metrics = compute_adaln_diversity_loss(
                actual_model.adaLN_attn, actual_model.adaLN_mlp, fwd_outputs["t_emb"], w_adaln_l2=args.w_adaln_l2
            )
            metrics["div_loss"] = div_loss.detach()
            metrics.update(adaln_metrics)

            loss = loss + div_loss * args.w_div

            if loss.dim() > 0:
                loss = loss.mean()

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0).item()

            optimizer.step()
            ema.step(actual_model)
            scheduler.step()

            # Warmup logic
            current_optim_step = step + 1
            if current_optim_step <= warmup_steps:
                lr_warmup_factor = max(0.01, current_optim_step / warmup_steps)
                for idx_p, param_group in enumerate(optimizer.param_groups):
                    param_group["lr"] = scheduler.base_lrs[idx_p] * lr_warmup_factor
            else:
                rel_step = current_optim_step % cosine_T0
                if rel_step < restart_warmup_steps:
                    lr_warmup_factor = max(0.01, rel_step / restart_warmup_steps)
                    for idx_p, param_group in enumerate(optimizer.param_groups):
                        param_group["lr"] = param_group["lr"] * lr_warmup_factor

            if step % args.log_steps == 0:
                metrics_dict = {
                    k: (v.mean().item() if isinstance(v, torch.Tensor) else v)
                    for k, v in metrics.items()
                }
                metrics_dict["loss"]      = loss.item()
                metrics_dict["lr_dus"]   = optimizer.param_groups[0]["lr"]
                metrics_dict["lr_new"]   = optimizer.param_groups[1]["lr"]
                metrics_dict["step"]      = step
                metrics_dict["grad_norm"] = grad_norm
                metrics_history.append(metrics_dict)

                if args.wandb_project:
                    wandb.log(metrics_dict, step=step)

                pbar.set_postfix({
                    "loss": f"{metrics_dict['loss']:.4f}",
                    "grad": f"{grad_norm:.3f}",
                    "h39_low": f"{metrics_dict.get('cos_h39_t_low', 0):.3f}",
                })

            if step > start_step and step % args.val_steps == 0:
                actual_model_ref = model.module if isinstance(model, nn.DataParallel) else model
                layer_div = compute_layer_divergence(actual_model_ref.dus)

                model.eval()
                val_metrics_sum = {}
                val_batches = 0
                with torch.no_grad():
                    for val_batch in val_dataloader:
                        v_ids  = val_batch["input_ids"].to(device)
                        v_mask = val_batch["attention_mask"].to(device)
                        v_out  = model(v_ids, attention_mask=v_mask, t_min=args.t_min, t_max=args.t_max, t_sample_alpha=args.t_sample_alpha)
                        v_loss, v_metrics = compute_phase3_loss(v_out, w_prior=args.w_prior, w_var_match=args.w_var_match, w_seq_rkd=args.w_seq_rkd)
                        
                        v_div_loss, v_adaln_metrics = compute_adaln_diversity_loss(
                            actual_model_ref.adaLN_attn, actual_model_ref.adaLN_mlp, v_out["t_emb"], w_adaln_l2=args.w_adaln_l2
                        )
                        v_metrics["div_loss"] = v_div_loss.detach()
                        v_metrics.update(v_adaln_metrics)
                        v_loss = v_loss + v_div_loss * args.w_div

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
                        if val_batches >= 20: break

                if 'v_out' in locals():
                    del v_out, v_loss, v_metrics
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if val_batches > 0:
                    val_metrics_avg = {k: v / val_batches for k, v in val_metrics_sum.items()}
                    if layer_div is not None:
                        val_metrics_avg["val_layer_divergence"] = layer_div
                    if args.wandb_project: 
                        wandb.log(val_metrics_avg, step=step)
                    print(f"\n[VAL] Step {step} | val_loss: {val_metrics_avg.get('val_loss', 0):.4f} | val_cos_h39_all: {val_metrics_avg.get('val_cos_h39_all', 0):.4f}")
                model.train()

            if step > start_step and step % args.save_steps == 0:
                ckpt_path = os.path.join(args.output_dir, f"phase3_step_{step}.pth")
                actual_model = model.module if isinstance(model, nn.DataParallel) else model

                dus_state = {k: v.cpu() for k, v in actual_model.dus.state_dict().items()}
                adaLN_attn_state = {k: v.cpu() for k, v in actual_model.adaLN_attn.state_dict().items()}
                adaLN_mlp_state  = {k: v.cpu() for k, v in actual_model.adaLN_mlp.state_dict().items()}
                t_proj_state     = {k: v.cpu() for k, v in actual_model.t_proj.state_dict().items()}

                ema.apply(actual_model)
                dus_ema_state = {k: v.cpu() for k, v in actual_model.dus.state_dict().items()}
                adaLN_attn_ema = {k: v.cpu() for k, v in actual_model.adaLN_attn.state_dict().items()}
                adaLN_mlp_ema  = {k: v.cpu() for k, v in actual_model.adaLN_mlp.state_dict().items()}
                t_proj_ema     = {k: v.cpu() for k, v in actual_model.t_proj.state_dict().items()}
                ema.restore(actual_model)

                torch.save({
                    "dus": dus_state,
                    "adaLN_attn": adaLN_attn_state,
                    "adaLN_mlp": adaLN_mlp_state,
                    "t_proj": t_proj_state,
                    "dus_ema": dus_ema_state,
                    "adaLN_attn_ema": adaLN_attn_ema,
                    "adaLN_mlp_ema": adaLN_mlp_ema,
                    "t_proj_ema": t_proj_ema,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "metrics_history": metrics_history
                }, ckpt_path)
                print(f"\n[SAVE] Checkpoint saved → {ckpt_path}")
                try:
                    subprocess.Popen(["gsutil", "-q", "cp", ckpt_path, args.gcs_checkpoint_dir])
                except Exception: pass

            step += 1
            pbar.update(1)

        if step >= total_steps:
            break

    pbar.close()

    final_path = os.path.join(args.output_dir, "phase3_final.pth")
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    dus_state = {k: v.cpu() for k, v in actual_model.dus.state_dict().items()}
    adaLN_attn_state = {k: v.cpu() for k, v in actual_model.adaLN_attn.state_dict().items()}
    adaLN_mlp_state  = {k: v.cpu() for k, v in actual_model.adaLN_mlp.state_dict().items()}
    t_proj_state     = {k: v.cpu() for k, v in actual_model.t_proj.state_dict().items()}

    ema.apply(actual_model)
    dus_ema_state = {k: v.cpu() for k, v in actual_model.dus.state_dict().items()}
    adaLN_attn_ema = {k: v.cpu() for k, v in actual_model.adaLN_attn.state_dict().items()}
    adaLN_mlp_ema  = {k: v.cpu() for k, v in actual_model.adaLN_mlp.state_dict().items()}
    t_proj_ema     = {k: v.cpu() for k, v in actual_model.t_proj.state_dict().items()}
    ema.restore(actual_model)

    torch.save({
        "dus": dus_state,
        "adaLN_attn": adaLN_attn_state,
        "adaLN_mlp": adaLN_mlp_state,
        "t_proj": t_proj_state,
        "dus_ema": dus_ema_state,
        "adaLN_attn_ema": adaLN_attn_ema,
        "adaLN_mlp_ema": adaLN_mlp_ema,
        "t_proj_ema": t_proj_ema,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "metrics_history": metrics_history
    }, final_path)

    print(f"[SAVE] Final weights saved → {final_path}")
    try:
        subprocess.run(["gsutil", "-q", "cp", final_path, args.gcs_checkpoint_dir])
    except Exception: pass

    if args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    train()
