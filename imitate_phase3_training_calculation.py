import math
import os
import re
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.beb_la_dii.model.dus import DUSModel
from src.beb_la_dii.model.vae import LatentEncoder
from src.beb_la_dii.model.modern_decoder import ModernLatentDecoder
from src.beb_la_dii.utils.data import get_dataloader, DistillationDataset
from src.beb_la_dii.utils.loss import safe_normalize

# === Monkey-patching для Clean Text Diffusion (ADR 061) ===
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

DistillationDataset._apply_mapper = _clean_apply_mapper

# === Noise & Sinusoidal Embedding ===
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

def cosine_noise_schedule(t: torch.Tensor) -> torch.Tensor:
    return torch.cos(t * (math.pi / 2))

def spherical_noise(x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    B, T, D = x0.shape
    eps = safe_normalize(torch.randn_like(x0), dim=-1)
    mu = cosine_noise_schedule(t).view(B, 1, 1)
    sigma = torch.sin(t * (math.pi / 2)).view(B, 1, 1)
    x_t = mu * x0 + sigma * eps
    return safe_normalize(x_t, dim=-1)

# === AdaLN Modulation ===
class AdaLNModulation(nn.Module):
    def __init__(self, t_emb_dim: int, hidden_dim: int):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, 2 * hidden_dim, bias=True),
        )
        nn.init.normal_(self.modulation[-1].weight, std=1e-3)
        bias = torch.zeros(2 * hidden_dim)
        bias[hidden_dim:] = 1.0
        self.modulation[-1].bias = nn.Parameter(bias)

    def forward(self, t_emb: torch.Tensor) -> tuple:
        out = self.modulation(t_emb)
        shift, scale = out.chunk(2, dim=-1)
        return shift.unsqueeze(1), scale.unsqueeze(1)

# === Full BEBLaDIIPhase3 Architecture ===
class BEBLaDIIPhase3(nn.Module):
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
        print("[Init] Loading Qwen Embeddings...")
        _qwen_base = AutoModel.from_pretrained(
            embedding_model_path, torch_dtype=torch.bfloat16
        )
        self.qwen_embeddings = _qwen_base.get_input_embeddings()
        del _qwen_base
        for p in self.qwen_embeddings.parameters():
            p.requires_grad = False

        print("[Init] Loading LatentEncoder...")
        self.encoder = LatentEncoder()
        if encoder_weights and os.path.exists(encoder_weights):
            state = torch.load(encoder_weights, map_location="cpu", weights_only=False)
            if "encoder" in state:
                state = state["encoder"]
            self.encoder.load_state_dict(state, strict=False)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.to(torch.bfloat16)

        print("[Init] Loading ModernLatentDecoder...")
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
        for p in self.decoder.parameters():
            p.requires_grad = False
        self.decoder.to(torch.bfloat16)

        print("[Init] Loading DUS Backbone...")
        dus_wrapper = DUSModel.from_scratch(
            config={"base_model_id": modernbert_path}, weights_path=None
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
                if "confidence_proj" in k_clean or "c_embed_alphas" in k_clean:
                    continue
                clean_state[k_clean] = v
            dus_wrapper.model.load_state_dict(clean_state, strict=False)

        self.dus = dus_wrapper.model
        hidden_dim = 1024
        self.t_sin_embed = SinusoidalEmbedding(t_emb_dim)
        self.t_proj = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(t_emb_dim * 4, t_emb_dim),
        )

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

        if sep_token_path and os.path.exists(sep_token_path):
            sep_tensor = torch.load(sep_token_path, map_location="cpu", weights_only=False).float()
            self.register_buffer("sep_embed", sep_tensor)
        else:
            raise FileNotFoundError(f"CRITICAL: sep_token.pt not found at {sep_token_path}")

        self.self_cond_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.self_cond_proj.weight)

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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        t: torch.Tensor | None = None,
        t_min: float = 0.02,
        t_max: float = 1.0,
        t_sample_alpha: float = 1.0,
        self_cond: torch.Tensor | None = None,
        z_noisy_input: torch.Tensor | None = None,
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

            z_clean_f = safe_normalize(z_clean.float(), dim=-1)
            if z_noisy_input is not None:
                z_noisy = z_noisy_input
            else:
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
        dus_final = h_39

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

# === Loss & Metrics Functions ===
def compute_adaln_diversity_loss(adaLN_attn, adaLN_mlp, t_emb_batch, w_adaln_l2: float = 1.0):
    B = t_emb_batch.shape[0]
    mask = ~torch.eye(B, dtype=torch.bool, device=t_emb_batch.device)

    all_outs_attn = torch.stack([m.modulation(t_emb_batch) for m in adaLN_attn], dim=0)
    all_outs_mlp  = torch.stack([m.modulation(t_emb_batch) for m in adaLN_mlp], dim=0)

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

def compute_phase3_loss(outputs: dict, w_prior: float = 0.05, w_seq_rkd: float = 10.0):
    z_clean   = outputs["z_clean"].float()
    dus_final = outputs["dus_final"].float()
    attn_f    = outputs["attention_mask"].float()
    t         = outputs["t"]

    metrics = {}
    B, T, D = z_clean.size()
    active_tokens = attn_f.sum().clamp(min=1.0)

    target  = safe_normalize(z_clean, dim=-1)
    cos_sim = (dus_final * target).sum(dim=-1)
    loss_el = 1.0 - cos_sim

    mu_val = torch.cos(t * (math.pi / 2))
    sigma_val = torch.sin(t * (math.pi / 2))
    snr = (mu_val / sigma_val) ** 2
    w_snr = snr / (snr + 1.0)
    w_snr = w_snr.view(B, 1)

    main_loss = (loss_el * w_snr * attn_f).sum() / active_tokens

    metrics["denoising_loss"] = main_loss.detach()
    metrics["cos_sim_all"]    = (cos_sim * attn_f).sum().detach() / active_tokens
    metrics["t_mean"]         = t.mean().detach()

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

    total_loss = main_loss + w_prior * prior_loss + w_seq_rkd * seq_rkd_loss
    return total_loss, metrics

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Running on device: {device} ===")

    ckpt_path = r"C:\Experiments\BEBLaDII\experiments\phase3\local_checkpoints\phase3_step_6995.pth"
    print(f"Loading checkpoint metadata from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"Checkpoint Step: {ckpt.get('step')}")
    history = ckpt.get("metrics_history", [])
    print(f"Metrics history records in checkpoint: {len(history)}")
    if history:
        print("\n--- Last 5 Logged Train Metrics in Checkpoint (WandB History) ---")
        for rec in history[-5:]:
            step = rec.get('step', 0)
            loss = rec.get('loss', 0.0)
            denoise = rec.get('denoising_loss', 0.0)
            cos_all = rec.get('cos_sim_all', 0.0)
            h39_hi = rec.get('cos_h39_t_high')
            h39_mid = rec.get('cos_h39_t_mid')
            h39_lo = rec.get('cos_h39_t_low')
            div = rec.get('div_loss', 0.0)
            
            def fmt(v):
                return f"{v:.4f}" if v is not None else " N/A  "
            
            print(f"Step {step:6d} | Loss: {loss:.4f} | Denoise: {denoise:.4f} | Cos All: {cos_all:.4f} | H39 Hi: {fmt(h39_hi)} | H39 Mid: {fmt(h39_mid)} | H39 Lo: {fmt(h39_lo)} | Div: {div:.4f}")

    # Build Model
    model = BEBLaDIIPhase3(
        embedding_model_path="Qwen/Qwen2.5-1.5B",
        modernbert_path="answerdotai/ModernBERT-large",
        dus_weights=r"C:\Experiments\BEBLaDII\kaggle_upload_1_2\AWAKENED_WEIGHTS_FINAL.pt",
        encoder_weights=r"C:\Experiments\BEBLaDII\experiments\phase 1\planB_phase1_checkpoints_phase1_vae_step_20000.pth",
        decoder_weights=r"C:\Experiments\BEBLaDII\experiments\phase3\kaggle\kaggle_dataset\planB_phase2_phase2_decoder_step_8000.pth",
        sep_token_path=r"C:\Experiments\BEBLaDII\storage\components\sep_token.pt",
    ).to(device)

    # Load trained checkpoint weights
    print("\n[Loading Checkpoint Weights into Model]")
    clean_dus = {k.replace("_orig_module.", ""): v for k, v in ckpt["dus"].items()}
    missing, unexpected = model.dus.load_state_dict(clean_dus, strict=False)
    print(f"DUS loaded (missing={len(missing)}, unexpected={len(unexpected)})")
    model.adaLN_attn.load_state_dict(ckpt["adaLN_attn"])
    model.adaLN_mlp.load_state_dict(ckpt["adaLN_mlp"])
    model.t_proj.load_state_dict(ckpt["t_proj"])
    model.self_cond_proj.load_state_dict(ckpt["self_cond_proj"])
    print("All Phase 3 modules loaded successfully.")

    # Data Loader
    data_dir = r"C:\Experiments\BEBLaDII\storage\experiments\phase3\data"
    print(f"\n[Loading Data from {data_dir}]")
    val_loader = get_dataloader(
        stage="reasoning",
        batch_size=8,
        max_length=512,
        split="val",
        val_ratio=0.05,
        data_dir=data_dir,
    )

    model.eval()
    print("\n" + "="*80)
    print("  SIMULATION 1: Validation Forward Pass (eval mode, self_cond=None)")
    print("="*80)

    val_metrics_accum = {}
    n_batches = 2
    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            if b_idx >= n_batches:
                break
            v_ids = batch["input_ids"].to(device)
            v_mask = batch["attention_mask"].to(device)
            v_out = model(v_ids, attention_mask=v_mask, t_min=0.02, t_max=1.0, t_sample_alpha=2.0)
            v_loss, v_metrics = compute_phase3_loss(v_out, w_prior=0.05, w_seq_rkd=10.0)
            v_div_loss, v_adaln_metrics = compute_adaln_diversity_loss(
                model.adaLN_attn, model.adaLN_mlp, v_out["t_emb"], w_adaln_l2=0.2
            )
            v_metrics["div_loss"] = v_div_loss.detach()
            v_metrics.update(v_adaln_metrics)
            v_loss = v_loss + v_div_loss * 0.1

            for k, v in v_metrics.items():
                val_metrics_accum[k] = val_metrics_accum.get(k, 0.0) + (v.mean().item() if isinstance(v, torch.Tensor) else v)
            val_metrics_accum["total_loss"] = val_metrics_accum.get("total_loss", 0.0) + v_loss.item()

    print(f"\nResults averaged over {n_batches} validation batches:")
    for k, v in sorted(val_metrics_accum.items()):
        print(f"  {k:25s}: {v / n_batches:10.6f}")

    print("\n" + "="*80)
    print("  SIMULATION 2: Noise Level Sweep (Fixed t values: 0.05, 0.2, 0.5, 0.8, 0.95)")
    print("="*80)
    test_batch = next(iter(val_loader))
    t_ids = test_batch["input_ids"].to(device)
    t_mask = test_batch["attention_mask"].to(device)
    B = t_ids.shape[0]

    t_levels = [0.05, 0.20, 0.50, 0.80, 0.95]
    print(f"{'t level':<8} | {'cos_h39':<10} | {'denoise_loss':<14} | {'w_snr':<10} | {'Prior Loss':<12} | {'Seq RKD':<10}")
    print("-" * 75)
    with torch.no_grad():
        for t_fixed in t_levels:
            t_tensor = torch.full((B,), t_fixed, device=device, dtype=torch.float32)
            out_fixed = model(t_ids, attention_mask=t_mask, t=t_tensor)
            loss_fixed, m_fixed = compute_phase3_loss(out_fixed, w_prior=0.05, w_seq_rkd=10.0)
            
            mu = math.cos(t_fixed * math.pi / 2)
            sigma = math.sin(t_fixed * math.pi / 2)
            snr = (mu / sigma) ** 2
            w_snr = snr / (snr + 1.0)
            
            cos_h39 = m_fixed.get('cos_h39_all', 0.0).item()
            d_loss = m_fixed.get('denoising_loss', 0.0).item()
            p_loss = m_fixed.get('prior_loss', 0.0).item()
            r_loss = m_fixed.get('seq_rkd_loss', 0.0).item()
            print(f"{t_fixed:<8.2f} | {cos_h39:<10.4f} | {d_loss:<14.6f} | {w_snr:<10.4f} | {p_loss:<12.4f} | {r_loss:<10.4f}")

    print("\n" + "="*80)
    print("  SIMULATION 3: Train-Mode Step with Self-Conditioning (Pass 1 -> Pass 2)")
    print("="*80)
    with torch.no_grad():
        out_sc1 = model(t_ids, attention_mask=t_mask, t_min=0.02, t_max=1.0, t_sample_alpha=2.0)
        self_cond_est = out_sc1["dus_final"].detach()
        t_samp = out_sc1["t"].detach()
        z_noisy_samp = out_sc1["z_noisy"].detach()

        out_sc2 = model(
            t_ids, attention_mask=t_mask,
            t=t_samp,
            self_cond=self_cond_est,
            z_noisy_input=z_noisy_samp
        )
        l1, m1 = compute_phase3_loss(out_sc1, w_prior=0.05, w_seq_rkd=10.0)
        l2, m2 = compute_phase3_loss(out_sc2, w_prior=0.05, w_seq_rkd=10.0)

        print(f"Pass 1 (No Self-Cond)  -> cos_h39: {m1['cos_h39_all'].item():.4f}, Loss: {l1.item():.4f}")
        print(f"Pass 2 (With Self-Cond)-> cos_h39: {m2['cos_h39_all'].item():.4f}, Loss: {l2.item():.4f}")

if __name__ == "__main__":
    main()
