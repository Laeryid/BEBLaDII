import os
import sys
import math
import re
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = "C:/Experiments/BEBLaDII"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.beb_la_dii.model.dus import DUSModel
from src.beb_la_dii.model.vae import LatentEncoder
from src.beb_la_dii.utils.loss import safe_normalize

def resolve_model_path(base_path: str) -> str:
    import pathlib
    p = pathlib.Path(base_path)
    def check_dir(dir_path):
        return (dir_path / "config.json").exists()
    if check_dir(p): return str(p)
    for parent in list(p.parents)[:4]:
        if check_dir(parent): return str(parent)
    if p.exists():
        for config_file in sorted(p.rglob("config.json")):
            return str(config_file.parent)
    
    keyword = ""
    if "qwen" in base_path.lower(): keyword = "qwen"
    elif "modernbert" in base_path.lower(): keyword = "modernbert"
    
    raise FileNotFoundError(f"КРИТИЧЕСКАЯ ОШИБКА: config.json не найден для {keyword} по пути {base_path}")

# --- Phase 4 Architecture Components ---
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / (half - 1))
        args = t.unsqueeze(-1) * freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

def cosine_noise_schedule(t: torch.Tensor) -> torch.Tensor:
    return torch.cos(t * (math.pi / 2))

def spherical_noise(x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    B, T, D = x0.shape
    eps = safe_normalize(torch.randn_like(x0), dim=-1)
    if t.dim() == 1:
        t = t.view(B, 1, 1)
    elif t.dim() == 2:
        t = t.unsqueeze(-1)
    mu = cosine_noise_schedule(t)
    sigma = torch.sin(t * (math.pi / 2))
    x_t = mu * x0 + sigma * eps
    return safe_normalize(x_t, dim=-1)

class AdaLNModulation(nn.Module):
    def __init__(self, t_emb_dim: int, hidden_dim: int):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, 2 * hidden_dim, bias=True),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        bias = torch.zeros(2 * hidden_dim)
        bias[hidden_dim:] = 1.0
        self.modulation[-1].bias = nn.Parameter(bias)

    def forward(self, t_emb: torch.Tensor) -> tuple:
        out = self.modulation(t_emb)
        shift, scale = out.chunk(2, dim=-1)
        if shift.dim() == 2:
            return shift.unsqueeze(1), scale.unsqueeze(1)
        return shift, scale

class AdaLNWrappedLayerNorm(nn.Module):
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

class BEBLaDIIPhase4aEval(nn.Module):
    def __init__(self, embedding_model_path: str, modernbert_path: str, t_emb_dim: int = 256):
        super().__init__()
        _qwen_base = AutoModel.from_pretrained(embedding_model_path, torch_dtype=torch.bfloat16)
        self.qwen_embeddings = _qwen_base.get_input_embeddings()
        del _qwen_base
        for p in self.qwen_embeddings.parameters(): p.requires_grad = False
        
        self.encoder = LatentEncoder()
        for p in self.encoder.parameters(): p.requires_grad = False
        self.encoder.to(torch.bfloat16)

        dus_wrapper = DUSModel.from_scratch(config={"base_model_id": modernbert_path}, weights_path=None)
        self.dus = dus_wrapper.model
        if hasattr(self.dus, "gradient_checkpointing_disable"):
            self.dus.gradient_checkpointing_disable()
        if hasattr(self.dus, "_maybe_set_compile"):
            self.dus._maybe_set_compile = lambda *a, **kw: None
            
        type(self.dus).device = property(lambda self: next(self.parameters()).device)
        type(self.dus).dtype  = property(lambda self: torch.float32)

        hidden_dim = 1024
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
        nn.init.zeros_(self.t_joint_proj.weight)
        nn.init.zeros_(self.t_joint_proj.bias)
        with torch.no_grad():
            self.t_joint_proj.weight[:, t_emb_dim:] = torch.eye(t_emb_dim)

        num_layers = len(self.dus.layers)
        self.adaLN_attn = nn.ModuleList([AdaLNModulation(t_emb_dim, hidden_dim) for _ in range(num_layers)])
        self.adaLN_mlp = nn.ModuleList([AdaLNModulation(t_emb_dim, hidden_dim) for _ in range(num_layers)])
        
        for i, layer in enumerate(self.dus.layers):
            layer.attn_norm = AdaLNWrappedLayerNorm(layer.attn_norm, self.adaLN_attn[i])
            layer.mlp_norm = AdaLNWrappedLayerNorm(layer.mlp_norm, self.adaLN_mlp[i])

        sep_token_path = "C:/Experiments/BEBLaDII/storage/components/sep_token.pt"
        if os.path.exists(sep_token_path):
            sep_tensor = torch.load(sep_token_path, map_location="cpu", weights_only=False).float()
            self.register_buffer("sep_embed", sep_tensor)
        else:
            self.register_buffer("sep_embed", torch.zeros(hidden_dim, dtype=torch.float32))
        self.self_cond_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.self_cond_proj.weight)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, t_global: torch.Tensor, t_reported: torch.Tensor | None = None, self_cond: torch.Tensor | None = None, z_noisy_override: torch.Tensor | None = None):
        with torch.no_grad():
            qwen_embeds = self.qwen_embeddings(input_ids)
            z_clean, _, _ = self.encoder(qwen_embeds)
            B, T, D = z_clean.shape
            z_clean_f = safe_normalize(z_clean.float(), dim=-1)
            
            if z_noisy_override is not None:
                z_noisy = z_noisy_override
            else:
                # If t_reported is passed, use it for noise generation as well
                t_actual = t_reported if t_reported is not None else t_global
                z_noisy = spherical_noise(z_clean_f, t_actual)

        if t_reported is None:
            t_reported = t_global.unsqueeze(1).expand(B, T)
        
        t_sin_global = self.t_sin_embed(t_global)
        t_emb_global = self.t_proj_global(t_sin_global)

        t_sin_token = self.t_sin_embed(t_reported)
        t_emb_token = self.t_proj_token(t_sin_token)

        cond = torch.cat([t_emb_token, t_emb_global.unsqueeze(1).expand(-1, T, -1)], dim=-1)
        t_emb = self.t_joint_proj(cond)

        sep_t_emb = torch.zeros(B, 1, t_emb.shape[-1], device=t_emb.device, dtype=t_emb.dtype)
        t_emb_extended = torch.cat([sep_t_emb, t_emb], dim=1)

        x_in = z_noisy.float()
        if self_cond is not None:
            x_in = x_in + self.self_cond_proj(self_cond.float().to(x_in.device))
        
        sep_prefix = self.sep_embed.unsqueeze(0).unsqueeze(0).expand(B, 1, -1).to(x_in.dtype)
        dus_input_extended = torch.cat([sep_prefix, x_in], dim=1)
        attention_mask_extended = F.pad(attention_mask, (1, 0), value=1)

        for layer in self.dus.layers:
            if isinstance(layer.attn_norm, AdaLNWrappedLayerNorm):
                layer.attn_norm._current_t_emb = t_emb_extended
            if isinstance(layer.mlp_norm, AdaLNWrappedLayerNorm):
                layer.mlp_norm._current_t_emb = t_emb_extended

        dus_outputs = self.dus(inputs_embeds=dus_input_extended, attention_mask=attention_mask_extended, output_hidden_states=False)
        
        pre_norm = dus_outputs.last_hidden_state[:, 1:, :].float()
        dus_final_raw = self.dus.final_norm(pre_norm.to(self.dus.dtype)).float()
        h_39 = safe_normalize(dus_final_raw, dim=-1)

        # Identity Gate Inference (ADR 065) - t_global acts as the gate
        gate_t = t_global.view(B, 1, 1).float()
        dus_final_blended = gate_t * h_39 + (1.0 - gate_t) * x_in
        dus_final = safe_normalize(dus_final_blended, dim=-1)

        return {"z_clean": z_clean_f, "z_noisy": z_noisy, "t": t_global, "dus_final": dus_final, "h_39": h_39}


# --- Diagnostic Utilities ---
def output_msg(msg, file=None):
    print(msg)
    if file:
        file.write(msg + "\n")

def analyze_weight_norms(diff_model, file):
    output_msg("\n" + "=" * 70, file)
    output_msg("БЛОК 1: Нормы весов (DUS + AdaLN)", file)
    output_msg("=" * 70, file)
    
    layer_norms = {}
    for name, param in diff_model.dus.named_parameters():
        m = re.search(r'layers\.([0-9]+)\.', name)
        if m:
            idx = int(m.group(1))
            layer_norms.setdefault(idx, []).append(param.data.float().norm().item())
            
    avg_norms = {i: sum(v) / len(v) for i, v in layer_norms.items()}
    output_msg(f"  {'Layer':>6} | {'AvgWeightNorm':>14}", file)
    for i in sorted(avg_norms.keys()):
        if i % 10 == 0 or i == max(avg_norms.keys()):
            output_msg(f"  {i:>6} | {avg_norms[i]:>14.4f}", file)

    for comp_name, comp_list in [("adaLN_attn", diff_model.adaLN_attn), ("adaLN_mlp", diff_model.adaLN_mlp)]:
        norms = [m.modulation[-1].weight.data.float().norm().item() for m in comp_list]
        avg = sum(norms) / len(norms)
        output_msg(f"  {comp_name}: mean_w_norm={avg:.4f}, min={min(norms):.4f}, max={max(norms):.4f}", file)

def analyze_adaln_sensitivity(diff_model, device, file):
    output_msg("\n" + "=" * 70, file)
    output_msg("БЛОК 2: AdaLN Sensitivity (delta t_global=0.01 vs 0.99)", file)
    output_msg("=" * 70, file)
    
    t_vals = torch.tensor([0.01, 0.99], device=device)
    with torch.no_grad():
        t_sin_global = diff_model.t_sin_embed(t_vals)
        t_emb_global = diff_model.t_proj_global(t_sin_global)
        
        # t_reported is also varied
        t_sin_token = diff_model.t_sin_embed(t_vals.unsqueeze(1))
        t_emb_token = diff_model.t_proj_token(t_sin_token)
        
        cond = torch.cat([t_emb_token, t_emb_global.unsqueeze(1)], dim=-1)
        t_emb = diff_model.t_joint_proj(cond).squeeze(1) # [2, dim]

    output_msg(f"  Первые 5 слоев: delta(t=0.01, t=0.99)")
    output_msg(f"  {'L':>3} | {'attn|Δshift|':>12} | {'attn|Δscale|':>12} | {'mlp|Δshift|':>11} | {'mlp|Δscale|':>11}", file)
    output_msg(f"  {'-'*3}-+-{'-'*12}-+-{'-'*12}-+-{'-'*11}-+-{'-'*11}", file)
    
    with torch.no_grad():
        for i in range(min(5, len(diff_model.adaLN_attn))):
            sh_a1, sc_a1 = diff_model.adaLN_attn[i](t_emb[0:1])
            sh_a2, sc_a2 = diff_model.adaLN_attn[i](t_emb[1:2])
            sh_m1, sc_m1 = diff_model.adaLN_mlp[i](t_emb[0:1])
            sh_m2, sc_m2 = diff_model.adaLN_mlp[i](t_emb[1:2])
            output_msg(f"  {i:>3} | {(sh_a2-sh_a1).abs().mean().item():>12.6f} | {(sc_a2-sc_a1).abs().mean().item():>12.6f} | "
                       f"{(sh_m2-sh_m1).abs().mean().item():>11.6f} | {(sc_m2-sc_m1).abs().mean().item():>11.6f}", file)

def compute_topological_metrics(hidden_states):
    centered = hidden_states - hidden_states.mean(dim=0, keepdim=True)
    _, s, _ = torch.linalg.svd(centered.float(), full_matrices=False)
    iso = (s.sum()**2) / (len(s) * (s**2).sum()) if len(s) > 0 else torch.tensor(0.0)
    rank1 = (s[0]**2) / (s**2).sum() if len(s) > 0 else torch.tensor(0.0)
    norms = hidden_states.norm(dim=-1)
    cv = norms.std() / norms.mean() if norms.mean() > 0 else torch.tensor(0.0)
    return iso.item(), rank1.item(), cv.item()

def analyze_topology_and_identity(diff_model, tokenizer, device, file):
    output_msg("\n" + "=" * 70, file)
    output_msg("БЛОК 3 & 4: Topology & Identity Preservation", file)
    output_msg("=" * 70, file)

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Мама мыла раму, а папа чинил телевизор.",
        "Quantum mechanics governs subatomic particle interactions."
    ]
    
    all_input_ids, all_masks = [], []
    for txt in texts:
        tok = tokenizer(txt, return_tensors="pt", add_special_tokens=False)
        all_input_ids.append(tok.input_ids[0])
        all_masks.append(tok.attention_mask[0])
        
    max_len = max(len(ids) for ids in all_input_ids)
    input_ids_batch = torch.stack([F.pad(ids, (0, max_len - len(ids)), value=tokenizer.pad_token_id or 0) for ids in all_input_ids]).to(device)
    mask_batch = torch.stack([F.pad(m, (0, max_len - len(m)), value=0) for m in all_masks]).to(device)

    with torch.no_grad():
        qwen_embeds = diff_model.qwen_embeddings(input_ids_batch)
        z_clean, _, _ = diff_model.encoder(qwen_embeds)
        z_clean = safe_normalize(z_clean.float(), dim=-1)
    active = z_clean[mask_batch.bool()]
    iso0, r0, cv0 = compute_topological_metrics(active)
    output_msg(f"  [Baseline z_clean] Iso={iso0:.4f} Rank1={r0:.4f} CV={cv0:.4f}", file)

    snap = {}
    def mk_hook(idx):
        def hook(m, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            snap[idx] = x.detach().float()
        return hook
        
    hooks = [
        diff_model.dus.layers[4].register_forward_hook(mk_hook(4)),
        diff_model.dus.layers[19].register_forward_hook(mk_hook(19)),
        diff_model.dus.layers[-1].register_forward_hook(mk_hook(39))
    ]

    t_list = [0.1, 0.5, 0.9]
    output_msg(f"\n  {'Text (Identity)':<10} | " + " | ".join(f"t={t:>5.1f}" for t in t_list), file)
    output_msg(f"  {'-'*10}-+-" + "-+-".join(["-"*7] * len(t_list)), file)

    cos_sims = {i: [] for i in range(len(texts))}
    cos_sims_h39 = {i: [] for i in range(len(texts))}

    for t_val in t_list:
        with torch.no_grad():
            t_tensor = torch.tensor([t_val] * len(texts), device=device)
            z_noisy = spherical_noise(z_clean, t_tensor)
            
            out_sc = diff_model(input_ids_batch, mask_batch, t_global=t_tensor, z_noisy_override=z_noisy)
            self_cond_est = out_sc["dus_final"].detach()
            out = diff_model(input_ids_batch, mask_batch, t_global=t_tensor, self_cond=self_cond_est, z_noisy_override=z_noisy)
        
        for i in range(len(texts)):
            m_i = mask_batch[i].bool()
            pred = out["dus_final"][i][m_i]
            pred_h39 = out["h_39"][i][m_i]
            target = z_clean[i][m_i]
            cs = (pred * target).sum(dim=-1).mean().item()
            cs_h39 = (pred_h39 * target).sum(dim=-1).mean().item()
            cos_sims[i].append(cs)
            cos_sims_h39[i].append(cs_h39)

        output_msg(f"\n  --- Topology at t={t_val} ---", file)
        for layer_idx, label in [(4, "L4 "), (19, "L19"), (39, "L39")]:
            if layer_idx in snap:
                active_h = snap[layer_idx][:, 1:, :][mask_batch.bool()]
                iso, r1, cv = compute_topological_metrics(active_h)
                flag = " *** COLLAPSE ***" if r1 > 0.8 else (" ! rank1 high" if r1 > 0.5 else "")
                output_msg(f"  {label}: Iso={iso:.4f} R1={r1:.4f} CV={cv:.4f}{flag}", file)
                
    for h in hooks: h.remove()
    
    output_msg(f"\n  Identity Cos(Pred, Clean) - POST GATE:", file)
    for i, (label, txt) in enumerate(zip(["English", "Russian", "Science"], texts)):
        output_msg(f"  {label:<10} | " + " | ".join(f"{v:>7.4f}" for v in cos_sims[i]), file)
        
    output_msg(f"\n  Identity Cos(h39, Clean) - PRE GATE:", file)
    for i, (label, txt) in enumerate(zip(["English", "Russian", "Science"], texts)):
        output_msg(f"  {label:<10} | " + " | ".join(f"{v:>7.4f}" for v in cos_sims_h39[i]), file)


def analyze_hierarchical_denoising(diff_model, texts, tokenizer, device, file):
    output_msg("\n======================================================================", file)
    output_msg("БЛОК 5: Hierarchical Token Denoising (Phase 4 Simulation)", file)
    output_msg("======================================================================", file)

    encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt", add_special_tokens=False)
    input_ids_batch = encoded.input_ids.to(device)
    mask_batch = encoded.attention_mask.to(device)
    B, T = input_ids_batch.shape

    with torch.no_grad():
        qwen_embeds = diff_model.qwen_embeddings(input_ids_batch)
        z_clean, _, _ = diff_model.encoder(qwen_embeds)
        z_clean = safe_normalize(z_clean.float(), dim=-1)

    t_globals = [0.2, 0.5, 0.8]
    
    for t_g in t_globals:
        output_msg(f"\n  --- t_global = {t_g} ---", file)
        
        t_min = max(0.02, t_g - 0.3)
        t_max = min(1.0, t_g + 0.3)
        
        # Simulate hierarchical noise for sequence
        t_actual = torch.rand(B, T, device=device) * (t_max - t_min) + t_min
        t_reported = t_actual  # The true noise level of each token
        
        t_global_tensor = torch.tensor([t_g] * B, device=device)
        
        with torch.no_grad():
            z_noisy = spherical_noise(z_clean, t_actual)
            
            # Pass 1: Self-Conditioning estimate
            out_sc = diff_model(input_ids_batch, mask_batch, t_global=t_global_tensor, t_reported=t_reported, z_noisy_override=z_noisy)
            self_cond_est = out_sc["dus_final"].detach()
            
            # Pass 2: Denoise
            out = diff_model(input_ids_batch, mask_batch, t_global=t_global_tensor, t_reported=t_reported, self_cond=self_cond_est, z_noisy_override=z_noisy)
            
        h_39 = out["h_39"]
        
        for i in range(len(texts)):
            m_i = mask_batch[i].bool()
            z_c_i = z_clean[i][m_i]
            h_39_i = h_39[i][m_i]
            t_a_i = t_actual[i][m_i]
            
            cos_per_token = (h_39_i * z_c_i).sum(dim=-1)
            
            # Group into bins based on t_actual
            lo_mask = t_a_i < 0.3
            mid_mask = (t_a_i >= 0.3) & (t_a_i <= 0.7)
            hi_mask = t_a_i > 0.7
            
            lo_cos = cos_per_token[lo_mask].mean().item() if lo_mask.sum() > 0 else 0.0
            mid_cos = cos_per_token[mid_mask].mean().item() if mid_mask.sum() > 0 else 0.0
            hi_cos = cos_per_token[hi_mask].mean().item() if hi_mask.sum() > 0 else 0.0
            
            cat = "English" if i == 0 else "Russian" if i == 1 else "Science"
            output_msg(f"  {cat[:7]:<7} | lo (t<0.3): {lo_cos:.4f} | mid (0.3<t<0.7): {mid_cos:.4f} | hi (t>0.7): {hi_cos:.4f}", file)


def load_and_evaluate_checkpoint(ckpt_path: str, diff_model: nn.Module, tokenizer, device: torch.device, file):
    output_msg(f"\n{'='*80}\nEvaluating Checkpoint: {os.path.basename(ckpt_path)}\n{'='*80}", file)
    
    state = torch.load(ckpt_path, map_location="cpu")
    
    # In EMA training, keys might be under dus_ema
    dus_raw = state.get("dus", {})
    if dus_raw:
        clean_dus_raw = {}
        for k, v in dus_raw.items():
            clean_k = k.replace("_orig_module.", "").replace("student.model.", "").replace("model.", "")
            clean_dus_raw[clean_k] = v
        diff_model.dus.load_state_dict(clean_dus_raw, strict=False)

    dus_ema = state.get("dus_ema", {})
    if dus_ema:
        clean_dus_ema = {}
        for k, v in dus_ema.items():
            clean_k = k.replace("_orig_module.", "").replace("student.model.", "").replace("model.", "")
            clean_dus_ema[clean_k] = v
        
        missing, unexpected = diff_model.dus.load_state_dict(clean_dus_ema, strict=False)
        if missing:
            output_msg(f"  [WARN] DUS_EMA missing keys: {len(missing)} (expected if EMA only tracks trainable params)", file)
        
    # Phase 4 specific components
    for name in ["adaLN_attn", "adaLN_mlp", "t_proj_global", "t_proj_token", "t_joint_proj", "sep_embed", "self_cond_proj"]:
        ema = state.get(f"{name}_ema", state.get(name, {}))
        if ema:
            if name == "sep_embed":
                diff_model.sep_embed.copy_(ema)
            else:
                getattr(diff_model, name).load_state_dict(ema, strict=True)
        else:
            # Fallback for simple state_dict structure
            comp_state = {k.replace(f"{name}.", ""): v for k, v in state.items() if k.startswith(f"{name}.")}
            if comp_state:
                getattr(diff_model, name).load_state_dict(comp_state, strict=True)

    # Load LatentEncoder
    enc_paths = [
        "C:/Experiments/BEBLaDII/experiments/phase 1/planB_phase1_checkpoints_phase1_vae_step_20000.pth",
        "C:/Experiments/BEBLaDII/BEBLaDII-planB-Phase3-Data/planB_phase1_checkpoints_phase1_vae_step_20000.pth",
    ]
    enc_path = next((p for p in enc_paths if os.path.exists(p)), None)
    if enc_path:
        st = torch.load(enc_path, map_location="cpu")
        diff_model.encoder.load_state_dict({k.replace("encoder.", ""): v for k, v in st.get("encoder", st).items()}, strict=False)
    else:
        output_msg("  [WARN] LatentEncoder weights not found! Evaluation will use random VAE weights!", file)

    diff_model.eval()
    
    analyze_weight_norms(diff_model, file)
    analyze_adaln_sensitivity(diff_model, device, file)
    analyze_topology_and_identity(diff_model, tokenizer, device, file)
    test_phrases = [
        "The quick brown fox jumps over the lazy dog.",
        "Мама мыла раму, а папа чинил телевизор.",
        "Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics to solve problems too complex for classical computers."
    ]
    analyze_hierarchical_denoising(diff_model, test_phrases, tokenizer, device, file)

def compute_layer_divergence(model_module, file):
    layer_params = {}
    for name, param in model_module.named_parameters():
        m = re.search(r'layers\.([0-9]+)\.', name)
        if m:
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
    
    avg_div = sum(divergences) / len(divergences) if divergences else 0.0
    output_msg(f"\n  Average Layer Divergence (Block 1 vs Block 2): {avg_div:.4f}", file)


def main():
    checkpoints_dir = "C:/Experiments/BEBLaDII/experiments/phase 4/local_checkpoints"
    out_path = os.path.join(checkpoints_dir, "evaluation_results.txt")
    f = open(out_path, "w", encoding="utf-8")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_msg(f"Device: {device}", f)
    
    base_qwen = "Qwen/Qwen2.5-1.5B"
    base_modernbert = "answerdotai/ModernBERT-large"

    tokenizer = AutoTokenizer.from_pretrained(base_qwen)
    diff_model = BEBLaDIIPhase4aEval(embedding_model_path=base_qwen, modernbert_path=base_modernbert)
    diff_model.to(device)
    
    ckpts = glob.glob(os.path.join(checkpoints_dir, "*.pth"))
    if not ckpts:
        output_msg(f"No .pth checkpoints found in {checkpoints_dir}", f)
        f.close()
        return

    # Evaluate each checkpoint found
    for ckpt in sorted(ckpts):
        load_and_evaluate_checkpoint(ckpt, diff_model, tokenizer, device, f)
        compute_layer_divergence(diff_model.dus, f)
        
    output_msg(f"\nAll done! Results saved to {out_path}", f)
    f.close()
    print(f"Results written to {out_path}")

if __name__ == "__main__":
    main()
