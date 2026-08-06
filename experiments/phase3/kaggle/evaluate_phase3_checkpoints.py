import os
import sys
import math
import subprocess
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# --- Environment Setup ---
PROJECT_ROOT = "/kaggle/working/BEBLaDII"
REPO_URL = "https://github.com/Laeryid/BEBLaDII.git"

if os.path.exists("/kaggle/input"):
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
    from src.beb_la_dii.utils.loss import safe_normalize
except ImportError as e:
    print(f"Warning: Не удалось импортировать модули проекта. Ошибка: {e}")

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
    
    if keyword:
        kaggle_input = pathlib.Path("/kaggle/input")
        if kaggle_input.exists():
            for config_file in kaggle_input.rglob("config.json"):
                if keyword in str(config_file).lower():
                    return str(config_file.parent)
                    
    raise FileNotFoundError(f"КРИТИЧЕСКАЯ ОШИБКА: Директория с 'config.json' для модели '{keyword}' не найдена по пути {base_path}. Убедись, что ты прикрепил (Add Data) нужные датасеты (Qwen/ModernBERT) к своему Kaggle ноутбуку!")

# --- Architecture Components (Standalone for Kaggle) ---
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
    return torch.cos(t * (math.pi / 2))

def spherical_noise(x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    B, T, D = x0.shape
    eps = safe_normalize(torch.randn_like(x0), dim=-1)
    mu = cosine_noise_schedule(t).view(B, 1, 1)
    x_t = mu * x0 + (1.0 - mu) * eps
    return safe_normalize(x_t, dim=-1)

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

class BEBLaDIIPhase3(nn.Module):
    def __init__(self, embedding_model_path: str, modernbert_path: str, t_emb_dim: int = 256):
        super().__init__()
        _qwen_base = AutoModel.from_pretrained(embedding_model_path, torch_dtype=torch.bfloat16, local_files_only=True)
        self.qwen_embeddings = _qwen_base.get_input_embeddings()
        del _qwen_base
        for p in self.qwen_embeddings.parameters(): p.requires_grad = False
        
        self.encoder = LatentEncoder()
        for p in self.encoder.parameters(): p.requires_grad = False
        self.encoder.to(torch.bfloat16)

        dus_wrapper = DUSModel.from_scratch(config={"base_model_id": modernbert_path}, weights_path=None, local_files_only=True)
        self.dus = dus_wrapper.model
        if hasattr(self.dus, "gradient_checkpointing_disable"):
            self.dus.gradient_checkpointing_disable()
        if hasattr(self.dus, "_maybe_set_compile"):
            self.dus._maybe_set_compile = lambda *a, **kw: None
            
        type(self.dus).device = property(lambda self: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        type(self.dus).dtype  = property(lambda self: torch.float32)

        hidden_dim = 1024
        self.t_sin_embed = SinusoidalEmbedding(t_emb_dim)
        self.t_proj = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(t_emb_dim * 4, t_emb_dim),
        )

        num_layers = len(self.dus.layers)
        self.adaLN_attn = nn.ModuleList([AdaLNModulation(t_emb_dim, hidden_dim) for _ in range(num_layers)])
        self.adaLN_mlp = nn.ModuleList([AdaLNModulation(t_emb_dim, hidden_dim) for _ in range(num_layers)])
        
        for i, layer in enumerate(self.dus.layers):
            layer.attn_norm.register_forward_hook(self._make_adaLN_hook(i, target="attn"))
            layer.mlp_norm.register_forward_hook(self._make_adaLN_hook(i, target="mlp"))

        self.register_buffer("sep_embed", torch.zeros(hidden_dim, dtype=torch.float32))
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

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, t: torch.Tensor):
        with torch.no_grad():
            qwen_embeds = self.qwen_embeddings(input_ids)
            z_clean, _, _ = self.encoder(qwen_embeds)
            B, T, D = z_clean.shape
            z_clean_f = safe_normalize(z_clean.float(), dim=-1)
            z_noisy = spherical_noise(z_clean_f, t)

        t_sin = self.t_sin_embed(t)
        t_emb = self.t_proj(t_sin)
        self._current_t_emb = t_emb

        x_in = z_noisy.float()
        sep_prefix = self.sep_embed.unsqueeze(0).unsqueeze(0).expand(B, 1, -1).to(x_in.dtype)
        dus_input_extended = torch.cat([sep_prefix, x_in], dim=1)
        attention_mask_extended = F.pad(attention_mask, (1, 0), value=1)

        dus_outputs = self.dus(inputs_embeds=dus_input_extended, attention_mask=attention_mask_extended, output_hidden_states=False)
        
        pre_norm = dus_outputs.last_hidden_state[:, 1:, :].float()
        dus_final_raw = self.dus.final_norm(pre_norm.to(self.dus.dtype)).float()
        h_39 = safe_normalize(dus_final_raw, dim=-1)

        gate_t = t.view(B, 1, 1).float()
        dus_final_blended = gate_t * h_39 + (1.0 - gate_t) * x_in
        dus_final = safe_normalize(dus_final_blended, dim=-1)

        return {"z_clean": z_clean_f, "z_noisy": z_noisy, "t": t, "dus_final": dus_final, "h_39": h_39}

# --- Diagnostic Utilities ---
def output_msg(msg, file=None):
    print(msg)
    if file:
        file.write(msg + "\n")

def encode_clean_text(text: str, tokenizer, embeddings, encoder, device, dtype):
    tok = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = tok.input_ids.to(device)
    attn_mask = tok.attention_mask.to(device)
    with torch.no_grad():
        embs = embeddings(input_ids)
        z, _, _ = encoder(embs)
    return z.to(dtype), attn_mask

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
    output_msg("БЛОК 2: AdaLN Sensitivity (delta t=0.01 vs 0.99)", file)
    output_msg("=" * 70, file)
    
    t_vals = torch.tensor([0.01, 0.99], device=device)
    with torch.no_grad():
        t_sin = diff_model.t_sin_embed(t_vals)
        t_emb = diff_model.t_proj(t_sin)

    output_msg(f"  Первые 5 слоев: delta(t=0.01, t=0.99)", file)
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

    # Topology Baseline
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

    # Dictionary to store cos sim per text
    cos_sims = {i: [] for i in range(len(texts))}

    for t_val in t_list:
        with torch.no_grad():
            out = diff_model(input_ids_batch, mask_batch, torch.tensor([t_val] * len(texts), device=device))
        
        # Identity
        for i in range(len(texts)):
            m_i = mask_batch[i].bool()
            pred = out["dus_final"][i][m_i]
            target = z_clean[i][m_i]
            cs = (pred * target).sum(dim=-1).mean().item()
            cos_sims[i].append(cs)

        # Topology
        output_msg(f"\n  --- Topology at t={t_val} ---", file)
        for layer_idx, label in [(4, "L4 "), (19, "L19"), (39, "L39")]:
            if layer_idx in snap:
                active_h = snap[layer_idx][mask_batch.bool()]
                iso, r1, cv = compute_topological_metrics(active_h)
                flag = " *** COLLAPSE ***" if r1 > 0.8 else (" ! rank1 high" if r1 > 0.5 else "")
                output_msg(f"  {label}: Iso={iso:.4f} R1={r1:.4f} CV={cv:.4f}{flag}", file)
                
    for h in hooks: h.remove()
    
    output_msg(f"\n  Identity Cos(Pred, Clean):", file)
    for i, (label, txt) in enumerate(zip(["English", "Russian", "Science"], texts)):
        output_msg(f"  {label:<10} | " + " | ".join(f"{v:>7.4f}" for v in cos_sims[i]), file)


def load_and_evaluate_checkpoint(ckpt_path: str, diff_model: nn.Module, tokenizer, device, file):
    output_msg(f"\n{'='*80}\nEvaluating Checkpoint: {os.path.basename(ckpt_path)}\n{'='*80}", file)
    
    state = torch.load(ckpt_path, map_location="cpu")
    
    # Extract keys
    dus_ema = state.get("dus_ema", state.get("dus", {}))
    if dus_ema:
        clean_dus = {k.replace("_orig_module.", ""): v for k, v in dus_ema.items()}
        diff_model.dus.load_state_dict(clean_dus, strict=False)
    else:
        output_msg("  [WARN] 'dus_ema' or 'dus' not found in checkpoint.", file)
        
    for name in ["adaLN_attn", "adaLN_mlp", "t_proj", "sep_embed"]:
        ema = state.get(f"{name}_ema", state.get(name, {}))
        if ema:
            if name == "sep_embed":
                diff_model.sep_embed.copy_(ema)
            else:
                getattr(diff_model, name).load_state_dict(ema, strict=True)

    diff_model.eval()
    
    analyze_weight_norms(diff_model, file)
    analyze_adaln_sensitivity(diff_model, device, file)
    analyze_topology_and_identity(diff_model, tokenizer, device, file)


def main():
    out_path = "evaluation_results.txt"
    f = open(out_path, "w", encoding="utf-8")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_msg(f"Device: {device}", f)
    
    # 1. Resolve paths
    base_qwen = resolve_model_path("/kaggle/input/datasets/ragnar123/qwen2-5-1-5b")
    base_modernbert = resolve_model_path("/kaggle/input/models/answer-ai/modernbert/transformers/large/2")
    
    output_msg(f"Qwen path: {base_qwen}", f)
    output_msg(f"ModernBERT path: {base_modernbert}", f)
    
    # 2. Init model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_qwen, local_files_only=True)
    diff_model = BEBLaDIIPhase3(embedding_model_path=base_qwen, modernbert_path=base_modernbert)
    diff_model.to(device)
    
    # 3. Find checkpoints
    input_dir = "/kaggle/input"
    checkpoints = []
    if os.path.exists(input_dir):
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.startswith("phase3_step_") and file.endswith(".pth") and "_opt" not in file:
                    checkpoints.append(os.path.join(root, file))
                    
    # Optional local fallback
    if not checkpoints and os.path.exists("local_checkpoints"):
        for file in os.listdir("local_checkpoints"):
            if file.startswith("phase3_step_") and file.endswith(".pth") and "_opt" not in file:
                checkpoints.append(os.path.join("local_checkpoints", file))
                
    if not checkpoints:
        output_msg("No checkpoints found. Terminating.", f)
        f.close()
        return
        
    def extract_step(path):
        m = re.search(r"step_(\d+)", path)
        return int(m.group(1)) if m else -1
        
    checkpoints.sort(key=extract_step)
    output_msg(f"Found {len(checkpoints)} checkpoints.", f)
    
    # 4. Evaluate
    for ckpt in checkpoints:
        load_and_evaluate_checkpoint(ckpt, diff_model, tokenizer, device, f)
        
    output_msg(f"\nAll done! Results saved to {out_path}", f)
    f.close()

if __name__ == "__main__":
    main()
