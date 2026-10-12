import os
import sys
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["WANDB_MODE"] = "disabled"

PROJECT_ROOT = "C:/Experiments/BEBLaDII"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.beb_la_dii.model.dus import DUSModel
from src.beb_la_dii.model.vae import LatentEncoder
from src.beb_la_dii.utils.loss import safe_normalize

def resolve_file_path(filename: str) -> str:
    import pathlib
    p_root = pathlib.Path(PROJECT_ROOT)
    for sub in ["BEBLaDII-planB-Phase6-Data", "BEBLaDII-planB-Phase5-Data", "BEBLaDII-planB-Phase4-Data", "BEBLaDII-planB-Phase3-Data", "experiments", "storage", "checkpoints"]:
        s = p_root / sub
        if s.exists():
            for m in s.rglob(filename):
                return str(m)
    return filename

def sample_token_noise_levels(t_global: torch.Tensor, T: int, kappa_min: float = 1.0, kappa_max: float = 8.0) -> torch.Tensor:
    dev = t_global.device
    B = t_global.shape[0]

    is_full_noise = (t_global >= 0.999)

    t_clamped = t_global.clamp(1e-4, 0.99)
    u = torch.rand(B, device=dev)
    kappa = torch.exp(math.log(kappa_min) + u * (math.log(kappa_max) - math.log(kappa_min)))
    a = (t_clamped * kappa).unsqueeze(1).expand(B, T)
    b = ((1.0 - t_clamped) * kappa).unsqueeze(1).expand(B, T)

    try:
        from torch.distributions import Beta
        t_actual = Beta(a, b).sample()
    except Exception:
        t_actual = t_clamped.unsqueeze(1).expand(B, T)

    t_actual = torch.where(is_full_noise.unsqueeze(1), torch.ones_like(t_actual), t_actual)
    t_actual = t_actual.clamp(1e-4, 1.0)
    return t_actual

def content_cos_stats(outputs):
    target = safe_normalize(outputs["z_clean"], dim=-1)
    cp = (outputs["dus_final"] * target).sum(dim=-1)
    cb = (outputs["z_noisy"] * target).sum(dim=-1)
    cm = (~outputs["void_mask"].bool()).float()
    return cp, cb, cm

def resolve_hf_snapshot(repo_id: str) -> str:
    user_profile = os.environ.get("USERPROFILE", "")
    folder_name = "models--" + repo_id.replace("/", "--")
    snap_dir = os.path.join(user_profile, ".cache", "huggingface", "hub", folder_name, "snapshots")
    if os.path.exists(snap_dir):
        subdirs = [os.path.join(snap_dir, d) for d in os.listdir(snap_dir) if os.path.isdir(os.path.join(snap_dir, d))]
        if subdirs:
            return subdirs[0]
    return repo_id

class Config:
    embedding_model_path = resolve_hf_snapshot("Qwen/Qwen2.5-1.5B")
    modernbert_path      = resolve_hf_snapshot("answerdotai/ModernBERT-large")
    local_files_only     = True
    dataset_path = resolve_file_path("train_phase6.parquet")
    val_dataset_path = resolve_file_path("val_phase6.parquet")
    encoder_weights = resolve_file_path("planB_phase1_checkpoints_phase1_vae_step_20000.pth")
    dus_weights     = resolve_file_path("phase4_step_85995.pth")
    sep_token       = os.path.join(PROJECT_ROOT, "storage/components/sep_token.pt")
    void_token      = os.path.join(PROJECT_ROOT, "storage/components/void_token.pt")
    latent_dict     = resolve_file_path("latent_dict.pt")
    batch_size    = 2
    max_length_q  = 64
    max_length_a  = 64
    learning_rate = 1e-4
    warmup_steps  = 1000
    ema_decay     = 0.998
    pace_alpha    = 0.001
    unfreeze_k_after_ca = 4
    use_gradient_checkpointing = False
    loss_t_power  = 0.0

class EMATracker:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone().float()

    def update(self, model, is_safe=None):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if is_safe is not None:
                        new_shadow = self.decay * self.shadow[name] + (1.0 - self.decay) * param.float()
                        self.shadow[name].copy_(torch.where(is_safe, new_shadow, self.shadow[name]))
                    else:
                        self.shadow[name].copy_(self.decay * self.shadow[name] + (1.0 - self.decay) * param.float())

class QADataset(Dataset):
    def __init__(self, parquet_path, tokenizer, max_length_q=512, max_length_a=512):
        self.df = pd.read_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.max_length_q = max_length_q
        self.max_length_a = max_length_a

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        q_text, a_text = str(row['Q']), str(row['A'])
        q_enc = self.tokenizer(q_text, truncation=True, max_length=self.max_length_q, padding='max_length', return_tensors='pt')
        a_enc = self.tokenizer(a_text, truncation=True, max_length=self.max_length_a, padding='max_length', return_tensors='pt')
        return {
            'input_ids_q': q_enc.input_ids.squeeze(0),
            'attention_mask_q': q_enc.attention_mask.squeeze(0),
            'input_ids_a': a_enc.input_ids.squeeze(0),
            'attention_mask_a': a_enc.attention_mask.squeeze(0),
        }

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
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
    if t.dim() == 1: t = t.view(B, 1, 1)
    elif t.dim() == 2: t = t.unsqueeze(-1)
    mu = cosine_noise_schedule(t)
    sigma = torch.sin(t * (math.pi / 2))
    return safe_normalize(mu * x0 + sigma * eps, dim=-1)

class AdaLNModulation(nn.Module):
    def __init__(self, t_emb_dim: int, hidden_dim: int):
        super().__init__()
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, 2 * hidden_dim))
        nn.init.zeros_(self.modulation[-1].weight)
        bias = torch.zeros(2 * hidden_dim)
        bias[hidden_dim:] = 1.0
        self.modulation[-1].bias = nn.Parameter(bias)
    def forward(self, t_emb: torch.Tensor) -> tuple:
        out = self.modulation(t_emb)
        shift, scale = out.chunk(2, dim=-1)
        if shift.dim() == 2: return shift.unsqueeze(1), scale.unsqueeze(1)
        return shift, scale

class AdaLNWrappedLayerNorm(nn.Module):
    def __init__(self, original_norm, adaln_module):
        super().__init__()
        self.original_norm = original_norm
        self.adaln = adaln_module
        self._current_t_emb = None
    def forward(self, x):
        out = self.original_norm(x)
        if self._current_t_emb is None: return out
        shift, scale = self.adaln(self._current_t_emb)
        return out * scale.to(out.dtype) + shift.to(out.dtype)

class CAPromptLayer(nn.Module):
    def __init__(self, dim=1024, t_emb_dim=256):
        super().__init__()
        self._disable = False
        self.last_ca_ratio = 0.0

        self.norm1 = nn.RMSNorm(dim)
        self.norm_q = nn.RMSNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.adaln_ca = AdaLNModulation(t_emb_dim, dim)

        self.norm2 = nn.RMSNorm(dim)
        self.qkv_proj_sa = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj_sa = nn.Linear(dim, dim, bias=False)
        self.adaln_sa = AdaLNModulation(t_emb_dim, dim)

        self.gate = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.qkv_proj_sa.weight)
        nn.init.zeros_(self.out_proj_sa.weight)

    def forward(self, A, Q, mask_Q=None, warmup_factor=1.0, t_emb=None):
        if getattr(self, '_disable', False):
            return A

        # --- Cross-Attention ---
        Q_norm = self.norm_q(Q)
        A_norm = self.norm1(A)
        if t_emb is not None:
            shift_ca, scale_ca = self.adaln_ca(t_emb)
            A_norm = A_norm * scale_ca + shift_ca

        q = self.q_proj(A_norm)
        kv = self.kv_proj(Q_norm)
        k, v = kv.chunk(2, dim=-1)

        attn_mask = None
        if mask_Q is not None:
            attn_mask = mask_Q.view(A.shape[0], 1, 1, -1).expand(-1, 1, A.shape[1], -1).bool()

        B, T_a, D = A.shape
        heads = 16
        head_dim = D // heads

        q = q.view(B, T_a, heads, head_dim).transpose(1, 2)
        k = k.view(B, -1, heads, head_dim).transpose(1, 2)
        v = v.view(B, -1, heads, head_dim).transpose(1, 2)

        ca_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        ca_out = ca_out.transpose(1, 2).reshape(B, T_a, D)
        ca_out = self.out_proj(ca_out)
        with torch.no_grad():
            self.last_ca_ratio = ca_out.norm(dim=-1).mean() / (A.norm(dim=-1).mean() + 1e-8)
        A = A + warmup_factor * ca_out

        # --- Self-Attention ---
        A_norm2 = self.norm2(A)
        if t_emb is not None:
            shift_sa, scale_sa = self.adaln_sa(t_emb)
            A_norm2 = A_norm2 * scale_sa + shift_sa

        qkv = self.qkv_proj_sa(A_norm2)
        q_sa, k_sa, v_sa = qkv.chunk(3, dim=-1)

        q_sa = q_sa.view(B, T_a, heads, head_dim).transpose(1, 2)
        k_sa = k_sa.view(B, T_a, heads, head_dim).transpose(1, 2)
        v_sa = v_sa.view(B, T_a, heads, head_dim).transpose(1, 2)

        sa_out = F.scaled_dot_product_attention(q_sa, k_sa, v_sa)
        sa_out = sa_out.transpose(1, 2).reshape(B, T_a, D)
        sa_out = self.out_proj_sa(sa_out)

        A = A + warmup_factor * sa_out
        return A

def _unwrap_module(m):
    while hasattr(m, "module") or hasattr(m, "_orig_module"):
        m = getattr(m, "module", None) or getattr(m, "_orig_module", None)
    return m

def _ca_inner(ca):
    return _unwrap_module(ca)

class Phase6BlockWrapper(nn.Module):
    def __init__(self, original_layer, ca_layer=None):
        super().__init__()
        self.original_layer = original_layer
        self.ca_layer = ca_layer
    @property
    def attention_type(self): return self.original_layer.attention_type
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        out = self.original_layer(hidden_states, attention_mask=attention_mask, **kwargs)
        is_tuple = isinstance(out, (tuple, list))
        hidden = out[0] if is_tuple else out

        if self.ca_layer is not None:
            ca_inner = _ca_inner(self.ca_layer)
            if not getattr(ca_inner, '_disable', False):
                Z_prompt = getattr(self.ca_layer, '_current_Z_prompt', None)
                if Z_prompt is None:
                    Z_prompt = getattr(ca_inner, '_current_Z_prompt', None)
                mask_Q = getattr(self.ca_layer, '_current_mask_Q', None)
                if mask_Q is None:
                    mask_Q = getattr(ca_inner, '_current_mask_Q', None)
                warmup_factor = getattr(self.ca_layer, '_current_warmup_factor', 1.0)
                t_emb = getattr(self.ca_layer, '_current_t_emb', None)
                if t_emb is None:
                    t_emb = getattr(ca_inner, '_current_t_emb', None)
                if Z_prompt is not None:
                    sep = hidden[:, 0:1, :]
                    ans = hidden[:, 1:, :]
                    ans = self.ca_layer(ans, Z_prompt, mask_Q, warmup_factor, t_emb=t_emb)
                    hidden = torch.cat([sep, ans], dim=1)

        if is_tuple:
            return (hidden,) + tuple(out[1:])
        return hidden

class BEBLaDIIPhase6(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        _qwen = AutoModel.from_pretrained(config.embedding_model_path, torch_dtype=torch.bfloat16, local_files_only=config.local_files_only)
        self.qwen_embeddings = _qwen.get_input_embeddings()
        del _qwen

        self.encoder = LatentEncoder()
        if os.path.exists(config.encoder_weights):
            state = torch.load(config.encoder_weights, map_location="cpu", weights_only=False)
            if "encoder" in state: state = state["encoder"]
            self.encoder.load_state_dict(state, strict=False)
        self.encoder.to(torch.bfloat16)

        dus_wrapper = DUSModel.from_scratch(config={"base_model_id": config.modernbert_path}, weights_path=None, local_files_only=config.local_files_only)
        self.dus = dus_wrapper.model

        t_emb_dim = 256
        hidden_dim = 1024
        self.t_sin_embed = SinusoidalEmbedding(t_emb_dim)
        self.t_proj_global = nn.Sequential(nn.Linear(t_emb_dim, t_emb_dim * 4), nn.SiLU(), nn.Linear(t_emb_dim * 4, t_emb_dim))
        self.t_proj_token = nn.Sequential(nn.Linear(t_emb_dim, t_emb_dim * 4), nn.SiLU(), nn.Linear(t_emb_dim * 4, t_emb_dim))
        self.t_joint_proj = nn.Linear(t_emb_dim * 2, t_emb_dim)

        self.adaLN_attn = nn.ModuleList([AdaLNModulation(t_emb_dim, hidden_dim) for _ in range(40)])
        self.adaLN_mlp = nn.ModuleList([AdaLNModulation(t_emb_dim, hidden_dim) for _ in range(40)])
        for i, layer in enumerate(self.dus.layers):
            layer.attn_norm = AdaLNWrappedLayerNorm(layer.attn_norm, self.adaLN_attn[i])
            layer.mlp_norm = AdaLNWrappedLayerNorm(layer.mlp_norm, self.adaLN_mlp[i])

        self.register_buffer("sep_embed", torch.load(config.sep_token).float())
        self.register_buffer("void_embed", torch.load(config.void_token).float())
        self.register_buffer("latent_dict", torch.load(config.latent_dict).to(torch.bfloat16))

        if os.path.exists(config.dus_weights):
            state = torch.load(config.dus_weights, map_location="cpu", weights_only=False)
            dus_state = state.get("dus_ema", state.get("dus", state.get("model_state_dict", state.get("model", state))))
            clean_dus_state = {k.replace("student.model.", "").replace("model.", "").replace("_orig_module.", ""): v for k, v in dus_state.items()}
            self.dus.load_state_dict(clean_dus_state, strict=False)

            if "adaLN_attn_ema" in state or "adaLN_attn" in state:
                ada_attn_st = state.get("adaLN_attn_ema", state.get("adaLN_attn"))
                clean_ada_attn = {k.replace("_orig_module.", ""): v for k, v in ada_attn_st.items()}
                self.adaLN_attn.load_state_dict(clean_ada_attn, strict=False)

            if "adaLN_mlp_ema" in state or "adaLN_mlp" in state:
                ada_mlp_st = state.get("adaLN_mlp_ema", state.get("adaLN_mlp"))
                clean_ada_mlp = {k.replace("_orig_module.", ""): v for k, v in ada_mlp_st.items()}
                self.adaLN_mlp.load_state_dict(clean_ada_mlp, strict=False)

            if "t_proj_global_ema" in state or "t_proj_global" in state:
                st_glob = state.get("t_proj_global_ema", state.get("t_proj_global"))
                self.t_proj_global.load_state_dict({k.replace("_orig_module.", ""): v for k, v in st_glob.items()}, strict=False)

            if "t_proj_token_ema" in state or "t_proj_token" in state:
                st_tok = state.get("t_proj_token_ema", state.get("t_proj_token"))
                self.t_proj_token.load_state_dict({k.replace("_orig_module.", ""): v for k, v in st_tok.items()}, strict=False)

            if "t_joint_proj_ema" in state or "t_joint_proj" in state:
                st_joint = state.get("t_joint_proj_ema", state.get("t_joint_proj"))
                self.t_joint_proj.load_state_dict({k.replace("_orig_module.", ""): v for k, v in st_joint.items()}, strict=False)

        for p in self.parameters(): p.requires_grad = False

        self.ca_layers = nn.ModuleDict({
            "12": CAPromptLayer(1024, t_emb_dim=256),
            "24": CAPromptLayer(1024, t_emb_dim=256),
            "36": CAPromptLayer(1024, t_emb_dim=256),
        })
        for i in [11, 23, 35]:
            self.dus.layers[i] = Phase6BlockWrapper(self.dus.layers[i], self.ca_layers[str(i+1)])

        for p in self.ca_layers.parameters(): p.requires_grad = True
        unfreeze_k = getattr(config, "unfreeze_k_after_ca", 0)
        if unfreeze_k > 0:
            ca_indices = [11, 23, 35]
            unfreeze_indices = []
            for idx in ca_indices:
                for k in range(1, unfreeze_k + 1):
                    if idx + k < len(self.dus.layers):
                        unfreeze_indices.append(idx + k)
            for i, layer in enumerate(self.dus.layers):
                if i in unfreeze_indices:
                    for p in layer.parameters():
                        p.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "qwen_embeddings"): self.qwen_embeddings.eval()
        if hasattr(self, "encoder"): self.encoder.eval()

    def forward(self, input_ids_q, attention_mask_q, input_ids_a, attention_mask_a, warmup_factor=1.0, t_global=None, disable_ca: bool = False):
        B, T_a = input_ids_a.shape
        with torch.no_grad():
            qwen_embeds_a = self.qwen_embeddings(input_ids_a)
            Z_A_clean, _, _ = self.encoder(qwen_embeds_a)
            Z_A_clean = safe_normalize(Z_A_clean.float(), dim=-1)

            if not disable_ca:
                qwen_embeds_q = self.qwen_embeddings(input_ids_q)
                Z_prompt, _, _ = self.encoder(qwen_embeds_q)
                Z_prompt = safe_normalize(Z_prompt.float(), dim=-1)

            void_embed = self.void_embed.to(Z_A_clean.dtype)
            is_void = torch.rand((B, T_a), device=Z_A_clean.device) < 0.05
            shift_k = torch.randint(0, 11, (B,), device=Z_A_clean.device)
            seq_indices = torch.arange(T_a, device=Z_A_clean.device).unsqueeze(0).expand(B, T_a)
            prefix_mask = seq_indices < shift_k.unsqueeze(1)
            is_void = is_void | prefix_mask

            content_indices = torch.cumsum((~is_void).long(), dim=1) - 1
            orig_len = attention_mask_a.sum(dim=1)
            overflow_mask = content_indices[:, -1] < (orig_len - 1)

            is_void = is_void & ~overflow_mask.unsqueeze(1)
            content_indices = torch.cumsum((~is_void).long(), dim=1) - 1
            content_indices = content_indices.clamp(min=0, max=T_a - 1)

            batch_indices = torch.arange(B, device=Z_A_clean.device).unsqueeze(1).expand(B, T_a)
            shifted_Z_A_clean = Z_A_clean[batch_indices, content_indices, :]
            shifted_attention_mask_a = attention_mask_a[batch_indices, content_indices]

            void_mask = is_void | (shifted_attention_mask_a == 0)
            void_mask_expanded = void_mask.unsqueeze(-1).expand(-1, -1, Z_A_clean.shape[-1])
            void_embed_expanded = void_embed.view(1, 1, -1).expand(B, T_a, -1)
            Z_A_clean = torch.where(void_mask_expanded, void_embed_expanded, shifted_Z_A_clean)
            Z_A_clean = safe_normalize(Z_A_clean.float(), dim=-1).to(void_embed.dtype)
            attention_mask_a = shifted_attention_mask_a

            cpu_dev = torch.device('cpu')
            if t_global is None:
                t_global_cpu = torch.randint(1, 26, (B,), device=cpu_dev).float() / 25.0
            else:
                t_global_cpu = t_global.detach().to(cpu_dev).float()
            t_actual_cpu = sample_token_noise_levels(t_global_cpu, T_a)
            t_global = t_global_cpu.to(Z_A_clean.device)
            t_actual = t_actual_cpu.to(Z_A_clean.device)

            z_noisy = spherical_noise(Z_A_clean, t_actual)

            z_noisy_bf16 = z_noisy.to(torch.bfloat16)
            RawDProx = torch.full((B, T_a), -1.0, device=z_noisy.device, dtype=torch.bfloat16)
            for dict_chunk in self.latent_dict.chunk(16, dim=0):
                sims_chunk = torch.matmul(z_noisy_bf16, dict_chunk.T)
                RawDProx = torch.maximum(RawDProx, sims_chunk.max(dim=-1).values)
            t_reported = (1.0 - RawDProx.float()).clamp(0.0, 1.0)

        # ПРОГОН ПРОМПТА
        if not disable_ca:
            t_global_prompt = torch.zeros((B,), device=Z_A_clean.device, dtype=torch.float32)
            t_sin_prompt_global = self.t_sin_embed(t_global_prompt)
            t_emb_prompt_global = self.t_proj_global(t_sin_prompt_global)

            t_sin_prompt_token = self.t_sin_embed(t_global_prompt)
            t_emb_prompt_token = self.t_proj_token(t_sin_prompt_token)

            cond_prompt = torch.cat([t_emb_prompt_token.unsqueeze(1).expand(-1, Z_prompt.shape[1], -1), t_emb_prompt_global.unsqueeze(1).expand(-1, Z_prompt.shape[1], -1)], dim=-1)
            t_emb_prompt = self.t_joint_proj(cond_prompt)

            sep_t_emb_prompt = torch.zeros(B, 1, t_emb_prompt.shape[-1], device=t_emb_prompt.device, dtype=t_emb_prompt.dtype)
            t_emb_prompt_extended = torch.cat([sep_t_emb_prompt, t_emb_prompt], dim=1)

            for layer in self.dus.layers:
                layer_to_check = layer.original_layer if isinstance(layer, Phase6BlockWrapper) else layer
                layer_to_check = _unwrap_module(layer_to_check)
                if hasattr(layer_to_check, "attn_norm"): layer_to_check.attn_norm._current_t_emb = t_emb_prompt_extended
                if hasattr(layer_to_check, "mlp_norm"): layer_to_check.mlp_norm._current_t_emb = t_emb_prompt_extended

            prompt_in = Z_prompt.float()
            sep_prefix_prompt = self.sep_embed.unsqueeze(0).unsqueeze(0).expand(B, 1, -1).to(prompt_in.dtype)
            dus_prompt_extended = torch.cat([sep_prefix_prompt, prompt_in], dim=1)

            ones_sep_prompt = torch.ones((B, 1), device=attention_mask_q.device, dtype=attention_mask_q.dtype)
            prompt_mask_extended = torch.cat([ones_sep_prompt, attention_mask_q], dim=1)

            with torch.no_grad():
                for ca in self.ca_layers.values():
                    _ca_inner(ca)._disable = True
                    ca._disable = True

                prompt_outputs = self.dus(
                    inputs_embeds=dus_prompt_extended,
                    attention_mask=prompt_mask_extended,
                    output_hidden_states=True,
                )

                Z_prompt_12 = prompt_outputs.hidden_states[12][:, 1:, :].to(Z_prompt.dtype)

                for ca in self.ca_layers.values():
                    _ca_inner(ca)._disable = False
                    ca._disable = False

            for ca in self.ca_layers.values():
                ca._current_Z_prompt = Z_prompt_12
                _ca_inner(ca)._current_Z_prompt = Z_prompt_12
        else:
            for ca in self.ca_layers.values():
                _ca_inner(ca)._disable = True
                ca._disable = True

        # ПРОГОН ХОЛСТА
        t_sin_global = self.t_sin_embed(t_global)
        t_emb_global = self.t_proj_global(t_sin_global)

        t_sin_token = self.t_sin_embed(t_reported)
        t_emb_token = self.t_proj_token(t_sin_token)

        cond = torch.cat([t_emb_token, t_emb_global.unsqueeze(1).expand(-1, T_a, -1)], dim=-1)
        t_emb = self.t_joint_proj(cond)

        sep_t_emb = torch.zeros(B, 1, t_emb.shape[-1], device=t_emb.device, dtype=t_emb.dtype)
        t_emb_extended = torch.cat([sep_t_emb, t_emb], dim=1)

        for layer in self.dus.layers:
            layer_to_check = layer.original_layer if isinstance(layer, Phase6BlockWrapper) else layer
            layer_to_check = _unwrap_module(layer_to_check)
            if hasattr(layer_to_check, "attn_norm"): layer_to_check.attn_norm._current_t_emb = t_emb_extended
            if hasattr(layer_to_check, "mlp_norm"): layer_to_check.mlp_norm._current_t_emb = t_emb_extended

        for ca in self.ca_layers.values():
            ca._current_mask_Q = attention_mask_q
            ca._current_warmup_factor = warmup_factor
            ca._current_t_emb = t_emb
            _ca_inner(ca)._current_mask_Q = attention_mask_q
            _ca_inner(ca)._current_warmup_factor = warmup_factor
            _ca_inner(ca)._current_t_emb = t_emb

        x_in = z_noisy.float()
        sep_prefix = self.sep_embed.unsqueeze(0).unsqueeze(0).expand(B, 1, -1).to(x_in.dtype)
        dus_input_extended = torch.cat([sep_prefix, x_in], dim=1)
        attention_mask_extended = torch.ones((B, T_a + 1), device=x_in.device, dtype=torch.long)

        dus_outputs = self.dus(
            inputs_embeds=dus_input_extended,
            attention_mask=attention_mask_extended,
            output_hidden_states=False,
        )

        pre_norm = dus_outputs.last_hidden_state[:, 1:, :].float()
        dus_final_raw = self.dus.final_norm(pre_norm.to(self.dus.dtype)).float()
        h_39 = safe_normalize(dus_final_raw, dim=-1)

        gate = torch.sin(t_global * (math.pi / 2)).view(B, 1, 1).to(h_39.dtype)
        dus_gated = gate * h_39 + (1.0 - gate) * x_in
        dus_final = safe_normalize(dus_gated, dim=-1)

        return {
            "z_clean": Z_A_clean,
            "z_noisy": z_noisy,
            "dus_final": dus_final,
            "t_actual": t_actual,
            "t_global": t_global,
            "void_mask": void_mask,
            "void_embed": void_embed
        }

def compute_phase6_loss(outputs, void_margin=0.0, loss_t_power=0.0):
    z_clean = outputs["z_clean"].float()
    dus_final = outputs["dus_final"].float()
    t_actual = outputs["t_actual"].float()
    void_mask = outputs["void_mask"].float()
    void_embed = outputs["void_embed"].float()

    target = safe_normalize(z_clean, dim=-1)
    cos_sim = (dus_final * target).sum(dim=-1)
    loss_el = 1.0 - cos_sim

    cos_sim_to_void = (dus_final * void_embed.view(1, 1, -1)).sum(dim=-1)
    cos_sim_target_to_void = (target * void_embed.view(1, 1, -1)).sum(dim=-1)
    content_mask = 1.0 - void_mask

    dynamic_margin = 1.0 - 0.8 * (1.0 - cos_sim_target_to_void)

    void_penalty = F.relu(cos_sim_to_void - dynamic_margin) * content_mask
    loss_el = loss_el + 0.5 * void_penalty

    if loss_t_power > 0.0:
        w_weighted = (1.0 - t_actual).pow(loss_t_power)
    else:
        w_weighted = torch.ones_like(t_actual)
    loss = (w_weighted * loss_el).sum() / w_weighted.sum().clamp(min=1e-8)

    avg_cos_sim = cos_sim.mean()
    true_void_sim = (cos_sim_to_void * void_mask).sum() / void_mask.sum().clamp(min=1e-8)
    content_to_void_sim = (cos_sim_to_void * content_mask).sum() / content_mask.sum().clamp(min=1e-8)

    return loss, avg_cos_sim, true_void_sim, content_to_void_sim


def main():
    print("Starting local diagnosis...")
    cfg = Config()
    tokenizer = AutoTokenizer.from_pretrained(cfg.embedding_model_path, local_files_only=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    dataset = QADataset(cfg.dataset_path, tokenizer, cfg.max_length_q, cfg.max_length_a)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    print(f"Dataset samples: {len(dataset)}. Initializing model...")
    model = BEBLaDIIPhase6(cfg)
    device = torch.device("cpu")
    model.to(device)

    for key in list(model.ca_layers.keys()):
        layer_idx = int(key) - 1
        model.dus.layers[layer_idx].ca_layer = model.ca_layers[key]

    ca_out_params, ca_qkv_params, ca_other_params = [], [], []
    for name, p in model.ca_layers.named_parameters():
        if p.requires_grad:
            if "out_proj" in name: ca_out_params.append(p)
            elif "q_proj" in name or "k_proj" in name or "v_proj" in name or "qkv_proj" in name: ca_qkv_params.append(p)
            else: ca_other_params.append(p)
    dus_params = []
    for name, p in model.dus.named_parameters():
        if p.requires_grad and "ca_layer" not in name:
            dus_params.append(p)

    trainable_params = ca_out_params + ca_qkv_params + ca_other_params + dus_params
    print(f"Trainable params: {len(trainable_params)}")

    optimizer = torch.optim.AdamW([
        {'params': ca_out_params, 'lr': 2e-4},
        {'params': ca_qkv_params, 'lr': 8e-5},
        {'params': ca_other_params, 'lr': 2.5e-5},
        {'params': dus_params, 'lr': 2.5e-5}
    ])
    ema_tracker = EMATracker(model, decay=cfg.ema_decay)

    named_params_dict = dict(model.named_parameters())
    param_to_name = {p: n for n, p in named_params_dict.items()}

    model.train()
    torch.manual_seed(42)

    for step, batch in enumerate(dataloader):
        if step >= 12: break
        print(f"\n=================== STEP {step + 1} ===================")
        input_ids_q = batch['input_ids_q'].to(device)
        mask_q = batch['attention_mask_q'].to(device)
        input_ids_a = batch['input_ids_a'].to(device)
        mask_a = batch['attention_mask_a'].to(device)

        print(f"Batch shapes: Q={input_ids_q.shape}, A={input_ids_a.shape}")
        print(f"Q mask active tokens: {mask_q.sum(dim=-1).tolist()}, A mask active: {mask_a.sum(dim=-1).tolist()}")

        optimizer.zero_grad()
        warmup_factor = torch.tensor(0.05, dtype=torch.float32, device=device)

        print("Forward pass...")
        outputs = model(input_ids_q, mask_q, input_ids_a, mask_a, warmup_factor=warmup_factor)

        for k, v in outputs.items():
            if torch.is_tensor(v):
                is_fin = torch.isfinite(v).all().item()
                print(f"  outputs['{k}']: finite={is_fin}, min={v.float().min().item():.4f}, max={v.float().max().item():.4f}")

        print("Loss calculation...")
        loss, avg_cos_sim, true_void_sim, content_to_void_sim = compute_phase6_loss(outputs, void_margin=0.5, loss_t_power=cfg.loss_t_power)
        print(f"  loss: {loss.item():.6f}, is_finite={torch.isfinite(loss).item()}")
        print(f"  avg_cos_sim: {avg_cos_sim.item():.4f}, true_void_sim: {true_void_sim.item():.4f}, content_to_void_sim: {content_to_void_sim.item():.4f}")

        print("Backward pass...")
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        is_finite = torch.isfinite(grad_norm)
        print(f"  grad_norm: {grad_norm.item()}, is_finite={is_finite.item()}")

        nan_params = []
        finite_params = []
        zero_params = []
        no_grad_params = []

        for p in trainable_params:
            p_name = param_to_name.get(p, "unknown")
            if p.grad is None:
                no_grad_params.append(p_name)
            elif not torch.isfinite(p.grad).all():
                nan_count = (~torch.isfinite(p.grad)).sum().item()
                total_count = p.grad.numel()
                nan_params.append((p_name, nan_count, total_count))
            elif p.grad.abs().max() == 0:
                zero_params.append(p_name)
            else:
                finite_params.append((p_name, p.grad.abs().mean().item(), p.grad.abs().max().item()))

        print(f"\n--- Gradient Summary for Step {step + 1} ---")
        print(f"Trainable params: {len(trainable_params)}")
        print(f"Finite non-zero grad: {len(finite_params)}")
        print(f"All-zero grad: {len(zero_params)}")
        print(f"None grad: {len(no_grad_params)}")
        print(f"NaN/Inf grad: {len(nan_params)}")

        if nan_params:
            print("\n!!! NaN/Inf DETECTED IN PARAMETERS !!!")
            for p_name, nan_c, total_c in nan_params:
                print(f"  [NaN] {p_name}: {nan_c}/{total_c} ({nan_c/total_c*100:.1f}%)")
        else:
            print(">>> ALL GRADIENTS ARE FINITE <<<")

        # Step
        for p in trainable_params:
            if p.grad is not None:
                p.grad.copy_(torch.where(is_finite, p.grad.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0), torch.zeros_like(p.grad)))

        optimizer.step()
        ema_tracker.update(model, is_safe=is_finite)
        print(f"Step {step + 1} finished successfully. is_finite={is_finite.item()}")

if __name__ == "__main__":
    main()
