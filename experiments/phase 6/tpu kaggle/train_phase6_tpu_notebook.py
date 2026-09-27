import math
import os
import subprocess
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
import numpy as np

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

# Handle torch_xla versions
try:
    import torch_xla.experimental.xla_sharding as xs
except ImportError:
    import torch_xla.distributed.spmd as xs
    if not hasattr(xs, 'mark_sharding'):
        import torch_xla.distributed.spmd.xla_sharding as xs

try:
    from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel
except ImportError:
    try:
        from torch_xla.distributed.spmd import SpmdFullyShardedDataParallel
    except ImportError:
        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as SpmdFullyShardedDataParallel

xr.use_spmd()

def setup_spmd_mesh():
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (num_devices, 1)
    device_ids = np.array(range(num_devices))
    mesh = xs.Mesh(device_ids, mesh_shape, ("fsdp", "model"))
    xs.set_global_mesh(mesh)
    return mesh

PROJECT_ROOT = "/kaggle/working/BEBLaDII"
REPO_URL = "https://github.com/Laeryid/BEBLaDII.git"

if not os.path.exists(PROJECT_ROOT):
    subprocess.run(["git", "clone", REPO_URL, PROJECT_ROOT], check=True)
else:
    subprocess.run(["git", "-C", PROJECT_ROOT, "pull"], check=True)

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
    return base_path

def resolve_file_path(filename: str, fallback_dir="/kaggle/input") -> str:
    import pathlib
    p = pathlib.Path(fallback_dir)
    if p.exists():
        for f in p.rglob(filename):
            return str(f)
    return filename

def sync_to_gcs_and_delete(local_path: str, gcs_dir: str):
    if not gcs_dir.endswith("/"): gcs_dir += "/"
    gcs_path = gcs_dir + os.path.basename(local_path)
    try:
        subprocess.run(["gsutil", "-q", "cp", local_path, gcs_path], check=True)
        os.remove(local_path)
    except Exception as e:
        print(f"[GCS] Error syncing {local_path}: {e}")

def get_latest_gcs_checkpoint(gcs_dir: str, prefix: str = "phase6_ca_layers_step_"):
    try:
        if not gcs_dir.startswith("gs://"):
            return None, 0
        result = subprocess.run(["gsutil", "ls", gcs_dir], capture_output=True, text=True)
        if result.returncode != 0: return None, 0

        files = result.stdout.strip().split("\n")
        ckpt_files = [f for f in files if prefix in f and f.endswith(".pth")]
        if not ckpt_files: return None, 0

        def extract_step(filename):
            try: return int(filename.split("_step_")[-1].replace(".pth", ""))
            except ValueError: return -1

        ckpt_files.sort(key=extract_step)
        latest_file = ckpt_files[-1]
        return latest_file, extract_step(latest_file)
    except Exception:
        return None, 0

class Config:
    embedding_model_path = resolve_model_path("/kaggle/input/datasets/ragnar123/qwen2-5-1-5b")
    modernbert_path      = resolve_model_path("/kaggle/input/models/answer-ai/modernbert/transformers/large/2")
    dataset_path = resolve_file_path("train_phase6.parquet")
    val_dataset_path = resolve_file_path("val_phase6.parquet")
    encoder_weights = resolve_file_path("planB_phase1_checkpoints_phase1_vae_step_20000.pth")
    dus_weights     = resolve_file_path("phase4_step_85995.pth")
    sep_token       = "/kaggle/working/BEBLaDII/storage/components/sep_token.pt"
    latent_dict     = resolve_file_path("latent_dict.pt")
    output_dir = "/kaggle/working/checkpoints/phase6"
    resume_from_checkpoint = True
    gcs_checkpoint_dir = "gs://bebladii-weigths-us/planB/phase6/checkpoints/"
    batch_size    = 64 * 4
    max_length_q  = 512
    max_length_a  = 512
    learning_rate = 2e-4
    epochs        = 50
    max_steps     = 200000
    log_steps     = 10
    val_steps     = 200
    save_steps    = 1000
    warmup_steps  = 1000
    ema_decay     = 0.999
    pace_alpha    = 0.001
    use_gradient_checkpointing = True
    wandb_project = "BEBLaDII-Phase6-Kaggle"

args = Config()

class EMATracker:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach().float()

    def update(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name].copy_(self.decay * self.shadow[name] + (1.0 - self.decay) * param.data.float())

    def pace_pullback(self, model, alpha):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    ema_casted = self.shadow[name].to(param.dtype)
                    param.data.sub_(alpha * (param.data - ema_casted))

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].to(param.dtype))
        xm.mark_step()

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}
        xm.mark_step()

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
    def __init__(self, dim=1024):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = nn.RMSNorm(dim)
        self.qkv_proj_sa = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj_sa = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.qkv_proj_sa.weight)
        nn.init.xavier_uniform_(self.out_proj_sa.weight)

    def forward(self, A, Q, mask_Q=None, warmup_factor=1.0):
        A_norm = self.norm1(A)
        Q_norm = self.norm1(Q)
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
        A = A + ca_out * (torch.tanh(self.gate) * warmup_factor)

        A_norm2 = self.norm2(A)
        qkv = self.qkv_proj_sa(A_norm2)
        q_sa, k_sa, v_sa = qkv.chunk(3, dim=-1)

        q_sa = q_sa.view(B, T_a, heads, head_dim).transpose(1, 2)
        k_sa = k_sa.view(B, T_a, heads, head_dim).transpose(1, 2)
        v_sa = v_sa.view(B, T_a, heads, head_dim).transpose(1, 2)

        sa_out = F.scaled_dot_product_attention(q_sa, k_sa, v_sa)
        sa_out = sa_out.transpose(1, 2).reshape(B, T_a, D)
        sa_out = self.out_proj_sa(sa_out)

        A = A + sa_out * (torch.tanh(self.gate) * warmup_factor)
        return A

class Phase6BlockWrapper(nn.Module):
    def __init__(self, original_layer, ca_layer=None):
        super().__init__()
        self.original_layer = original_layer
        self.ca_layer = ca_layer
    @property
    def attention_type(self): return self.original_layer.attention_type
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        out = self.original_layer(hidden_states, attention_mask=attention_mask, **kwargs)
        if self.ca_layer is not None:
            Z_prompt = getattr(self.ca_layer, '_current_Z_prompt', None)
            mask_Q = getattr(self.ca_layer, '_current_mask_Q', None)
            warmup_factor = getattr(self.ca_layer, '_current_warmup_factor', 1.0)
            if Z_prompt is not None:
                sep = out[0][:, 0:1, :]
                ans = out[0][:, 1:, :]
                ans = self.ca_layer(ans, Z_prompt, mask_Q, warmup_factor)
                out = (torch.cat([sep, ans], dim=1),) + out[1:]
        return out

class BEBLaDIIPhase6(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        _qwen = AutoModel.from_pretrained(config.embedding_model_path, torch_dtype=torch.bfloat16, local_files_only=True)
        self.qwen_embeddings = _qwen.get_input_embeddings()
        del _qwen

        self.encoder = LatentEncoder()
        if os.path.exists(config.encoder_weights):
            state = torch.load(config.encoder_weights, map_location="cpu", weights_only=False)
            if "encoder" in state: state = state["encoder"]
            self.encoder.load_state_dict(state, strict=False)
        self.encoder.to(torch.bfloat16)

        dus_wrapper = DUSModel.from_scratch(config={"base_model_id": config.modernbert_path}, weights_path=None, local_files_only=True)
        self.dus = dus_wrapper.model
        if config.use_gradient_checkpointing and hasattr(self.dus, "gradient_checkpointing_enable"):
            self.dus.gradient_checkpointing_enable({"use_reentrant": False})

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
        self.register_buffer("latent_dict", torch.load(config.latent_dict).float())

        if os.path.exists(config.dus_weights):
            state = torch.load(config.dus_weights, map_location="cpu", weights_only=False)
            if "dus_ema" in state: state = state["dus_ema"]
            elif "dus" in state: state = state["dus"]
            elif "model_state_dict" in state: state = state["model_state_dict"]
            elif "model" in state: state = state["model"]
            clean_state = {k.replace("student.model.", "").replace("model.", "").replace("_orig_module.", ""): v for k, v in state.items()}
            self.load_state_dict(clean_state, strict=False)

        for p in self.parameters(): p.requires_grad = False

        self.ca_layers = nn.ModuleDict({
            "12": CAPromptLayer(1024),
            "24": CAPromptLayer(1024),
            "36": CAPromptLayer(1024),
        })
        for i in [11, 23, 35]:
            self.dus.layers[i] = Phase6BlockWrapper(self.dus.layers[i], self.ca_layers[str(i+1)])

        for p in self.ca_layers.parameters(): p.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "qwen_embeddings"): self.qwen_embeddings.eval()
        if hasattr(self, "encoder"): self.encoder.eval()

    def forward(self, input_ids_q, attention_mask_q, input_ids_a, attention_mask_a, warmup_factor=1.0):
        B, T_a = input_ids_a.shape
        with torch.no_grad():
            qwen_embeds_a = self.qwen_embeddings(input_ids_a)
            Z_A_clean, _, _ = self.encoder(qwen_embeds_a)
            Z_A_clean = safe_normalize(Z_A_clean.float(), dim=-1)

            qwen_embeds_q = self.qwen_embeddings(input_ids_q)
            Z_prompt, _, _ = self.encoder(qwen_embeds_q)
            Z_prompt = safe_normalize(Z_prompt.float(), dim=-1)

            t_actual = torch.randint(1, 26, (B, T_a), device=Z_A_clean.device) / 25.0
            z_noisy = spherical_noise(Z_A_clean, t_actual)

            sims = torch.matmul(z_noisy, self.latent_dict.T)
            RawDProx, _ = sims.max(dim=-1)
            t_reported = (1.0 - RawDProx).clamp(0.0, 1.0)

        t_global = torch.mean(t_actual, dim=-1)
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
            if hasattr(layer_to_check, "attn_norm"): layer_to_check.attn_norm._current_t_emb = t_emb_extended
            if hasattr(layer_to_check, "mlp_norm"): layer_to_check.mlp_norm._current_t_emb = t_emb_extended

        for ca in self.ca_layers.values():
            ca._current_Z_prompt = Z_prompt
            ca._current_mask_Q = attention_mask_q
            ca._current_warmup_factor = warmup_factor

        x_in = z_noisy.float()
        sep_prefix = self.sep_embed.unsqueeze(0).unsqueeze(0).expand(B, 1, -1).to(x_in.dtype)
        dus_input_extended = torch.cat([sep_prefix, x_in], dim=1)
        attention_mask_extended = F.pad(attention_mask_a, (1, 0), value=1)

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
            "dus_final": dus_final,
            "t_actual": t_actual,
            "attention_mask_a": attention_mask_a
        }

def compute_phase6_loss(outputs):
    z_clean = outputs["z_clean"].float()
    dus_final = outputs["dus_final"].float()
    attn_f = outputs["attention_mask_a"].float()
    t_actual = outputs["t_actual"].float()

    target = safe_normalize(z_clean, dim=-1)
    cos_sim = (dus_final * target).sum(dim=-1)
    loss_el = 1.0 - cos_sim

    w_weighted = (1.0 - t_actual).pow(2.0) * attn_f
    loss = (w_weighted * loss_el).sum() / w_weighted.sum().clamp(min=1e-8)
    avg_cos_sim = (cos_sim * attn_f).sum() / attn_f.sum().clamp(min=1e-8)
    return loss, avg_cos_sim

def main():
    mesh = setup_spmd_mesh()
    device = xm.xla_device()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.wandb_project:
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            wandb.login(key=user_secrets.get_secret("WANDB_API_KEY"))
            gcp_sa = user_secrets.get_secret("GCP_SA_JSON")
            with open("gcp_sa.json", "w") as f: f.write(gcp_sa)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath("gcp_sa.json")
        except Exception: pass
        wandb.init(project=args.wandb_project, config=vars(args))

    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    try:
        dataset = QADataset(args.dataset_path, tokenizer, args.max_length_q, args.max_length_a)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    model = BEBLaDIIPhase6(args).to(device)

    def shard_output(output, mesh): return None
    model.ca_layers = SpmdFullyShardedDataParallel(model.ca_layers, mesh=mesh, shard_output=shard_output)

    ema_tracker = EMATracker(model.ca_layers, decay=args.ema_decay)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    model.train()
    global_step = 0

    if args.resume_from_checkpoint and args.gcs_checkpoint_dir:
        latest_ckpt, step = get_latest_gcs_checkpoint(args.gcs_checkpoint_dir)
        if latest_ckpt:
            local_ckpt = os.path.join(args.output_dir, "resume_ca_layers.pth")
            try:
                subprocess.run(["gsutil", "-q", "cp", latest_ckpt, local_ckpt], check=True)
                ckpt_state = torch.load(local_ckpt, map_location="cpu", weights_only=False)
                model.ca_layers.load_state_dict(ckpt_state)
                ema_tracker = EMATracker(model.ca_layers, decay=args.ema_decay)
                global_step = step
                os.remove(local_ckpt)
            except Exception as e: print(f"Resume failed: {e}")

    for epoch in range(args.epochs):
        for batch in dataloader:
            if global_step >= args.max_steps: return

            warmup_factor_val = min(1.0, global_step / args.warmup_steps)
            warmup_factor = torch.tensor(warmup_factor_val, dtype=torch.float32, device=device)

            input_ids_q = batch['input_ids_q'].to(device)
            mask_q = batch['attention_mask_q'].to(device)
            input_ids_a = batch['input_ids_a'].to(device)
            mask_a = batch['attention_mask_a'].to(device)

            xs.mark_sharding(input_ids_q, mesh, ("fsdp", None))
            xs.mark_sharding(mask_q, mesh, ("fsdp", None))
            xs.mark_sharding(input_ids_a, mesh, ("fsdp", None))
            xs.mark_sharding(mask_a, mesh, ("fsdp", None))

            optimizer.zero_grad()
            outputs = model(input_ids_q, mask_q, input_ids_a, mask_a, warmup_factor=warmup_factor)
            loss, avg_cos_sim = compute_phase6_loss(outputs)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            xm.optimizer_step(optimizer)

            ema_tracker.update(model.ca_layers)
            ema_tracker.pace_pullback(model.ca_layers, alpha=args.pace_alpha)

            if global_step % args.log_steps == 0:
                xm.add_step_closure(lambda s, l: wandb.log({"loss": l.item()}, step=s), args=(global_step, loss))

            if global_step % args.save_steps == 0 and global_step > 0:
                ema_tracker.apply_shadow(model.ca_layers)
                ckpt_path = os.path.join(args.output_dir, f"phase6_ca_layers_step_{global_step}.pth")
                xm.save(model.ca_layers.state_dict(), ckpt_path)
                ema_tracker.restore(model.ca_layers)
                if args.gcs_checkpoint_dir:
                    xm.add_step_closure(lambda: sync_to_gcs_and_delete(ckpt_path, args.gcs_checkpoint_dir))

            global_step += 1

if __name__ == "__main__":
    main()
