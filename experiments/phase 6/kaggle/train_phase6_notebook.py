# %% [markdown]
# # BEBLaDII Phase 6a Training — Contextual CA_Prompt Layer
# *Архитектура: Внедрение CAPromptLayer (Cross-Attention + Self-Attention) в DUS.*
# *Ключевые изменения:*
# *- Заморожены Embeddings, Encoder и 40 слоев ModernBERT (из Phase 4).*
# *- t_actual сэмплируется из 25 дискретных шагов.*
# *- t_reported = 1.0 - RawDProx (из словаря).*
# *- max_length_q = 512, max_length_a = 512.*

# %% [markdown]
# ## 1. Setup Environment

# %%
# !pip install -q einops wandb transformers pandas pyarrow

# %%
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
                if f.endswith(".json") or f.endswith(".pt") or f.endswith(".pth") or f.endswith(".parquet"):
                    print(f"{subindent}{f}")
    print("==============================")
# -----------------------------------------

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
    from src.beb_la_dii.utils.loss import safe_normalize
except ImportError as e:
    print(f"Warning: Не удалось импортировать модули проекта. Ошибка: {e}")

def resolve_model_path(base_path: str) -> str:
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

def resolve_file_path(filename: str, fallback_dir="/kaggle/input") -> str:
    import pathlib
    p = pathlib.Path(fallback_dir)
    if p.exists():
        for f in p.rglob(filename):
            return str(f)
    return filename

# %% [markdown]
# ## 2. Configuration

# %%
class Config:
    # Пути к базовым моделям
    embedding_model_path = resolve_model_path("/kaggle/input/datasets/ragnar123/qwen2-5-1-5b")
    modernbert_path      = resolve_model_path("/kaggle/input/models/answer-ai/modernbert/transformers/large/2")

    # Пути к данным
    dataset_path = resolve_file_path("train_phase6.parquet")

    # Пути к весам
    encoder_weights = resolve_file_path("planB_phase1_checkpoints_phase1_vae_step_20000.pth")
    dus_weights     = "/kaggle/input/bebladii-planb-phase4-data/phase4_step_85995.pth" # Чекпоинт Phase 4
    sep_token       = resolve_file_path("sep_token.pt")
    latent_dict     = "/kaggle/input/bebladii-planb-phase5-data/latent_dict.pt"

    # Директория вывода
    output_dir = "/kaggle/working/checkpoints/phase6"

    # Гиперпараметры Phase 6
    batch_size    = 8
    max_length_q  = 512
    max_length_a  = 512
    learning_rate = 1e-4
    epochs        = 50
    max_steps     = 200000
    log_steps     = 10
    val_steps     = 500
    save_steps    = 1000
    warmup_steps  = 1000

    use_gradient_checkpointing = True
    wandb_project = "BEBLaDII-Phase6-Kaggle"

args = Config()


# %% [markdown]
# ## 3. Data & Tokenization

# %%
class QADataset(Dataset):
    def __init__(self, parquet_path, tokenizer, max_length_q=512, max_length_a=512):
        print(f"[Dataset] Loading from {parquet_path}...")
        self.df = pd.read_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.max_length_q = max_length_q
        self.max_length_a = max_length_a
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        q_text = str(row['Q'])
        a_text = str(row['A'])
        
        q_enc = self.tokenizer(q_text, truncation=True, max_length=self.max_length_q, padding='max_length', return_tensors='pt')
        a_enc = self.tokenizer(a_text, truncation=True, max_length=self.max_length_a, padding='max_length', return_tensors='pt')
        
        return {
            'input_ids_q': q_enc.input_ids.squeeze(0),
            'attention_mask_q': q_enc.attention_mask.squeeze(0),
            'input_ids_a': a_enc.input_ids.squeeze(0),
            'attention_mask_a': a_enc.attention_mask.squeeze(0),
        }


# %% [markdown]
# ## 4. Noise & ADA Modules (Phase 4 Compat)

# %%
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


# %% [markdown]
# ## 5. CAPromptLayer (Phase 6 Core)

# %%
class CAPromptLayer(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        # Q-KV sharing: A -> q, Q (prompt) -> k, v
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
        # --- Cross-Attention ---
        A_norm = self.norm1(A)
        Q_norm = self.norm1(Q) 
        
        q = self.q_proj(A_norm)
        kv = self.kv_proj(Q_norm)
        k, v = kv.chunk(2, dim=-1)
        
        attn_mask = None
        if mask_Q is not None:
            # mask_Q shape [B, T_q]. F.scaled_dot_product_attention expects [B, 1, 1, T_q] for bool mask
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
        
        # --- Self-Attention ---
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
    """Обертка для ModernBERT слоя, позволяющая инжектировать CAPromptLayer."""
    def __init__(self, original_layer, ca_layer=None):
        super().__init__()
        self.original_layer = original_layer
        self.ca_layer = ca_layer
        
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


# %% [markdown]
# ## 6. Model Definition (Phase 6)

# %%
class BEBLaDIIPhase6(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        
        # 1. Загрузка Qwen Embeddings
        _qwen = AutoModel.from_pretrained(config.embedding_model_path, torch_dtype=torch.bfloat16, local_files_only=True)
        self.qwen_embeddings = _qwen.get_input_embeddings()
        del _qwen
        
        # 2. Загрузка Latent Encoder
        self.encoder = LatentEncoder()
        if os.path.exists(config.encoder_weights):
            state = torch.load(config.encoder_weights, map_location="cpu", weights_only=False)
            if "encoder" in state: state = state["encoder"]
            self.encoder.load_state_dict(state, strict=False)
        self.encoder.to(torch.bfloat16)

        # 3. DUS Backbone
        dus_wrapper = DUSModel.from_scratch(config={"base_model_id": config.modernbert_path}, weights_path=None, local_files_only=True)
        self.dus = dus_wrapper.model
        if config.use_gradient_checkpointing and hasattr(self.dus, "gradient_checkpointing_enable"):
            self.dus.gradient_checkpointing_enable({"use_reentrant": False})
        
        # 4. Phase 4 Time Projections (для совместимости)
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
            
        # 5. Separator Token & Latent Dict
        self.register_buffer("sep_embed", torch.load(config.sep_token).float())
        self.register_buffer("latent_dict", torch.load(config.latent_dict).float()) # [150000, 1024]
        
        # --- ЗАГРУЗКА ВЕСОВ PHASE 4 ---
        if os.path.exists(config.dus_weights):
            state = torch.load(config.dus_weights, map_location="cpu", weights_only=False)
            if "dus_ema" in state: state = state["dus_ema"]
            elif "dus" in state: state = state["dus"]
            elif "model_state_dict" in state: state = state["model_state_dict"]
            elif "model" in state: state = state["model"]
            clean_state = {k.replace("student.model.", "").replace("model.", "").replace("_orig_module.", ""): v for k, v in state.items()}
            self.load_state_dict(clean_state, strict=False)
            print(f"[Init] Phase 4 weights loaded from {config.dus_weights}")
            
        # --- ПОЛНАЯ ЗАМОРОЗКА ---
        for p in self.parameters():
            p.requires_grad = False
            
        # --- ВНЕДРЕНИЕ CAPromptLayer (Trainable) ---
        self.ca_layers = nn.ModuleDict({
            "12": CAPromptLayer(1024),
            "24": CAPromptLayer(1024),
            "36": CAPromptLayer(1024),
        })
        for i in [11, 23, 35]:
            self.dus.layers[i] = Phase6BlockWrapper(self.dus.layers[i], self.ca_layers[str(i+1)])
            
        # Убедимся, что новые слои обучаются
        for p in self.ca_layers.parameters():
            p.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        # Оставляем замороженные компоненты в eval
        if hasattr(self, "qwen_embeddings"): self.qwen_embeddings.eval()
        if hasattr(self, "encoder"): self.encoder.eval()

    def forward(self, input_ids_q, attention_mask_q, input_ids_a, attention_mask_a, warmup_factor=1.0):
        B, T_a = input_ids_a.shape
        
        with torch.no_grad():
            # 1. Подготовка чистого латентного представления для Ответа (A)
            qwen_embeds_a = self.qwen_embeddings(input_ids_a)
            Z_A_clean, _, _ = self.encoder(qwen_embeds_a)
            Z_A_clean = safe_normalize(Z_A_clean.float(), dim=-1)
            
            # 2. Подготовка латентного представления для Промпта (Q)
            qwen_embeds_q = self.qwen_embeddings(input_ids_q)
            Z_prompt, _, _ = self.encoder(qwen_embeds_q)
            Z_prompt = safe_normalize(Z_prompt.float(), dim=-1)
            
            # 3. Шум и t_actual (25 дискретных шагов)
            t_actual = torch.randint(1, 26, (B, T_a), device=Z_A_clean.device) / 25.0
            z_noisy = spherical_noise(Z_A_clean, t_actual)
            
            # 4. Вычисление RawDProx и t_reported
            # z_noisy: [B, T_a, 1024], latent_dict: [V, 1024] -> sims: [B, T_a, V]
            sims = torch.matmul(z_noisy, self.latent_dict.T)
            RawDProx, _ = sims.max(dim=-1)
            t_reported = (1.0 - RawDProx).clamp(0.0, 1.0)
            
        # 5. Вычисление Time Embeddings (заморожено, как в Phase 4)
        t_global = torch.mean(t_actual, dim=-1)
        t_sin_global = self.t_sin_embed(t_global)
        t_emb_global = self.t_proj_global(t_sin_global)
        
        t_sin_token = self.t_sin_embed(t_reported)
        t_emb_token = self.t_proj_token(t_sin_token)
        
        cond = torch.cat([t_emb_token, t_emb_global.unsqueeze(1).expand(-1, T_a, -1)], dim=-1)
        t_emb = self.t_joint_proj(cond)
        
        sep_t_emb = torch.zeros(B, 1, t_emb.shape[-1], device=t_emb.device, dtype=t_emb.dtype)
        t_emb_extended = torch.cat([sep_t_emb, t_emb], dim=1)
        
        # Инъекция t_emb и контекста CA
        for layer in self.dus.layers:
            layer_to_check = layer.original_layer if isinstance(layer, Phase6BlockWrapper) else layer
            if hasattr(layer_to_check, "attn_norm"): layer_to_check.attn_norm._current_t_emb = t_emb_extended
            if hasattr(layer_to_check, "mlp_norm"): layer_to_check.mlp_norm._current_t_emb = t_emb_extended
            
        for ca in self.ca_layers.values():
            ca._current_Z_prompt = Z_prompt
            ca._current_mask_Q = attention_mask_q
            ca._current_warmup_factor = warmup_factor

        # 6. Прогон через DUS
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
        dus_final = safe_normalize(dus_final_raw, dim=-1)
        
        return {
            "z_clean": Z_A_clean,
            "dus_final": dus_final,
            "t_actual": t_actual,
            "attention_mask_a": attention_mask_a
        }


# %% [markdown]
# ## 7. Loss & Training Loop

# %%
def compute_phase6_loss(outputs):
    z_clean = outputs["z_clean"].float()
    dus_final = outputs["dus_final"].float()
    attn_f = outputs["attention_mask_a"].float()
    t_actual = outputs["t_actual"].float()
    
    target = safe_normalize(z_clean, dim=-1)
    cos_sim = (dus_final * target).sum(dim=-1)
    loss_el = 1.0 - cos_sim
    
    # Weighting: (1 - t_actual)^2.0 * mask_A
    w_weighted = (1.0 - t_actual).pow(2.0) * attn_f
    loss = (w_weighted * loss_el).sum() / w_weighted.sum().clamp(min=1e-8)
    return loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        wandb_api = user_secrets.get_secret("WANDB_API_KEY")
        wandb.login(key=wandb_api)
    except Exception as e:
        print(f"Kaggle secrets not available or failed to login to wandb: {e}")
        
    wandb.init(project=args.wandb_project, config=vars(args))
    
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    try:
        dataset = QADataset(args.dataset_path, tokenizer, args.max_length_q, args.max_length_a)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    except Exception as e:
        print(f"Failed to load dataset: {e}. Running dummy loop for compilation check.")
        dataloader = []
        
    model = BEBLaDIIPhase6(args).to(device)
    
    # Оптимизатор обновляет только CA_layers
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    
    model.train()
    global_step = 0
    
    for epoch in range(args.epochs):
        for batch in dataloader:
            if global_step >= args.max_steps: return
            
            warmup_factor = min(1.0, global_step / args.warmup_steps)
            
            input_ids_q = batch['input_ids_q'].to(device)
            mask_q = batch['attention_mask_q'].to(device)
            input_ids_a = batch['input_ids_a'].to(device)
            mask_a = batch['attention_mask_a'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids_q, mask_q, input_ids_a, mask_a, warmup_factor=warmup_factor)
            loss = compute_phase6_loss(outputs)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            wandb.log({
                "loss": loss.item(),
                "warmup_factor": warmup_factor,
                "gate_12": model.ca_layers["12"].gate.item(),
                "gate_24": model.ca_layers["24"].gate.item(),
                "gate_36": model.ca_layers["36"].gate.item(),
            }, step=global_step)
            
            if global_step % args.log_steps == 0:
                print(f"Step {global_step} | Loss: {loss.item():.4f} | Gate_12: {model.ca_layers['12'].gate.item():.4f}")
                
            if global_step % args.save_steps == 0 and global_step > 0:
                torch.save(model.ca_layers.state_dict(), os.path.join(args.output_dir, f"phase6_ca_layers_step_{global_step}.pth"))
                
            global_step += 1

if __name__ == "__main__":
    main()
