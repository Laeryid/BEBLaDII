import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import argparse
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
PROJECT_ROOT = "C:/Experiments/BEBLaDII"
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments", "phase 4"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments", "phase 5"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from transformers import AutoTokenizer, AutoModel
from beb_la_dii.model.sensor_ensemble import SensorEnsemble
from beb_la_dii.model.vae import LatentEncoder
from beb_la_dii.model.dus import DUSModel
from beb_la_dii.utils.loss import safe_normalize

class CAPromptLayer(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.norm1 = getattr(nn, 'RMSNorm', nn.LayerNorm)(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        self.norm2 = getattr(nn, 'RMSNorm', nn.LayerNorm)(dim)
        self.qkv_proj_sa = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj_sa = nn.Linear(dim, dim, bias=False)
        
        self.gate = nn.Parameter(torch.zeros(1))
        
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.qkv_proj_sa.weight)
        nn.init.xavier_uniform_(self.out_proj_sa.weight)
        
    def forward(self, A, Q, mask_Q=None, warmup_factor=1.0):
        # Cross-Attention
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
        
        # Self-Attention
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
    def attention_type(self):
        return self.original_layer.attention_type
        
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

class BEBLaDIIPhase6Eval(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.qwen_embeddings = AutoModel.from_pretrained(config['qwen_path'], torch_dtype=torch.bfloat16).get_input_embeddings()
        
        self.encoder = LatentEncoder()
        if os.path.exists(config['encoder_path']):
            state = torch.load(config['encoder_path'], map_location="cpu", weights_only=False)
            if "encoder" in state: state = state["encoder"]
            self.encoder.load_state_dict(state, strict=False)
        self.encoder.to(torch.bfloat16)

        dus_wrapper = DUSModel.from_scratch(config={"base_model_id": config['modernbert_path']}, weights_path=None, local_files_only=True)
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
            
        self.register_buffer("sep_embed", torch.load(config['sep_token_path']).float())
        
        if os.path.exists(config['phase4_path']):
            state = torch.load(config['phase4_path'], map_location="cpu", weights_only=False)
            if "dus_ema" in state: state = state["dus_ema"]
            elif "dus" in state: state = state["dus"]
            elif "model_state_dict" in state: state = state["model_state_dict"]
            clean_state = {k.replace("student.model.", "").replace("model.", "").replace("_orig_module.", ""): v for k, v in state.items()}
            self.load_state_dict(clean_state, strict=False)
            
        for p in self.parameters():
            p.requires_grad = False
            
        self.ca_layers = nn.ModuleDict({
            "12": CAPromptLayer(1024),
            "24": CAPromptLayer(1024),
            "36": CAPromptLayer(1024),
        })
        for i in [11, 23, 35]:
            self.dus.layers[i] = Phase6BlockWrapper(self.dus.layers[i], self.ca_layers[str(i+1)])
            
        if config.get('phase6_ckpt') and os.path.exists(config['phase6_ckpt']):
            st = torch.load(config['phase6_ckpt'], map_location="cpu", weights_only=False)
            self.ca_layers.load_state_dict(st, strict=False)
            print(f"Loaded Phase 6 CA layers from {config['phase6_ckpt']}")
        else:
            print("Warning: Phase 6 checkpoint not found. CA layers are randomly initialized!")

        self.eval()

    def forward_step(self, x_in, Z_prompt, mask_Q, t_actual, t_reported):
        B, T_a, _ = x_in.shape
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
            ca._current_mask_Q = mask_Q
            ca._current_warmup_factor = 1.0

        sep_prefix = self.sep_embed.unsqueeze(0).unsqueeze(0).expand(B, 1, -1).to(x_in.dtype)
        dus_input_extended = torch.cat([sep_prefix, x_in], dim=1)
        attention_mask_extended = torch.ones(B, T_a + 1, device=x_in.device)
        
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
        return h_39, dus_final

def run_generation(model, sensor_ensemble, latent_dict, tokenizer, query_text, device, max_len=64, steps=25):
    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Generating answer for: '{query_text}'")
    q_enc = tokenizer(query_text, return_tensors='pt').to(device)
    
    with torch.no_grad():
        Z_prompt, _, _ = model.encoder(model.qwen_embeddings(q_enc.input_ids))
        Z_prompt = safe_normalize(Z_prompt.float(), dim=-1)
        
        B = 1
        T_a = max_len
        z_canvas = safe_normalize(torch.randn(B, T_a, 1024, device=device), dim=-1)
        
        dt = 1.0 / steps
        run_history = []
        
        for i in range(steps):
            t_val = 1.0 - i * dt
            t_actual = torch.full((B, T_a), t_val, device=device)
            
            sims = torch.matmul(z_canvas, latent_dict.T)
            raw_dprox, w1_ids = sims.max(dim=-1)
            t_reported = (1.0 - raw_dprox).clamp(0.0, 1.0)
            
            h39, z_pred = model.forward_step(z_canvas, Z_prompt, q_enc.attention_mask, t_actual, t_reported)
            
            w1_words = [tokenizer.decode([idx]) for idx in w1_ids[0].tolist()]
            step_text = "".join(w1_words).replace("Ġ", " ")
            run_history.append({'step': i, 't': t_val, 'text': step_text, 'raw_dprox': raw_dprox[0].mean().item()})
            
            if i % 5 == 0 or i == steps - 1:
                print(f"Step {i:02d} (t={t_val:.2f}, avg_dprox={raw_dprox[0].mean().item():.3f}): {step_text[:100]}...")
                
            t_next = torch.full((B, T_a), max(0.0, t_val - dt), device=device)
            theta_now = t_actual.unsqueeze(-1) * (math.pi / 2)
            theta_next = t_next.unsqueeze(-1) * (math.pi / 2)
            
            sin_now = torch.sin(theta_now)
            sin_now = torch.where(sin_now < 1e-5, torch.ones_like(sin_now) * 1e-5, sin_now)
            w_pred = torch.sin(theta_now - theta_next) / sin_now
            w_cur = torch.sin(theta_next) / sin_now
            
            z_next = w_pred * z_pred + w_cur * z_canvas
            z_canvas = torch.where(theta_now > 1e-5, safe_normalize(z_next, dim=-1), safe_normalize(z_pred, dim=-1))
            
    final_sims = torch.matmul(z_canvas, latent_dict.T)
    _, final_ids = final_sims.max(dim=-1)
    final_text = tokenizer.decode(final_ids[0])
    print(f"\n=== FINAL ANSWER ===\n{final_text}\n====================")
    
    return {'query': query_text, 'history': run_history, 'final': final_text}

def generate_html(results, html_path):
    html = "<html><head><meta charset='utf-8'><title>Phase 6 Generation</title>"
    html += "<style>body{font-family:sans-serif; margin:20px; background:#f9f9f9} .run{background:#fff; padding:15px; margin-bottom:20px; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1)} .step{font-family:monospace; color:#555; margin:5px 0; padding-bottom:5px; border-bottom:1px solid #eee} .final{font-weight:bold; color:#2c3e50; margin-top:10px; font-size:1.1em}</style></head><body>"
    html += "<h1>Phase 6 CA_Prompt Generation Demos</h1>"
    
    for res in results:
        html += f"<div class='run'><h3>Q: {res['query']}</h3>"
        for h in res['history']:
            if h['step'] % 5 == 0 or h['step'] == len(res['history'])-1:
                html += f"<div class='step'>[Step {h['step']:02d} | t={h['t']:.2f} | DProx={h['raw_dprox']:.3f}] {h['text']}</div>"
        html += f"<div class='final'>A: {res['final']}</div></div>"
        
    html += "</body></html>"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report saved to {html_path}")

def main():
    parser = argparse.ArgumentParser(description="Phase 6 CA_Prompt Evaluation")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--ckpt", type=str, default="", help="Path to CA_Prompt weights (Phase 6)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Phase 6 on {device}...")
    
    config = {
        'qwen_path': "Qwen/Qwen2.5-1.5B",
        'modernbert_path': "answerdotai/ModernBERT-large",
        'encoder_path': os.path.join(PROJECT_ROOT, "experiments", "phase 1", "planB_phase1_checkpoints_phase1_vae_step_20000.pth"),
        'phase4_path': os.path.join(PROJECT_ROOT, "experiments", "phase 4", "local_checkpoints", "phase4_step_85995.pth"),
        'sep_token_path': os.path.join(PROJECT_ROOT, "storage", "components", "sep_token.pt"),
        'phase6_ckpt': args.ckpt
    }
    
    tokenizer = AutoTokenizer.from_pretrained(config['qwen_path'])
    model = BEBLaDIIPhase6Eval(config).to(device)
    sensor_ensemble = SensorEnsemble().to(device).eval()
    
    dict_path = os.path.join(PROJECT_ROOT, "experiments", "phase 5", "local", "latent_dict.pt")
    if os.path.exists(dict_path):
        latent_dict = torch.load(dict_path, map_location=device)
    else:
        print("latent_dict.pt not found! Need to build it via Phase 5 script first.")
        return
        
    results = []
    
    # Dataset facts
    ds_path = os.path.join(PROJECT_ROOT, "BEBLaDII-planB-Phase6-Data", "phase 6", "data", "train_phase6.parquet")
    if os.path.exists(ds_path):
        try:
            df = pd.read_parquet(ds_path)
            if len(df) > 0:
                print(f"Loaded {len(df)} samples from {ds_path}. Sampling 1...")
                sample_q = df.iloc[0]['Q']
                res = run_generation(model, sensor_ensemble, latent_dict, tokenizer, sample_q, device, max_len=64)
                results.append(res)
        except Exception as e:
            print(f"Could not read dataset: {e}")
            
    # Hardcoded Out-of-distribution
    ood_q = "Explain why the sky is blue using simple words."
    res = run_generation(model, sensor_ensemble, latent_dict, tokenizer, ood_q, device, max_len=64)
    results.append(res)
    
    html_path = os.path.join(PROJECT_ROOT, "experiments", "phase 6", "phase6_generation_demo.html")
    generate_html(results, html_path)
    
    if args.interactive:
        while True:
            try:
                user_q = input("\nEnter prompt (or 'q' to quit): ")
                if user_q.lower() in ['q', 'quit', 'exit']:
                    break
                if not user_q.strip():
                    continue
                res = run_generation(model, sensor_ensemble, latent_dict, tokenizer, user_q, device, max_len=64)
                results.append(res)
                generate_html(results, html_path)
            except KeyboardInterrupt:
                break
                
if __name__ == "__main__":
    main()
