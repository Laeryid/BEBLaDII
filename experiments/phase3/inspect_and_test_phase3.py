import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

project_root = r"C:\Experiments\BEBLaDII"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.beb_la_dii.model.dus import DUSModel
from src.beb_la_dii.model.vae import LatentEncoder
from src.beb_la_dii.model.modern_decoder import ModernLatentDecoder
from src.beb_la_dii.utils.tokenizer import get_tokenizer
from src.beb_la_dii.utils.loss import safe_normalize

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
    sigma = torch.sin(t * (math.pi / 2)).view(B, 1, 1)
    x_t = mu * x0 + sigma * eps
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
        _qwen_base = AutoModel.from_pretrained(embedding_model_path, torch_dtype=torch.bfloat16, local_files_only=False)
        self.qwen_embeddings = _qwen_base.get_input_embeddings()
        del _qwen_base
        for p in self.qwen_embeddings.parameters():
            p.requires_grad = False

        self.encoder = LatentEncoder(1536, 1024)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.to(torch.bfloat16)

        dus_wrapper = DUSModel.from_scratch(config={"base_model_id": modernbert_path}, weights_path=None, local_files_only=False)
        self.dus = dus_wrapper.model
        if hasattr(self.dus, "gradient_checkpointing_disable"):
            self.dus.gradient_checkpointing_disable()
        if hasattr(self.dus, "_maybe_set_compile"):
            self.dus._maybe_set_compile = lambda *a, **kw: None
            
        type(self.dus).device = property(lambda self: next(self.parameters()).device)
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

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, t: torch.Tensor, z_noisy_override: torch.Tensor = None, self_cond: torch.Tensor = None):
        with torch.no_grad():
            if z_noisy_override is not None:
                z_noisy = z_noisy_override
                z_clean_f = None
                B, T, D = z_noisy.shape
            else:
                qwen_embeds = self.qwen_embeddings(input_ids)
                z_clean, _, _ = self.encoder(qwen_embeds)
                B, T, D = z_clean.shape
                z_clean_f = safe_normalize(z_clean.float(), dim=-1)
                z_noisy = spherical_noise(z_clean_f, t)

        t_sin = self.t_sin_embed(t)
        t_emb = self.t_proj(t_sin)
        self._current_t_emb = t_emb

        x_in = z_noisy.float()
        
        if self_cond is not None and getattr(self, 'self_cond_proj', None) is not None:
            x_in = x_in + self.self_cond_proj(self_cond.float().to(x_in.device))
        
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

        self._current_t_emb = None

        return {
            "z_clean": z_clean_f,
            "z_noisy": z_noisy,
            "t": t,
            "dus_final": dus_final,
            "h_39": h_39
        }

def load_checkpoint_and_inspect(ckpt_path):
    print(f"[*] Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return state

def load_models(device, dtype, ckpt_state):
    print("\n[*] Initializing models...")
    embed_model_id = "Qwen/Qwen2.5-1.5B"
    tokenizer = get_tokenizer(embed_model_id)
    
    causal_model = AutoModelForCausalLM.from_pretrained(embed_model_id, torch_dtype=dtype)
    lm_head_weight = causal_model.lm_head.weight.data.to(device)
    
    enc_path = r"C:\Experiments\BEBLaDII\experiments\phase 1\planB_phase1_checkpoints_phase1_vae_step_20000.pth"
    dec_path = r"C:\Experiments\BEBLaDII\experiments\phase 2\planB_phase2_checkpoints_decoder_step_9000.pth"
    dus_weights_path = r"C:\Experiments\BEBLaDII\kaggle_upload_1_2\AWAKENED_WEIGHTS_FINAL.pt"
    
    decoder = ModernLatentDecoder(latent_dim=1024, qwen_dim=1536, num_layers=3, dus_weights_path=dus_weights_path).to(device).to(dtype)
    if os.path.exists(dec_path):
        st = torch.load(dec_path, map_location="cpu", weights_only=False)
        decoder.load_state_dict({k.replace("decoder.", ""): v for k, v in st.get("decoder", st).items()}, strict=False)
        print("[*] Phase 2 ModernLatentDecoder loaded.")
    decoder.eval()
    
    diff_model = BEBLaDIIPhase3(embedding_model_path=embed_model_id, modernbert_path="answerdotai/ModernBERT-large")
    diff_model.to(device)
    
    if os.path.exists(enc_path):
        st = torch.load(enc_path, map_location="cpu", weights_only=False)
        diff_model.encoder.load_state_dict({k.replace("encoder.", ""): v for k, v in st.get("encoder", st).items()}, strict=False)
        print("[*] Phase 1 VAE LatentEncoder loaded.")
        
    dus_ema = ckpt_state.get("dus_ema", ckpt_state.get("dus", {}))
    clean_dus = {k.replace("_orig_module.", ""): v for k, v in dus_ema.items()}
    diff_model.dus.load_state_dict(clean_dus, strict=False)
    print("[*] DUS weights loaded.")
    
    for name in ["adaLN_attn", "adaLN_mlp", "t_proj", "self_cond_proj", "sep_embed"]:
        ema = ckpt_state.get(f"{name}_ema", ckpt_state.get(name, {}))
        if ema:
            if name == "sep_embed":
                diff_model.sep_embed.copy_(ema)
            else:
                getattr(diff_model, name).load_state_dict(ema, strict=True)
            print(f"[*] {name} weights loaded.")
            
    # Removed sep_token.pt fallback since Phase 3 training expects it to be zeros if not saved.
        
    diff_model.eval()
    
    return tokenizer, diff_model, decoder, lm_head_weight

def decode_z(z, decoder, lm_head_weight, tokenizer):
    with torch.no_grad():
        dec_out = decoder(z.to(next(decoder.parameters()).dtype))
        logits = F.linear(dec_out, lm_head_weight)
        token_ids = logits.argmax(dim=-1).squeeze(0)
        return tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)

def run_step_by_step_diffusion(phrase_title, text, diff_model, decoder, lm_head_weight, tokenizer, device, steps=20):
    print("\n" + "=" * 120)
    print(f"EXPERIMENT: {phrase_title}")
    print(f"Clean Text: \"{text}\"")
    print("=" * 120)
    
    tok = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = tok.input_ids.to(device)
    attn_mask = tok.attention_mask.to(device)
    mask_b = attn_mask.bool()
    
    with torch.no_grad():
        qwen_embeds = diff_model.qwen_embeddings(input_ids)
        z_clean, _, _ = diff_model.encoder(qwen_embeds)
        z_clean = safe_normalize(z_clean.float(), dim=-1)
    
    clean_decoded = decode_z(z_clean, decoder, lm_head_weight, tokenizer)
    print(f"[AE Baseline] Clean Encoder -> Decoder text:\n  \"{clean_decoded.strip()}\"\n")
    
    z_current = safe_normalize(torch.randn_like(z_clean), dim=-1)
    t_schedule = torch.linspace(1.0, 0.0, steps + 1, device=device)
    
    print(f"{'Step':<5} | {'t_now':<6} | {'Cos(z_t,clean)':<15} | {'Cos(x0_pred,clean)':<18} | {'Decoded Current Trajectory (z_t)':<40}")
    print("-" * 120)
    
    for i in range(steps):
        t_now = t_schedule[i:i+1]
        t_next = t_schedule[i+1:i+2]
        
        with torch.no_grad():
            out = diff_model(input_ids, attn_mask, t_now, z_noisy_override=z_current)
            z_pred = out["dus_final"]
            
        mu_t = cosine_noise_schedule(t_now).view(1,1,1)
        eps_pred = z_current - mu_t * z_pred
        eps_pred_norm = eps_pred.norm(dim=-1, keepdim=True)
        eps_pred = torch.where(eps_pred_norm > 1e-5, eps_pred / eps_pred_norm, torch.randn_like(eps_pred))
        
        mu_next = cosine_noise_schedule(t_next).view(1,1,1)
        sigma_next = torch.sin(t_next * (math.pi / 2)).view(1,1,1)
        z_next = mu_next * z_pred + sigma_next * eps_pred
        z_current = safe_normalize(z_next, dim=-1)
        
        cos_sim_clean = (z_current[mask_b] * z_clean[mask_b]).sum(dim=-1).mean().item()
        cos_sim_pred = (z_pred[mask_b] * z_clean[mask_b]).sum(dim=-1).mean().item()
        
        dec_zt = decode_z(z_current, decoder, lm_head_weight, tokenizer).strip().replace('\n', ' ')
        dec_x0 = decode_z(z_pred, decoder, lm_head_weight, tokenizer).strip().replace('\n', ' ')
        
        print(f"{i+1:<5} | {t_now.item():<6.2f} | {cos_sim_clean:<15.4f} | {cos_sim_pred:<18.4f} | z_t: \"{dec_zt[:45]}\"")
        if i % 4 == 0 or i == steps - 1:
            print(f"      |        |                 |                    | -> Pred x0: \"{dec_x0[:50]}\"")
            print("-" * 120)

def main():
    ckpt_path = r"C:\Experiments\BEBLaDII\experiments\phase3\local_checkpoints\phase3_step_18995.pth"
    if not os.path.exists(ckpt_path):
        print(f"[!] File not found: {ckpt_path}")
        return
        
    state = load_checkpoint_and_inspect(ckpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    tokenizer, diff_model, decoder, lm_head_weight = load_models(device, dtype, state)
    
    p1_title = "Sanity Check Phrase 1 (English)"
    p1_text = "The quick brown fox jumps over the lazy dog."
    run_step_by_step_diffusion(p1_title, p1_text, diff_model, decoder, lm_head_weight, tokenizer, device, steps=20)

    p2_title = "Sanity Check Phrase 2 (Russian)"
    p2_text = "Мама мыла раму, а папа чинил телевизор."
    run_step_by_step_diffusion(p2_title, p2_text, diff_model, decoder, lm_head_weight, tokenizer, device, steps=20)

if __name__ == "__main__":
    main()
