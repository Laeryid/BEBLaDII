import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

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

def spherical_noise(x0: torch.Tensor, t: torch.Tensor, noise_eps=None) -> torch.Tensor:
    B, T, D = x0.shape
    if noise_eps is None:
        noise_eps = safe_normalize(torch.randn_like(x0), dim=-1)
    mu = cosine_noise_schedule(t).view(B, 1, 1)
    x_t = mu * x0 + (1.0 - mu) * noise_eps
    return safe_normalize(x_t, dim=-1)

class AdaLNModulation(nn.Module):
    def __init__(self, t_emb_dim: int, hidden_dim: int):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, 2 * hidden_dim, bias=True),
        )
    def forward(self, t_emb: torch.Tensor) -> tuple:
        out = self.modulation(t_emb)
        shift, scale = out.chunk(2, dim=-1)
        return shift.unsqueeze(1), scale.unsqueeze(1)

class BEBLaDIIPhase3Eval(nn.Module):
    def __init__(self, dus_model, t_emb_dim=256, hidden_dim=1024):
        super().__init__()
        self.dus = dus_model
        
        self.t_sin_embed = SinusoidalEmbedding(t_emb_dim)
        self.t_proj = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(t_emb_dim * 4, t_emb_dim),
        )
        
        num_layers = len(self.dus.layers)
        self.adaLN = nn.ModuleList([
            AdaLNModulation(t_emb_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        self._hooks = []
        for i, layer in enumerate(self.dus.layers):
            self._hooks.append(layer.register_forward_pre_hook(self._make_adaLN_hook(i)))
            
    def _make_adaLN_hook(self, layer_idx: int):
        def hook(module, args):
            if not hasattr(self, "_current_t_emb") or self._current_t_emb is None:
                return args
            hidden_states = args[0]
            shift, scale = self.adaLN[layer_idx](self._current_t_emb)
            shift = shift.to(hidden_states.dtype)
            scale = scale.to(hidden_states.dtype)
            hidden_states = hidden_states * scale + shift
            return (hidden_states,) + args[1:]
        return hook

    def forward(self, z_in, attention_mask, t, sep_embed=None):
        B, T, D = z_in.shape
        t_sin = self.t_sin_embed(t)
        t_emb = self.t_proj(t_sin)
        self._current_t_emb = t_emb

        x_in = z_in.float()
        
        if sep_embed is not None:
            sep_prefix = sep_embed.unsqueeze(0).unsqueeze(0).expand(B, 1, -1).to(x_in.dtype)
            dus_input = torch.cat([sep_prefix, x_in], dim=1)
            attention_mask_ext = F.pad(attention_mask, (1, 0), value=1)
        else:
            dus_input = x_in
            attention_mask_ext = attention_mask

        dus_outputs = self.dus(
            inputs_embeds=dus_input,
            attention_mask=attention_mask_ext,
            output_hidden_states=True,
        )
        
        if sep_embed is not None:
            pre_norm = dus_outputs.hidden_states[-1][:, 1:, :].float()
        else:
            pre_norm = dus_outputs.hidden_states[-1].float()
            
        dus_final_raw = self.dus.final_norm(pre_norm.to(self.dus.dtype)).float()
        dus_final = safe_normalize(dus_final_raw, dim=-1)
        
        self._current_t_emb = None
        return dus_final

def load_checkpoint_and_inspect(ckpt_path):
    print(f"[*] Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return state

def load_models(device, dtype, ckpt_state):
    print("\n[*] Initializing models...")
    embed_model_id = "Qwen/Qwen2.5-1.5B"
    tokenizer = get_tokenizer(embed_model_id)
    causal_model = AutoModelForCausalLM.from_pretrained(embed_model_id, torch_dtype=dtype)
    embeddings = causal_model.get_input_embeddings().to(device)
    embeddings.requires_grad_(False)
    lm_head_weight = causal_model.lm_head.weight.data.to(device)
    
    enc_path = r"C:\Experiments\BEBLaDII\experiments\phase 1\planB_phase1_checkpoints_phase1_vae_step_20000.pth"
    dec_path = r"C:\Experiments\BEBLaDII\experiments\phase 2\planB_phase2_checkpoints_decoder_step_9000.pth"
    
    encoder = LatentEncoder(1536, 1024).to(device).to(dtype)
    decoder = ModernLatentDecoder(latent_dim=1024, qwen_dim=1536, num_layers=3).to(device).to(dtype)
    
    if os.path.exists(enc_path):
        st = torch.load(enc_path, map_location="cpu", weights_only=False)
        encoder.load_state_dict({k.replace("encoder.", ""): v for k, v in st.get("encoder", st).items()}, strict=False)
        print("[*] Phase 1 VAE LatentEncoder loaded.")
        
    if os.path.exists(dec_path):
        st = torch.load(dec_path, map_location="cpu", weights_only=False)
        decoder.load_state_dict({k.replace("decoder.", ""): v for k, v in st.get("decoder", st).items()}, strict=False)
        print("[*] Phase 2 ModernLatentDecoder loaded.")

    encoder.eval()
    decoder.eval()
    
    dus_wrapper = DUSModel.from_scratch(config={"base_model_id": "answerdotai/ModernBERT-large"}, weights_path=None, local_files_only=False)
    dus_model = dus_wrapper.model
    type(dus_model).device = property(lambda self: device)
    type(dus_model).dtype  = property(lambda self: torch.float32)
    
    diff_model = BEBLaDIIPhase3Eval(dus_model).to(device)
    
    dus_ema = ckpt_state.get("dus_ema", ckpt_state.get("dus", {}))
    clean_dus = {k.replace("_orig_module.", ""): v for k, v in dus_ema.items()}
    diff_model.dus.load_state_dict(clean_dus, strict=False)
    
    adaLN_ema = ckpt_state.get("adaLN_ema", ckpt_state.get("adaLN", {}))
    if adaLN_ema:
        diff_model.adaLN.load_state_dict(adaLN_ema, strict=True)
        
    t_proj_ema = ckpt_state.get("t_proj_ema", ckpt_state.get("t_proj", {}))
    if t_proj_ema:
        diff_model.t_proj.load_state_dict(t_proj_ema, strict=True)
    
    sep_path = r"C:\Experiments\BEBLaDII\storage\components\sep_token.pt"
    if os.path.exists(sep_path):
        sep_embed = torch.load(sep_path, map_location=device).float()
        print(f"[*] Separator token loaded from {sep_path}")
    else:
        sep_embed = torch.zeros(1024, dtype=torch.float32, device=device)
        print("[!] Warning: Separator token NOT found, using zeros.")
        
    diff_model.eval()
    return tokenizer, embeddings, encoder, decoder, diff_model, lm_head_weight, sep_embed

def decode_z(z, decoder, lm_head_weight, tokenizer):
    with torch.no_grad():
        dec_out = decoder(z.to(next(decoder.parameters()).dtype))
        logits = F.linear(dec_out, lm_head_weight)
        token_ids = logits.argmax(dim=-1).squeeze(0)
        return tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)

def encode_chatml_with_mask(user_text, assistant_text, tokenizer, embeddings, encoder, device, dtype):
    prefix_u = "<|im_start|>user\n"
    suffix_u = "<|im_end|>\n<|im_start|>assistant\n<|thought|>\n"
    
    p_u_tokens = tokenizer.encode(prefix_u, add_special_tokens=False)
    u_tokens = tokenizer.encode(user_text, add_special_tokens=False)
    s_u_tokens = tokenizer.encode(suffix_u, add_special_tokens=False)
    a_tokens = tokenizer.encode(assistant_text, add_special_tokens=False) if assistant_text else []
    
    full_tokens = p_u_tokens + u_tokens + s_u_tokens + a_tokens
    
    u_start = len(p_u_tokens)
    u_end = u_start + len(u_tokens)
    content_indices = list(range(u_start, u_end))
    
    if a_tokens:
        a_start = u_end + len(s_u_tokens)
        a_end = a_start + len(a_tokens)
        content_indices.extend(list(range(a_start, a_end)))
        
    input_ids = torch.tensor([full_tokens]).to(device)
    attn_mask = torch.tensor([[1] * len(full_tokens)]).to(device)
    
    with torch.no_grad():
        qwen_embeds = embeddings(input_ids)
        z_clean, _, _ = encoder(qwen_embeds)
        z_clean = safe_normalize(z_clean.float(), dim=-1)
        
    return z_clean, attn_mask, content_indices, full_tokens

def run_step_by_step_diffusion(phrase_title, user_text, assistant_text, diff_model, embeddings, encoder, decoder, lm_head_weight, tokenizer, sep_embed, device, dtype, steps=20):
    print("\n" + "=" * 120)
    print(f"EXPERIMENT: {phrase_title}")
    print(f"User Text: \"{user_text}\"")
    if assistant_text:
        print(f"Assistant Text: \"{assistant_text}\"")
    print("=" * 120)
    
    z_clean, attn_mask, content_indices, full_tokens = encode_chatml_with_mask(user_text, assistant_text, tokenizer, embeddings, encoder, device, dtype)
    mask_b = attn_mask.bool()
    content_idx_tensor = torch.tensor(content_indices, device=device)
    
    z_clean_content = z_clean[:, content_idx_tensor, :]
    clean_decoded = decode_z(z_clean_content, decoder, lm_head_weight, tokenizer)
    print(f"[AE Baseline (Content Only)] Clean Encoder -> Decoder text:\n  \"{clean_decoded.strip()}\"\n")
    
    z_current = safe_normalize(torch.randn_like(z_clean), dim=-1)
    t_schedule = torch.linspace(1.0, 0.0, steps + 1, device=device)
    
    print(f"{'Step':<5} | {'t_now':<6} | {'Cos(z_t,clean)':<15} | {'Cos(x0_pred,clean)':<18} | {'Decoded Current Trajectory Content (z_t)':<40}")
    print("-" * 120)
    
    for i in range(steps):
        t_now = t_schedule[i:i+1]
        t_next = t_schedule[i+1:i+2]
        
        with torch.no_grad():
            z_pred = diff_model(z_current, attn_mask, t_now, sep_embed=sep_embed)
            
        mu_t = cosine_noise_schedule(t_now).view(1,1,1)
        eps_pred = z_current - mu_t * z_pred
        eps_pred_norm = eps_pred.norm(dim=-1, keepdim=True)
        eps_pred = torch.where(eps_pred_norm > 1e-5, eps_pred / eps_pred_norm, torch.randn_like(eps_pred))
        
        mu_next = cosine_noise_schedule(t_next).view(1,1,1)
        z_next = mu_next * z_pred + (1.0 - mu_next) * eps_pred
        z_current = safe_normalize(z_next, dim=-1)
        
        cos_sim_clean = (z_current[mask_b] * z_clean[mask_b]).sum(dim=-1).mean().item()
        cos_sim_pred = (z_pred[mask_b] * z_clean[mask_b]).sum(dim=-1).mean().item()
        
        z_t_content = z_current[:, content_idx_tensor, :]
        z_pred_content = z_pred[:, content_idx_tensor, :]
        
        dec_zt = decode_z(z_t_content, decoder, lm_head_weight, tokenizer).strip().replace('\n', ' ')
        dec_x0 = decode_z(z_pred_content, decoder, lm_head_weight, tokenizer).strip().replace('\n', ' ')
        
        print(f"{i+1:<5} | {t_now.item():<6.2f} | {cos_sim_clean:<15.4f} | {cos_sim_pred:<18.4f} | z_t: \"{dec_zt[:45]}\"")
        if i % 4 == 0 or i == steps - 1:
            print(f"      |        |                 |                    | -> Pred x0: \"{dec_x0[:50]}\"")
            print("-" * 120)

def main():
    ckpt_path = r"C:\Experiments\BEBLaDII\experiments\phase3\planB_phase3_checkpoints_phase3_step_9995.pth"
    if not os.path.exists(ckpt_path):
        print(f"[!] File not found: {ckpt_path}")
        return
        
    state = load_checkpoint_and_inspect(ckpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 
    
    tokenizer, embeddings, encoder, decoder, diff_model, lm_head_weight, sep_embed = load_models(device, dtype, state)
    
    p1_title = "Sanity Check Phrase 1 (English)"
    p1_u = "The quick brown fox jumps over the lazy dog."
    p1_a = "Neural networks learn representations from data."
    run_step_by_step_diffusion(p1_title, p1_u, p1_a, diff_model, embeddings, encoder, decoder, lm_head_weight, tokenizer, sep_embed, device, dtype, steps=20)

    p2_title = "Sanity Check Phrase 2 (Russian)"
    p2_u = "Мама мыла раму, а папа чинил телевизор."
    p2_a = "Искусственный интеллект способствовал развитию технологий."
    run_step_by_step_diffusion(p2_title, p2_u, p2_a, diff_model, embeddings, encoder, decoder, lm_head_weight, tokenizer, sep_embed, device, dtype, steps=20)

if __name__ == "__main__":
    main()
