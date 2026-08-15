import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

project_root = r"C:\Experiments\BEBLaDII"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.beb_la_dii.model.dus import DUSModel
from src.beb_la_dii.model.vae import LatentEncoder
from src.beb_la_dii.model.modern_decoder import ModernLatentDecoder
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

class BEBLaDIIPhase3Eval(nn.Module):
    def __init__(self, embedding_model_path: str, modernbert_path: str, t_emb_dim: int = 256):
        super().__init__()
        _qwen_base = AutoModel.from_pretrained(embedding_model_path, torch_dtype=torch.bfloat16)
        self.qwen_embeddings = _qwen_base.get_input_embeddings()
        del _qwen_base
        for p in self.qwen_embeddings.parameters():
            p.requires_grad = False

        self.encoder = LatentEncoder(1536, 1024)
        for p in self.encoder.parameters():
            p.requires_grad = False
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

def decode_z(z: torch.Tensor, decoder: nn.Module, lm_head_weight: torch.Tensor, tokenizer) -> str:
    with torch.no_grad():
        projected = decoder(z.to(next(decoder.parameters()).dtype))
        logits = F.linear(projected.float(), lm_head_weight.float())
        pred_ids = logits.argmax(dim=-1)
        return tokenizer.decode(pred_ids[0], skip_special_tokens=True)

def load_all_eval_models(ckpt_path: str, device: str = "cpu"):
    print(f"[*] Загрузка чекпоинта Phase 3: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    embed_model_id = "Qwen/Qwen2.5-1.5B"
    print(f"[*] Загрузка Qwen Tokenizer & LM Head...")
    tokenizer = AutoTokenizer.from_pretrained(embed_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 151643

    causal_model = AutoModelForCausalLM.from_pretrained(embed_model_id, torch_dtype=torch.float32)
    lm_head_weight = causal_model.lm_head.weight.detach().clone().to(device)
    del causal_model

    print(f"[*] Загрузка Decoder (Phase 2)...")
    dec_path = r"C:\Experiments\BEBLaDII\experiments\phase 2\planB_phase2_checkpoints_decoder_step_9000.pth"
    dus_weights_path = r"C:\Experiments\BEBLaDII\kaggle_upload_1_2\AWAKENED_WEIGHTS_FINAL.pt"
    decoder = ModernLatentDecoder(latent_dim=1024, qwen_dim=1536, num_layers=3, dus_weights_path=dus_weights_path).to(device)
    if os.path.exists(dec_path):
        st = torch.load(dec_path, map_location="cpu", weights_only=False)
        decoder.load_state_dict({k.replace("decoder.", ""): v for k, v in st.get("decoder", st).items()}, strict=False)
    decoder.eval()

    print(f"[*] Сборка Phase 3 DUS + VAE...")
    diff_model = BEBLaDIIPhase3Eval(embedding_model_path=embed_model_id, modernbert_path="answerdotai/ModernBERT-large")
    diff_model.to(device)

    enc_path = r"C:\Experiments\BEBLaDII\experiments\phase 1\planB_phase1_checkpoints_phase1_vae_step_20000.pth"
    if os.path.exists(enc_path):
        st = torch.load(enc_path, map_location="cpu", weights_only=False)
        diff_model.encoder.load_state_dict({k.replace("encoder.", ""): v for k, v in st.get("encoder", st).items()}, strict=False)

    dus_ema = state.get("dus_ema", state.get("dus", {}))
    clean_dus = {k.replace("_orig_module.", ""): v for k, v in dus_ema.items()}
    diff_model.dus.load_state_dict(clean_dus, strict=False)

    for name in ["adaLN_attn", "adaLN_mlp", "t_proj", "self_cond_proj", "sep_embed"]:
        ema = state.get(f"{name}_ema", state.get(name, {}))
        if ema:
            if name == "sep_embed":
                diff_model.sep_embed.copy_(ema)
            else:
                getattr(diff_model, name).load_state_dict(ema, strict=True)

    diff_model.eval()
    print("[*] Модели успешно готовы к тестированию!\n")
    return tokenizer, diff_model, decoder, lm_head_weight

def run_denoising_sanity_test(ckpt_path: str, device: str = "cpu"):
    tokenizer, diff_model, decoder, lm_head_weight = load_all_eval_models(ckpt_path, device)
    
    test_phrases = [
        "The quick brown fox jumps over the lazy dog.",
        "Мама мыла раму, а папа чинил телевизор.",
        "Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics to solve problems too complex for classical computers.",
        "Neural networks learn representations from data.",
        "Artificial intelligence and machine learning have revolutionized the way we interact with technology."
    ]

    t_noise_levels = [0.05, 0.10, 0.20, 0.30, 0.50]

    print("=" * 110)
    print(f"PHASE 3 DENOISING SANITY CHECK — ЧЕКПОИНТ: {os.path.basename(ckpt_path)}")
    print("=" * 110)

    for idx, phrase in enumerate(test_phrases, 1):
        print(f"\n--- TEST #{idx} ---")
        print(f"ORIGINAL TEXT : {phrase}")
        
        # 1. Encode to latent space
        tok = tokenizer(phrase, return_tensors="pt", add_special_tokens=False)
        input_ids = tok.input_ids.to(device)
        attn_mask = tok.attention_mask.to(device)
        mask_b = attn_mask.bool()

        with torch.no_grad():
            qwen_embeds = diff_model.qwen_embeddings(input_ids)
            z_clean, _, _ = diff_model.encoder(qwen_embeds)
            z_clean = safe_normalize(z_clean.float(), dim=-1)

        # Baseline clean decode
        clean_decoded = decode_z(z_clean, decoder, lm_head_weight, tokenizer)
        print(f"CLEAN DECODED : {clean_decoded}")
        print("-" * 110)

        for t_val in t_noise_levels:
            # Generate spherical noise for exact t
            x_t = spherical_noise(z_clean, torch.tensor([t_val], device=device))
            x_t = safe_normalize(x_t, dim=-1)

            # Cosine similarity of noisy input with clean
            cos_noisy = (x_t[mask_b] * z_clean[mask_b]).sum(dim=-1).mean().item()

            # Raw noisy decode (without DUS)
            raw_noisy_decoded = decode_z(x_t, decoder, lm_head_weight, tokenizer)

            # DUS 1-step prediction (with self-conditioning if t >= 0.5)
            with torch.no_grad():
                if getattr(diff_model, 'self_cond_proj', None) is not None and t_val >= 0.5:
                    out_sc = diff_model(input_ids, attn_mask, torch.tensor([t_val], device=device), z_noisy_override=x_t, self_cond=None)
                    sc_est = out_sc["dus_final"].detach()
                    out = diff_model(input_ids, attn_mask, torch.tensor([t_val], device=device), z_noisy_override=x_t, self_cond=sc_est)
                else:
                    out = diff_model(input_ids, attn_mask, torch.tensor([t_val], device=device), z_noisy_override=x_t)
                pred_x0 = out["dus_final"]
                h_39 = out["h_39"]

            cos_pred = (pred_x0[mask_b] * z_clean[mask_b]).sum(dim=-1).mean().item()
            cos_h39 = (h_39[mask_b] * z_clean[mask_b]).sum(dim=-1).mean().item()

            # Denoised decode
            denoised_decoded = decode_z(pred_x0, decoder, lm_head_weight, tokenizer)

            print(f"[t = {t_val:4.2f}] | Cos(x_t)={cos_noisy:.4f} -> Cos(x0)={cos_pred:.4f} (h39={cos_h39:.4f})")
            print(f"  └─ NOISY RAW : {raw_noisy_decoded}")
            print(f"  └─ DUS CLEAN : {denoised_decoded}")

    print("\n" + "=" * 110)
    print("SANITY CHECK COMPLETE!")
    print("=" * 110)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = r"C:\Experiments\BEBLaDII\experiments\phase3\local_checkpoints\phase3_step_32995.pth"
    run_denoising_sanity_test(ckpt, device=device)
