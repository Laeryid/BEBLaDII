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
    """Синусоидальное позиционное кодирование для t in [0, 1]."""
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
    """Slerp-шум по сфере: x_t = normalize(cos(t * pi / 2) * x0 + sin(t * pi / 2) * eps)."""
    B, T, D = x0.shape
    eps = safe_normalize(torch.randn_like(x0), dim=-1)
    mu = cosine_noise_schedule(t).view(B, 1, 1)
    sigma = torch.sin(t * (math.pi / 2)).view(B, 1, 1)
    x_t = mu * x0 + sigma * eps
    return safe_normalize(x_t, dim=-1)


class AdaLNModulation(nn.Module):
    """Zero Init AdaLN per ADR 074."""
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
        return shift.unsqueeze(1), scale.unsqueeze(1)


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
        type(self.dus).dtype = property(lambda self: torch.float32)

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
            layer.attn_norm = AdaLNWrappedLayerNorm(layer.attn_norm, self.adaLN_attn[i])
            layer.mlp_norm = AdaLNWrappedLayerNorm(layer.mlp_norm, self.adaLN_mlp[i])

        sep_token_path = r"C:\Experiments\BEBLaDII\storage\components\sep_token.pt"
        if os.path.exists(sep_token_path):
            sep_tensor = torch.load(sep_token_path, map_location="cpu", weights_only=False).float()
            self.register_buffer("sep_embed", sep_tensor)
        else:
            self.register_buffer("sep_embed", torch.zeros(hidden_dim, dtype=torch.float32))

        self.self_cond_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.self_cond_proj.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        t: torch.Tensor,
        z_noisy_override: torch.Tensor | None = None,
        self_cond: torch.Tensor | None = None
    ):
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

        x_in = z_noisy.float()

        if self_cond is not None and getattr(self, "self_cond_proj", None) is not None:
            x_in = x_in + self.self_cond_proj(self_cond.float().to(x_in.device))

        sep_prefix = self.sep_embed.unsqueeze(0).unsqueeze(0).expand(B, 1, -1).to(x_in.dtype)
        dus_input_extended = torch.cat([sep_prefix, x_in], dim=1)
        attention_mask_extended = F.pad(attention_mask, (1, 0), value=1)

        for layer in self.dus.layers:
            if hasattr(layer, "attn_norm") and isinstance(layer.attn_norm, AdaLNWrappedLayerNorm):
                layer.attn_norm._current_t_emb = t_emb
            if hasattr(layer, "mlp_norm") and isinstance(layer.mlp_norm, AdaLNWrappedLayerNorm):
                layer.mlp_norm._current_t_emb = t_emb

        dus_outputs = self.dus(
            inputs_embeds=dus_input_extended,
            attention_mask=attention_mask_extended,
            output_hidden_states=False
        )

        for layer in self.dus.layers:
            if hasattr(layer, "attn_norm") and isinstance(layer.attn_norm, AdaLNWrappedLayerNorm):
                layer.attn_norm._current_t_emb = None
            if hasattr(layer, "mlp_norm") and isinstance(layer.mlp_norm, AdaLNWrappedLayerNorm):
                layer.mlp_norm._current_t_emb = None

        pre_norm = dus_outputs.last_hidden_state[:, 1:, :].float()
        dus_final_raw = self.dus.final_norm(pre_norm.to(self.dus.dtype)).float()
        h_39 = safe_normalize(dus_final_raw, dim=-1)

        # Strict x0-prediction (ADR 072)
        dus_final = h_39

        return {
            "z_clean": z_clean_f,
            "z_noisy": z_noisy,
            "t": t,
            "dus_final": dus_final,
            "h_39": h_39,
        }


def load_checkpoint_and_inspect(ckpt_path: str):
    print("=" * 100)
    print(f"[*] ИНСПЕКЦИЯ ЧЕКПОИНТА: {ckpt_path}")
    print("=" * 100)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    print(f"Keys in checkpoint: {list(state.keys())}")
    step = state.get("step", "N/A")
    print(f"Training Step: {step}")

    history = state.get("metrics_history", [])
    print(f"Metrics history count: {len(history)}")
    if history:
        print("\n--- Последние метрики обучения (из чекпоинта) ---")
        def fmt(val):
            if isinstance(val, (int, float)):
                return f"{val:.4f}"
            if isinstance(val, torch.Tensor):
                return f"{val.item():.4f}"
            return str(val)

        for entry in history[-5:]:
            st = entry.get("step", "?")
            loss = fmt(entry.get("loss", entry.get("denoising_loss", "?")))
            h39_hi = fmt(entry.get("cos_h39_t_high", entry.get("val_cos_h39_t_high", "?")))
            h39_mid = fmt(entry.get("cos_h39_t_mid", entry.get("val_cos_h39_t_mid", "?")))
            h39_lo = fmt(entry.get("cos_h39_t_low", entry.get("val_cos_h39_t_low", "?")))
            adaln_w = fmt(entry.get("adaln_w_norm", "?"))
            lr_dus = entry.get("lr_dus", "?")
            lr_str = f"{lr_dus:.2e}" if isinstance(lr_dus, (int, float)) else str(lr_dus)
            print(f"  Step {st:>6} | Loss: {loss} | h39_hi: {h39_hi} | h39_mid: {h39_mid} | h39_lo: {h39_lo} | adaln_w: {adaln_w} | lr: {lr_str}")
    return state


def load_models(device, dtype, ckpt_state):
    print("\n[*] Инициализация моделей...")
    embed_model_id = "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(embed_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 151643

    causal_model = AutoModelForCausalLM.from_pretrained(embed_model_id, torch_dtype=dtype)
    lm_head_weight = causal_model.lm_head.weight.data.to(device)
    del causal_model

    enc_paths = [
        r"C:\Experiments\BEBLaDII\experiments\phase 1\planB_phase1_checkpoints_phase1_vae_step_20000.pth",
        r"C:\Experiments\BEBLaDII\BEBLaDII-planB-Phase3-Data\planB_phase1_checkpoints_phase1_vae_step_20000.pth",
    ]
    dec_paths = [
        r"C:\Experiments\BEBLaDII\experiments\phase 2\planB_phase2_checkpoints_decoder_step_9000.pth",
        r"C:\Experiments\BEBLaDII\BEBLaDII-planB-Phase3-Data\planB_phase2_phase2_decoder_step_8000.pth",
    ]
    dus_weights_path = r"C:\Experiments\BEBLaDII\kaggle_upload_1_2\AWAKENED_WEIGHTS_FINAL.pt"

    dec_path = next((p for p in dec_paths if os.path.exists(p)), None)
    decoder = ModernLatentDecoder(latent_dim=1024, qwen_dim=1536, num_layers=3, dus_weights_path=dus_weights_path).to(device).to(dtype)
    if dec_path:
        st = torch.load(dec_path, map_location="cpu", weights_only=False)
        decoder.load_state_dict({k.replace("decoder.", ""): v for k, v in st.get("decoder", st).items()}, strict=False)
        print(f"[*] Phase 2 ModernLatentDecoder loaded from {dec_path}")
    decoder.eval()

    diff_model = BEBLaDIIPhase3(embedding_model_path=embed_model_id, modernbert_path="answerdotai/ModernBERT-large")
    diff_model.to(device)

    enc_path = next((p for p in enc_paths if os.path.exists(p)), None)
    if enc_path:
        st = torch.load(enc_path, map_location="cpu", weights_only=False)
        diff_model.encoder.load_state_dict({k.replace("encoder.", ""): v for k, v in st.get("encoder", st).items()}, strict=False)
        print(f"[*] Phase 1 VAE LatentEncoder loaded from {enc_path}")

    dus_ema = ckpt_state.get("dus_ema", ckpt_state.get("dus", {}))
    clean_dus = {k.replace("_orig_module.", ""): v for k, v in dus_ema.items()}
    diff_model.dus.load_state_dict(clean_dus, strict=False)
    print(f"[*] DUS weights loaded ({'EMA' if 'dus_ema' in ckpt_state else 'RAW'}).")

    for name in ["adaLN_attn", "adaLN_mlp", "t_proj", "self_cond_proj"]:
        ema = ckpt_state.get(f"{name}_ema", ckpt_state.get(name, None))
        if ema:
            getattr(diff_model, name).load_state_dict(ema, strict=True)
            print(f"[*] {name} weights loaded ({'EMA' if f'{name}_ema' in ckpt_state else 'RAW'}).")

    diff_model.eval()

    # Diagnostic inspection of AdaLN weights
    with torch.no_grad():
        w_attn_norms = [m.modulation[-1].weight.norm().item() for m in diff_model.adaLN_attn]
        w_mlp_norms = [m.modulation[-1].weight.norm().item() for m in diff_model.adaLN_mlp]
        print(f"[*] AdaLN Attn Mean Weight Norm: {sum(w_attn_norms)/len(w_attn_norms):.6f}")
        print(f"[*] AdaLN MLP Mean Weight Norm:  {sum(w_mlp_norms)/len(w_mlp_norms):.6f}")

    return tokenizer, diff_model, decoder, lm_head_weight


def decode_z(z: torch.Tensor, decoder: nn.Module, lm_head_weight: torch.Tensor, tokenizer) -> str:
    with torch.no_grad():
        dec_out = decoder(z.to(next(decoder.parameters()).dtype))
        logits = F.linear(dec_out.float(), lm_head_weight.float())
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

    # Инициализация чистым шумом на сфере (t=1.0)
    z_current = safe_normalize(torch.randn_like(z_clean), dim=-1)
    t_schedule = torch.linspace(1.0, 0.0, steps + 1, device=device)

    print(f"{'Step':<5} | {'t_now':<6} | {'Cos(z_t,clean)':<15} | {'Cos(x0_pred,clean)':<18} | {'Decoded Current Trajectory (z_t)':<40}")
    print("-" * 120)

    for i in range(steps):
        t_now = t_schedule[i:i + 1]
        t_next = t_schedule[i + 1:i + 2]

        with torch.no_grad():
            out = diff_model(input_ids, attn_mask, t_now, z_noisy_override=z_current)
            z_pred = out["dus_final"]

        # Ортогональное извлечение шума через процесс Грама-Шмидта (ADR 072)
        if t_now.item() > 0.0:
            eps_pred = z_current - (z_current * z_pred).sum(dim=-1, keepdim=True) * z_pred
            eps_pred = safe_normalize(eps_pred, dim=-1)
        else:
            eps_pred = torch.zeros_like(z_current)

        # Slerp шаг к t_next
        theta_next = t_next * (math.pi / 2)
        z_next = math.cos(theta_next) * z_pred + math.sin(theta_next) * eps_pred
        z_current = safe_normalize(z_next, dim=-1)

        cos_sim_clean = (z_current[mask_b] * z_clean[mask_b]).sum(dim=-1).mean().item()
        cos_sim_pred = (z_pred[mask_b] * z_clean[mask_b]).sum(dim=-1).mean().item()

        dec_zt = decode_z(z_current, decoder, lm_head_weight, tokenizer).strip().replace('\n', ' ')
        dec_x0 = decode_z(z_pred, decoder, lm_head_weight, tokenizer).strip().replace('\n', ' ')

        print(f"{i + 1:<5} | {t_now.item():<6.2f} | {cos_sim_clean:<15.4f} | {cos_sim_pred:<18.4f} | z_t: \"{dec_zt[:45]}\"")
        if i % 4 == 0 or i == steps - 1:
            print(f"      |        |                 |                    | -> Pred x0: \"{dec_x0[:50]}\"")
            print("-" * 120)


def main():
    default_ckpt = r"C:\Experiments\BEBLaDII\experiments\phase3\local_checkpoints\phase3_step_4995.pth"
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
    else:
        ckpt_path = default_ckpt

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
