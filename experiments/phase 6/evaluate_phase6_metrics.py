import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import argparse
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = "C:/Experiments/BEBLaDII"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from transformers import AutoTokenizer, AutoModel
from src.beb_la_dii.model.vae import LatentEncoder
from src.beb_la_dii.model.dus import DUSModel
from src.beb_la_dii.model.modern_decoder import ModernLatentDecoder
from src.beb_la_dii.utils.loss import safe_normalize
from src.beb_la_dii.model.diffusion_blocks import (
    SinusoidalEmbedding,
    AdaLNModulation,
    AdaLNWrappedLayerNorm,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    patch_modernbert_xla_safe_mask,
)


# =========================================================================
# 1. State Dict Validation & Key Normalization (ADR 100, 101, 102)
# =========================================================================
def is_state_dict_finite(state_dict: dict) -> bool:
    """ADR 101: Проверяет отсутствие NaN и Inf во всех тензорах state_dict."""
    for k, v in state_dict.items():
        if torch.is_tensor(v) and not torch.isfinite(v).all():
            return False
    return True


def normalize_key(k: str) -> str:
    """
    ADR 100, 102: Очищает FSDP, DDP, Dynamo и EMA префиксы из ключей чекпоинта
    для гарантированного маппинга на целевую архитектуру.
    """
    return (
        k.replace("._orig_module.", ".")
        .replace("_orig_module.", "")
        .replace(".module.", ".")
        .replace("module.", "")
        .replace("student.model.", "")
        .replace("student.", "")
    )


# =========================================================================
# 2. Geodesic Noise Schedule in Tangent Space (ADR 105)
# =========================================================================
def cosine_noise_schedule(t: torch.Tensor) -> torch.Tensor:
    return torch.cos(t * (math.pi / 2))


def spherical_noise(x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    ADR 105: Точная геодезическая траектория на сфере с ортогонализацией шума
    в касательном пространстве сферы T_{x0}S^{D-1}.
    """
    B, T, D = x0.shape
    eps = torch.randn_like(x0)
    # Ортогонализация Грама-Шмидта: исключаем параллельную проекцию на x0
    proj = (eps * x0).sum(dim=-1, keepdim=True) * x0
    eps_ortho = safe_normalize(eps - proj, dim=-1)

    if t.dim() == 1:
        t = t.view(B, 1, 1)
    elif t.dim() == 2:
        t = t.unsqueeze(-1)

    mu = cosine_noise_schedule(t)
    sigma = torch.sin(t * (math.pi / 2))
    return safe_normalize(mu * x0 + sigma * eps_ortho, dim=-1)


# =========================================================================
# 3. CAPromptLayer & Block Wrapper (ADR 103, 104, 105)
# =========================================================================
class CAPromptLayer(nn.Module):
    def __init__(self, dim=1024, t_emb_dim=256):
        super().__init__()
        self._disable = False
        self.last_ca_ratio = 0.0

        # RMSNorm для Q-промпта и A-холста
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

        heads = 16
        head_dim = dim // heads
        self.rope_sa = RotaryEmbedding(dim=head_dim)

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

        B, T_a, D = A.shape
        heads = 16
        head_dim = D // heads

        q = q.view(B, T_a, heads, head_dim).transpose(1, 2)
        k = k.view(B, -1, heads, head_dim).transpose(1, 2)
        v = v.view(B, -1, heads, head_dim).transpose(1, 2)

        attn_mask = None
        if mask_Q is not None:
            # ADR 103 / 104: Безопасная аддитивная float-маска для SDPA
            mask_q_4d = mask_Q.view(B, 1, 1, -1).bool()
            attn_mask = torch.where(
                mask_q_4d,
                torch.zeros(1, dtype=q.dtype, device=q.device),
                torch.tensor(-10000.0, dtype=q.dtype, device=q.device)
            )

        ca_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        ca_out = ca_out.transpose(1, 2).reshape(B, T_a, D)
        ca_out = self.out_proj(ca_out)

        with torch.no_grad():
            self.last_ca_ratio = (ca_out.norm(dim=-1).mean() / (A.norm(dim=-1).mean() + 1e-8)).item()

        A = A + warmup_factor * ca_out

        # --- Self-Attention с RoPE (ADR 105) ---
        A_norm2 = self.norm2(A)
        if t_emb is not None:
            shift_sa, scale_sa = self.adaln_sa(t_emb)
            A_norm2 = A_norm2 * scale_sa + shift_sa

        qkv = self.qkv_proj_sa(A_norm2)
        q_sa, k_sa, v_sa = qkv.chunk(3, dim=-1)

        q_sa = q_sa.view(B, T_a, heads, head_dim).transpose(1, 2)
        k_sa = k_sa.view(B, T_a, heads, head_dim).transpose(1, 2)
        v_sa = v_sa.view(B, T_a, heads, head_dim).transpose(1, 2)

        cos_sa, sin_sa = self.rope_sa(q_sa, T_a)
        q_sa, k_sa = apply_rotary_pos_emb(q_sa, k_sa, cos_sa, sin_sa)

        sa_out = F.scaled_dot_product_attention(q_sa, k_sa, v_sa)
        sa_out = sa_out.transpose(1, 2).reshape(B, T_a, D)
        sa_out = self.out_proj_sa(sa_out)

        A = A + warmup_factor * sa_out
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
        is_tuple = isinstance(out, (tuple, list))
        hidden = out[0] if is_tuple else out

        if self.ca_layer is not None:
            Z_prompt = getattr(self.ca_layer, '_current_Z_prompt', None)
            mask_Q = getattr(self.ca_layer, '_current_mask_Q', None)
            warmup_factor = getattr(self.ca_layer, '_current_warmup_factor', 1.0)
            t_emb = getattr(self.ca_layer, '_current_t_emb', None)

            if Z_prompt is not None:
                sep = hidden[:, 0:1, :]
                ans = hidden[:, 1:, :]
                ans = self.ca_layer(ans, Z_prompt, mask_Q, warmup_factor, t_emb=t_emb)
                hidden = torch.cat([sep, ans], dim=1)

        if is_tuple:
            return (hidden,) + tuple(out[1:])
        return hidden


# =========================================================================
# 4. Phase 6 Evaluation Model Architecture
# =========================================================================
class BEBLaDIIPhase6Eval(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 1. Qwen Embeddings
        _qwen = AutoModel.from_pretrained(config['qwen_path'], torch_dtype=torch.float32)
        self.qwen_embeddings = _qwen.get_input_embeddings()
        del _qwen

        # 2. VAE Latent Encoder
        self.encoder = LatentEncoder()
        if os.path.exists(config['encoder_path']):
            enc_st = torch.load(config['encoder_path'], map_location="cpu", weights_only=False)
            if "encoder" in enc_st:
                enc_st = enc_st["encoder"]
            self.encoder.load_state_dict(enc_st, strict=False)
        else:
            raise FileNotFoundError(f"Encoder not found: {config['encoder_path']}")

        # 3. ModernBERT DUS (40 layers) в FP32 с патчем масок (ADR 103, 104)
        dus_wrapper = DUSModel.from_scratch(
            config={"base_model_id": config['modernbert_path']},
            weights_path=None,
            local_files_only=True
        )
        self.dus = dus_wrapper.model
        self.dus.to(torch.float32)
        type(self.dus).dtype = property(lambda self: torch.float32)
        patch_modernbert_xla_safe_mask(self.dus)

        # 4. Time Projections & AdaLN (Phase 4 фундамент)
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

        self.register_buffer("sep_embed", torch.load(config['sep_token_path'], map_location="cpu").float())
        if 'void_token_path' in config and os.path.exists(config['void_token_path']):
            self.register_buffer("void_embed", torch.load(config['void_token_path'], map_location="cpu").float())
        else:
            self.register_buffer("void_embed", torch.zeros(1024))

        # Загрузка весов Фазы 4 (ADR 102)
        if os.path.exists(config['phase4_path']):
            p4 = torch.load(config['phase4_path'], map_location="cpu", weights_only=False)
            dus_st = p4.get("dus_ema", p4.get("dus", p4.get("model_state_dict", p4)))
            clean_dus = {normalize_key(k): v for k, v in dus_st.items()}
            self.dus.load_state_dict(clean_dus, strict=False)

            if "adaLN_attn_ema" in p4 or "adaLN_attn" in p4:
                st = p4.get("adaLN_attn_ema", p4.get("adaLN_attn"))
                self.adaLN_attn.load_state_dict({normalize_key(k): v for k, v in st.items()}, strict=False)
            if "adaLN_mlp_ema" in p4 or "adaLN_mlp" in p4:
                st = p4.get("adaLN_mlp_ema", p4.get("adaLN_mlp"))
                self.adaLN_mlp.load_state_dict({normalize_key(k): v for k, v in st.items()}, strict=False)
            if "t_proj_global_ema" in p4 or "t_proj_global" in p4:
                st = p4.get("t_proj_global_ema", p4.get("t_proj_global"))
                self.t_proj_global.load_state_dict({normalize_key(k): v for k, v in st.items()}, strict=False)
            if "t_proj_token_ema" in p4 or "t_proj_token" in p4:
                st = p4.get("t_proj_token_ema", p4.get("t_proj_token"))
                self.t_proj_token.load_state_dict({normalize_key(k): v for k, v in st.items()}, strict=False)
            if "t_joint_proj_ema" in p4 or "t_joint_proj" in p4:
                st = p4.get("t_joint_proj_ema", p4.get("t_joint_proj"))
                self.t_joint_proj.load_state_dict({normalize_key(k): v for k, v in st.items()}, strict=False)

        # Заморозка базовых весов
        for p in self.parameters():
            p.requires_grad = False

        # Внедрение CAPromptLayers (слои 12, 24, 36 -> индексы 11, 23, 35)
        self.ca_layers = nn.ModuleDict({
            "12": CAPromptLayer(1024, t_emb_dim=256),
            "24": CAPromptLayer(1024, t_emb_dim=256),
            "36": CAPromptLayer(1024, t_emb_dim=256),
        })
        for i in [11, 23, 35]:
            self.dus.layers[i] = Phase6BlockWrapper(self.dus.layers[i], self.ca_layers[str(i+1)])

        # Загрузка обученных весов Фазы 6 (ADR 100, 101, 102)
        if config.get('phase6_ckpt') and os.path.exists(config['phase6_ckpt']):
            self.load_phase6_weights(config['phase6_ckpt'])
        else:
            print("Warning: Phase 6 checkpoint not provided or not found. CA layers are randomly initialized!")

        self.eval()

    def load_phase6_weights(self, ckpt_path: str):
        """Интеллектуальная нормализация и валидация чекпоинта Фазы 6."""
        ckpt_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw_weights = ckpt_state.get("trainable_state", ckpt_state.get("model", ckpt_state)) if isinstance(ckpt_state, dict) else ckpt_state

        # Нормализация ключей модели
        model_keys = list(self.state_dict().keys())
        norm_to_model = {normalize_key(k): k for k in model_keys}

        clean_state = {}
        for ckpt_k, v in raw_weights.items():
            if not torch.is_tensor(v):
                continue
            norm_k = normalize_key(ckpt_k)
            if norm_k in norm_to_model:
                clean_state[norm_to_model[norm_k]] = v
            elif ckpt_k in self.state_dict():
                clean_state[ckpt_k] = v

        if not is_state_dict_finite(clean_state):
            print(f"[ERROR] Чекпоинт {ckpt_path} содержит NaN/Inf! Загрузка отменена.")
            return

        res = self.load_state_dict(clean_state, strict=False)
        loaded_count = len(clean_state)
        print(f"Loaded Phase 6 weights from {ckpt_path}: {loaded_count} tensors mapped, missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)}")

        # Диагностика загрузки слоев CA
        ca_keys_loaded = [k for k in clean_state.keys() if "ca_layer" in k or "ca_layers" in k]
        dus_keys_loaded = [k for k in clean_state.keys() if "dus.layers" in k and "ca_layer" not in k]
        print(f"   -> CA weights loaded: {len(ca_keys_loaded)} tensors")
        print(f"   -> DUS unfrozen layers loaded: {len(dus_keys_loaded)} tensors")

    def encode_prompt(self, input_ids_q, attention_mask_q):
        """
        ADR 099 & ADR 105:
        Контекстуализация промпта Q через слои 0-11 ModernBERT DUS при t=0.
        Ограничение self.dus.layers[:12] дает ускорение >3x и экономит HBM.
        """
        B = input_ids_q.shape[0]
        device = input_ids_q.device
        with torch.no_grad():
            qwen_embeds_q = self.qwen_embeddings(input_ids_q)
            Z_prompt, _, _ = self.encoder(qwen_embeds_q)
            Z_prompt = safe_normalize(Z_prompt.float(), dim=-1)

            t_zero = torch.zeros((B,), device=device, dtype=torch.float32)
            t_sin_global = self.t_sin_embed(t_zero)
            t_emb_global = self.t_proj_global(t_sin_global)
            t_sin_token = self.t_sin_embed(t_zero)
            t_emb_token = self.t_proj_token(t_sin_token)

            cond_p = torch.cat([
                t_emb_token.unsqueeze(1).expand(-1, Z_prompt.shape[1], -1),
                t_emb_global.unsqueeze(1).expand(-1, Z_prompt.shape[1], -1)
            ], dim=-1)
            t_emb_p = self.t_joint_proj(cond_p)
            sep_t_emb_p = torch.zeros(B, 1, t_emb_p.shape[-1], device=device, dtype=t_emb_p.dtype)
            t_emb_p_ext = torch.cat([sep_t_emb_p, t_emb_p], dim=1)

            for layer in self.dus.layers:
                l = layer.original_layer if isinstance(layer, Phase6BlockWrapper) else layer
                if hasattr(l, "attn_norm"):
                    l.attn_norm._current_t_emb = t_emb_p_ext
                if hasattr(l, "mlp_norm"):
                    l.mlp_norm._current_t_emb = t_emb_p_ext

            prompt_in = Z_prompt.float()
            sep_prefix = self.sep_embed.unsqueeze(0).unsqueeze(0).expand(B, 1, -1).to(device).to(prompt_in.dtype)
            dus_prompt_ext = torch.cat([sep_prefix, prompt_in], dim=1)
            ones_sep = torch.ones((B, 1), device=device, dtype=attention_mask_q.dtype)
            prompt_mask_ext = torch.cat([ones_sep, attention_mask_q], dim=1)

            for ca in self.ca_layers.values():
                ca._disable = True

            orig_layers = self.dus.layers
            try:
                self.dus.layers = orig_layers[:12]
                prompt_outputs = self.dus(
                    inputs_embeds=dus_prompt_ext,
                    attention_mask=prompt_mask_ext,
                    output_hidden_states=True,
                )
                Z_prompt_12 = prompt_outputs.hidden_states[12][:, 1:, :].to(Z_prompt.dtype)
            finally:
                self.dus.layers = orig_layers

            for ca in self.ca_layers.values():
                ca._disable = False

            return Z_prompt_12

    def forward_step(self, x_in, Z_prompt_12, mask_Q, t_actual, t_reported):
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
            if hasattr(layer_to_check, "attn_norm"):
                layer_to_check.attn_norm._current_t_emb = t_emb_extended
            if hasattr(layer_to_check, "mlp_norm"):
                layer_to_check.mlp_norm._current_t_emb = t_emb_extended

        for ca in self.ca_layers.values():
            ca._current_Z_prompt = Z_prompt_12
            ca._current_mask_Q = mask_Q
            ca._current_warmup_factor = 1.0
            ca._current_t_emb = t_emb

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
        return h_39, dus_final


# =========================================================================
# 5. Loss & Metrics Computation (ADR 097, 100, 105)
# =========================================================================
def compute_phase6_loss(outputs, void_margin=0.0, loss_t_power=0.0):
    z_clean = outputs["z_clean"].float()
    dus_final = outputs["dus_final"].float()
    t_actual = outputs["t_actual"].float()
    void_mask = outputs["void_mask"].float()
    void_embed = outputs["void_embed"].float()

    target = safe_normalize(z_clean, dim=-1)
    cos_sim = (dus_final * target).sum(dim=-1)
    loss_el = 1.0 - cos_sim

    # Dynamic Void Penalty
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


def evaluate_dataset_metrics(model, tokenizer, df_val, device, max_samples=50):
    """
    Вычисляет количественные метрики качества модели на валидационной выборке:
    loss, avg_cos_sim, true_void_sim, content_to_void_sim, ca_ratio.
    """
    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Evaluating metrics on {min(len(df_val), max_samples)} dataset samples...")
    model.eval()

    total_loss = 0.0
    total_cos = 0.0
    total_void_cos = 0.0
    total_content_cos = 0.0
    ca_ratios = {"12": [], "24": [], "36": []}

    n_samples = min(len(df_val), max_samples)
    with torch.no_grad():
        for idx in range(n_samples):
            row = df_val.iloc[idx]
            q_text = row['Q']
            a_text = row['A']

            q_enc = tokenizer(q_text, return_tensors='pt', max_length=128, truncation=True).to(device)
            a_enc = tokenizer(a_text, return_tensors='pt', max_length=64, padding="max_length", truncation=True).to(device)

            input_ids_q = q_enc.input_ids
            attention_mask_q = q_enc.attention_mask
            input_ids_a = a_enc.input_ids
            attention_mask_a = a_enc.attention_mask

            B, T_a = input_ids_a.shape

            # Чистый таргет
            qwen_embeds_a = model.qwen_embeddings(input_ids_a)
            Z_A_clean, _, _ = model.encoder(qwen_embeds_a)
            Z_A_clean = safe_normalize(Z_A_clean.float(), dim=-1)

            # Void mask (PAD positions)
            void_mask = (attention_mask_a == 0).float()
            void_embed = model.void_embed.to(Z_A_clean.dtype)

            # Фиксированный уровень шума t=0.5 для сопоставимости оценки
            t_val = 0.5
            t_actual = torch.full((B, T_a), t_val, device=device)
            z_noisy = spherical_noise(Z_A_clean, t_actual)

            # dprox
            t_reported = torch.full((B, T_a), t_val, device=device)

            # Encode prompt
            Z_prompt_12 = model.encode_prompt(input_ids_q, attention_mask_q)

            # Forward step
            _, dus_final = model.forward_step(z_noisy, Z_prompt_12, attention_mask_q, t_actual, t_reported)

            outputs = {
                "z_clean": Z_A_clean,
                "dus_final": dus_final,
                "t_actual": t_actual,
                "void_mask": void_mask,
                "void_embed": void_embed
            }

            loss, avg_cos, true_v_cos, cont_v_cos = compute_phase6_loss(outputs)
            total_loss += loss.item()
            total_cos += avg_cos.item()
            total_void_cos += true_v_cos.item()
            total_content_cos += cont_v_cos.item()

            for k in ["12", "24", "36"]:
                ca_ratios[k].append(model.ca_layers[k].last_ca_ratio)

    avg_loss = total_loss / n_samples
    avg_cos = total_cos / n_samples
    avg_void_cos = total_void_cos / n_samples
    avg_content_cos = total_content_cos / n_samples

    print("\n" + "="*50)
    print("PHASE 6 VALIDATION METRICS SUMMARY (at t=0.50):")
    print(f"  Total Loss               : {avg_loss:.4f}")
    print(f"  Cos Sim (Z_pred vs Z_tgt): {avg_cos:.4f}")
    print(f"  True Void Cos Sim        : {avg_void_cos:.4f}")
    print(f"  Content-to-Void Cos Sim  : {avg_content_cos:.4f}")
    print(f"  CA Ratio (Layer 12)      : {sum(ca_ratios['12']) / n_samples:.4f}")
    print(f"  CA Ratio (Layer 24)      : {sum(ca_ratios['24']) / n_samples:.4f}")
    print(f"  CA Ratio (Layer 36)      : {sum(ca_ratios['36']) / n_samples:.4f}")
    print("="*50 + "\n")

    return {
        "loss": avg_loss,
        "cos_sim": avg_cos,
        "true_void_cos": avg_void_cos,
        "content_void_cos": avg_content_cos,
        "ca_ratio_12": sum(ca_ratios['12']) / n_samples,
        "ca_ratio_24": sum(ca_ratios['24']) / n_samples,
        "ca_ratio_36": sum(ca_ratios['36']) / n_samples,
    }


# =========================================================================
# 6. Text Decoding & Slerp Reverse Diffusion Generation
# =========================================================================
def decode_latents_to_text(decoder, qwen_embed_weight, tokenizer, z):
    """
    Декодирует непрерывные векторы Z в текст с помощью ModernLatentDecoder
    и LM-головы Qwen (Phase 2).
    """
    with torch.no_grad():
        projected = decoder(z.float())  # [B, T, 1536]
        logits = F.linear(projected, qwen_embed_weight.float())
        pred_ids = logits.argmax(dim=-1)
        text = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
        return text


def run_generation(model, decoder, latent_dict, tokenizer, query_text, device, max_len=64, steps=25):
    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Generating answer for: '{query_text}'")
    q_enc = tokenizer(query_text, return_tensors='pt').to(device)

    with torch.no_grad():
        # ADR 099 & ADR 105: Contextualized prompt cache Z_prompt_12
        Z_prompt_12 = model.encode_prompt(q_enc.input_ids, q_enc.attention_mask)

        B = 1
        T_a = max_len
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        z_canvas = safe_normalize(torch.randn(B, T_a, 1024, device=device), dim=-1)
        dt = 1.0 / steps
        run_history = []

        for i in range(steps):
            # Точная дискретная сетка с шагом 0.04 (k / 25), как в обучении
            t_val = round((1.0 - i * dt) / 0.04) * 0.04
            t_val = max(0.04, min(1.0, round(t_val, 4)))
            t_actual = torch.full((B, T_a), t_val, device=device)

            sims = torch.matmul(z_canvas, latent_dict.T)
            raw_dprox, nn_ids = sims.max(dim=-1)
            t_reported = (1.0 - raw_dprox).clamp(0.0, 1.0)

            h39, z_pred = model.forward_step(z_canvas, Z_prompt_12, q_enc.attention_mask, t_actual, t_reported)

            # Phase 2 Decoder text
            step_text = decode_latents_to_text(decoder, model.qwen_embeddings.weight, tokenizer, z_canvas)
            # Latent Dict nearest neighbor sample
            nn_text = tokenizer.decode(nn_ids[0][:20], skip_special_tokens=True)

            run_history.append({
                'step': i,
                't': t_val,
                'text': step_text,
                'raw_dprox': raw_dprox[0].mean().item()
            })

            if i % 5 == 0 or i == steps - 1:
                print(f"Step {i:02d} (t={t_val:.2f}, avg_dprox={raw_dprox[0].mean().item():.3f}):")
                print(f"   [Phase 2 Decoder]: {step_text[:90]}...")
                print(f"   [Latent Dict NN ]: {nn_text[:60]}...")

            t_next_val = max(0.0, round((t_val - 0.04) / 0.04) * 0.04)
            t_next_val = round(t_next_val, 4)
            t_next = torch.full((B, T_a), t_next_val, device=device)
            theta_now = t_actual.unsqueeze(-1) * (math.pi / 2)
            theta_next = t_next.unsqueeze(-1) * (math.pi / 2)

            sin_now = torch.sin(theta_now)
            sin_now = torch.where(sin_now < 1e-5, torch.ones_like(sin_now) * 1e-5, sin_now)
            w_pred = torch.sin(theta_now - theta_next) / sin_now
            w_cur = torch.sin(theta_next) / sin_now

            z_next = w_pred * z_pred + w_cur * z_canvas
            z_canvas = torch.where(theta_now > 1e-5, safe_normalize(z_next, dim=-1), safe_normalize(z_pred, dim=-1))

        # Финальное декодирование
        final_text = decode_latents_to_text(decoder, model.qwen_embeddings.weight, tokenizer, z_canvas)
        final_sims = torch.matmul(z_canvas, latent_dict.T)
        _, final_nn_ids = final_sims.max(dim=-1)
        final_nn_text = tokenizer.decode(final_nn_ids[0], skip_special_tokens=True)

        print(f"\n=== FINAL ANSWER (ModernLatentDecoder) ===\n{final_text}\n" + "="*42)
        print(f"=== FINAL ANSWER (LatentDict NN) ===\n{final_nn_text}\n" + "="*42)

    return {'query': query_text, 'history': run_history, 'final': final_text, 'final_nn': final_nn_text}


def generate_html(results, html_path):
    html = "<html><head><meta charset='utf-8'><title>Phase 6 Generation (ModernLatentDecoder)</title>"
    html += "<style>body{font-family:sans-serif; margin:20px; background:#f9f9f9} "
    html += ".run{background:#fff; padding:15px; margin-bottom:20px; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1)} "
    html += ".step{font-family:monospace; color:#555; margin:5px 0; padding-bottom:5px; border-bottom:1px solid #eee} "
    html += ".final{font-weight:bold; color:#2c3e50; margin-top:10px; font-size:1.1em; background:#eef6ff; padding:10px; border-radius:4px}</style></head><body>"
    html += "<h1>Phase 6 CA_Prompt Generation Demos (ModernLatentDecoder + Z_prompt_12)</h1>"

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


# =========================================================================
# 7. Main Entry Point
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Phase 6 CA_Prompt Evaluation with ModernLatentDecoder & Metrics")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive generation mode")
    parser.add_argument("--ckpt", type=str, default="", help="Path to CA_Prompt weights (Phase 6)")
    parser.add_argument("--steps", type=int, default=25, help="Number of Slerp diffusion steps")
    parser.add_argument("--max_len", type=int, default=64, help="Maximum generated answer length")
    parser.add_argument("--eval_samples", type=int, default=50, help="Number of dataset samples for quantitative metric evaluation (0 to skip)")
    parser.add_argument("--output_html", type=str, default="", help="Path to output HTML report")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Phase 6 on {device}...")

    # Дефолтный путь к чекпоинту шага 9000 или переданный через CLI
    default_ckpt_9000 = os.path.join(PROJECT_ROOT, "experiments", "phase 6", "checkpoints", "planB_phase6_checkpoints_phase6_ca_layers_step_9000.pth")
    default_ckpt_fallback = os.path.join(PROJECT_ROOT, "experiments", "phase 6", "checkpoints", "planB_phase6_checkpoints_phase6_ca_layers_step_6000.pth")
    target_ckpt = args.ckpt if args.ckpt else (default_ckpt_9000 if os.path.exists(default_ckpt_9000) else default_ckpt_fallback)

    config = {
        'qwen_path': "Qwen/Qwen2.5-1.5B",
        'modernbert_path': "answerdotai/ModernBERT-large",
        'encoder_path': os.path.join(PROJECT_ROOT, "experiments", "phase 1", "planB_phase1_checkpoints_phase1_vae_step_20000.pth"),
        'phase4_path': os.path.join(PROJECT_ROOT, "experiments", "phase 4", "local_checkpoints", "phase4_step_85995.pth"),
        'phase2_decoder_path': os.path.join(PROJECT_ROOT, "experiments", "phase 2", "planB_phase2_checkpoints_decoder_step_9000.pth"),
        'sep_token_path': os.path.join(PROJECT_ROOT, "storage", "components", "sep_token.pt"),
        'void_token_path': os.path.join(PROJECT_ROOT, "storage", "components", "void_token.pt"),
        'phase6_ckpt': target_ckpt
    }

    tokenizer = AutoTokenizer.from_pretrained(config['qwen_path'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading BEBLaDIIPhase6 model...")
    model = BEBLaDIIPhase6Eval(config).to(device)

    print("Loading Phase 2 ModernLatentDecoder...")
    decoder = ModernLatentDecoder(
        latent_dim=1024,
        qwen_dim=1536,
        num_layers=3,
        dus_weights_path=config['phase4_path']
    ).to(device)
    ckpt_dec = torch.load(config['phase2_decoder_path'], map_location=device, weights_only=False)
    st_dec = ckpt_dec.get("decoder", ckpt_dec)
    clean_st_dec = {normalize_key(k).replace("decoder.", ""): v for k, v in st_dec.items()}
    decoder.load_state_dict(clean_st_dec, strict=False)
    decoder.eval()
    print("ModernLatentDecoder loaded successfully.")

    dict_path = os.path.join(PROJECT_ROOT, "experiments", "phase 5", "local", "latent_dict.pt")
    if os.path.exists(dict_path):
        latent_dict = torch.load(dict_path, map_location=device).float()
    else:
        print("latent_dict.pt not found! Building from fallback...")
        latent_dict = torch.load(os.path.join(PROJECT_ROOT, "BEBLaDII-planB-Phase5-Data", "latent_dict.pt"), map_location=device).float()

    # 1. Количественная оценка метрик датасета
    ds_path = os.path.join(PROJECT_ROOT, "BEBLaDII-planB-Phase6-Data", "phase 6", "data", "train_phase6.parquet")
    df = None
    if os.path.exists(ds_path):
        try:
            df = pd.read_parquet(ds_path)
            if args.eval_samples > 0 and len(df) > 0:
                evaluate_dataset_metrics(model, tokenizer, df, device, max_samples=args.eval_samples)
        except Exception as e:
            print(f"Dataset metrics evaluation skipped: {e}")

    # 2. Демонстрационная диффузионная генерация
    results = []

    if df is not None and len(df) > 0:
        sample_q = df.iloc[0]['Q']
        res = run_generation(model, decoder, latent_dict, tokenizer, sample_q, device, max_len=args.max_len, steps=args.steps)
        results.append(res)

    ood_q = "Explain why the sky is blue using simple words."
    res = run_generation(model, decoder, latent_dict, tokenizer, ood_q, device, max_len=args.max_len, steps=args.steps)
    results.append(res)

    html_path = args.output_html if args.output_html else os.path.join(PROJECT_ROOT, "experiments", "phase 6", "phase6_generation_demo.html")
    generate_html(results, html_path)

    # 3. Интерактивный режим
    if args.interactive:
        while True:
            try:
                user_q = input("\nEnter prompt (or 'q' to quit): ")
                if user_q.lower() in ['q', 'quit', 'exit']:
                    break
                if not user_q.strip():
                    continue
                res = run_generation(model, decoder, latent_dict, tokenizer, user_q, device, max_len=args.max_len, steps=args.steps)
                results.append(res)
                generate_html(results, html_path)
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
