import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional/timestep embedding."""
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
    """
    Adaptive LayerNorm (AdaLN) modulation module.
    Predicts shift and scale parameters from timestep embedding.
    """
    def __init__(self, t_emb_dim: int, hidden_dim: int):
        super().__init__()
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, 2 * hidden_dim))
        nn.init.zeros_(self.modulation[-1].weight)
        bias = torch.zeros(2 * hidden_dim)
        bias[hidden_dim:] = 1.0
        self.modulation[-1].bias = nn.Parameter(bias)

    def forward(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.modulation(t_emb)
        shift, scale = out.chunk(2, dim=-1)
        if shift.dim() == 2:
            return shift.unsqueeze(1), scale.unsqueeze(1)
        return shift, scale


class AdaLNWrappedLayerNorm(nn.Module):
    """
    Wraps standard LayerNorm / RMSNorm with AdaLN conditioning.
    """
    def __init__(self, original_norm, adaln_module: AdaLNModulation):
        super().__init__()
        self.original_norm = original_norm
        self.adaln = adaln_module
        self._current_t_emb = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.original_norm(x)
        if self._current_t_emb is None:
            return out
        shift, scale = self.adaln(self._current_t_emb)
        return out * scale.to(out.dtype) + shift.to(out.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embedding to query and key tensors.
    q, k: [B, heads, seq_len, head_dim]
    cos, sin: [1, 1, seq_len, head_dim]
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) without learnable parameters.
    XLA-friendly and deterministic.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().view(1, 1, seq_len, self.dim).to(dtype=x.dtype)
        sin = emb.sin().view(1, 1, seq_len, self.dim).to(dtype=x.dtype)
        return cos, sin


def patch_modernbert_xla_safe_mask(modernbert_model):
    """
    Патчит генерацию и применение масок в ModernBertModel для стабильности на TPU XLA (ADR 103, ADR 104).
    1. Заменяет булевы маски (True=attend, False=mask) на аддитивные float-маски (0.0 и -10000.0).
    2. Заменяет экстремальные отрицательные числа (< -10000.0, -inf, -3.4e38) на безопасный сентинел -10000.0.
    3. Перехватывает как attention_mask, так и sliding_window_mask на всех уровнях:
       - legacy _update_attention_mask (transformers <= 4.49)
       - create_bidirectional_mask, create_bidirectional_sliding_window_mask, sdpa_mask
       - MODERNBERT_ATTENTION_FUNCTION["sdpa"]
       - layer.attn.forward
    """
    def _clamp_mask_tensor(mask, target_dtype=torch.float32):
        if mask is None or not torch.is_tensor(mask):
            return mask
        if mask.dtype == torch.bool:
            zero = torch.tensor(0.0, dtype=target_dtype, device=mask.device)
            neg = torch.tensor(-10000.0, dtype=target_dtype, device=mask.device)
            return torch.where(mask, zero, neg)
        if mask.is_floating_point():
            safe_min = torch.tensor(-10000.0, dtype=mask.dtype, device=mask.device)
            return torch.where((mask < -10000.0) | torch.isneginf(mask), safe_min, mask)
        return mask

    def _wrap_mask_fn(orig_fn):
        def safe_mask_fn(*args, **kwargs):
            mask = orig_fn(*args, **kwargs)
            return _clamp_mask_tensor(mask)
        return safe_mask_fn

    # 1. Если есть legacy-метод _update_attention_mask (transformers <= 4.49)
    if hasattr(modernbert_model, "_update_attention_mask"):
        orig_update_mask = modernbert_model._update_attention_mask
        def safe_update_attention_mask(attention_mask: torch.Tensor, output_attentions: bool = False):
            global_mask, sliding_mask = orig_update_mask(attention_mask, output_attentions=output_attentions)
            return _clamp_mask_tensor(global_mask), _clamp_mask_tensor(sliding_mask)
        modernbert_model._update_attention_mask = safe_update_attention_mask

    # 2. Модуль modeling_modernbert (transformers >= 4.50)
    mod = sys.modules.get(modernbert_model.__class__.__module__)
    if mod is not None:
        for fn_name in ["create_bidirectional_mask", "create_bidirectional_sliding_window_mask"]:
            if hasattr(mod, fn_name):
                setattr(mod, fn_name, _wrap_mask_fn(getattr(mod, fn_name)))
        if hasattr(mod, "MODERNBERT_ATTENTION_FUNCTION") and "sdpa" in mod.MODERNBERT_ATTENTION_FUNCTION:
            orig_sdpa = mod.MODERNBERT_ATTENTION_FUNCTION["sdpa"]
            def safe_sdpa(module, qkv, attention_mask=None, sliding_window_mask=None, *args, **kwargs):
                td = qkv.dtype if torch.is_tensor(qkv) else torch.float32
                if attention_mask is not None:
                    attention_mask = _clamp_mask_tensor(attention_mask, target_dtype=td)
                if sliding_window_mask is not None:
                    sliding_window_mask = _clamp_mask_tensor(sliding_window_mask, target_dtype=td)
                return orig_sdpa(module, qkv, attention_mask=attention_mask, sliding_window_mask=sliding_window_mask, *args, **kwargs)
            mod.MODERNBERT_ATTENTION_FUNCTION["sdpa"] = safe_sdpa

    # 3. Общий модуль transformers.masking_utils
    try:
        import transformers.masking_utils as mu
        for fn_name in ["create_bidirectional_mask", "create_bidirectional_sliding_window_mask", "sdpa_mask"]:
            if hasattr(mu, fn_name):
                setattr(mu, fn_name, _wrap_mask_fn(getattr(mu, fn_name)))
    except (ImportError, AttributeError):
        pass

    # 4. Непосредственный перехват на уровне слоев внимания (layer.attn.forward)
    if hasattr(modernbert_model, "layers"):
        for layer in modernbert_model.layers:
            attn_mod = getattr(layer, "attn", None)
            if attn_mod is not None and hasattr(attn_mod, "forward"):
                orig_attn_forward = attn_mod.forward
                def _make_safe_attn_forward(orig_fn):
                    def safe_attn_forward(*args, **kwargs):
                        td = args[0].dtype if (len(args) > 0 and torch.is_tensor(args[0])) else torch.float32
                        if "hidden_states" in kwargs and torch.is_tensor(kwargs["hidden_states"]):
                            td = kwargs["hidden_states"].dtype
                        if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                            kwargs["attention_mask"] = _clamp_mask_tensor(kwargs["attention_mask"], target_dtype=td)
                        if "sliding_window_mask" in kwargs and kwargs["sliding_window_mask"] is not None:
                            kwargs["sliding_window_mask"] = _clamp_mask_tensor(kwargs["sliding_window_mask"], target_dtype=td)
                        if len(args) >= 3 and args[2] is not None:
                            args_list = list(args)
                            args_list[2] = _clamp_mask_tensor(args_list[2], target_dtype=td)
                            args = tuple(args_list)
                        return orig_fn(*args, **kwargs)
                    return safe_attn_forward
                attn_mod.forward = _make_safe_attn_forward(orig_attn_forward)
