"""Автономная компонента Энкодера Plan C (Burer-Monteiro Gr(8, 128), Phase 1 step_19500)."""

import json
import math
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ..plan_c_layers import (
        ModernBERTLayer,
        RMSNorm,
        RotaryEmbedding128,
        SphereLinear,
        apply_rope_16d,
        canonicalize_cell_basis,
        clamp_to_unit_ball,
        safe_lorentzian_eigh,
    )
    from ..quantization import (
        Int8RowwiseEmbedding,
        System1QuantPolicy,
        is_quantized_checkpoint,
        load_quantized_encoder_payload,
        quantize_encoder_in_place,
        resolve_component_weights_path,
        resolve_system1_quant_policy,
    )
except ImportError:
    from plan_c_layers import (
        ModernBERTLayer,
        RMSNorm,
        RotaryEmbedding128,
        SphereLinear,
        apply_rope_16d,
        canonicalize_cell_basis,
        clamp_to_unit_ball,
        safe_lorentzian_eigh,
    )
    from quantization import (
        Int8RowwiseEmbedding,
        System1QuantPolicy,
        is_quantized_checkpoint,
        load_quantized_encoder_payload,
        quantize_encoder_in_place,
        resolve_component_weights_path,
        resolve_system1_quant_policy,
    )

ENCODER_DIR = Path(__file__).resolve().parent


class LexicalCoreMLP(nn.Module):
    """Token-wise 2-слойный SwiGLU MLP: раскладывает эмбеддинг (1024) на v_sem (8x96) и v_pos (8x16)."""

    def __init__(self, hidden_size: int = 1024, intermediate_size: int = 2048, num_axes: int = 8):
        super().__init__()
        self.num_axes = num_axes
        self.out_dim = num_axes * (96 + 16)

        self.norm = RMSNorm(hidden_size)
        self.gate_proj = SphereLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = SphereLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = SphereLinear(intermediate_size, self.out_dim, bias=False)
        self.out_norm = RMSNorm(self.out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        normed = self.norm(x)
        gate = F.silu(self.gate_proj(normed))
        up = self.up_proj(normed)
        h = self.out_norm(self.down_proj(gate * up))

        B, T, _ = h.shape
        h = h.view(B, T, self.num_axes, 112)
        return h[..., :96], h[..., 96:]


class ContextSyntaxEncoder(nn.Module):
    """2-слойный контекстный ModernBERT энкодер синтаксиса (8x16 каналов на токен)."""

    def __init__(self, hidden_size: int = 1024, intermediate_size: int = 2048, num_axes: int = 8):
        super().__init__()
        self.num_axes = num_axes
        self.layers = nn.ModuleList([ModernBERTLayer(hidden_size, intermediate_size) for _ in range(2)])
        self.final_norm = RMSNorm(hidden_size)
        self.syntax_proj = nn.Linear(hidden_size, num_axes * 16, bias=False)
        self.syntax_norm = RMSNorm(num_axes * 16)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.final_norm(x)
        proj_w_dtype = self.syntax_proj.weight.dtype
        x_proj = x if (torch.is_autocast_enabled(x.device.type) or x.dtype == proj_w_dtype) else x.to(proj_w_dtype)
        h_syn = self.syntax_norm(self.syntax_proj(x_proj))
        B, T, _ = h_syn.shape
        return h_syn.view(B, T, self.num_axes, 16)


class BlockDiagonalVAEHead(nn.Module):
    """Изолированный блочно-диагональный mu-VAE проектор (96 | 16 | 16)."""

    def __init__(self, num_axes: int = 8):
        super().__init__()
        self.num_axes = num_axes
        self.mu_sem = nn.Linear(96, 96)
        self.mu_syn = nn.Linear(16, 16)
        self.mu_pos = nn.Linear(16, 16)
        self.logvar_sem = nn.Linear(96, 96)
        self.logvar_syn = nn.Linear(16, 16)
        self.logvar_pos = nn.Linear(16, 16)

    def forward(
        self, h_sem: torch.Tensor, h_syn: torch.Tensor, h_pos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Защищённый островок: при FP32 весах VAE-головы считаем строго в их точности (тонкие каналы syn/pos)
        w_dtype = self.mu_sem.weight.dtype
        if w_dtype == torch.float32:
            with torch.autocast(device_type=h_sem.device.type, enabled=False):
                hs, hy, hp = h_sem.float(), h_syn.float(), h_pos.float()
                mu = torch.cat([self.mu_sem(hs), self.mu_syn(hy), self.mu_pos(hp)], dim=-1)
                logvar = torch.cat([self.logvar_sem(hs), self.logvar_syn(hy), self.logvar_pos(hp)], dim=-1)
            return mu, logvar
        hs, hy, hp = h_sem.to(w_dtype), h_syn.to(w_dtype), h_pos.to(w_dtype)
        mu = torch.cat([self.mu_sem(hs), self.mu_syn(hy), self.mu_pos(hp)], dim=-1)
        logvar = torch.cat([self.logvar_sem(hs), self.logvar_syn(hy), self.logvar_pos(hp)], dim=-1)
        return mu, logvar


class GrassmannianTokenNormalizer(nn.Module):
    """Сжатие в единичный шар B^128, расчет rk_eff и Axis Dropout при обучении."""

    def __init__(self, axis_dim: int = 128, num_axes: int = 8, rank_floor: float = 1.5, dropout_prob: float = 0.20):
        super().__init__()
        self.axis_dim = axis_dim
        self.num_axes = num_axes
        self.rank_floor = rank_floor
        self.dropout_prob = dropout_prob
        self.alpha = 1.0 / math.sqrt(axis_dim)

    def safe_squash(self, z: torch.Tensor) -> torch.Tensor:
        norms = torch.norm(z, p=2, dim=-1, keepdim=True)
        direction = z / (norms + 1e-7)
        return direction * torch.tanh(self.alpha * norms)

    def clamp_to_unit_ball(self, V: torch.Tensor) -> torch.Tensor:
        return clamp_to_unit_ball(V)

    def compute_rank_eff(self, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        G = torch.matmul(V, V.transpose(-2, -1))
        frob_sq = torch.diagonal(G, dim1=-2, dim2=-1).sum(dim=-1)
        gram_frob_sq = (G ** 2).sum(dim=[-2, -1])
        rk_eff = (frob_sq ** 2) / (gram_frob_sq + 1e-7)
        return rk_eff, G

    def forward(
        self, z: torch.Tensor, non_pad_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        V = self.safe_squash(z)
        rk_eff, G = self.compute_rank_eff(V)

        rank_loss_elem = F.relu(self.rank_floor - rk_eff) ** 2
        if non_pad_mask is not None:
            mask_f = non_pad_mask.float()
            rank_loss = (rank_loss_elem * mask_f).sum() / mask_f.sum().clamp(min=1.0)
        else:
            rank_loss = rank_loss_elem.mean()

        if self.training and self.dropout_prob > 0.0 and torch.rand(1).item() < self.dropout_prob:
            with torch.no_grad():
                q = G.new_ones(*G.shape[:-1], 1)
                q = q / (torch.norm(q, dim=-2, keepdim=True) + 1e-7)
                for _ in range(5):
                    q = torch.matmul(G, q)
                    q = q / (torch.norm(q, dim=-2, keepdim=True) + 1e-7)
                q_top = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0).to(V.dtype)
            proj = torch.matmul(q_top.transpose(-2, -1), V)
            V = V - torch.matmul(q_top, proj)

        return V, rk_eff, rank_loss


class GrassmannianSpectralPooler(nn.Module):
    """Спектральный пулинг окна 4 -> 1 (Partial RoPE 112:128, Lorentzian Eigh, Sign Fixing)."""

    def __init__(self, chunk_size: int = 4, num_axes_per_tok: int = 8, out_axes: int = 8, axis_dim: int = 128):
        super().__init__()
        self.chunk_size = chunk_size
        self.num_axes_per_tok = num_axes_per_tok
        self.out_axes = out_axes
        self.axis_dim = axis_dim
        self.register_buffer("positions", torch.arange(chunk_size), persistent=False)

    def forward(self, V: torch.Tensor, weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N_cells, L, K, D = V.shape
        assert L == self.chunk_size and K == self.num_axes_per_tok and D == self.axis_dim

        v_sem = V[..., :112]
        v_pos = apply_rope_16d(V[..., 112:], self.positions)
        V_tagged = torch.cat([v_sem, v_pos], dim=-1)

        if weights is None:
            w_norm = torch.full((B, N_cells, L, 1, 1), 1.0 / math.sqrt(L), device=V.device, dtype=V.dtype)
        else:
            w_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-5)
            w_norm = torch.sqrt(weights / w_sum).unsqueeze(-1).unsqueeze(-1)

        W_tilde = (V_tagged * w_norm).view(B, N_cells, L * K, D)
        G_win = torch.matmul(W_tilde.float(), W_tilde.float().transpose(-2, -1))
        G_win = 0.5 * (G_win + G_win.transpose(-2, -1))

        eigvals, eigvecs = safe_lorentzian_eigh(G_win, eps=1e-4)
        top_eigvals = F.relu(torch.flip(eigvals[..., -self.out_axes :], dims=[-1]))
        top_eigvecs = torch.flip(eigvecs[..., -self.out_axes :], dims=[-1])

        u = top_eigvecs.transpose(-2, -1)
        max_idx = torch.argmax(torch.abs(u), dim=-1, keepdim=True)
        signs = torch.sign(torch.gather(u, dim=-1, index=max_idx))
        signs = torch.where(signs == 0.0, torch.ones_like(signs), signs)
        u_fixed = u * signs

        V_cell = torch.matmul(u_fixed.to(V.dtype), W_tilde)
        singular_vals = torch.sqrt(top_eigvals.clamp(min=1e-8)).to(V.dtype)
        return V_cell, singular_vals


class GrassmannianEncoder(nn.Module):
    """Энкодер Plan C: отображает input_ids (B, 4N) в концепт-клетки Gr(8, 128) формы (B, N, 8, 128)."""

    def __init__(
        self,
        vocab_size: int = 131072,
        hidden_size: int = 1024,
        intermediate_size: int = 2048,
        num_axes: int = 8,
        axis_dim: int = 128,
        chunk_size: int = 4,
        pad_token_id: int = 10,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_axes = num_axes
        self.axis_dim = axis_dim
        self.chunk_size = chunk_size
        self.pad_token_id = pad_token_id

        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.embed_norm = RMSNorm(hidden_size)
        self.rope = RotaryEmbedding128(dim=axis_dim, max_seq_len=max_seq_len)
        self.lexical_mlp = LexicalCoreMLP(hidden_size, intermediate_size, num_axes)
        self.context_encoder = ContextSyntaxEncoder(hidden_size, intermediate_size, num_axes)
        self.vae_head = BlockDiagonalVAEHead(num_axes=num_axes)
        self.normalizer = GrassmannianTokenNormalizer(axis_dim=axis_dim, num_axes=num_axes)
        self.pooler = GrassmannianSpectralPooler(chunk_size=chunk_size, num_axes_per_tok=num_axes, out_axes=num_axes, axis_dim=axis_dim)

    def encode_and_pool(
        self, input_ids: torch.Tensor, weights: Optional[torch.Tensor] = None, canonicalize: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Кодирует последовательность (B, 4N) в V_cell (B, N, 8, 128) с возвратом kl_loss, rank_loss и rk_eff."""
        B, seq_len = input_ids.shape
        assert seq_len % self.chunk_size == 0, f"Seq len {seq_len} must be divisible by {self.chunk_size}"
        num_cells = seq_len // self.chunk_size

        x_embed = self.embed_norm(self.embeddings(input_ids))
        cos, sin = self.rope(x_embed, seq_len)

        h_sem, h_pos = self.lexical_mlp(x_embed)
        h_syn = self.context_encoder(x_embed, cos, sin)
        mu, logvar = self.vae_head(h_sem, h_syn, h_pos)

        non_pad_mask = input_ids != self.pad_token_id
        mask_4d = non_pad_mask.unsqueeze(-1).unsqueeze(-1).float()
        var = torch.exp(torch.clamp(logvar, min=-10.0, max=10.0))
        kl_elem = torch.clamp(-0.5 * (1.0 + logvar - var), min=0.0) * mask_4d
        var_floor = F.relu(0.05 - var).pow(2) * mask_4d
        kl_loss = (kl_elem + 2.0 * var_floor).sum() / (mask_4d.sum() * self.num_axes * self.axis_dim).clamp(min=1.0)

        z_raw = mu + torch.randn_like(mu) * torch.sqrt(var) if self.training else mu
        V_tok, rk_eff, rank_loss = self.normalizer(z_raw, non_pad_mask=non_pad_mask)

        is_pad = ~non_pad_mask
        if is_pad.any():
            V_pad = torch.zeros(self.num_axes, self.axis_dim, device=V_tok.device, dtype=V_tok.dtype)
            for k in range(self.num_axes):
                amp = math.sqrt(0.01 * (0.6 ** k))
                V_pad[k, 104 + k] = amp
                V_pad[k, 112 + k] = amp
            V_tok = torch.where(is_pad.unsqueeze(-1).unsqueeze(-1), V_pad, V_tok)
            rk_eff = torch.where(is_pad, torch.full_like(rk_eff, 2.0), rk_eff)

        V_tok_windows = V_tok.view(B, num_cells, self.chunk_size, self.num_axes, self.axis_dim)
        weights_windows = weights.view(B, num_cells, self.chunk_size) if weights is not None else None
        V_cell, _ = self.pooler(V_tok_windows, weights_windows)
        if canonicalize:
            V_cell, _ = canonicalize_cell_basis(V_cell)
        return V_cell, kl_loss, rank_loss, rk_eff

    def forward(
        self, input_ids: torch.Tensor, weights: Optional[torch.Tensor] = None, canonicalize: bool = False
    ) -> torch.Tensor:
        V_cell, _, _, _ = self.encode_and_pool(input_ids, weights, canonicalize=canonicalize)
        return V_cell


def get_encoder(
    weights_path: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
    device: Union[str, torch.device] = "cpu",
    dtype: Optional[torch.dtype] = None,
    eval_mode: bool = True,
    quant_mode: str = "fp32",
    mode: str = "inference",
    amp_dtype: Optional[torch.dtype] = None,
) -> GrassmannianEncoder:
    """
    Загружает GrassmannianEncoder из storage/components/encoder с поддержкой квантизации.

    Параметры квантизации и переключения:
      - quant_mode: 'fp32' (по умолчанию, полная обратная совместимость), 'auto' (автовыбор по mode+device),
                    'half' ('fp16'/'bf16'), 'emb_int8', 'int8'.
      - mode: 'train' (слои в FP16/BF16/FP32, словарь per-row INT8) или 'inference' (полный per-row INT8 на CPU/GPU).
      - amp_dtype: желаемый compute dtype (например env.amp_dtype).
    Если готовый квантизованный файл (weights_emb_int8_fp16.pt / weights_int8.pt / weights_fp16.pt) отсутствует
    рядом с weights_path (например на Kaggle со старым датасетом), конвертация выполняется автоматически на лету.
    """
    cfg_file = Path(config_path) if config_path else ENCODER_DIR / "config.json"
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    policy: System1QuantPolicy = resolve_system1_quant_policy(
        mode=mode,
        device=device,
        quant_mode=quant_mode,
        amp_dtype=(amp_dtype if amp_dtype is not None else dtype),
    )

    model = GrassmannianEncoder(
        vocab_size=cfg.get("vocab_size", 131072),
        hidden_size=cfg.get("hidden_size", 1024),
        intermediate_size=cfg.get("intermediate_size", 2048),
        num_axes=cfg.get("num_axes", 8),
        axis_dim=cfg.get("axis_dim", 128),
        chunk_size=cfg.get("chunk_size", 4),
        pad_token_id=cfg.get("pad_token_id", 10),
        max_seq_len=cfg.get("max_seq_len", 4096),
    )

    fallback_name = cfg.get("weights_file", "weights.pt")
    w_file, _ = resolve_component_weights_path(
        base_dir=ENCODER_DIR,
        preset_filename=policy.encoder_weights_filename,
        explicit_weights_path=weights_path,
        fallback_filename=fallback_name,
    )
    raw = torch.load(w_file, map_location="cpu", weights_only=False)

    if is_quantized_checkpoint(raw):
        loaded_preset = load_quantized_encoder_payload(
            model,
            payload=raw,
            layer_dtype=policy.layer_dtype,
            use_cpu_onednn=policy.use_cpu_onednn_int8,
        )
        if policy.encoder_preset == "int8" and loaded_preset != "int8":
            quantize_encoder_in_place(
                model,
                preset="int8",
                layer_dtype=policy.layer_dtype,
                use_cpu_onednn=policy.use_cpu_onednn_int8,
            )
    else:
        sd = raw["model_state_dict"] if (isinstance(raw, dict) and "model_state_dict" in raw) else raw
        model.load_state_dict(sd, strict=True)
        if policy.encoder_preset != "fp32":
            quantize_encoder_in_place(
                model,
                preset=policy.encoder_preset,
                layer_dtype=policy.layer_dtype,
                use_cpu_onednn=policy.use_cpu_onednn_int8,
            )
        elif dtype is not None:
            model = model.to(dtype=dtype)

    model.quant_policy = policy
    model = model.to(device)
    if eval_mode:
        model.eval()
    return model


__all__ = [
    "ENCODER_DIR",
    "LexicalCoreMLP",
    "ContextSyntaxEncoder",
    "BlockDiagonalVAEHead",
    "GrassmannianTokenNormalizer",
    "GrassmannianSpectralPooler",
    "GrassmannianEncoder",
    "get_encoder",
]
