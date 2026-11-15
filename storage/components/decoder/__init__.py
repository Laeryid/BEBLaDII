"""Автономная компонента Декодера Plan C (O(8)-инвариантный декомпрессор 1 -> 4, Phase 1 step_19500)."""

import json
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ..plan_c_layers import (
        ModernBERTLayer,
        RMSNorm,
        RotaryEmbedding128,
        clamp_to_unit_ball,
    )
except ImportError:
    from plan_c_layers import (
        ModernBERTLayer,
        RMSNorm,
        RotaryEmbedding128,
        clamp_to_unit_ball,
    )

DECODER_DIR = Path(__file__).resolve().parent


class SetCrossAttentionUnpacker(nn.Module):
    """Архитектурно O(8)-инвариантная распаковка клетки 1 -> 4 через Cross-Attention и Gram(V) в float32."""

    def __init__(self, axis_dim: int = 128, num_slots: int = 4, hidden_size: int = 1024):
        super().__init__()
        self.axis_dim = axis_dim
        self.num_slots = num_slots
        self.hidden_size = hidden_size

        self.query_slots = nn.Parameter(torch.randn(num_slots, axis_dim) * 0.02)
        self.q_norm = RMSNorm(axis_dim)
        self.k_norm = RMSNorm(axis_dim)

        self.gram_mlp = nn.Sequential(
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            RMSNorm(128),
        )
        self.out_proj = nn.Linear(axis_dim + 128, hidden_size)
        self.out_norm = RMSNorm(hidden_size)

    def forward(self, V_cell: torch.Tensor) -> torch.Tensor:
        B, N, _, D = V_cell.shape
        Q = self.q_norm(self.query_slots).unsqueeze(0).unsqueeze(0).expand(B, N, self.num_slots, D).to(V_cell.dtype)
        K_mat = self.k_norm(V_cell)

        scores = torch.matmul(Q, K_mat.transpose(-2, -1)) / (D ** 0.5)
        attn = F.softmax(scores.float(), dim=-1).to(V_cell.dtype)
        h_attn = torch.matmul(attn, V_cell)

        # Строгий float32 с отключенным autocast для матрицы Грама и gram_mlp (ADR 126)
        device_type = V_cell.device.type
        target_dtype = V_cell.dtype
        with torch.autocast(device_type=device_type, enabled=False):
            V_f32 = V_cell.float()
            G_flat_f32 = torch.matmul(V_f32, V_f32.transpose(-2, -1)).view(B, N, 64)

            mlp_w0 = self.gram_mlp[0].weight.float()
            mlp_b0 = self.gram_mlp[0].bias.float() if self.gram_mlp[0].bias is not None else None
            h1 = F.silu(F.linear(G_flat_f32, mlp_w0, mlp_b0))
            mlp_w2 = self.gram_mlp[2].weight.float()
            mlp_b2 = self.gram_mlp[2].bias.float() if self.gram_mlp[2].bias is not None else None
            h2 = F.linear(h1, mlp_w2, mlp_b2)
            norm_w = self.gram_mlp[3].weight.float()
            variance = h2.pow(2).mean(-1, keepdim=True)
            h_normed = h2 * torch.rsqrt(variance + 1e-6) * norm_w
            h_gram = h_normed.to(target_dtype).unsqueeze(2).expand(B, N, self.num_slots, 128)

        h_comb = torch.cat([h_attn, h_gram], dim=-1)
        h_out = self.out_norm(self.out_proj(h_comb))
        return h_out.view(B, N * self.num_slots, self.hidden_size)


class GrassmannianDecoder(nn.Module):
    """Иерархический O(8)-инвариантный декомпрессор: (B, N, 8, 128) -> logits (B, 4N, 131072)."""

    def __init__(
        self,
        vocab_size: int = 131072,
        hidden_size: int = 1024,
        intermediate_size: int = 2048,
        num_axes: int = 8,
        axis_dim: int = 128,
        chunk_size: int = 4,
        max_seq_len: int = 4096,
        cell_jitter_std: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_axes = num_axes
        self.axis_dim = axis_dim
        self.chunk_size = chunk_size
        self.cell_jitter_std = cell_jitter_std

        self.rope = RotaryEmbedding128(dim=axis_dim, max_seq_len=max_seq_len)
        self.cell_in_norm = RMSNorm(hidden_size)
        self.dec_context_layer = ModernBERTLayer(hidden_size, intermediate_size)
        self.unpacker = SetCrossAttentionUnpacker(axis_dim=axis_dim, num_slots=chunk_size, hidden_size=hidden_size)
        self.dec_smooth_layers = nn.ModuleList([ModernBERTLayer(hidden_size, intermediate_size) for _ in range(2)])
        self.dec_final_norm = RMSNorm(hidden_size)

        # Разделяемая матрица эмбеддингов для tied LM head (ADR 125)
        self.register_buffer("lm_head_weight", None, persistent=False)

    def tie_embeddings(self, embeddings_weight: torch.Tensor) -> None:
        """Привязывает матрицу эмбеддингов (131072 x 1024) как выходную голову декодера."""
        self.lm_head_weight = embeddings_weight

    def decode_hidden(self, V_cell: torch.Tensor) -> torch.Tensor:
        """Декомпрессия клеток V_cell (B, N, 8, 128) в скрытые состояния (B, 4N, 1024)."""
        B, N, K, D = V_cell.shape
        if self.training and self.cell_jitter_std > 0.0:
            noise_scale = torch.rand(B, N, 1, 1, device=V_cell.device, dtype=V_cell.dtype) * self.cell_jitter_std
            V_cell_in = clamp_to_unit_ball(V_cell + torch.randn_like(V_cell) * noise_scale)
        else:
            V_cell_in = V_cell

        cell_vec = self.cell_in_norm(V_cell_in.reshape(B, N, K * D))
        cos_N, sin_N = self.rope(cell_vec, N)
        h_cell = self.dec_context_layer(cell_vec, cos_N, sin_N)
        V_cell_mod = V_cell_in + 0.1 * h_cell.reshape(B, N, K, D)

        h_unpacked = self.unpacker(V_cell_mod)
        full_len = N * self.chunk_size
        cos_full, sin_full = self.rope(h_unpacked, full_len)
        x = h_unpacked
        for layer in self.dec_smooth_layers:
            x = layer(x, cos_full, sin_full)
        return self.dec_final_norm(x)

    def get_tied_head_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Вычисление логитов строго в float32 с отключенным autocast (ADR 125)."""
        if self.lm_head_weight is None:
            raise RuntimeError(
                "Не задана матрица lm_head_weight. Передайте embeddings_weight в get_decoder() или вызовите tie_embeddings()."
            )
        device_type = hidden_states.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            return F.linear(hidden_states.float(), self.lm_head_weight.float())

    def forward(self, V_cell: torch.Tensor) -> torch.Tensor:
        x = self.decode_hidden(V_cell)
        return self.get_tied_head_logits(x)


def get_decoder(
    weights_path: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
    embeddings_weight: Optional[torch.Tensor] = None,
    encoder: Optional[nn.Module] = None,
    device: Union[str, torch.device] = "cpu",
    dtype: Optional[torch.dtype] = None,
    eval_mode: bool = True,
) -> GrassmannianDecoder:
    """Загружает GrassmannianDecoder и привязывает матрицу эмбеддингов (от encoder или из ../encoder/weights.pt)."""
    cfg_file = Path(config_path) if config_path else DECODER_DIR / "config.json"
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model = GrassmannianDecoder(
        vocab_size=cfg.get("vocab_size", 131072),
        hidden_size=cfg.get("hidden_size", 1024),
        intermediate_size=cfg.get("intermediate_size", 2048),
        num_axes=cfg.get("num_axes", 8),
        axis_dim=cfg.get("axis_dim", 128),
        chunk_size=cfg.get("chunk_size", 4),
        max_seq_len=cfg.get("max_seq_len", 4096),
        cell_jitter_std=cfg.get("cell_jitter_std", 0.0),
    )

    w_file = Path(weights_path) if weights_path else DECODER_DIR / cfg.get("weights_file", "weights.pt")
    sd = torch.load(w_file, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd, strict=True)

    if embeddings_weight is not None:
        model.tie_embeddings(embeddings_weight)
    elif encoder is not None and hasattr(encoder, "embeddings"):
        model.tie_embeddings(encoder.embeddings.weight)
    else:
        enc_weights = DECODER_DIR.parent / "encoder" / "weights.pt"
        if enc_weights.exists():
            enc_sd = torch.load(enc_weights, map_location="cpu", weights_only=False)
            if "embeddings.weight" in enc_sd:
                model.tie_embeddings(enc_sd["embeddings.weight"])

    if dtype is not None:
        model = model.to(dtype=dtype)
    model = model.to(device)
    if eval_mode:
        model.eval()
    return model


__all__ = [
    "DECODER_DIR",
    "SetCrossAttentionUnpacker",
    "GrassmannianDecoder",
    "get_decoder",
]
