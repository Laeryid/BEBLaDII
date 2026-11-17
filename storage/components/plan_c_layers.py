"""Общие математические примитивы и архитектурные слои Plan C (Burer-Monteiro Gr(8, 128))."""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class LorentzianEigh(torch.autograd.Function):
    """Спектральное разложение симметричной матрицы с лоренцевой регуляризацией градиентов."""

    @staticmethod
    def forward(ctx, G: torch.Tensor, eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = G.dtype
        G_f32 = 0.5 * (G.float() + G.float().transpose(-2, -1))
        K = G_f32.size(-1)

        diag_shift = torch.arange(1, K + 1, device=G_f32.device, dtype=torch.float32) * 1e-6
        G_reg = G_f32 + torch.diag(diag_shift)

        L_reg: Optional[torch.Tensor] = None
        V: Optional[torch.Tensor] = None
        try:
            L_reg, V = torch.linalg.eigh(G_reg)
            is_valid = torch.isfinite(L_reg).all() and torch.isfinite(V).all()
        except Exception:
            is_valid = False

        if not is_valid:
            try:
                L_64, V_64 = torch.linalg.eigh(G_reg.double())
                L_reg = L_64.float()
                V = V_64.float()
            except Exception:
                L_reg = torch.full(G_reg.shape[:-1], 1e-5, device=G_reg.device, dtype=torch.float32)
                V = torch.eye(K, device=G_reg.device, dtype=torch.float32).expand_as(G_reg).clone()
            L_reg = torch.nan_to_num(L_reg, nan=1e-5, posinf=1.0, neginf=1e-5)
            V = torch.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)

        ctx.save_for_backward(L_reg, V)
        ctx.eps = eps
        ctx.orig_dtype = orig_dtype

        L_clean = torch.clamp(L_reg - diag_shift, min=1e-8)
        return L_clean.to(orig_dtype), V.to(orig_dtype)

    @staticmethod
    def backward(ctx, grad_L: torch.Tensor, grad_V: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        L_reg, V = ctx.saved_tensors
        eps = ctx.eps

        diff = L_reg.unsqueeze(-2) - L_reg.unsqueeze(-1)
        lorentz = diff / (diff ** 2 + eps ** 2)
        lorentz.diagonal(dim1=-2, dim2=-1).zero_()

        vt_gv = torch.matmul(V.transpose(-2, -1), grad_V.float())
        vt_gv_skew = 0.5 * (vt_gv - vt_gv.transpose(-2, -1))
        mid = torch.diag_embed(grad_L.float()) + lorentz * vt_gv_skew

        grad_G = torch.matmul(torch.matmul(V, mid), V.transpose(-2, -1))
        grad_G = 0.5 * (grad_G + grad_G.transpose(-2, -1))
        grad_G = torch.nan_to_num(grad_G, nan=0.0, posinf=0.0, neginf=0.0)
        return grad_G.to(ctx.orig_dtype), None


def safe_lorentzian_eigh(G: torch.Tensor, eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    """Безопасное спектральное разложение с лоренцевым ядром в backward."""
    return LorentzianEigh.apply(G, eps)


def apply_rope_16d(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Partial RoPE для последних 16 координат (8 пар) по гармонической сетке [pi/8, pi/2]."""
    thetas = torch.linspace(math.pi / 8.0, math.pi / 2.0, steps=8, device=x.device, dtype=x.dtype)
    angles = positions.unsqueeze(-1).to(x.dtype) * thetas.unsqueeze(0)
    sin = torch.sin(angles).unsqueeze(-2)
    cos = torch.cos(angles).unsqueeze(-2)

    shape_prefix = x.shape[:-1]
    x_pairs = x.view(*shape_prefix, 8, 2)
    x0 = x_pairs[..., 0]
    x1 = x_pairs[..., 1]

    x0_rot = x0 * cos - x1 * sin
    x1_rot = x0 * sin + x1 * cos
    return torch.stack([x0_rot, x1_rot], dim=-1).view(*shape_prefix, 16)


def clamp_to_unit_ball(V: torch.Tensor) -> torch.Tensor:
    """Радиальная проекция строк в единичный шар B^128 без сжатия векторов с нормой <= 1."""
    norms = torch.norm(V, p=2, dim=-1, keepdim=True)
    return V / norms.clamp(min=1.0)


def canonicalize_cell_basis(V: torch.Tensor, eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Детерминированная O(K)-инвариантная канонизация базиса клетки V in R^{..., K, D}:
    1. Вычисляет матрицу Грама G = V V^T в float32 с отключенным autocast (ADR 126).
    2. Диагонализирует G через safe_lorentzian_eigh (строго монотонный diag_shift, ADR 122).
    3. Сортирует собственные векторы Q по убыванию собственных значений lambda_1 >= ... >= lambda_K.
    4. Переводит базис в канонические главные оси V_aligned = Q^T V (инвариантно к R in O(K)).
    5. Детерминированно фиксирует знак каждой строки по максимальной по модулю координате в R^D.
    Возвращает (V_canon, eigvals_desc).
    """
    device_type = V.device.type
    orig_dtype = V.dtype
    with torch.autocast(device_type=device_type, enabled=False):
        V_f32 = torch.nan_to_num(V.float(), nan=0.0, posinf=65000.0, neginf=-65000.0)
        G = torch.matmul(V_f32, V_f32.transpose(-2, -1))
        G = 0.5 * (G + G.transpose(-2, -1))

        eigvals, eigvecs = safe_lorentzian_eigh(G, eps=eps)
        eigvals_desc = F.relu(torch.flip(eigvals, dims=[-1]))
        Q_desc = torch.flip(eigvecs, dims=[-1])

        V_aligned = torch.matmul(Q_desc.transpose(-2, -1), V_f32)
        max_idx = torch.argmax(torch.abs(V_aligned), dim=-1, keepdim=True)
        max_vals = torch.gather(V_aligned, dim=-1, index=max_idx)
        signs = torch.where(max_vals >= 0.0, torch.ones_like(max_vals), -torch.ones_like(max_vals))
        V_canon = V_aligned * signs.detach()

    return V_canon.to(orig_dtype), eigvals_desc.to(orig_dtype)


def _build_gpt2_byte_decoder() -> Dict[str, int]:
    """Обратная таблица ByteLevel BPE (символ юникода GPT-2 -> исходный байт 0..255)."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class WordAwareElasticChunker:
    """
    Словно-ориентированный эластичный чанкер (1..4 BPE токена на концепт-клетку).
    Исключает разрыв слов длины <= chunk_size между соседними клетками (Теорема 9 Эккарта-Янга),
    гасит фазовый сдвиг внутри клетки и формирует двухуровневые веса/таргеты (plan C.md п. 6.2):
    - Внутренние пустые слоты смешанных клеток имеют вес eps_pad=0.001 и target=pad_token_id.
    - Хвостовые полностью пустые VOID-клетки имеют target=-100 (ignore_index).
    """

    def __init__(
        self,
        tokenizer: Any = None,
        vocab_size: int = 131072,
        chunk_size: int = 4,
        pad_token_id: int = 10,
        eps_stop: float = 0.02,
        eps_pad: float = 0.001,
        stopword_mask: Optional[torch.Tensor] = None,
        special_token_limit: int = 1000,
    ):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        self.pad_token_id = pad_token_id
        self.eps_stop = eps_stop
        self.eps_pad = eps_pad
        self.special_token_limit = special_token_limit

        if stopword_mask is not None:
            if stopword_mask.numel() < vocab_size:
                padded = torch.zeros(vocab_size, dtype=torch.bool)
                padded[: stopword_mask.numel()] = stopword_mask.bool()
                self.stopword_mask = padded
            else:
                self.stopword_mask = stopword_mask[:vocab_size].bool()
        else:
            self.stopword_mask = torch.zeros(vocab_size, dtype=torch.bool)
            self.stopword_mask[: min(64, vocab_size)] = True

        self.starts_new, self.ends_boundary, self.is_partial_utf8 = self._build_vocab_boundary_masks()

    def _build_vocab_boundary_masks(self) -> Tuple[List[bool], List[bool], List[bool]]:
        starts_new = [True] * self.vocab_size
        ends_boundary = [True] * self.vocab_size
        is_partial_utf8 = [False] * self.vocab_size

        if self.tokenizer is None:
            return starts_new, ends_boundary, is_partial_utf8

        backend = getattr(self.tokenizer, "_tokenizer", None) or getattr(self.tokenizer, "backend_tokenizer", None) or self.tokenizer
        id_to_token = getattr(backend, "id_to_token", None)
        convert_fn = getattr(self.tokenizer, "convert_ids_to_tokens", None)
        byte_dec = _build_gpt2_byte_decoder()

        tok_vocab_size = self.vocab_size
        if hasattr(backend, "get_vocab_size"):
            tok_vocab_size = min(self.vocab_size, int(backend.get_vocab_size()))

        for tid in range(self.special_token_limit, tok_vocab_size):
            tok_str = id_to_token(tid) if callable(id_to_token) else (convert_fn(tid) if callable(convert_fn) else None)
            if not tok_str:
                continue
            try:
                raw_bytes = bytes([byte_dec[ch] for ch in tok_str if ch in byte_dec])
                decoded = raw_bytes.decode("utf-8", errors="replace")
            except Exception:
                decoded = self.tokenizer.decode([tid]) if hasattr(self.tokenizer, "decode") else ""

            has_rep = "\ufffd" in decoded
            is_partial_utf8[tid] = has_rep
            if has_rep:
                starts_new[tid] = False
                ends_boundary[tid] = False
            elif len(decoded) == 0:
                starts_new[tid] = True
                ends_boundary[tid] = True
            else:
                ch0, ch_last = decoded[0], decoded[-1]
                starts_new[tid] = ch0.isspace() or (not ch0.isalnum() and ch0 != "_")
                ends_boundary[tid] = ch_last.isspace() or (not ch_last.isalnum() and ch_last != "_")

        return starts_new, ends_boundary, is_partial_utf8

    def split_into_words(self, token_ids: List[int]) -> List[List[int]]:
        """Группирует поток BPE-токенов в неделимые слова и знаки препинания."""
        words: List[List[int]] = []
        current_word: List[int] = []
        prev_tid = -1
        n_vocab = self.vocab_size

        for raw_tid in token_ids:
            tid = int(raw_tid)
            if tid == self.pad_token_id:
                continue
            safe_tid = tid if 0 <= tid < n_vocab else 0

            if not current_word:
                start_new = True
            elif safe_tid < self.special_token_limit:
                start_new = True
            elif self.is_partial_utf8[safe_tid]:
                start_new = prev_tid < self.special_token_limit
            elif self.starts_new[safe_tid]:
                start_new = True
            elif prev_tid >= 0 and self.ends_boundary[prev_tid]:
                start_new = True
            else:
                start_new = False

            if start_new and current_word:
                words.append(current_word)
                current_word = [tid]
            else:
                current_word.append(tid)
            prev_tid = safe_tid

        if current_word:
            words.append(current_word)
        return words

    def pack_cells_from_words(
        self,
        words: List[List[int]],
        max_cells: int,
        elastic_drop_prob: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[List[List[int]], int]:
        """Упаковывает слова в клетки размера chunk_size без разрыва слов <= chunk_size."""
        cells: List[List[int]] = []
        current_cell: List[int] = []
        cs = self.chunk_size

        def flush_cell() -> None:
            nonlocal current_cell
            if current_cell and len(cells) < max_cells:
                padded = current_cell + [self.pad_token_id] * (cs - len(current_cell))
                cells.append(padded)
            current_cell = []

        for word in words:
            if len(cells) >= max_cells:
                break
            w_len = len(word)
            if w_len > cs:
                flush_cell()
                for offset in range(0, w_len, cs):
                    if len(cells) >= max_cells:
                        break
                    piece = word[offset : offset + cs]
                    if len(piece) == cs:
                        cells.append(piece)
                    else:
                        current_cell = list(piece)
            else:
                if current_cell and (len(current_cell) + w_len > cs):
                    flush_cell()
                    if len(cells) >= max_cells:
                        break
                current_cell.extend(word)
                if len(current_cell) == cs:
                    flush_cell()
                elif elastic_drop_prob > 0.0 and current_cell:
                    r = torch.rand(1, generator=generator).item()
                    if r < elastic_drop_prob:
                        flush_cell()

        flush_cell()
        n_content_cells = len(cells)
        while len(cells) < max_cells:
            cells.append([self.pad_token_id] * cs)
        return cells, n_content_cells

    def chunk_tokens(
        self,
        token_ids: Union[List[int], torch.Tensor],
        max_cells: int,
        elastic_drop_prob: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        """Возвращает input_ids (4N,), targets (4N,), weights (4N,), content_cell_mask (N,)."""
        if isinstance(token_ids, torch.Tensor):
            token_list = token_ids.tolist()
        else:
            token_list = list(token_ids)

        words = self.split_into_words(token_list)
        cells, n_content_cells = self.pack_cells_from_words(
            words, max_cells=max_cells, elastic_drop_prob=elastic_drop_prob, generator=generator
        )

        cs = self.chunk_size
        seq_len = max_cells * cs
        input_ids = torch.tensor(cells, dtype=torch.long).view(seq_len)

        targets = input_ids.clone()
        if n_content_cells < max_cells:
            targets[n_content_cells * cs :] = -100

        is_pad = input_ids == self.pad_token_id
        safe_ids = torch.clamp(input_ids, 0, self.stopword_mask.numel() - 1)
        is_stop = self.stopword_mask[safe_ids] & (~is_pad)

        raw_w = torch.where(
            is_pad,
            torch.tensor(self.eps_pad, dtype=torch.float32),
            torch.where(
                is_stop,
                torch.tensor(self.eps_stop, dtype=torch.float32),
                torch.tensor(1.0, dtype=torch.float32),
            ),
        ).view(max_cells, cs)
        weights = (raw_w / raw_w.sum(dim=-1, keepdim=True).clamp(min=1e-8)).view(seq_len)

        content_cell_mask = torch.zeros(max_cells, dtype=torch.bool)
        if n_content_cells > 0:
            content_cell_mask[:n_content_cells] = True

        return {
            "input_ids": input_ids,
            "targets": targets,
            "weights": weights,
            "content_cell_mask": content_cell_mask,
            "n_content_cells": torch.tensor(n_content_cells, dtype=torch.long),
        }

    @staticmethod
    def detokenize_non_pad(
        token_ids: Union[List[int], torch.Tensor],
        tokenizer: Any,
        pad_token_id: int = 10,
        skip_special_tokens: bool = True,
    ) -> str:
        """Сборка текста из 4N слотов с удалением закрывающих SLOT_PAD (pad_token_id)."""
        raw_ids = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else list(token_ids)
        clean_ids = [int(t) for t in raw_ids if int(t) != pad_token_id]
        if not clean_ids or tokenizer is None:
            return ""
        try:
            return tokenizer.decode(clean_ids, skip_special_tokens=skip_special_tokens)
        except TypeError:
            return tokenizer.decode(clean_ids)


class RMSNorm(nn.Module):
    """RMSNorm с расчетом в float32 и защитой от нечисловых переполнений."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = torch.nan_to_num(x.float(), nan=0.0, posinf=65000.0, neginf=-65000.0)
        var = torch.mean(x_f ** 2, dim=-1, keepdim=True)
        normed = x_f * torch.rsqrt(var + self.eps)
        return (normed * self.weight.float()).to(x.dtype)


class SphereLinear(nn.Module):
    """Линейный слой со сферической параметризацией строк для OrthMuon (ADR 131): W = diag(row_scale) @ q_weight."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        w_init = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(w_init, a=math.sqrt(5))
        row_norms = w_init.norm(p=2, dim=-1).clamp(min=1e-7)

        self.q_weight = nn.Parameter(w_init / row_norms.unsqueeze(-1))
        self.row_scale = nn.Parameter(row_norms)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def merged_weight(self) -> torch.Tensor:
        """Собирает эффективную плотную матрицу весов W = diag(row_scale) @ q_weight."""
        return self.q_weight * self.row_scale.unsqueeze(-1).to(self.q_weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.merged_weight(), self.bias)


class RotaryEmbedding128(nn.Module):
    """RoPE для ModernBERT attention (head_dim=128)."""

    def __init__(self, dim: int = 128, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len].to(x.dtype), self.sin_cached[:seq_len].to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class ModernBERTLayer(nn.Module):
    """Слой трансформера ModernBERT с Pre-RMSNorm, per-head QK-norm, RoPE и SwiGLU на SphereLinear."""

    def __init__(self, hidden_size: int = 1024, intermediate_size: int = 2048, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.attn_norm = RMSNorm(hidden_size)
        self.q_proj = SphereLinear(hidden_size, hidden_size, bias=False)
        self.k_proj = SphereLinear(hidden_size, hidden_size, bias=False)
        self.v_proj = SphereLinear(hidden_size, hidden_size, bias=False)
        self.o_proj = SphereLinear(hidden_size, hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.mlp_norm = RMSNorm(hidden_size)
        self.gate_proj = SphereLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = SphereLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = SphereLinear(intermediate_size, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        normed_x = self.attn_norm(x)
        q = self.q_norm(self.q_proj(normed_x).view(B, T, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(normed_x).view(B, T, self.num_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(normed_x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.o_proj(attn_out)

        normed_mlp = self.mlp_norm(x)
        gate = F.silu(self.gate_proj(normed_mlp).float())
        up = self.up_proj(normed_mlp).float()
        gate_up = (gate * up).clamp(-65000.0, 65000.0).to(x.dtype)
        x = x + self.down_proj(gate_up)
        return x


__all__ = [
    "LorentzianEigh",
    "safe_lorentzian_eigh",
    "apply_rope_16d",
    "clamp_to_unit_ball",
    "canonicalize_cell_basis",
    "WordAwareElasticChunker",
    "RMSNorm",
    "SphereLinear",
    "RotaryEmbedding128",
    "rotate_half",
    "apply_rotary_pos_emb",
    "ModernBERTLayer",
]
