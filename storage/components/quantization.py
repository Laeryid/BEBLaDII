"""
Квантизация и автоматический выбор точности для Системы 1 Plan C (Encoder + Decoder).

Принципы (ADR 125, ADR 126, ADR 131):
- Все веса на диске хранятся в 100% переносимом формате (int8 + float32 scales, либо fp16/bf16 baked).
- Защищённые FP32-островки НИКОГДА не квантизируются:
  * Encoder: BlockDiagonalVAEHead (96|16|16), RMSNorm, GrassmannianSpectralPooler (eigh), canonicalize.
  * Decoder: SetCrossAttentionUnpacker.gram_mlp (ADR 126), NoiseRadarHead MLP, noise_cond_proj,
    RMSNorm, canonicalize, а также финальные логиты / logsumexp словаря (ADR 125).
- Одна и та же per-row INT8 матрица embeddings (131,072 x 1024, ~134 MB) обслуживает сразу:
  1) encoder.embeddings(input_ids),
  2) decoder.lm_head (chunked FP32 logits & Online Chunked Exact CE),
  3) NoiseRadarHead.compute_dict_stats (через unit_scale = row_scale * inv_norms, без 256 MB дубликата).
- На CPU (PyTorch oneDNN) Int8RowwiseLinear и Int8RowwiseEmbedding (NoiseRadar) лениво используют
  torch.ops.quantized.linear_dynamic под no_grad (3.8x - 6.5x быстрее FP32) с прозрачным fallback.
- На GPU (CUDA) и TPU (XLA) используются нативные Tensor Cores / MXU с сохранением 100% static shapes.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .plan_c_layers import RMSNorm, SphereLinear
except ImportError:
    from plan_c_layers import RMSNorm, SphereLinear

QUANT_FORMAT_VERSION = "planc_sys1_quant_v1"

# Соответствие пресетов именам файлов в storage/components/{encoder,decoder}/
ENCODER_PRESET_FILES: Dict[str, str] = {
    "fp32": "weights.pt",
    "half": "weights_fp16.pt",
    "emb_int8_half": "weights_emb_int8_fp16.pt",
    "int8": "weights_int8.pt",
}

DECODER_PRESET_FILES: Dict[str, str] = {
    "fp32": "weights.pt",
    "half": "weights_fp16.pt",
    "int8": "weights_int8.pt",
}


def _onednn_int8_available() -> bool:
    """Проверяет доступность быстрого CPU oneDNN dynamic int8 ядра в текущей сборке PyTorch."""
    try:
        engines = getattr(torch.backends.quantized, "supported_engines", [])
        if not engines or ("onednn" not in engines and "fbgemm" not in engines and "x86" not in engines):
            return False
        return hasattr(torch.ops, "quantized") and hasattr(torch.ops.quantized, "linear_prepack") and hasattr(
            torch.ops.quantized, "linear_dynamic"
        )
    except Exception:
        return False


def _prepack_int8_linear_cpu(
    q_weight: torch.Tensor,
    row_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> Optional[Any]:
    """Упаковывает per-row int8 матрицу в oneDNN/FBGEMM packed params для быстрого CPU linear_dynamic."""
    if not _onednn_int8_available():
        return None
    try:
        qw = q_weight.detach().to(device="cpu", dtype=torch.int8).contiguous()
        sc = row_scale.detach().to(device="cpu", dtype=torch.float64).clamp(min=1e-12).contiguous()
        zp = torch.zeros(qw.shape[0], dtype=torch.int64, device="cpu")
        qt = torch._make_per_channel_quantized_tensor(qw, sc, zp, axis=0)
        b_cpu = bias.detach().to(device="cpu", dtype=torch.float32).contiguous() if bias is not None else None
        return torch.ops.quantized.linear_prepack(qt, b_cpu)
    except Exception:
        return None


def quantize_rowwise_int8(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Симметричная per-row INT8 квантизация 2D матрицы весов (out_features, in_features)."""
    w_f = weight.detach().float()
    row_scale = w_f.abs().amax(dim=-1).clamp(min=1e-8) / 127.0
    q_w = torch.round(w_f / row_scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8).contiguous()
    return q_w, row_scale.to(torch.float32).contiguous()


def is_quantized_checkpoint(obj: Any) -> bool:
    """Возвращает True, если загруженный объект является квантизованным/запечённым пакетом Системы 1."""
    return isinstance(obj, dict) and obj.get("quant_format") == QUANT_FORMAT_VERSION


# ---------------------------------------------------------------------------
# Модули: BakedLinear, Int8RowwiseLinear, Int8RowwiseEmbedding
# ---------------------------------------------------------------------------


class BakedLinear(nn.Module):
    """
    Замороженный линейный слой с однократно слитой матрицей W = diag(row_scale) @ q_weight.
    Убирает повторное умножение q_weight * row_scale на каждом шаге и поддерживает нативный FP16/BF16 GEMM.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_module(cls, mod: nn.Module, dtype: torch.dtype = torch.float32) -> "BakedLinear":
        if isinstance(mod, SphereLinear):
            with torch.no_grad():
                w = (mod.q_weight.detach().float() * mod.row_scale.detach().float().unsqueeze(-1)).to(dtype=dtype)
                b = mod.bias.detach().to(dtype=dtype) if mod.bias is not None else None
            out = cls(mod.in_features, mod.out_features, bias=(b is not None), dtype=dtype)
        elif isinstance(mod, (nn.Linear, BakedLinear)):
            with torch.no_grad():
                w = mod.weight.detach().to(dtype=dtype)
                b = mod.bias.detach().to(dtype=dtype) if mod.bias is not None else None
            out = cls(mod.in_features, mod.out_features, bias=(b is not None), dtype=dtype)
        else:
            raise TypeError(f"BakedLinear.from_module не поддерживает {type(mod).__name__}")
        with torch.no_grad():
            out.weight.copy_(w.contiguous())
            if b is not None and out.bias is not None:
                out.bias.copy_(b.contiguous())
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        b = self.bias
        if torch.is_autocast_enabled(x.device.type):
            return F.linear(x, w, b)
        if x.dtype == w.dtype:
            return F.linear(x, w, b)
        # На CPU избегаем медленного/неподдерживаемого FP16 GEMM; на GPU/TPU считаем в dtype весов (Tensor Cores)
        if x.device.type == "cpu" and w.dtype == torch.float16:
            return F.linear(x.float(), w.float(), b.float() if b is not None else None).to(x.dtype)
        return F.linear(x.to(w.dtype), w, b).to(x.dtype)


class Int8RowwiseLinear(nn.Module):
    """
    Per-row симметричный INT8 линейный слой (идеален для SphereLinear, где строки q_weight единичной нормы).
    - На CPU под no_grad использует быстрый oneDNN linear_dynamic (3.5x-4x быстрее FP32).
    - На GPU/TPU или под grad использует поблочную деквантизацию в compute_dtype (100% XLA-совместимо).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        use_cpu_onednn: bool = True,
        compute_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.use_cpu_onednn = bool(use_cpu_onednn)
        self.compute_dtype = compute_dtype
        self._cpu_packed: Optional[Any] = None
        self._cpu_pack_attempted: bool = False

        self.register_buffer("q_weight", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("row_scale", torch.ones(out_features, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_buffer("bias", None)

    @classmethod
    def from_module(
        cls,
        mod: nn.Module,
        use_cpu_onednn: bool = True,
        compute_dtype: Optional[torch.dtype] = None,
    ) -> "Int8RowwiseLinear":
        if isinstance(mod, SphereLinear):
            with torch.no_grad():
                w = mod.q_weight.detach().float() * mod.row_scale.detach().float().unsqueeze(-1)
                b = mod.bias.detach().float() if mod.bias is not None else None
            in_f, out_f = mod.in_features, mod.out_features
        elif isinstance(mod, (nn.Linear, BakedLinear)):
            with torch.no_grad():
                w = mod.weight.detach().float()
                b = mod.bias.detach().float() if mod.bias is not None else None
            in_f, out_f = mod.in_features, mod.out_features
        elif isinstance(mod, Int8RowwiseLinear):
            out = cls(
                mod.in_features,
                mod.out_features,
                bias=(mod.bias is not None),
                use_cpu_onednn=use_cpu_onednn,
                compute_dtype=compute_dtype,
            )
            out.q_weight.copy_(mod.q_weight)
            out.row_scale.copy_(mod.row_scale)
            if mod.bias is not None and out.bias is not None:
                out.bias.copy_(mod.bias)
            return out
        else:
            raise TypeError(f"Int8RowwiseLinear.from_module не поддерживает {type(mod).__name__}")

        q_w, scale = quantize_rowwise_int8(w)
        layer = cls(
            in_f,
            out_f,
            bias=(b is not None),
            use_cpu_onednn=use_cpu_onednn,
            compute_dtype=compute_dtype,
        )
        layer.q_weight.copy_(q_w)
        layer.row_scale.copy_(scale)
        if b is not None and layer.bias is not None:
            layer.bias.copy_(b)
        return layer

    def _apply(self, fn):
        # Сохраняем q_weight в int8 и row_scale/bias строго в float32 даже при вызове model.to(dtype=bf16/fp16)
        self._cpu_packed = None
        self._cpu_pack_attempted = False
        for key, buf in list(self._buffers.items()):
            if buf is None:
                continue
            out = fn(buf)
            if key == "q_weight" and out.dtype != torch.int8:
                out = out.to(dtype=torch.int8)
            elif key in ("row_scale", "bias") and out.dtype != torch.float32:
                out = out.to(dtype=torch.float32)
            self._buffers[key] = out
        return self

    def _ensure_cpu_packed(self) -> Optional[Any]:
        if self._cpu_packed is not None:
            return self._cpu_packed
        if self._cpu_pack_attempted:
            return None
        self._cpu_pack_attempted = True
        self._cpu_packed = _prepack_int8_linear_cpu(self.q_weight, self.row_scale, self.bias)
        return self._cpu_packed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == "cpu" and self.use_cpu_onednn and (not torch.is_grad_enabled()):
            packed = self._ensure_cpu_packed()
            if packed is not None:
                out = torch.ops.quantized.linear_dynamic(x.float().contiguous(), packed, reduce_range=False)
                return out.to(dtype=x.dtype)

        if self.compute_dtype is not None and not (x.device.type == "cpu" and self.compute_dtype == torch.float16):
            w_dtype = self.compute_dtype
        elif x.is_floating_point():
            w_dtype = torch.float32 if (x.device.type == "cpu" and x.dtype == torch.float16) else x.dtype
        else:
            w_dtype = torch.float32

        w = self.q_weight.to(dtype=w_dtype) * self.row_scale.to(dtype=w_dtype).unsqueeze(-1)
        b = self.bias.to(dtype=w_dtype) if self.bias is not None else None
        out = F.linear(x.to(dtype=w_dtype), w, b)
        return out.to(dtype=x.dtype)


class Int8RowwiseEmbedding(nn.Module):
    """
    Per-row симметричный INT8 словарь (131,072 x 1024 -> ~134 MB вместо 512 MB FP32).
    Единая матрица q_weight обслуживает:
      1) encoder.embeddings(input_ids),
      2) decoder.get_tied_head_logits / Online Chunked Exact CE (строго FP32 по ADR 125),
      3) NoiseRadarHead.compute_dict_stats (через unit_scale = row_scale * inv_norms, без 256 MB копии).
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        out_dtype: torch.dtype = torch.float32,
        use_cpu_onednn: bool = True,
    ):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.out_dtype = out_dtype
        self.use_cpu_onednn = bool(use_cpu_onednn)
        self._cpu_radar_packed: Optional[List[Tuple[int, int, Any]]] = None
        self._cpu_radar_chunk_size: int = 0
        self._cpu_radar_pack_attempted: bool = False

        self.register_buffer("q_weight", torch.zeros(num_embeddings, embedding_dim, dtype=torch.int8))
        self.register_buffer("row_scale", torch.ones(num_embeddings, dtype=torch.float32))
        self.register_buffer("inv_norms", torch.ones(num_embeddings, dtype=torch.float32))
        self.register_buffer("unit_scale", torch.ones(num_embeddings, dtype=torch.float32))

    @classmethod
    def from_weight(
        cls,
        weight: torch.Tensor,
        out_dtype: torch.dtype = torch.float32,
        use_cpu_onednn: bool = True,
    ) -> "Int8RowwiseEmbedding":
        w_f = weight.detach().float()
        v_size, d_emb = int(w_f.shape[0]), int(w_f.shape[1])
        q_w, row_scale = quantize_rowwise_int8(w_f)
        inv_norms = 1.0 / w_f.norm(p=2, dim=-1).clamp(min=1e-6)
        unit_scale = (row_scale * inv_norms).to(torch.float32).contiguous()

        emb = cls(v_size, d_emb, out_dtype=out_dtype, use_cpu_onednn=use_cpu_onednn)
        emb.q_weight.copy_(q_w)
        emb.row_scale.copy_(row_scale)
        emb.inv_norms.copy_(inv_norms)
        emb.unit_scale.copy_(unit_scale)
        return emb

    @property
    def device(self) -> torch.device:
        return self.q_weight.device

    @property
    def shape(self) -> torch.Size:
        return self.q_weight.shape

    def _apply(self, fn):
        self._cpu_radar_packed = None
        self._cpu_radar_chunk_size = 0
        self._cpu_radar_pack_attempted = False
        for key, buf in list(self._buffers.items()):
            if buf is None:
                continue
            out = fn(buf)
            if key == "q_weight" and out.dtype != torch.int8:
                out = out.to(dtype=torch.int8)
            elif key in ("row_scale", "inv_norms", "unit_scale") and out.dtype != torch.float32:
                out = out.to(dtype=torch.float32)
            self._buffers[key] = out
        return self

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        q = F.embedding(input_ids, self.q_weight)
        s = F.embedding(input_ids, self.row_scale.unsqueeze(-1))
        out = q.float() * s
        if self.out_dtype == torch.float16 and out.device.type == "cpu":
            return out
        return out.to(dtype=self.out_dtype)

    def dequantize_slice(
        self,
        start: int,
        end: int,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        q_c = self.q_weight[start:end]
        s_c = self.row_scale[start:end]
        if device is not None and q_c.device != device:
            q_c = q_c.to(device=device)
            s_c = s_c.to(device=device)
        return q_c.to(dtype=dtype) * s_c.to(dtype=dtype).unsqueeze(-1)

    def dequantize_rows(
        self,
        indices: torch.Tensor,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        idx = indices.to(device=self.q_weight.device)
        q_r = self.q_weight.index_select(0, idx)
        s_r = self.row_scale.index_select(0, idx)
        if device is not None and q_r.device != device:
            q_r = q_r.to(device=device)
            s_r = s_r.to(device=device)
        return q_r.to(dtype=dtype) * s_r.to(dtype=dtype).unsqueeze(-1)

    def linear_chunked_fp32(
        self,
        hidden_states: torch.Tensor,
        chunk_size: int = 32768,
    ) -> torch.Tensor:
        """Вычисляет полные логиты словаря (..., V) по чанкам строго в float32 (ADR 125)."""
        device = hidden_states.device
        with torch.autocast(device_type=device.type, enabled=False):
            h_f32 = hidden_states.float()
            chunks: List[torch.Tensor] = []
            c_sz = max(1, int(chunk_size))
            for start in range(0, self.num_embeddings, c_sz):
                end = min(start + c_sz, self.num_embeddings)
                w_c = self.dequantize_slice(start, end, dtype=torch.float32, device=device)
                chunks.append(F.linear(h_f32, w_c))
            return torch.cat(chunks, dim=-1)

    def _ensure_cpu_radar_packed(self, chunk_size: int) -> Optional[List[Tuple[int, int, Any]]]:
        if self._cpu_radar_packed is not None and self._cpu_radar_chunk_size == chunk_size:
            return self._cpu_radar_packed
        if self._cpu_radar_pack_attempted and self._cpu_radar_chunk_size == chunk_size:
            return None
        self._cpu_radar_chunk_size = chunk_size
        self._cpu_radar_pack_attempted = True
        if not _onednn_int8_available():
            return None
        packed_chunks: List[Tuple[int, int, Any]] = []
        for start in range(0, self.num_embeddings, chunk_size):
            end = min(start + chunk_size, self.num_embeddings)
            p = _prepack_int8_linear_cpu(self.q_weight[start:end], self.unit_scale[start:end], None)
            if p is None:
                self._cpu_radar_packed = None
                return None
            packed_chunks.append((start, end, p))
        self._cpu_radar_packed = packed_chunks
        return self._cpu_radar_packed

    @torch.no_grad()
    def compute_radar_top2_cos(
        self,
        h_unit_f32: torch.Tensor,
        chunk_size: int = 32768,
        gemm_dtype: torch.dtype = torch.bfloat16,
        use_cpu_onednn: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Вычисляет (cos_max, cos_2nd) формы (M, 1) в float32 по единичным запросам h_unit_f32 (M, D).
        - На CPU с oneDNN: ~6x быстрее FP32 без дубликата словаря.
        - На GPU/TPU: 100% static shapes, GEMM в gemm_dtype (без переполнения FP16, т.к. ||h||=1, |q|<=127).
        """
        device = h_unit_f32.device
        m_rows = h_unit_f32.shape[0]
        c_sz = max(1, int(chunk_size))
        allow_onednn = self.use_cpu_onednn if use_cpu_onednn is None else bool(use_cpu_onednn)

        top2_vals = torch.full((m_rows, 2), -10.0, device=device, dtype=torch.float32)

        if device.type == "cpu" and allow_onednn and self.q_weight.device.type == "cpu":
            packed_chunks = self._ensure_cpu_radar_packed(c_sz)
            if packed_chunks is not None:
                h_cont = h_unit_f32.float().contiguous()
                for start, end, p_chunk in packed_chunks:
                    cos_chunk = torch.ops.quantized.linear_dynamic(h_cont, p_chunk, reduce_range=False)
                    k_take = min(2, end - start)
                    chunk_top = torch.topk(cos_chunk, k=k_take, dim=-1).values
                    merged = torch.cat([top2_vals, chunk_top], dim=-1)
                    top2_vals = torch.topk(merged, k=2, dim=-1).values
                return top2_vals[:, 0:1], top2_vals[:, 1:2]

        eff_dtype = gemm_dtype
        if device.type == "cpu" and eff_dtype == torch.float16:
            eff_dtype = torch.float32

        with torch.autocast(device_type=device.type, enabled=False):
            h_g = h_unit_f32.to(dtype=eff_dtype)
            for start in range(0, self.num_embeddings, c_sz):
                end = min(start + c_sz, self.num_embeddings)
                q_c = self.q_weight[start:end]
                us_c = self.unit_scale[start:end]
                if q_c.device != device:
                    q_c = q_c.to(device=device)
                    us_c = us_c.to(device=device)
                cos_chunk = F.linear(h_g, q_c.to(dtype=eff_dtype)).float() * us_c.unsqueeze(0)
                k_take = min(2, end - start)
                chunk_top = torch.topk(cos_chunk, k=k_take, dim=-1).values
                merged = torch.cat([top2_vals, chunk_top], dim=-1)
                top2_vals = torch.topk(merged, k=2, dim=-1).values
        return top2_vals[:, 0:1], top2_vals[:, 1:2]


# ---------------------------------------------------------------------------
# Политика автоматического переключения по режиму (train/inference) и ускорителю
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class System1QuantPolicy:
    mode: str  # "train" | "inference"
    device_type: str  # "cpu" | "cuda" | "xla"
    quant_mode: str  # "fp32" | "half" | "emb_int8" | "int8"
    encoder_preset: str  # "fp32" | "half" | "emb_int8_half" | "int8"
    decoder_preset: str  # "fp32" | "half" | "int8"
    embedding_int8: bool
    encoder_layers_int8: bool
    decoder_layers_int8: bool
    bake_sphere_linear: bool
    layer_dtype: torch.dtype
    radar_dtype: torch.dtype
    use_cpu_onednn_int8: bool
    encoder_weights_filename: str
    decoder_weights_filename: str
    reason: str


def _detect_device_type_and_half_dtype(
    device: Union[str, torch.device],
    amp_dtype: Optional[torch.dtype] = None,
) -> Tuple[str, torch.dtype]:
    dev_str = str(device).lower()
    if "xla" in dev_str or "tpu" in dev_str:
        return "xla", (amp_dtype if amp_dtype in (torch.bfloat16, torch.float16, torch.float32) else torch.bfloat16)

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type == "cuda":
        if amp_dtype in (torch.bfloat16, torch.float16, torch.float32):
            return "cuda", amp_dtype
        try:
            idx = dev.index if dev.index is not None else torch.cuda.current_device()
            major, _ = torch.cuda.get_device_capability(idx)
            # ADR 126: BF16 строго на Ampere+ (CC >= 8.0), на Turing (T4, CC 7.5) — нативный FP16
            return "cuda", (torch.bfloat16 if major >= 8 else torch.float16)
        except Exception:
            return "cuda", torch.float16

    # CPU: избегаем FP16 (нет аппаратной поддержки GEMM), используем BF16 только если явно запрошен amp_dtype=bf16
    if amp_dtype == torch.bfloat16:
        return "cpu", torch.bfloat16
    return "cpu", torch.float32


def resolve_system1_quant_policy(
    mode: str = "train",
    device: Union[str, torch.device] = "cpu",
    quant_mode: str = "auto",
    amp_dtype: Optional[torch.dtype] = None,
) -> System1QuantPolicy:
    """
    Автоматически выбирает оптимальный режим точности/квантизации Энкодера и Декодера
    в зависимости от:
      - mode: 'train' (есть backward через Декодер для L_ce, таргеты V_0 из Энкодера)
              или 'inference' (только forward, критичен размер в памяти и скорость NoiseRadar);
      - device: 'cpu', 'cuda' (Turing T4 -> FP16, Ampere+ L4/A100 -> BF16), 'xla'/TPU (BF16);
      - quant_mode: 'auto' | 'fp32' ('none') | 'half' ('fp16'/'bf16') | 'emb_int8' | 'int8'.
    """
    m = mode.lower().strip()
    if m not in ("train", "inference", "eval"):
        raise ValueError(f"Неизвестный mode={mode!r}; ожидалось 'train' или 'inference'.")
    norm_mode = "train" if m == "train" else "inference"

    qm = quant_mode.lower().strip()
    aliases = {"none": "fp32", "off": "fp32", "fp16": "half", "bf16": "half", "emb_int8_half": "emb_int8"}
    qm = aliases.get(qm, qm)
    if qm not in ("auto", "fp32", "half", "emb_int8", "int8"):
        raise ValueError(
            f"Неизвестный quant_mode={quant_mode!r}; допустимо: 'auto', 'fp32', 'half', 'emb_int8', 'int8'."
        )

    dev_type, nat_dtype = _detect_device_type_and_half_dtype(device, amp_dtype=amp_dtype)
    if quant_mode.lower().strip() == "fp16" and dev_type != "cpu":
        nat_dtype = torch.float16
    elif quant_mode.lower().strip() == "bf16":
        nat_dtype = torch.bfloat16

    cpu_onednn = (dev_type == "cpu") and _onednn_int8_available()

    if qm == "auto":
        if norm_mode == "train":
            if dev_type == "xla":
                # TPU: большой HBM, максимальный throughput на нативном BF16 MXU
                qm = "half"
                reason = "train+xla: нативный BF16 на MXU TPU (без dequant overhead, static shapes)"
            elif dev_type == "cuda":
                # GPU: словарь per-row INT8 (-377 MB VRAM, 4x меньше трафика T4), слои в FP16/BF16 для быстрого CE backward
                qm = "emb_int8"
                reason = f"train+cuda({nat_dtype}): словарь per-row INT8 + запечённые слои {nat_dtype} для быстрого L_ce backward"
            else:
                # CPU train: словарь per-row INT8 (-377 MB RAM, 6.5x NoiseRadar через oneDNN), слои FP32/BF16 для backward
                qm = "emb_int8"
                reason = f"train+cpu: словарь per-row INT8 (oneDNN={cpu_onednn}) + запечённые слои {nat_dtype} для backward"
        else:
            if dev_type == "xla":
                qm = "half"
                reason = "inference+xla: нативный BF16 на MXU TPU"
            elif dev_type == "cuda":
                qm = "int8"
                reason = f"inference+cuda({nat_dtype}): полный per-row INT8 Системы 1 (~217 MB VRAM, compute={nat_dtype})"
            else:
                qm = "int8"
                reason = f"inference+cpu: полный per-row INT8 Системы 1 (~217 MB RAM, oneDNN={cpu_onednn}, 3.8x-6.5x быстрее)"
    else:
        reason = f"ручной выбор quant_mode={qm} на {norm_mode}+{dev_type}"

    if qm == "fp32":
        return System1QuantPolicy(
            mode=norm_mode,
            device_type=dev_type,
            quant_mode="fp32",
            encoder_preset="fp32",
            decoder_preset="fp32",
            embedding_int8=False,
            encoder_layers_int8=False,
            decoder_layers_int8=False,
            bake_sphere_linear=False,
            layer_dtype=torch.float32,
            radar_dtype=torch.float32,
            use_cpu_onednn_int8=False,
            encoder_weights_filename=ENCODER_PRESET_FILES["fp32"],
            decoder_weights_filename=DECODER_PRESET_FILES["fp32"],
            reason=reason,
        )

    if qm == "half":
        layer_dt = nat_dtype if dev_type != "cpu" else (torch.bfloat16 if amp_dtype == torch.bfloat16 else torch.float32)
        return System1QuantPolicy(
            mode=norm_mode,
            device_type=dev_type,
            quant_mode="half",
            encoder_preset="half",
            decoder_preset="half",
            embedding_int8=False,
            encoder_layers_int8=False,
            decoder_layers_int8=False,
            bake_sphere_linear=True,
            layer_dtype=layer_dt,
            radar_dtype=layer_dt,
            use_cpu_onednn_int8=False,
            encoder_weights_filename=ENCODER_PRESET_FILES["half"],
            decoder_weights_filename=DECODER_PRESET_FILES["half"],
            reason=reason,
        )

    if qm == "emb_int8":
        layer_dt = nat_dtype if dev_type != "cpu" else (torch.bfloat16 if amp_dtype == torch.bfloat16 else torch.float32)
        radar_dt = nat_dtype if dev_type != "cpu" else torch.float32
        return System1QuantPolicy(
            mode=norm_mode,
            device_type=dev_type,
            quant_mode="emb_int8",
            encoder_preset="emb_int8_half",
            decoder_preset="half",
            embedding_int8=True,
            encoder_layers_int8=False,
            decoder_layers_int8=False,
            bake_sphere_linear=True,
            layer_dtype=layer_dt,
            radar_dtype=radar_dt,
            use_cpu_onednn_int8=cpu_onednn,
            encoder_weights_filename=ENCODER_PRESET_FILES["emb_int8_half"],
            decoder_weights_filename=DECODER_PRESET_FILES["half"],
            reason=reason,
        )

    # qm == "int8"
    layer_dt = nat_dtype if dev_type != "cpu" else torch.float32
    radar_dt = nat_dtype if dev_type != "cpu" else torch.float32
    return System1QuantPolicy(
        mode=norm_mode,
        device_type=dev_type,
        quant_mode="int8",
        encoder_preset="int8",
        decoder_preset="int8",
        embedding_int8=True,
        encoder_layers_int8=True,
        decoder_layers_int8=True,
        bake_sphere_linear=True,
        layer_dtype=layer_dt,
        radar_dtype=radar_dt,
        use_cpu_onednn_int8=cpu_onednn,
        encoder_weights_filename=ENCODER_PRESET_FILES["int8"],
        decoder_weights_filename=DECODER_PRESET_FILES["int8"],
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Преобразование моделей in-place, сохранение и загрузка компактных весов
# ---------------------------------------------------------------------------


def _replace_sphere_linear_in_module(
    root: nn.Module,
    mode: str,  # "bake" | "int8"
    layer_dtype: torch.dtype = torch.float32,
    use_cpu_onednn: bool = True,
) -> int:
    """Заменяет все SphereLinear внутри root на BakedLinear или Int8RowwiseLinear."""
    replaced = 0
    for name, child in list(root.named_children()):
        if isinstance(child, SphereLinear):
            if mode == "int8":
                new_mod: nn.Module = Int8RowwiseLinear.from_module(
                    child, use_cpu_onednn=use_cpu_onednn, compute_dtype=layer_dtype
                )
            else:
                new_mod = BakedLinear.from_module(child, dtype=layer_dtype)
            setattr(root, name, new_mod)
            replaced += 1
        elif isinstance(child, BakedLinear) and mode == "int8":
            setattr(
                root,
                name,
                Int8RowwiseLinear.from_module(child, use_cpu_onednn=use_cpu_onednn, compute_dtype=layer_dtype),
            )
            replaced += 1
        elif isinstance(child, BakedLinear) and mode == "bake":
            with torch.no_grad():
                child.weight.data = child.weight.data.to(dtype=layer_dtype)
                if child.bias is not None:
                    child.bias.data = child.bias.data.to(dtype=layer_dtype)
        else:
            replaced += _replace_sphere_linear_in_module(
                child, mode=mode, layer_dtype=layer_dtype, use_cpu_onednn=use_cpu_onednn
            )
    return replaced


def _install_quantized_skeleton_encoder(
    encoder: nn.Module,
    preset: str,
    layer_dtype: torch.dtype,
    use_cpu_onednn: bool,
) -> None:
    """Готовит структуру модулей Энкодера перед load_state_dict для заданного пресета."""
    if preset in ("emb_int8_half", "int8"):
        v_size = int(getattr(encoder, "vocab_size", 131072))
        d_emb = int(getattr(encoder, "hidden_size", 1024))
        encoder.embeddings = Int8RowwiseEmbedding(
            num_embeddings=v_size,
            embedding_dim=d_emb,
            out_dtype=layer_dtype,
            use_cpu_onednn=use_cpu_onednn,
        )
    elif preset == "half" and isinstance(encoder.embeddings, nn.Embedding):
        encoder.embeddings.to(dtype=layer_dtype)

    if preset in ("half", "emb_int8_half"):
        _replace_sphere_linear_in_module(encoder.lexical_mlp, mode="bake", layer_dtype=layer_dtype)
        _replace_sphere_linear_in_module(encoder.context_encoder, mode="bake", layer_dtype=layer_dtype)
    elif preset == "int8":
        _replace_sphere_linear_in_module(
            encoder.lexical_mlp, mode="int8", layer_dtype=layer_dtype, use_cpu_onednn=use_cpu_onednn
        )
        _replace_sphere_linear_in_module(
            encoder.context_encoder, mode="int8", layer_dtype=layer_dtype, use_cpu_onednn=use_cpu_onednn
        )


def _install_quantized_skeleton_decoder(
    decoder: nn.Module,
    preset: str,
    layer_dtype: torch.dtype,
    use_cpu_onednn: bool,
) -> None:
    """Готовит структуру модулей Декодера перед load_state_dict для заданного пресета."""
    if preset == "half":
        _replace_sphere_linear_in_module(decoder.dec_context_layer, mode="bake", layer_dtype=layer_dtype)
        _replace_sphere_linear_in_module(decoder.dec_smooth_layers, mode="bake", layer_dtype=layer_dtype)
    elif preset == "int8":
        _replace_sphere_linear_in_module(
            decoder.dec_context_layer, mode="int8", layer_dtype=layer_dtype, use_cpu_onednn=use_cpu_onednn
        )
        _replace_sphere_linear_in_module(
            decoder.dec_smooth_layers, mode="int8", layer_dtype=layer_dtype, use_cpu_onednn=use_cpu_onednn
        )


def preserve_sensitive_fp32(model: nn.Module) -> None:
    """
    Гарантирует, что все чувствительные островки Plan C (ADR 125/126) остаются строго в float32:
    RMSNorm, BlockDiagonalVAEHead, syntax_proj, unpacker (gram_mlp, query_slots, out_proj),
    noise_radar, noise_cond_proj.
    """
    sensitive_attr_names = (
        "vae_head",
        "unpacker",
        "noise_radar",
        "noise_cond_proj",
    )
    for attr in sensitive_attr_names:
        sub = getattr(model, attr, None)
        if isinstance(sub, nn.Module):
            sub.to(dtype=torch.float32)

    ctx_enc = getattr(model, "context_encoder", None)
    if isinstance(ctx_enc, nn.Module):
        sp = getattr(ctx_enc, "syntax_proj", None)
        if isinstance(sp, nn.Module):
            sp.to(dtype=torch.float32)

    for m in model.modules():
        if isinstance(m, RMSNorm):
            m.to(dtype=torch.float32)


def quantize_encoder_in_place(
    encoder: nn.Module,
    preset: str,
    layer_dtype: torch.dtype = torch.float16,
    use_cpu_onednn: bool = True,
) -> nn.Module:
    """Конвертирует загруженный FP32 GrassmannianEncoder в заданный пресет ('fp32'|'half'|'emb_int8_half'|'int8')."""
    if preset == "fp32":
        return encoder
    if preset not in ENCODER_PRESET_FILES:
        raise ValueError(f"Неизвестный encoder preset={preset!r}")

    with torch.no_grad():
        if preset in ("emb_int8_half", "int8"):
            if isinstance(encoder.embeddings, nn.Embedding):
                encoder.embeddings = Int8RowwiseEmbedding.from_weight(
                    encoder.embeddings.weight,
                    out_dtype=layer_dtype,
                    use_cpu_onednn=use_cpu_onednn,
                )
            elif isinstance(encoder.embeddings, Int8RowwiseEmbedding):
                encoder.embeddings.out_dtype = layer_dtype
                encoder.embeddings.use_cpu_onednn = bool(use_cpu_onednn)
        elif preset == "half":
            if isinstance(encoder.embeddings, nn.Embedding):
                encoder.embeddings.to(dtype=layer_dtype)

        layer_mode = "int8" if preset == "int8" else "bake"
        _replace_sphere_linear_in_module(
            encoder.lexical_mlp, mode=layer_mode, layer_dtype=layer_dtype, use_cpu_onednn=use_cpu_onednn
        )
        _replace_sphere_linear_in_module(
            encoder.context_encoder, mode=layer_mode, layer_dtype=layer_dtype, use_cpu_onednn=use_cpu_onednn
        )
        preserve_sensitive_fp32(encoder)
        for p in encoder.parameters():
            p.requires_grad_(False)
    return encoder


def quantize_decoder_in_place(
    decoder: nn.Module,
    preset: str,
    layer_dtype: torch.dtype = torch.float16,
    use_cpu_onednn: bool = True,
) -> nn.Module:
    """Конвертирует загруженный FP32 GrassmannianDecoder в заданный пресет ('fp32'|'half'|'int8')."""
    if preset == "fp32":
        return decoder
    if preset not in DECODER_PRESET_FILES:
        raise ValueError(f"Неизвестный decoder preset={preset!r}")

    with torch.no_grad():
        layer_mode = "int8" if preset == "int8" else "bake"
        _replace_sphere_linear_in_module(
            decoder.dec_context_layer, mode=layer_mode, layer_dtype=layer_dtype, use_cpu_onednn=use_cpu_onednn
        )
        _replace_sphere_linear_in_module(
            decoder.dec_smooth_layers, mode=layer_mode, layer_dtype=layer_dtype, use_cpu_onednn=use_cpu_onednn
        )
        preserve_sensitive_fp32(decoder)
        for p in decoder.parameters():
            p.requires_grad_(False)
    return decoder


def save_quantized_component(
    model: nn.Module,
    out_path: Union[str, Path],
    component: str,
    preset: str,
    source_file: str = "weights.pt",
) -> Path:
    """Сохраняет квантизованный/запечённый state_dict на диск в переносимом формате."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Исключаем неперсистентные буферы (lm_head_*, rope cache), state_dict() уже их пропускает
    sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    payload = {
        "quant_format": QUANT_FORMAT_VERSION,
        "component": component,
        "preset": preset,
        "source_file": source_file,
        "state_dict": sd,
    }
    torch.save(payload, out)
    return out


def load_quantized_encoder_payload(
    encoder: nn.Module,
    payload: Dict[str, Any],
    layer_dtype: torch.dtype,
    use_cpu_onednn: bool = True,
) -> str:
    """Подготавливает скелет Энкодера под пресет пакета и загружает state_dict."""
    preset = str(payload["preset"])
    sd = payload["state_dict"]
    _install_quantized_skeleton_encoder(encoder, preset=preset, layer_dtype=layer_dtype, use_cpu_onednn=use_cpu_onednn)
    encoder.load_state_dict(sd, strict=True)
    if preset in ("half", "emb_int8_half"):
        if isinstance(encoder.embeddings, nn.Embedding):
            encoder.embeddings.to(dtype=layer_dtype)
        elif isinstance(encoder.embeddings, Int8RowwiseEmbedding):
            encoder.embeddings.out_dtype = layer_dtype
            encoder.embeddings.use_cpu_onednn = bool(use_cpu_onednn)
        _replace_sphere_linear_in_module(encoder.lexical_mlp, mode="bake", layer_dtype=layer_dtype)
        _replace_sphere_linear_in_module(encoder.context_encoder, mode="bake", layer_dtype=layer_dtype)
    elif preset == "int8":
        if isinstance(encoder.embeddings, Int8RowwiseEmbedding):
            encoder.embeddings.out_dtype = layer_dtype
            encoder.embeddings.use_cpu_onednn = bool(use_cpu_onednn)
        for m in encoder.modules():
            if isinstance(m, Int8RowwiseLinear):
                m.compute_dtype = layer_dtype
                m.use_cpu_onednn = bool(use_cpu_onednn)
    preserve_sensitive_fp32(encoder)
    for p in encoder.parameters():
        p.requires_grad_(False)
    return preset


def load_quantized_decoder_payload(
    decoder: nn.Module,
    payload: Dict[str, Any],
    layer_dtype: torch.dtype,
    use_cpu_onednn: bool = True,
) -> str:
    """Подготавливает скелет Декодера под пресет пакета и загружает state_dict."""
    preset = str(payload["preset"])
    sd = payload["state_dict"]
    _install_quantized_skeleton_decoder(decoder, preset=preset, layer_dtype=layer_dtype, use_cpu_onednn=use_cpu_onednn)
    decoder.load_state_dict(sd, strict=True)
    if preset == "half":
        _replace_sphere_linear_in_module(decoder.dec_context_layer, mode="bake", layer_dtype=layer_dtype)
        _replace_sphere_linear_in_module(decoder.dec_smooth_layers, mode="bake", layer_dtype=layer_dtype)
    elif preset == "int8":
        for m in decoder.modules():
            if isinstance(m, Int8RowwiseLinear):
                m.compute_dtype = layer_dtype
                m.use_cpu_onednn = bool(use_cpu_onednn)
    preserve_sensitive_fp32(decoder)
    for p in decoder.parameters():
        p.requires_grad_(False)
    return preset


def resolve_component_weights_path(
    base_dir: Path,
    preset_filename: str,
    explicit_weights_path: Optional[Union[str, Path]] = None,
    fallback_filename: str = "weights.pt",
) -> Tuple[Path, bool]:
    """
    Ищет квантизованный файл весов рядом с explicit_weights_path (или в base_dir).
    Возвращает (path, is_exact_preset_file):
    - Если нужный файл пресета найден (например weights_int8.pt) -> (path, True);
    - Иначе (например на Kaggle подключён датасет только с weights.pt) -> (fallback_path, False),
      после чего загрузчик автоматически выполнит конвертацию на лету.
    """
    if explicit_weights_path is not None:
        exp = Path(explicit_weights_path)
        if exp.name == preset_filename and exp.exists():
            return exp, True
        sibling = exp.parent / preset_filename
        if sibling.exists():
            return sibling, True
        if exp.exists():
            return exp, (exp.name == preset_filename)

    candidate = base_dir / preset_filename
    if candidate.exists():
        return candidate, True
    return base_dir / fallback_filename, False


def estimate_model_memory_mb(model: nn.Module) -> float:
    """Оценивает фактический объём параметров и персистентных буферов модели в МБ (без дублей)."""
    seen = set()
    total_bytes = 0
    for t in list(model.parameters()) + list(model.buffers()):
        if t is None:
            continue
        ptr = t.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        total_bytes += t.numel() * t.element_size()
    return total_bytes / (1024.0 * 1024.0)


__all__ = [
    "QUANT_FORMAT_VERSION",
    "ENCODER_PRESET_FILES",
    "DECODER_PRESET_FILES",
    "System1QuantPolicy",
    "BakedLinear",
    "Int8RowwiseLinear",
    "Int8RowwiseEmbedding",
    "quantize_rowwise_int8",
    "is_quantized_checkpoint",
    "resolve_system1_quant_policy",
    "preserve_sensitive_fp32",
    "quantize_encoder_in_place",
    "quantize_decoder_in_place",
    "save_quantized_component",
    "load_quantized_encoder_payload",
    "load_quantized_decoder_payload",
    "resolve_component_weights_path",
    "estimate_model_memory_mb",
]
