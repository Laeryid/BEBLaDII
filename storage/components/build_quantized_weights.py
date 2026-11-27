"""
Генератор и валидатор квантизованных/запечённых весов Системы 1 Plan C в storage/components/.

Сохраняет переносимые компактные чекпоинты (исходные weights.pt в FP32 НЕ изменяются):
  - storage/components/encoder/weights_fp16.pt            (~323 MB, TPU/GPU half)
  - storage/components/encoder/weights_emb_int8_fp16.pt   (~188 MB, GPU/CPU train)
  - storage/components/encoder/weights_int8.pt            (~162 MB, CPU/GPU inference)
  - storage/components/decoder/weights_fp16.pt            (~106 MB, GPU/TPU train)
  - storage/components/decoder/weights_int8.pt            (~54 MB,  CPU/GPU inference)

Запуск:
  .venv\\Scripts\\python.exe -B "storage\\components\\build_quantized_weights.py"
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict

import torch

COMPONENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = COMPONENTS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from storage.components.decoder import DECODER_DIR, get_decoder  # noqa: E402
from storage.components.encoder import ENCODER_DIR, get_encoder  # noqa: E402
from storage.components.quantization import (  # noqa: E402
    DECODER_PRESET_FILES,
    ENCODER_PRESET_FILES,
    quantize_decoder_in_place,
    quantize_encoder_in_place,
    save_quantized_component,
)


def _file_mb(path: Path) -> float:
    return path.stat().st_size / (1024.0 * 1024.0) if path.exists() else 0.0


def _update_config(cfg_path: Path, quantized_map: Dict[str, str]) -> None:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["quantized_weights"] = quantized_map
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    t0 = time.time()
    enc_orig = ENCODER_DIR / "weights.pt"
    dec_orig = DECODER_DIR / "weights.pt"
    print(f"=== Генерация квантизованных весов Системы 1 ({COMPONENTS_DIR}) ===", flush=True)
    print(f"Исходники FP32: encoder={_file_mb(enc_orig):.1f} MB, decoder={_file_mb(dec_orig):.1f} MB", flush=True)

    # 1. Энкодер: half (fp16), emb_int8_half, int8
    for preset in ("half", "emb_int8_half", "int8"):
        t_p = time.time()
        enc = get_encoder(weights_path=enc_orig, device="cpu", eval_mode=True, quant_mode="fp32")
        quantize_encoder_in_place(enc, preset=preset, layer_dtype=torch.float16, use_cpu_onednn=False)
        out_path = ENCODER_DIR / ENCODER_PRESET_FILES[preset]
        save_quantized_component(enc, out_path=out_path, component="encoder", preset=preset, source_file="weights.pt")
        print(
            f"  [Encoder:{preset:14s}] -> {out_path.name:28s} | {_file_mb(out_path):6.1f} MB "
            f"({_file_mb(enc_orig) / max(_file_mb(out_path), 1e-6):.2f}x) | {time.time() - t_p:.1f}s",
            flush=True,
        )
        del enc

    # 2. Декодер: half (fp16), int8
    for preset in ("half", "int8"):
        t_p = time.time()
        dec = get_decoder(
            weights_path=dec_orig,
            embeddings_weight=torch.zeros(1, 1),  # не грузим 645 MB энкодер ради сохранения весов декодера
            device="cpu",
            eval_mode=True,
            quant_mode="fp32",
        )
        quantize_decoder_in_place(dec, preset=preset, layer_dtype=torch.float16, use_cpu_onednn=False)
        out_path = DECODER_DIR / DECODER_PRESET_FILES[preset]
        save_quantized_component(dec, out_path=out_path, component="decoder", preset=preset, source_file="weights.pt")
        print(
            f"  [Decoder:{preset:14s}] -> {out_path.name:28s} | {_file_mb(out_path):6.1f} MB "
            f"({_file_mb(dec_orig) / max(_file_mb(out_path), 1e-6):.2f}x) | {time.time() - t_p:.1f}s",
            flush=True,
        )
        del dec

    # 3. Обновляем config.json
    _update_config(ENCODER_DIR / "config.json", dict(ENCODER_PRESET_FILES))
    _update_config(DECODER_DIR / "config.json", dict(DECODER_PRESET_FILES))
    print(f"=== Готово за {time.time() - t0:.1f}s. config.json обновлены. ===", flush=True)


if __name__ == "__main__":
    main()
