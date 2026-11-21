"""
Инспекция и аудит чекпоинта Фазы 2 Plan C (Generator-Decoder & Noise Radar).

Проверяет:
  1. Загрузку весов (raw / EMA), отсутствие NaN/Inf.
  2. Качество реконструкции (Encode -> Decode) на реальных текстах (RU, EN, код, короткие фразы).
  3. O(8)-инвариантность (случайный ортогональный поворот концепта).
  4. Устойчивость к шуму (Gaussian + Ball Clamp, MixOther).
  5. Калибровку Noise Radar (TrueSigma vs RadarHat, MAE).

Запуск из корня репозитория:
    .venv\\Scripts\\python.exe "experiments/phase 2/inspect_phase2_decoder.py" --checkpoint "experiments/phase 2/checkpoints/phase2_decoder_step_25000.zip"
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

PHASE2_DIR = Path(__file__).resolve().parent
REPO_ROOT = PHASE2_DIR.parents[1]
for p in (str(REPO_ROOT), str(PHASE2_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from storage.components.decoder import GrassmannianDecoder, get_decoder
from storage.components.encoder import GrassmannianEncoder, get_encoder
from storage.components.plan_c_layers import WordAwareElasticChunker
from storage.components.tokenizer import get_tokenizer
from train_phase2 import (
    EMAModel,
    EnvironmentConfig,
    corrupt_latent_cells,
    grassmann_cell_similarity,
)

DEFAULT_SAMPLES: List[Dict[str, str]] = [
    {
        "name": "RU_prose",
        "text": (
            "Искусственный интеллект постепенно меняет способы анализа сложных научных данных. "
            "В отличие от классических алгоритмов, современные нейросетевые модели способны находить "
            "скрытые геометрические структуры в многомерных пространствах."
        ),
    },
    {
        "name": "EN_tech",
        "text": (
            "Grassmannian manifolds provide a natural geometric framework for representing subspaces "
            "invariant to orthogonal rotations. By canonicalizing each local basis, the decoder "
            "recovers exact token sequences even after arbitrary O(8) transformations."
        ),
    },
    {
        "name": "Python_code",
        "text": (
            "def normalize_basis(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:\n"
            "    norm = torch.linalg.norm(v, dim=-1, keepdim=True).clamp(min=eps)\n"
            "    return v / norm\n"
        ),
    },
    {
        "name": "RU_short",
        "text": "Привет! Как дела? Проверка короткой фразы и чисел: 42, 3.14159, 2026.",
    },
]


def find_latest_decoder_checkpoint(ckpt_dir: str) -> Optional[str]:
    patterns = [
        os.path.join(ckpt_dir, "phase2_decoder_*.pt"),
        os.path.join(ckpt_dir, "phase2_decoder_*.zip"),
    ]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def load_decoder_checkpoint(
    decoder: GrassmannianDecoder,
    ckpt: Dict[str, Any],
    use_ema: bool = True,
) -> Dict[str, Any]:
    if "decoder_state_dict" not in ckpt:
        raise KeyError(
            f"В чекпоинте нет ключа 'decoder_state_dict'. "
            f"Доступные ключи: {list(ckpt.keys())}"
        )

    sd = {k.replace("module.", ""): v for k, v in ckpt["decoder_state_dict"].items()}
    missing, unexpected = decoder.load_state_dict(sd, strict=False)
    allowed_missing = {"lm_head_weight"}
    bad_missing = [k for k in missing if k not in allowed_missing]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"Несовпадение ключей state_dict: missing={bad_missing}, unexpected={unexpected}"
        )

    used_ema = False
    if use_ema and ckpt.get("ema_state_dict"):
        ema = EMAModel(decoder, decay=0.999)
        ema.load_state_dict(ckpt["ema_state_dict"], device=torch.device("cpu"))
        ema.apply_shadow(decoder)
        used_ema = True

    nan_count = 0
    inf_count = 0
    total_params = 0
    for p in decoder.parameters():
        total_params += p.numel()
        if p.is_floating_point():
            nan_count += int(torch.isnan(p).sum().item())
            inf_count += int(torch.isinf(p).sum().item())

    return {
        "step": int(ckpt.get("step", 0)),
        "used_ema": used_ema,
        "total_params": total_params,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "config": ckpt.get("config", {}),
        "ckpt_train_metrics": ckpt.get("metrics", {}),
    }


def encode_texts_to_batch(
    texts: List[str],
    tokenizer: Any,
    chunker: WordAwareElasticChunker,
    num_cells: int = 32,
    pad_token_id: int = 10,
) -> Dict[str, torch.Tensor]:
    input_ids_list, targets_list, weights_list, masks_list = [], [], [], []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            ids = [1]
        packed = chunker.chunk_tokens(
            token_ids=ids,
            max_cells=num_cells,
            elastic_drop_prob=0.0,
        )
        input_ids_list.append(packed["input_ids"])
        targets_list.append(packed["targets"])
        weights_list.append(packed["weights"])
        masks_list.append(packed["content_cell_mask"])

    return {
        "input_ids": torch.stack(input_ids_list, dim=0),
        "targets": torch.stack(targets_list, dim=0),
        "weights": torch.stack(weights_list, dim=0),
        "content_cell_mask": torch.stack(masks_list, dim=0),
    }


@torch.no_grad()
def inspect_modes(
    encoder: GrassmannianEncoder,
    decoder: GrassmannianDecoder,
    tokenizer: Any,
    samples: List[Dict[str, str]],
    device: torch.device,
    num_cells: int = 32,
    pad_token_id: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    chunker = WordAwareElasticChunker(
        tokenizer=tokenizer,
        chunk_size=4,
        pad_token_id=pad_token_id,
    )
    texts = [s["text"] for s in samples]
    batch = encode_texts_to_batch(
        texts=texts,
        tokenizer=tokenizer,
        chunker=chunker,
        num_cells=num_cells,
        pad_token_id=pad_token_id,
    )

    input_ids = batch["input_ids"].to(device)
    targets = batch["targets"].to(device)
    weights = batch["weights"].to(device)
    content_mask = batch["content_cell_mask"].to(device)

    enc_w_dtype = encoder.embeddings.weight.dtype
    V_0 = encoder(input_ids, weights.to(enc_w_dtype), canonicalize=False).float()

    modes = ["clean", "random_o8", "gaussian", "mix_other"]
    results: Dict[str, Any] = {}

    for mode in modes:
        torch.manual_seed(seed + modes.index(mode) * 100)
        V_in, true_sigma, _ = corrupt_latent_cells(V_0, content_mask, mode_override=mode)
        logits, radar = decoder(V_in, return_radar=True)
        pred_ids = torch.argmax(logits.float(), dim=-1)
        sigma_hat = radar["sigma_cell_hat"].float()

        mode_items = []
        acc_w_list, acc_p_list, mae_list = [], [], []

        for b, sample in enumerate(samples):
            tgt_b = targets[b]
            pred_b = pred_ids[b]
            valid_tok = tgt_b != -100
            word_mask = valid_tok & (tgt_b != pad_token_id)
            pad_mask = valid_tok & (tgt_b == pad_token_id)

            acc_w = (
                float((pred_b[word_mask] == tgt_b[word_mask]).float().mean().item())
                if word_mask.any()
                else 1.0
            )
            acc_p = (
                float((pred_b[pad_mask] == tgt_b[pad_mask]).float().mean().item())
                if pad_mask.any()
                else 1.0
            )

            cm_b = content_mask[b]
            true_s_b = float(true_sigma[b, cm_b].mean().item()) if cm_b.any() else 0.0
            hat_s_b = float(sigma_hat[b, cm_b].mean().item()) if cm_b.any() else 0.0
            mae_b = float(torch.abs(sigma_hat[b, cm_b] - true_sigma[b, cm_b]).mean().item()) if cm_b.any() else 0.0
            sim_b = float(grassmann_cell_similarity(V_0[b : b + 1], V_in[b : b + 1])[0, cm_b].mean().item()) if cm_b.any() else 1.0

            ref_txt = WordAwareElasticChunker.detokenize_non_pad(tgt_b[valid_tok].cpu(), tokenizer, pad_token_id)
            gen_txt = WordAwareElasticChunker.detokenize_non_pad(pred_b[valid_tok].cpu(), tokenizer, pad_token_id)

            acc_w_list.append(acc_w)
            acc_p_list.append(acc_p)
            mae_list.append(mae_b)

            mode_items.append(
                {
                    "name": sample["name"],
                    "acc_words": round(acc_w, 4),
                    "acc_pad": round(acc_p, 4),
                    "latent_sim": round(sim_b, 4),
                    "true_sigma": round(true_s_b, 4),
                    "radar_hat": round(hat_s_b, 4),
                    "radar_mae": round(mae_b, 4),
                    "exact_match": ref_txt == gen_txt,
                    "ref": ref_txt,
                    "gen": gen_txt,
                }
            )

        results[mode] = {
            "mean_acc_words": round(sum(acc_w_list) / len(acc_w_list), 4),
            "mean_acc_pad": round(sum(acc_p_list) / len(acc_p_list), 4),
            "mean_radar_mae": round(sum(mae_list) / len(mae_list), 4),
            "samples": mode_items,
        }

    return results


def print_report(load_info: Dict[str, Any], results: Dict[str, Any]) -> None:
    print("=" * 90)
    print(
        f"Phase 2 Decoder Audit | step={load_info['step']} | EMA={load_info['used_ema']} | "
        f"params={load_info['total_params']:,} | NaN={load_info['nan_count']} | Inf={load_info['inf_count']}"
    )
    print("=" * 90)
    print(f"{'Mode':<12} | {'AccW':>7} | {'AccP':>7} | {'RadarMAE':>8}")
    print("-" * 44)
    for mode, data in results.items():
        print(
            f"{mode:<12} | {data['mean_acc_words']:7.4f} | "
            f"{data['mean_acc_pad']:7.4f} | {data['mean_radar_mae']:8.4f}"
        )

    for mode, data in results.items():
        print("\n" + "-" * 90)
        print(f"MODE: {mode.upper()}")
        print("-" * 90)
        for item in data["samples"]:
            status = "EXACT" if item["exact_match"] else f"AccW={item['acc_words']:.3f}"
            print(
                f"  [{item['name']}] {status} | Sim={item['latent_sim']:.3f} | "
                f"TrueSigma={item['true_sigma']:.3f} | RadarHat={item['radar_hat']:.3f} | MAE={item['radar_mae']:.3f}"
            )
            print(f"    REF: {item['ref'][:160]!r}")
            print(f"    GEN: {item['gen'][:160]!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Инспекция чекпоинта Фазы 2 Plan C (Generator-Decoder & Noise Radar)")
    parser.add_argument("--checkpoint", type=str, default="auto", help="Путь к .pt/.zip чекпоинту или 'auto'")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--no-ema", action="store_true", help="Использовать сырые веса вместо EMA")
    parser.add_argument("--num-cells", type=int, default=32, help="Число концепт-клеток (4 токена на клетку)")
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимости шума")
    parser.add_argument("--save-json", type=str, default=None, help="Путь для сохранения JSON-отчёта")
    args = parser.parse_args()

    env = EnvironmentConfig.detect(override_device=args.device)
    ckpt_dir = str(PHASE2_DIR / "checkpoints")
    ckpt_path = find_latest_decoder_checkpoint(ckpt_dir) if args.checkpoint == "auto" else args.checkpoint
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Чекпоинт не найден: {ckpt_path}")

    print(f"[Inspect] Чекпоинт: {ckpt_path}")
    print(f"[Inspect] Устройство: {env.device}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})

    tokenizer = get_tokenizer()
    encoder = get_encoder(weights_path=env.encoder_weights_path, device=env.device, eval_mode=True)
    decoder = get_decoder(
        weights_path=None,
        encoder=encoder,
        device=env.device,
        eval_mode=True,
        num_smooth_layers=int(cfg.get("num_smooth_layers", 4)),
        canonicalize_basis=bool(cfg.get("canonicalize_basis", True)),
        enable_noise_radar=bool(cfg.get("enable_noise_radar", True)),
    )

    load_info = load_decoder_checkpoint(decoder, ckpt, use_ema=not args.no_ema)
    decoder.to(env.device).eval()
    encoder.to(env.device).eval()

    results = inspect_modes(
        encoder=encoder,
        decoder=decoder,
        tokenizer=tokenizer,
        samples=DEFAULT_SAMPLES,
        device=env.device,
        num_cells=args.num_cells,
        seed=args.seed,
    )

    print_report(load_info, results)

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"checkpoint": ckpt_path, "load_info": load_info, "results": results}, f, ensure_ascii=False, indent=2)
        print(f"\n[Inspect] JSON-отчёт сохранён: {out_path}")


if __name__ == "__main__":
    main()
