"""
Скрипт детальной инспекции и валидации автоэнкодера Фазы 1 Plan C:
- Загрузка чекпоинта phase1_best_model.pt (с EMA shadow весами)
- Сквозная проверка реконструкции текста (Original vs Reconstructed)
- Расчет метрик качества восстановления (Token Accuracy, Perplexity, Levenshtein)
- Глубокая спектральная диагностика латентного пространства Грассманиана Gr(8, 128):
  * Эффективный ранг (rk_eff)
  * Распределение энергии по 8 ортогональным осям
  * Траектория непрерывности по холсту концепт-клеток
  * Матрица расстояний Грассмана между концептами
"""

import os
import sys
import math
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
from typing import List, Dict, Any, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from train_phase1 import (
    GrassmannianPhase1Model,
    EnvironmentConfig,
    load_tekken_tokenizer,
)


def load_model_and_tokenizer(
    ckpt_path: str, device: torch.device
) -> Tuple[GrassmannianPhase1Model, Any, Dict[str, Any]]:
    env = EnvironmentConfig.detect()
    tokenizer = load_tekken_tokenizer(env)

    print(f"\n[1/4] Загрузка чекпоинта: {ckpt_path} ...", flush=True)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Файл чекпоинта не найден: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    step = checkpoint.get("step", 0)
    metrics = checkpoint.get("metrics", {})
    best_val_loss = metrics.get("val/loss", checkpoint.get("val_loss", None))
    print(f"      -> Чекпоинт шага: {step} | val_loss: {best_val_loss}", flush=True)

    # Инициализация модели
    model = GrassmannianPhase1Model(
        vocab_size=131072,
        hidden_size=1024,
        intermediate_size=2048,
        num_axes=8,
        axis_dim=128,
        chunk_size=4,
        pad_token_id=10,
    )

    # Приоритет отдаем теневым весам EMA
    if "ema_state_dict" in checkpoint and checkpoint["ema_state_dict"] is not None:
        print("      -> Применяем сглаженные EMA shadow веса.", flush=True)
        raw_sd = checkpoint["ema_state_dict"]
        if isinstance(raw_sd, dict) and "shadow" in raw_sd:
            raw_sd = raw_sd["shadow"]
    else:
        print("      -> Применяем стандартные веса модели.", flush=True)
        raw_sd = checkpoint.get("model_state_dict", checkpoint)

    cleaned_sd = {k.replace("module.", ""): v for k, v in raw_sd.items()}
    model.load_state_dict(cleaned_sd, strict=True)
    model.to(device)
    model.eval()
    print("      -> Модель успешно загружена и переведена в режим eval().", flush=True)

    meta = {"step": step, "val_loss": best_val_loss}
    return model, tokenizer, meta


def compute_token_accuracy(orig_ids: List[int], pred_ids: List[int], pad_id: int = 10) -> Dict[str, float]:
    total_valid = 0
    correct = 0
    for o, p in zip(orig_ids, pred_ids):
        if o != pad_id and o != -100:
            total_valid += 1
            if o == p:
                correct += 1
    acc = (correct / total_valid * 100.0) if total_valid > 0 else 0.0
    return {"accuracy": acc, "correct": correct, "total": total_valid}


def analyze_grassmann_cell(V_cell: torch.Tensor) -> Dict[str, Any]:
    """
    Анализ геометрии клетки V_cell in R^{N x 8 x 128}.
    Возвращает спектральные метрики, эффективный ранг и непрерывность траектории.
    """
    N, K, D = V_cell.shape
    V_f32 = V_cell.float()

    # 1. Сингулярные числа каждой клетки (8 осей)
    # Матрица Грама G_i = V_i @ V_i^T in R^{8 x 8}
    G = torch.matmul(V_f32, V_f32.transpose(-2, -1))  # (N, 8, 8)
    eigs = torch.linalg.eigvalsh(G).clamp(min=1e-8)  # (N, 8)
    eigs = eigs.flip(dims=[-1])  # по убыванию lambda_1 >= lambda_2 >= ...

    # Энергия по осям (нормализованный спектр)
    eig_sum = eigs.sum(dim=-1, keepdim=True)
    prob = eigs / eig_sum
    # Эффективный ранг по Шеннону
    entropy = -(prob * torch.log(prob)).sum(dim=-1)
    rk_eff = torch.exp(entropy)

    # 2. Непрерывность траектории: косинус Грассмана (tr(P_i P_{i+1}) / 8) между соседними клетками
    proj_matrices = torch.matmul(V_f32.transpose(-2, -1), V_f32)  # P_i = V_i^T V_i in R^{128 x 128}
    smoothness = []
    for i in range(N - 1):
        # tr(P_i P_{i+1}) = ||V_i V_{i+1}^T||_F^2
        overlap = torch.matmul(V_f32[i], V_f32[i + 1].transpose(-2, -1))
        frob_sq = (overlap ** 2).sum().item()
        similarity = frob_sq / (K)  # нормализовано в [0, 1]
        smoothness.append(similarity)

    return {
        "eigs_mean": eigs.mean(dim=0).cpu().tolist(),
        "prob_mean": prob.mean(dim=0).cpu().tolist(),
        "rk_eff_mean": rk_eff.mean().item(),
        "rk_eff_min": rk_eff.min().item(),
        "rk_eff_max": rk_eff.max().item(),
        "smoothness_mean": (sum(smoothness) / len(smoothness)) if smoothness else 1.0,
    }


def inspect_text(
    text: str,
    model: GrassmannianPhase1Model,
    tokenizer: Any,
    device: torch.device,
    title: str = "Сэмпл",
    max_len: int = 256,
):
    print(f"\n{'=' * 80}")
    print(f"=== {title.upper()} ===")
    print(f"{'=' * 80}")
    print(f"Исходный текст ({len(text)} симв.):\n\"{text[:200]}{'...' if len(text) > 200 else ''}\"\n")

    # Токенизация
    encoded = tokenizer.encode(text)
    input_ids = encoded.ids[:max_len]
    pad_token_id = 10

    # Выравнивание до кратности 4
    rem = len(input_ids) % 4
    if rem != 0:
        pad_needed = 4 - rem
        input_ids = input_ids + [pad_token_id] * pad_needed

    seq_len = len(input_ids)
    num_cells = seq_len // 4

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    targets_tensor = input_tensor.clone()

    # Прямой проход энкодер + декодер
    with torch.no_grad():
        logits, V_cell, kl_loss, rank_loss, rk_eff = model(input_tensor)
        recon_loss = F.cross_entropy(logits.view(-1, 131072), targets_tensor.view(-1), ignore_index=pad_token_id)
        pred_ids = logits.argmax(dim=-1)[0].cpu().tolist()

    # Декодирование токенов обратно в текст
    rec_text = tokenizer.decode(pred_ids)

    # Метрики точности токенов
    metrics = compute_token_accuracy(input_ids, pred_ids, pad_id=pad_token_id)
    ppl = math.exp(min(recon_loss.item(), 50.0))

    print(f"[Метрики восстановления]:")
    print(f"  * Токенов в сэмпле: {seq_len} (концепт-клеток 4->1: {num_cells})")
    print(f"  * Точность токенов (Token Accuracy): {metrics['accuracy']:.2f}% ({metrics['correct']}/{metrics['total']})")
    print(f"  * Recon Loss (Cross-Entropy): {recon_loss.item():.4f}")
    print(f"  * Перплексия (Perplexity): {ppl:.2f}")

    print(f"\n[Сравнение текстов]:")
    print(f"  ОРИГИНАЛ:       \"{tokenizer.decode(input_ids)[:250]}\"")
    print(f"  РЕКОНСТРУКЦИЯ:  \"{rec_text[:250]}\"")

    # Анализ геометрии латентного пространства Грассманиана
    geom = analyze_grassmann_cell(V_cell[0])
    print(f"\n[Геометрия латентных концепт-клеток Gr(8, 128)]:")
    print(f"  * Средний эффективный ранг (rk_eff): {geom['rk_eff_mean']:.3f} (min: {geom['rk_eff_min']:.3f}, max: {geom['rk_eff_max']:.3f})")
    print(f"  * Непрерывность по холсту (Grassmann overlap): {geom['smoothness_mean']:.3f}")
    
    # Визуализация распределения спектра
    print(f"  * Спектральный профиль 8 осей (доля энергии в %):")
    bars = []
    for k, p in enumerate(geom['prob_mean']):
        pct = p * 100.0
        bar = "#" * int(round(pct / 2.5))
        bars.append(f"      Ось {k+1}: {pct:5.1f}% | {bar}")
    print("\n".join(bars))


def main():
    parser = argparse.ArgumentParser(description="Инспектор модели и латентного пространства Phase 1 Plan C")
    parser.add_argument("--ckpt", type=str, default="experiments/phase 1/checkpoints/phase1_best_model.pt")
    parser.add_argument("--num-samples", type=int, default=3, help="Количество валидационных сэмплов")
    parser.add_argument("--max-len", type=int, default=128, help="Максимальная длина последовательности токенов")
    parser.add_argument("--custom-text", type=str, default=None, help="Пользовательский текст для проверки")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, tokenizer, meta = load_model_and_tokenizer(args.ckpt, device)

    # 1. Если задан кастомный текст
    if args.custom_text:
        inspect_text(args.custom_text, model, tokenizer, device, title="Пользовательский текст", max_len=args.max_len)
        return

    # 2. Инспекция валидационных сэмплов из val.parquet
    env = EnvironmentConfig.detect()
    val_path = env.val_data_path
    if os.path.exists(val_path):
        print(f"\n[2/4] Загрузка валидационного датасета из: {val_path} ...", flush=True)
        df = pd.read_parquet(val_path)
        col = "text" if "text" in df.columns else df.columns[0]
        samples = df[col].dropna().tolist()[:args.num_samples]

        for idx, text in enumerate(samples):
            inspect_text(text, model, tokenizer, device, title=f"Валидационный сэмпл #{idx + 1}", max_len=args.max_len)
    else:
        # Fallback тестовые сэмплы (Русский, Английский, Код)
        print(f"\n[2/4] val.parquet не найден локально. Используем типовые синтетические сэмплы.", flush=True)
        test_samples = [
            ("Русский текст (Философия / Наука)", "Нейронная сеть представляет собой математическую модель, построенную по принципу организации биологических нейронных сетей. Она способна обобщать входные данные и выявлять скрытые смысловые закономерности в латентном пространстве."),
            ("English Text (Computer Science)", "The transformer architecture relies on self-attention mechanisms to compute representations of its input and output without using sequence-aligned recurrent networks or convolution operations."),
            ("Python Code & Reasoning", "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"),
        ]
        for name, text in test_samples:
            inspect_text(text, model, tokenizer, device, title=name, max_len=args.max_len)


if __name__ == "__main__":
    main()
