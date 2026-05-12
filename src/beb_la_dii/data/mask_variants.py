"""
mask_variants.py

Утилита для генерации 4 вариантов attention_mask из одного батча.
Маски вычисляются от реальной длины каждого примера (не от max_len),
что гарантирует наличие семантического градиента даже в коротких текстах.

Формы тензоров остаются неизменными (статические для XLA/TPU):
- input_ids:      (B, T) — токены вне маски заменены на 0
- attention_mask: (B, T) — 1 для активных, 0 для скрытых и padding
"""
import torch
from typing import Optional


# Доли от реальной длины текста для каждого варианта
MASK_FRACTIONS = [0.25, 0.5, 0.75, 1.0]

# Веса этих вариантов в суммарном лоссе
MASK_WEIGHTS = [0.4, 0.6, 0.8, 1.0]


def make_mask_variants(
    batch: dict,
    seq_lens: Optional[torch.Tensor] = None,
) -> list[dict]:
    """
    Генерирует 4 варианта батча с нарастающими масками.

    Args:
        batch: словарь с ключами 'input_ids' и 'attention_mask', оба (B, T).
        seq_lens: (B,) — реальные длины текстов без padding.
                  Если None — вычисляется из attention_mask.sum(dim=1).

    Returns:
        Список из 4 словарей, каждый содержит:
            'input_ids':      (B, T) — токены вне маски = 0
            'attention_mask': (B, T) — 1 только для активных позиций
            'variant_idx':    int 0..3 (для выбора веса в loss)
    """
    input_ids = batch["input_ids"]
    B, T = input_ids.shape
    device = input_ids.device

    if seq_lens is None:
        # Реальные длины — сумма единиц в оригинальной маске
        seq_lens = batch["attention_mask"].sum(dim=1)  # (B,)

    # Тензор индексов позиций, совместимый с XLA (статическая форма)
    positions = torch.arange(T, device=device).unsqueeze(0)  # (1, T)

    variants = []
    for idx, frac in enumerate(MASK_FRACTIONS):
        # Целевая длина для каждого примера в батче
        target_lens = (seq_lens.float() * frac).long().clamp(min=1)  # (B,)

        # Маска: True там, где позиция < target_len
        new_mask = (positions < target_lens.unsqueeze(1)).long()  # (B, T)

        # Зануляем токены вне маски — модель не "подглядит" в padding
        new_ids = input_ids * new_mask

        variants.append({
            "input_ids":      new_ids,
            "attention_mask": new_mask,
            "variant_idx":    idx,
        })

    return variants


if __name__ == "__main__":
    # Smoke-тест
    B, T = 3, 32
    batch = {
        "input_ids":      torch.randint(1, 1000, (B, T)),
        "attention_mask": torch.ones(B, T, dtype=torch.long),
    }
    # Симулируем разные реальные длины
    seq_lens = torch.tensor([32, 20, 8])

    variants = make_mask_variants(batch, seq_lens=seq_lens)

    assert len(variants) == 4, "Должно быть 4 варианта"
    for i, v in enumerate(variants):
        assert v["input_ids"].shape == (B, T), f"Bad input_ids shape in variant {i}"
        assert v["attention_mask"].shape == (B, T), f"Bad mask shape in variant {i}"
        active = v["attention_mask"].sum(dim=1)
        print(f"Variant {i} (frac={MASK_FRACTIONS[i]}): active_tokens = {active.tolist()}")

    # Monotonicity check: each next variant is longer
    for b in range(B):
        lengths = [v["attention_mask"][b].sum().item() for v in variants]
        assert lengths == sorted(lengths), f"Monotonicity broken for sample {b}: {lengths}"

    print("Smoke test passed OK")
