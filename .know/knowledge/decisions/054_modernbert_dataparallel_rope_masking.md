<!-- created: 2026-07-06 -->
# ADR 054: Совместимость ModernBERT с DataParallel и влияние RoPE на PAD-токены

## Context and Problem
В рамках Phase 2 декодер использует последние 3 слоя из `ModernBERT-large`. При обучении на двух GPU через `nn.DataParallel` (на Kaggle) мы столкнулись с ошибками `StopIteration`. 
Изначальный диагноз: реплики `DataParallel` не могут обработать вызовы `self.parameters()` внутри properties `device` и `dtype`, которые `ModernBertModel` использует для проверки компиляции и подготовки глобальной маски внимания.
Попытка вырезать сырые слои в чистый `RawModernBertDecoder` (полностью удалив обертку `ModernBertModel`) привела к `TypeError: cannot unpack non-iterable NoneType object`.
Параллельная проблема: отключение `attention_mask` ради ускорения `XLA` (предотвращение рекомпиляции графов из-за `unpadding`) привело к сильному плато функции потерь (0.3–0.4 против 0.03 у простого трансформера) и деградации сглаженности генерации (Smoothness = DEGRADED при инференсе).

## Decisions Made and Lessons Learned
- **What was tried and didn't work:** Извлечение сырых слоев `ModernBertLayer`. Выяснилось, что `ModernBERT` вычисляет `Rotary Position Embeddings` (RoPE) глобально (на уровне `ModernBertModel`) и прокидывает их внутрь каждого слоя. Без обертки слои лишаются позиционной информации и падают.
- **Why loss was plateauing:** Отсутствие `attention_mask` заставляло RoPE вращать PAD-токены (так как их позиция не была скрыта). Это превращало пустые векторы в "кричащий" позиционный шум, на подавление которого бессмысленно расходовалась ёмкость сети.
- **Successful Solution (Best Practice):** 
  1. **DataParallel Hotfix:** Оставить цельную обертку `ModernBertModel`, но переопределить её свойства `device` и `dtype` (через патч `SafeModernBertModel`), жестко зашив туда безопасные значения (например, `torch.bfloat16`). Это обходит баг `DataParallel`.
  2. **RoPE Masking:** Обязательно передавать `attention_mask` в `backbone`. При использовании `attn_implementation="sdpa"` HuggingFace не включает динамический `unpadding`, поэтому XLA не рекомпилирует графы на TPU, а PAD-токены корректно обнуляются и не искажают Attention через позиционный шум.

## Impact
- **Positive:** Стабильное многопроцессорное обучение (DataParallel) без архитектурных поломок. Маска внимания позволяет модели спускаться к истинному оптимуму лосса и восстанавливает гладкость (smoothness) латентного пространства.
- **Negative:** Вынужденное использование грязного патча (`monkey-patching`) класса `ModernBertModel` в `modern_decoder.py` для обхода ограничений HuggingFace.
