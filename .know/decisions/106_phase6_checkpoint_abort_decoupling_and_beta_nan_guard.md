<!-- created: 2026-10-16 -->
<!-- status: Accepted -->
# ADR 106: Phase 6 Checkpoint Abort Decoupling, Beta Sampler NaN Sanitization, and Consecutive NaN Guard

**Status**: Accepted  
**Date**: 2026-10-16  
**Git Commit**: `8cd5bee` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.3.0.dev3)  

## Context and Problem
После внедрения ADR 105 обучение Phase 6 на Kaggle TPU аварийно прервалось на шаге 9000 ложным срабатыванием проверки чекпоинта: валидация на шаге 9000 прошла штатно (Val Loss: 0.0871, Val Cos: 0.9134), а веса были валидны. Прерывание вызвало объединение проверок конечного состояния весов и текущего лосса батча. Дополнительно: в диагностическом тесте TEST 3 разморозка DUS не была защищена try/finally, сэмплирование Beta-распределения на CPU при экстремально низких концентрациях стохастически давало NaN, а при пропуске не-конечных шагов отсутствовал лимит подряд идущих пропусков.

## Decisions Made
1. Разделение проверки чекпоинта: [ABORT] выполняется строго по нарушению конечности обучаемых весов is_state_dict_finite. Не-конечный лосс текущего шага дает лишь [WARN] без прерывания сессии.
2. Пропуск не-конечных шагов: при not torch.isfinite(loss) backward и step оптимизатора пропускаются с очисткой градиентов optimizer.zero_grad(set_to_none=True).
3. Изоляция TEST 3: блок заморозки DUS обернут в try/finally для гарантированного восстановления requires_grad = True.
4. Санитизация Beta-сэмплера: в sample_token_noise_levels на CPU не-конечные значения заменяются на fallback_t = tg через torch.where(torch.isfinite(t), t, fallback_t).
5. Защита от бесконечного цикла пропуска (nan_streak): введен лимит в 20 подряд идущих не-конечных шагов, после чего сессия завершается для защиты квоты TPU.
6. Увеличение эффективного batch_size до 128 (64*2) для ускорения обучения на TPU.

## Consequences
Устранены ложные аварийные остановки длительных запусков на TPU при валидных весах. Стохастические выбросы гамма-сэмплера PyTorch изолируются на CPU до передачи в XLA-граф. Исключено выгорание квоты TPU при потенциальном повреждении градиентного графа благодаря жесткому лимиту nan_streak.

## Invariants & Rules for AI
1. Чекпоинт-аборт MUST зависеть только от is_state_dict_finite(trainable_state).
2. Выход функции sample_token_noise_levels MUST быть строго конечным (без NaN/Inf) до передачи на TPU.
3. Цикл обучения MUST прерывать выполнение при nan_streak >= 20.

## Scope & Affected Files
- `experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.py` (Related KI: `decisions/097_phase6_t_global_orchestrator_and_per_sample_beta_noise.md`)
- `experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.ipynb`
