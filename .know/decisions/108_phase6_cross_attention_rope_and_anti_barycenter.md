<!-- created: 2026-10-19 -->
<!-- status: Accepted -->
# ADR 108: Устранение позиционного коллапса в Phase 6: RoPE в Cross-Attention, отталкивающий анти-барицентровый лосс, обнуление weight decay и разделение live/EMA чекпоинтов

**Status**: Accepted  
**Date**: 2026-10-19  
**Git Commit**: `6779c2c` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.3.0.dev3)  

## Context and Problem
На шаге 14000 обучения Phase 6 (чекпоинт planB_phase6_checkpoints_phase6_ca_layers_step_14000.pth) была зафиксирована стагнация и выраженное заикание: модель генерировала одно и то же слово темы на всех позициях ответа подряд (например, 'computer data computer data...' или 'information information...'). 
Инструментальная диагностика показала:
1. Позиционный коллапс в CAPromptLayer: попарное косинусное сходство между различными позициями выхода CA достигало 0.9953 (при норме 0.0638 для истинных таргетов).
2. Запросы Q в Cross-Attention не использовали RoPE, из-за чего при сильном шуме (t -> 1.0) все позиции холста генерировали тождественные векторы запроса и извлекали из промпта один и тот же усредненный вектор.
3. Косинусный лосс провоцировал коллапс в барицентр распределения при высокой энтропии шума.
4. В AdamW для CA был включен weight_decay=1e-2, подавлявший выходные проекции.
5. При сохранении чекпоинтов сохранялись только EMA-тени вместо живых весов, вызывая откат на ~500 шагов назад при каждом перезапуске на Kaggle.

## Decisions Made
1. Внедрить Rotary Position Embedding (self.rope_sa) для q (холст) и k (промпт) в Cross-Attention блока CAPromptLayer.
2. Внедрить трехкомпонентный динамический отталкивающий лосс bary_penalty (global_bary_penalty, sample_bary_penalty, adj_repulsion), штрафующий предсказание за превышение близости к барицентрам и соседним позициям сверх уровня таргета (с безопасным запасом +0.05).
3. Обнулить weight_decay = 0.0 для всех обучаемых параметров CA в AdamW.
4. Разделить сохранение чекпоинта: живые веса сохранять в live_trainable_state до применения теней EMA, а тени — в trainable_state. При Resume живые веса загружать в модель, а тени — в трекер EMA.
5. В скрипте evaluate_phase6_metrics.py добавить расчет метрик val_content_gain_CA и intra_pos_sim, а также поддержку --cfg_scale.

## Consequences
- Устраняется позиционный коллапс в Cross-Attention: запросы холста на разных позициях вращаются RoPE и извлекают релевантные компоненты промпта вместо одинакового усредненного вектора темы.
- Динамический отталкивающий лосс выбивает латентное пространство из глобального и сэмплового барицентров при t -> 1.0, не мешая точной подгонке при малых t.
- Обнуление weight_decay устраняет искусственное подавление весов проекций CA.
- Разделение live_trainable_state и ema_shadow в чекпоинте исключает потерю накопленного опыта (амнезию) при перезапусках обучения на Kaggle/TPU.
- Полная обратная совместимость с архитектурой чекпоинтов шагов 9000 и 14000: форма и имена весов state_dict не изменились.

## Invariants & Rules for AI
- В Cross-Attention блока CAPromptLayer векторы q и k ДОЛЖНЫ обязательно поворачиваться через self.rope_sa перед SDPA.
- Для параметров CAPromptLayer в оптимайзере AdamW параметр weight_decay ДОЛЖЕН быть строго равен 0.0.
- При сохранении чекпоинта живые веса модели ОБЯЗАТЕЛЬНО должны сохраняться в ключ live_trainable_state до вызова ema_tracker.apply_shadow(model).

## Scope & Affected Files
- `experiments/phase 6/evaluate_phase6_metrics.py`
- `experiments/phase 6/kaggle/train_phase6_notebook.py` (Related KI: `decisions/091_global_scheduled_identity_gate.md`, `decisions/096_phase6_gradient_collapse_and_dynamic_void_penalty.md`, `decisions/097_phase6_t_global_orchestrator_and_per_sample_beta_noise.md`)
- `experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.py` (Related KI: `decisions/097_phase6_t_global_orchestrator_and_per_sample_beta_noise.md`)
- `experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.ipynb`

## Verification
1. AST и статический анализ через run_tpu_smoke_test.py.
2. Валидация выходов CA на синтетических и реальных батчах: замер intra_pos_sim и val_content_gain_CA в evaluate_phase6_metrics.py.
3. Проверка py_compile для всех затронутых скриптов.
