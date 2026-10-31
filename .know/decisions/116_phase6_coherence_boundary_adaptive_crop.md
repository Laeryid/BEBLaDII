<!-- created: 2026-10-31 -->
<!-- status: Accepted -->
<!-- supersedes: 114, 115 -->
# ADR 116: Phase 6 Coherence Restoration, Boundary Void Weighting, and Adaptive Inference Cropping

**Status**: Accepted  
**Date**: 2026-10-31  
**Supersedes**: [ADR 114](./114_phase6_void_balance_and_tail_erasing.md), [ADR 115](./115_phase6_variable_length_and_exposure_bias_synthesis.md)  
**Git Commit**: `07a0fd9` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.3.0.dev3)  

## Context and Problem
На чекпоинте step_129000 модель успешно освоила нативную терминацию холста токенами <|void|> (67% void, K_eff=1.1 на хвосте), однако качество связности текста деградировало из-за полного отключения transition loss (transition_loss_weight=0.0) при размороженных буферных слоях DUS, наличия Exposure Bias границы и шортката лексического копирования промпта. Кроме того, статический SmartCrop на шаге 0 (t=1.0) приводил к ошибочной обрезке из-за отсутствия войд-сигнала на чистом шуме.

## Decisions Made
1. Внедрить 4-компонентное решение проблемы связности и границы: Gated Transition Loss (t <= 0.28, вес 0.25), Prompt Anti-Copy Penalty (margin 0.05, вес 0.25), Causal Noise Ramp (35% flat, 35% span, 30% ramp), и Boundary-Aware Void Weighting (w_bound=1.20 на 24 токена границы, w_far=0.60 на дальний хвост с принудительным шумом t >= 0.75 на 50% батча). 2. Исключить Spurious-Tail Erasing из-за конфликта градиентов на flat noise. 3. Возобновить обучение TPU с чекпоинта step_107000 (max_resume_step = 107000), где биграммные слои DUS еще не были разрушены. 4. Внедрить на инференсе Adaptive SmartCrop, стабилизирующий границу на t <= 0.70 при удержании >= 90% войда в хвосте на протяжении 2 шагов.

## Consequences
Восстанавливается связность текста и синтаксическая структура без потери способности к нативной терминации. Cross-Attention вынужден генерировать новый семантический контент вместо повторения промпта. Инференс с Adaptive SmartCrop получает ускорение и защиту от галлюцинаций в хвосте без риска искажения траектории внимания на ранних диффузионных шагах.

## Alternatives Considered & Rejected
1. Spurious-Tail Erasing: отклонено, так как на flat noise зашумленный контент хвоста конфликтует с таргетом войда и сбивает модель в середине предложения. 2. Продолжение обучения с 129k: отклонено, так как биграммы DUS в слоях 13-39 деградировали из-за 22k шагов с нулевым transition loss.

## Invariants & Rules for AI
1. Transition loss ОБЯЗАН быть строго гейтирован по уровню шума t <= 0.28 во избежание градиентного взрыва нормализованных разностей на многообразии. 2. Вес дальнего хвоста войда w_far не должен опускаться ниже 0.50. 3. SmartCrop на шаге 0 (t=1.0) запрещен к жесткому усечению холста без подтверждения на шагах t <= 0.70.

## Scope & Affected Files
- `experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.py` (Related KI: `decisions/097_phase6_t_global_orchestrator_and_per_sample_beta_noise.md`)
- `experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.ipynb`
- `experiments/phase 6/evaluate_phase6_metrics.py`
- `experiments/phase 6/fixes_plan.md`

## Verification
.venv\Scripts\python.exe -m py_compile "experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.py" && .venv\Scripts\python.exe -m py_compile "experiments/phase 6/evaluate_phase6_metrics.py" && .venv\Scripts\python.exe "experiments/phase 6/evaluate_phase6_metrics.py" --check-files
