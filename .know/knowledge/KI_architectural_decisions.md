<!-- last_verified: 2026-05-13 -->
# KI: Architectural Decisions (ADR Archive)

## Overview
Центральный архив принятых архитектурных решений (ADR). Содержит ретроспективы и ссылки на тематические группы решений.

## Specialized Decision Groups
- [**Training Optimization**](KI_training_optimization.md) — Решения по AdamW8bit, LR, скейлингу и изотропизации.
- [**TPU Infrastructure & FSDP**](KI_tpu_infrastructure.md) — Технические аспекты работы с XLA, SPMD и FSDP.

## General ADR & Retrospectives
| ADR | Topic | Status | Purpose |
|---|---|---|---|
| `001` | [Kaggle Phase 1 Retrospective](../decisions/001_kaggle_phase1_retrospective.md) | Accepted | Итоги первой фазы обучения. |
| `068` | [Phase 3 Mode Collapse & Self-Conditioning](../decisions/068_phase3_mode_collapse_and_self_conditioning.md) | Accepted | Внедрение Self-Conditioning и фикс `sep_embed` для устранения коллапса на высоких шумах. |
| `072` | [Phase 3 Gradient Trap & x0-prediction](../decisions/072_phase3_gradient_trap_and_strict_x0_prediction.md) | Accepted | Удаление gate_t, Entropy Loss, переход на честный Slerp и x0-prediction. |
| `080` | [Phase 4 EMA Validation & PACE Pullback Calibration](../decisions/080_phase4_ema_validation_resume_fix_and_pace_pullback_calibration.md) | Accepted | Перевод валидации на EMA, in-place copy теней на TPU и калибровка pullback_alpha (0.001). |
| `081` | [Phase 4 Checkpoint Evaluation & Tensor-Based PACE Schedule](../decisions/081_phase4_checkpoints_evaluation_and_pace_schedule.md) | Accepted | Сравнение 8995 vs 10995 (устранение R1-коллапса), тензорный pullback_alpha, warmup (0.03->0.001) и косинусный цикл (0.001..0.01). |

## Usage
Агенты должны обращаться к ADR при возникновении вопросов "почему это реализовано именно так". Если вопрос касается TPU или Оптимизации — переходите в соответствующие KI выше.










## Related KIs

