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

## Usage
Агенты должны обращаться к ADR при возникновении вопросов "почему это реализовано именно так". Если вопрос касается TPU или Оптимизации — переходите в соответствующие KI выше.








## Related KIs

