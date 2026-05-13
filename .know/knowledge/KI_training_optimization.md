<!-- last_verified: 2026-05-13 -->
# KI: Training Optimization

## Overview
Консолидация архитектурных решений, направленных на повышение стабильности, эффективности использования ресурсов и качества сходимости моделей в процессе обучения.

## Key Components
| ADR | Topic | Purpose |
|---|---|---|
| `002` | [Phase 1 Training Optimization](../decisions/002_phase1_training_optimization.md) | AdamW8bit, Persistence, Cosine Annealing. |
| `009` | [Phase 1 LR Aggressive Strategy](../decisions/009_phase1_lr_aggressive_strategy.md) | CosineAnnealingWarmRestarts и борьба с плато. |
| `011` | [Projector Scale Optimization](../decisions/011_projector_scale_optimization.md) | Балансировка residual и MLP веток. |
| `012` | [Latent Space Isotropization](../decisions/012_latent_space_isotropization.md) | Изотропизация и диффузионно-ориентированная архитектура. |

## Non-obvious Details
- **AdamW8bit**: Выбран для экономии VRAM/HBM на Kaggle и TPU, позволяет увеличить размер батча.
- **Isotropization**: Критична для работы с латентными представлениями (System 2), предотвращает коллапс измерений.


## Related KIs

