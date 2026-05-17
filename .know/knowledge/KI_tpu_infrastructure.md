<!-- last_verified: 2026-05-13 -->
# KI: TPU Infrastructure & FSDP

## Overview
Архив технических решений и ограничений, связанных с обучением на TPU v6e с использованием стратегий SPMD и FSDP.

## Key Components
| ADR | Topic | Purpose |
|---|---|---|
| `003` | [FSDP TPU Transition](../decisions/003_fsdp_tpu_transition.md) | Переход на FSDP для решения проблем OOM. |
| `004` | [SPMD FSDP TPU Constraints](../decisions/004_spmd_fsdp_tpu_constraints.md) | Шардинг батча и лимиты HBM. |
| `005` | [ModernBERT Gradient Checkpointing](../decisions/005_modernbert_xla_gradient_checkpointing.md) | Отказ от GC из-за особенностей XLA и Sliding Window. |
| `006` | [FSDP Checkpoint Fix](../decisions/006_fsdp_checkpoint_loading_fix.md) | Исправление логики загрузки весов. |
| `007` | [TPU Launch Stabilization](../decisions/007_tpu_v6e_launch_stabilization.md) | Переменные окружения и топология. |
| `008` | [Single Process Victory](../decisions/008_single_process_spmd_victory.md) | Устранение дедлоков через однопроцессный SPMD. |
| `010` | [Reasoning Restart](../decisions/010_spmd_fsdp_reasoning_restart.md) | Структура чекпоинтов для возобновления обучения. |

## Non-obvious Details
- **XLA OOM**: Многие стандартные оптимизации (например, Gradient Checkpointing) могут вызывать OOM на XLA из-за статической компиляции графа.
- **Single Process**: Однопроцессный режим SPMD значительно упрощает отладку и исключает рассинхронизацию между процессами на одном хосте.


## Related KIs

