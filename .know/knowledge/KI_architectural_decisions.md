<!-- last_verified: 2026-05-13 -->
# KI: Architectural Decisions (ADR Archive)

## Overview
Архив принятых архитектурных решений (ADR). Содержит историю изменений логики проекта, обоснование выбора технологий и результаты ретроспектив.

## ADR List
| ADR | Topic | Status |
|---|---|---|
| `001` | Kaggle Phase 1 Retrospective | Accepted |
| `002` | Phase 1 Training Optimization | Accepted |
| `003` | FSDP TPU Transition | Accepted |
| `004` | SPMD FSDP TPU Constraints | Accepted |
| `005` | ModernBERT XLA Gradient Checkpointing | Accepted |
| `006` | FSDP Checkpoint Loading Fix | Accepted |
| `007` | TPU v6e Launch Stabilization | Accepted |
| `008` | Single Process SPMD Victory | Accepted |
| `009` | Phase 1 LR Aggressive Strategy | Accepted |
| `010` | SPMD FSDP Reasoning Restart | Accepted |
| `011` | Projector Scale Optimization | Accepted |
| `012` | Latent Space Isotropization | Accepted |

## Usage
Агенты должны обращаться к ADR при возникновении вопросов "почему это реализовано именно так" или перед внесением изменений в критические части системы (XLA, FSDP, Модели).

## Related KIs
- [Knowledge Management](KI_knowledge_management.md)
