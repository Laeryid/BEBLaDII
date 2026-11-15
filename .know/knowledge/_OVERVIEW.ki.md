<!-- last_verified: 2026-11-15 -->
# KI: Project Overview — BEBLaDII

## Overview
**BEBLaDII** (*Bidirectional Encoder Based Latent Diffusion with Information Injection*) — исследовательский проект непрерывной латентной диффузионной языковой модели, разделяющей лингвистическую генерацию (System 1) и непрерывное рассуждение (System 2) с прямой инъекцией внешней памяти (**CLM**).

## Эволюция архитектуры
- **Plan A (ADR 001–037, архив)**: ранняя дистилляция `latentBERT` (ModernBERT-large DUS 40 слоёв) и `OutputProjector`. Зафиксированы базовые уроки по XLA/TPU, SPMD/FSDP и сферической топологии.
- **Plan B (ADR 038–119, завершён)**: сферическая латентная диффузия ($S^{1023}$) на токенах Qwen2.5-1.5B (Phase 1 VAE → Phase 2 ModernLatentDecoder → Phase 3/4 AdaLN диффузия → Phase 5 Sensor Ensemble / Orchestrator → Phase 6 `CAPromptLayer` и `|void|`).
- **Plan C (ADR 120–133+, актуальный)**: перезапуск с токенизатором Tekken и спектральным латентным пространством подпространств **Burer–Monteiro Grassmannian Frames** ($V \in \mathbb{R}^{8 \times 128}$, оператор плотности $\rho = V^T V$, $\mathcal{O}(8)$-инвариантный декомпрессор, OrthMuon, квантовая логика).

## Структура репозитория
- `src/beb_la_dii/` — библиотечное ядро: `model/` (DUS, VAE, декомпрессор, проекторы, оркестратор), `rag/` (CLM geometric pooling), `utils/` (лоссы, данные, трекеры, токенизаторы), `data/`.
- `experiments/` — тренировочные и оценочные пайплайны: `phase 1/`, `phase 2 plan C/` (актуальный Plan C), а также архивные `Plan A/` и `Plan B/`.
- `scripts/` — утилиты подготовки датасетов, диагностики весов, проверки TPU/Kaggle и smoke-тестов.
- `ideas/` — теоретические заметки и статьи, включая [plan C.md](file:///C:/Experiments/BEBLaDII/experiments/plan%20C.md).
- `tests/` — модульные и интеграционные тесты.
- `.know/` — база знаний (`knowledge/*.ki.md`, `decisions/*.md`, `doc_config.json`, `DIR_INDEX.md`).

## Навигация для агентов
- Перед правкой кода вызывайте `ki_context_for(files=[...])` для получения актуальных инвариантов и ограничений.
- Для обзора активных KI и ADR используйте `ki_list(status="active")` и `ki_search(query="...")`.
