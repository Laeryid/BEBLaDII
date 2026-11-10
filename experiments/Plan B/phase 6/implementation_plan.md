# Implementation Plan

- **Affected layers**: Phase 6 Evaluation Scripts
- **Read KIs**: `experiments/phase 6/plan.md`, `experiments/phase 6/kaggle/train_phase6_notebook.py`
- **KIs Constraints**: Phase 6 model uses `CAPromptLayer` to inject query semantics into the diffusion canvas. Canvas starts as pure noise. Model needs to be reconstructed from Phase 4 weights + Phase 6 CA_Prompt trained weights (or zero-initialized if no weights exist yet, though I assume some weights exist or we will demonstrate the setup).

## Steps:
1. Создать скрипт `evaluate_phase6_metrics.py` в папке `experiments/phase 6/`.
2. Перенести определения `CAPromptLayer`, `Phase6BlockWrapper` и `BEBLaDIIPhase6` (в режиме eval) в скрипт, так как они существуют только в kaggle-ноутбуке.
3. Реализовать загрузку чекпоинта (базовый DUS из фазы 4 + веса CA слоев из фазы 6, если имеются).
4. Написать логику денойзинга (25 шагов от t=1.0 до 0.0) на основе `plan.md`. Canvas стартует из шума `F.normalize(randn)`.
5. Добавить парсинг аргументов и интерактивный ввод. Скрипт сможет как прогнать захардкоженные факты (из train_phase6.parquet и вне его), так и принять пользовательский ввод (вопрос `Q`).
6. Сохранять отчет (например, HTML/TXT) с демонстрацией шагов генерации `A`.
7. Проверить синтаксис скрипта `py_compile` перед завершением.
