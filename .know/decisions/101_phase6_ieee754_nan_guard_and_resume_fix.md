<!-- created: 2026-10-10 -->
<!-- status: Accepted -->
# ADR 101: Phase 6 IEEE 754 Safe NaN Guard, Checkpoint Finite Validation, and Optimizer Warmup

**Status**: Accepted  
**Date**: 2026-10-10  
**Git Commit**: `e72e83e` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.3.0.dev3)  

## Context and Problem
На шаге ~380 обучения Phase 6 TPU произошел сбой и появление NaN.
Анализ выявил:
1. Закон IEEE 754: операция p.grad.mul_(is_finite) при появлении Inf/NaN выполняла умножение на 0.0, что в стандарте IEEE 754 возвращает NaN! Градиенты не обнулялись, отравляя веса модели, а в EMATracker выражение (1.0 - eff_decay) * param при eff_decay=1.0 давало 0.0 * NaN = NaN, мгновенно разрушая тень.
2. В функции возобновления вызов _get_gcs_client() падал с NameError, из-за чего обучение всегда начиналось с шага 0.
3. Чекпоинты, сохраненные на шагах 1000-3000 после краша, содержали NaN и перезаписали бакет GCS.
4. Отсутствовал LR Warmup для оптимизатора, что вызывало резонансный всплеск по мере роста warmup_factor в CA-слоях.

## Decisions Made
1. Внедрить честный In-Graph NaN Guard через torch.where: p.grad.copy_(torch.where(is_safe, p.grad.nan_to_num(0.0), torch.zeros_like(p.grad))).
2. Изолировать EMATracker и pace_pullback через torch.where(is_safe, new_val, old_val), полностью исключая умножение на ноль потенциальных NaN.
3. Реализовать get_gcs_client() и функцию is_state_dict_finite() для валидации весов перед сохранением и при загрузке.
4. Внедрить безопасный перебор доступных чекпоинтов GCS: отбрасывать любые поврежденные чекпоинты (NaN/Inf) и загружать последний валидный.
5. Добавить ступенчатый прогрев LR (квантование каждые 20 шагов), исключающий мутации HLO/XLA графа на каждом шаге.
6. Раннее прерывание при сбое (Early Abort): при сохранении чекпоинта, если обнаружены NaN/Inf, сохраняется 1 аварийный снимок на GCS для диагностики, после чего обучение немедленно завершается для сбережения квоты TPU.

## Consequences
1. Полная математическая гарантия невозможности отравления модели через p.grad, EMA и PACE благодаря семантике torch.where вместо умножения на ноль (правило IEEE 754: 0.0 * NaN = NaN).
2. Защита GCS бакета от заражения нечисловыми чекпоинтами (is_state_dict_finite блокирует выгрузку битых весов).
3. Интеллектуальный Resume: последовательный перебор чекпоинтов с GCS от самых свежих к старым с автоматической отбраковкой поврежденных чекпоинтов (шаги 1000-3000 предыдущего краша будут пропущены).
4. Ступенчатый прогрев LR каждые 20 шагов устраняет гидроудар и предотвращает лишние перекомпиляции графа TPU.
5. Немедленное прерывание при появлении NaN в чекпоинте исключает повторение ситуации с тратой часов TPU-квоты на генерацию последующих пустых чекпоинтов.

## Scope & Affected Files
- `experiments/phase 6/tpu kaggle/__pycache__/train_phase6_tpu_notebook.cpython-313.pyc`
- `experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.ipynb`
- `experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.py` (Related KI: `decisions/097_phase6_t_global_orchestrator_and_per_sample_beta_noise.md`)
- `scratch/sync_phase6_notebook.py`
- `scratch/__pycache__/sync_phase6_notebook.cpython-313.pyc`
