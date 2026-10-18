<!-- created: 2026-10-18 -->
<!-- status: Accepted -->
# ADR 107: Phase 6 TPU HBM Program Reservation Fix, In-Graph Loss Guard, CPU Resume before FSDP, and TPU Skill Validator Calibration

**Status**: Accepted  
**Date**: 2026-10-18  
**Git Commit**: `6c787a1` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.3.0.dev3)  

## Context and Problem
После успешного устранения NaN в градиентах (ADR 104: 156/156 конечных градиентов во всех 3 тестах run_nan_diagnosis) запуск train_phase6_tpu_notebook.py на Kaggle TPU завершился ошибкой RESOURCE_EXHAUSTED при попытке загрузить скомпилированную программу первого шага обучения (Attempting to reserve 12.33G at the bottom of memory. There are 11.79G free). Причиной дефицита 0.54 ГБ стало наложение 6 факторов: (1) удвоенный batch_size = 64*2 = 128; (2) синхронная проверка if not torch.isfinite(loss): между Forward и Backward, разрезающая граф и удерживающая активации 40 слоев в HBM; (3) удержание словаря outputs ([B, 512, 1024]) после xm.mark_step(); (4) немедленный запуск run_validation и клонирование весов в EMATracker.apply_shadow в HBM на 1-м же шаге после Resume с шага 12000; (5) вызов model.load_state_dict после оборачивания в SpmdFullyShardedDataParallel; (6) активный run_diagnosis_first=True. Кроме того, валидатор estimate_tpu_memory.py в скилле tpu-script-crafting не распознавал тернарный ast.IfExp для batch_size и недооценивал глубину обратного прохода FP32 DUS (29 слоев от индекса 11 до 39).

## Decisions Made
1. В скилле tpu-script-crafting модернизированы verify_tpu_script.py (Level 2) и estimate_tpu_memory.py (Level 4): добавлен парсинг ast.IfExp, калибровка глубины обратного прохода FP32 ModernBERT (29 активных слоев от первого CA-слоя) и детекция 5 анти-паттернов памяти (MID_STEP_LOSS_SYNC, OUTPUTS_ALIVE_ACROSS_MARK_STEP, EMA_BACKUP_ON_DEVICE, RESUME_AFTER_FSDP, run_diagnosis_first). На исходном скрипте оценка точно воспроизвела сбой (12.35 ГБ программа при 11.67 ГБ свободных -> DANGER / OOM RISK).
2. В train_phase6_tpu_notebook.py (.py и .ipynb):
   - Возвращен batch_size = 64 и отключен run_diagnosis_first = False.
   - Загрузка чекпоинта Resume перенесена на CPU ДО model.to(device) и SpmdFullyShardedDataParallel, а шаг продолжения установлен в global_step = step + 1.
   - В EMATracker.apply_shadow бэкап весов перенесен в оперативную память хоста (param.detach().cpu().clone()).
   - Убран синхронный разрыв графа if not torch.isfinite(loss): между Forward и Backward (заменен на in-graph зануление градиентов через is_loss_finite и асинхронный контроль серии NaN в log_step).
   - Вычисление content_cos_stats(outputs), регистрация xm.add_step_closure и удаление del outputs перенесены ДО вызова xm.mark_step().

## Consequences
1. Снижение размера резервируемой XLA-программы шага обучения с 12.33 ГБ до ~5.18 ГБ (пиковое потребление 9.26 ГБ из 15.75 ГБ HBM, запас ~6.5 ГБ).
2. Исключение синхронного разрыва графа между прямым и обратным проходами и освобождение ~1.5 ГБ утечек жизненного цикла тензоров (outputs, backup весов EMA на CPU, отключение предпусковой диагностики).
3. Полная защита шардинга SPMD при возобновлении обучения с чекпоинта и исключение повторного запуска валидации/сохранения на первом шаге после Resume.
4. Точное заблаговременное обнаружение рисков переполнения HBM и анти-паттернов жизненного цикла тензоров валидаторами скилла tpu-script-crafting до отправки кода на Kaggle TPU.

## Scope & Affected Files
- `.agents/skills/tpu-script-crafting/scripts/__pycache__/estimate_tpu_memory.cpython-313.pyc`
- `.agents/skills/tpu-script-crafting/scripts/__pycache__/verify_tpu_script.cpython-313.pyc`
- `.agents/skills/tpu-script-crafting/scripts/estimate_tpu_memory.py`
- `.agents/skills/tpu-script-crafting/scripts/verify_tpu_script.py`
- `experiments/phase 6/kaggle/train_phase6_notebook.py` (Related KI: `decisions/091_global_scheduled_identity_gate.md`, `decisions/096_phase6_gradient_collapse_and_dynamic_void_penalty.md`, `decisions/097_phase6_t_global_orchestrator_and_per_sample_beta_noise.md`)
- `experiments/phase 6/tpu kaggle/__pycache__/train_phase6_tpu_notebook.cpython-313.pyc`
- `experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.ipynb`
- `experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.py` (Related KI: `decisions/097_phase6_t_global_orchestrator_and_per_sample_beta_noise.md`)
- `scratch/__pycache__/sync_phase6_notebook.cpython-313.pyc`
- `scratch/sync_phase6_notebook.py`
