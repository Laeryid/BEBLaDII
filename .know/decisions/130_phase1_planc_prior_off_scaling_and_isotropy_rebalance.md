<!-- created: 2026-11-13 -->
<!-- status: Accepted -->
# ADR 130: Масштабирование prior_off и ребалансировка prior_diag после спектральной диагностики step_28500 (уточнение ADR 129)

**Status**: Accepted  
**Date**: 2026-11-13  
**Git Commit**: `8932724` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.3.3.dev3)  

## Context and Problem
После внедрения ADR 129 и прогрева cell_jitter_std (коммит 06a7448) обучение Phase 1 Plan C до шага 29000 показало устойчивую сходимость val/recon (~1.52 на 15k -> ~1.35 на 29k) и линейный рост val/effective_dim (~15.3 на 10k -> ~21.3 на 29k, ~+0.32 измерения/1k шагов). Однако диагностика чекпоинта step_28500 через inspect_manifold_and_logic.py и поканальный спектральный анализ на val.parquet выявили причины замедления изотропизации:
1. Топ-64 измерения концентрируют 99.1% энергии на текстах (EffDim=22.81-23.15/128) и 99.8% на словаре (EffDim=17.21/128).
2. Покоординатные дисперсии в sem [0:96] равномерны (var=0.00104..0.00257), но каналы сильно скоррелированы между собой (mean(|r|)=0.121, p95(|r|)=0.294, max(|r|)=0.503; sem-only EffDim=28.84/96).
3. Блок syn [96:112] просажен до 5.65% энергии (EffDim=2.17/16), а блок pos [112:128] забирает 26.81% энергии (EffDim=3.90/16) из-за перекоса в последней RoPE-паре (126, 127) с частотой pi/2 (bar_rho*128 = 9.47 и 9.83).
4. В compute_prior_loss внедиагональные элементы bar_rho не были масштабированы на 128 (mean(|rho_ij|)=0.00092 < delta=0.01), что при (2/128) и delta=0.05 давало вклад в total_loss ~6e-6 (практически нулевой градиент на декорреляцию), а delta_diag=0.05 с Huber delta=0.5 давал слабый вклад ~0.006 (~0.4% от recon).

## Decisions Made
1. В compute_prior_loss (experiments/phase 1/train_phase1.py) масштабировать внедиагональные элементы матрицы Грама аналогично диагонали: off_diag_scaled = bar_rho[triu] * 128.0, и заменить (2/128)*sum Huber(delta=0.01) на F.huber_loss(off_diag_scaled, 0, delta=0.5, reduction="mean").
2. В diag_loss поднять порог Huber с delta=0.5 до delta=1.0 для усиления отклика на умеренные отклонения и выбросы в RoPE-хвосте.
3. В compute_phase1_loss увеличить веса: delta (prior_off) с 0.05 до 0.5 и delta_diag (prior_diag) с 0.05 до 0.25.
4. В TRAINING_CONFIG увеличить max_steps с 50000 до 500000 для обеспечения длинного горизонта дообучения без преждевременного затухания cosine LR.
5. Переэкспортировать Kaggle-блокнот experiments/phase 1/Kaggle/train_phase1_notebook.ipynb через export_to_notebook.py.

## Consequences
- На контрольном прогоне (batch_size=4, seq_len=256, CPU, step_28500) вклад 0.5 * loss/prior_off составил 0.00758 (~0.4% от total_loss) вместо прежних ~6e-6, вклад 0.25 * loss/prior_diag — 0.04192 (~2.1% от total_loss), grad_norm = 5.5232 (без взрыва градиентов).
- Создано устойчивое давление на декорреляцию каналов sem [0:96], подъем syn [96:112] и выравнивание перекоса RoPE-пары (126, 127) для роста effective_dim к целевым >= 80-90.
- Изменился масштаб логируемых метрик loss/prior_off и val/prior_off (~1e-4 -> ~1.5e-2), что необходимо учитывать при сопоставлении графиков до и после шага 28500.
- Время 1 шага forward+backward на локальном CPU (batch_size=4, seq_len=256, float32) замерено на уровне ~24 сек.

## Alternatives Considered & Rejected
- Продолжение обучения на 100-200k шагов без изменения compute_prior_loss: при вкладе prior_off ~6e-6 и delta_diag=0.05 рост effective_dim упирается в межканальные корреляции sem [0:96] и доминирование RoPE-пары (126, 127).
- Принудительное обнуление или жесткое нормирование pos [112:128]: отвергнуто в пользу мягкой единой регуляризации всех 128 каналов через prior_diag и prior_off без поломки кодирования позиций в окне 4->1.

## Invariants & Rules for AI
- [MUST] В compute_prior_loss обе компоненты (off_diag и diag) должны вычисляться на масштабированной матрице 128 * bar_rho в едином масштабе O(1).
- [MUST] При сопоставлении метрик loss/prior_off и val/prior_off до и после данного изменения учитывать переход от немасштабированного sum к 128 * bar_rho с reduction="mean".
- [MUST] После изменений в experiments/phase 1/train_phase1.py переэкспортировать Kaggle-блокнот через experiments/phase 1/export_to_notebook.py (наследуется от ADR 129).

## Scope & Affected Files
- `experiments/phase 1/train_phase1.py`
- `experiments/phase 1/Kaggle/train_phase1_notebook.ipynb`

## Verification
.venv\Scripts\python.exe -m py_compile "experiments\phase 1\train_phase1.py"; .venv\Scripts\python.exe "experiments\phase 1\export_to_notebook.py"; .venv\Scripts\python.exe "experiments\phase 1\inspect_manifold_and_logic.py" --ckpt "experiments\phase 1\checkpoints\planC_Phase 1_checkpoints_phase1_step_28500.pt"
