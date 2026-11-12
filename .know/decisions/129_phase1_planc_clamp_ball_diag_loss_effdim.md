<!-- created: 2026-11-12 -->
<!-- status: Accepted -->
# ADR 129: Исправление масштаба clamp_to_unit_ball после Denoising Jittering, выделение loss/prior_diag и метрика val/effective_dim (уточнение ADR 128)

**Status**: Accepted  
**Date**: 2026-11-12  
**Git Commit**: `e845bda` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.3.3.dev3)  

## Context and Problem
Развитие и исправление багов в решениях ADR 128 (128_phase1_planc_geometry_denoising_and_quantum_logic.md):\n1. Метод GrassmannianTokenNormalizer.safe_squash (alpha = 1/sqrt(128) ≈ 0.0884) предназначен для безграничных выходов VAE. Его применение к уже нормированным клеткам V_cell + jitter в GrassmannianPhase1Model.decode() и в инъекции шума/логике inspect_manifold_and_logic.py сжимало нормы осей примерно в 11.3 раза (и матрицу Грама V V^T в ~128 раз) по сравнению с eval()-режимом.\n2. Диагональный штраф матрицы Грама без масштабирования давал малую величину (~0.006) и был смешан с off-diag в одном prior_loss.\n3. В валидации отсутствовала метрика эффективной размерности (Participation Ratio), а в inspect_manifold_and_logic.py скан покрытия не подавлял PAD-токены (w=1.0 вместо 0.001) и скан словаря покрывал только первые 25000 токенов.

## Decisions Made
1. Добавить GrassmannianTokenNormalizer.clamp_to_unit_ball(V) = V / norms.clamp(min=1.0) и использовать его вместо safe_squash в GrassmannianPhase1Model.decode() и операциях логики/шума в inspect_manifold_and_logic.py (оставив safe_squash только для pre-ball выходов VAE и randn prior-сэмплирования).\n2. Разделить compute_prior_loss(V_cell) на (off_diag_loss, diag_loss), масштабируя torch.diagonal(bar_rho, 0) * 128.0 (цель 1.0, Huber delta=0.5). В compute_phase1_loss добавить отдельный вес delta_diag=0.05 и метрики loss/prior_off, loss/prior_diag.\n3. Добавить в evaluate() накопление ковариационной статистики в float64 по не-PAD клеткам и логирование val/effective_dim (EffDim: .../128), val/prior_off и val/prior_diag.\n4. В inspect_manifold_and_logic.py применить PAD-веса (0.001 для PAD, 1.0 для остальных) в run_manifold_coverage_scan и равномерную выборку по всему словарю Tekken в run_vocabulary_distribution_scan.\n5. Переэкспортировать Kaggle-блокнот experiments/phase 1/Kaggle/train_phase1_notebook.ipynb через export_to_notebook.py.

## Consequences
- Устранен train/eval сдвиг масштаба норм клеток в GrassmannianPhase1Model.decode() и при логических/шумовых тестах.\n- Обеспечены раздельная настройка веса (delta_diag=0.05) и наблюдаемость диагональной изотропизации через loss/prior_diag, val/prior_diag и val/effective_dim.\n- Повышена репрезентативность диагностики словаря (равномерная выборка 32768 токенов по всему 131072-словарю, EffDim словаря 19.40/128) и корректность PAD-взвешивания в оценке покрытия текстов.

## Alternatives Considered & Rejected
- Использование safe_squash поверх V_cell + jitter (отвергнуто из-за сжатия норм в ~11.3x и несоответствия train/eval).\n- Суммирование немасштабированного диагонального штрафа внутри единого prior_loss без отдельного веса и логирования.

## Invariants & Rules for AI
- [MUST] Для возмущенных клеток V_cell + jitter и результатов булевой логики на Gr(8, 128) применять clamp_to_unit_ball, а не safe_squash (уточняет инвариант ADR 128).\n- [MUST] Применять safe_squash исключительно к безграничным pre-ball выходам VAE и случайным векторам при безусловном prior-сэмплировании.\n- [MUST] Отслеживать loss/prior_diag, loss/prior_off и val/effective_dim раздельно для контроля изотропизации пространства.\n- [MUST] После изменений в experiments/phase 1/train_phase1.py переэкспортировать Kaggle-блокнот через experiments/phase 1/export_to_notebook.py.

## Scope & Affected Files
- `.know/doc_state.json`
- `experiments/phase 1/Kaggle/train_phase1_notebook.ipynb`
- `experiments/phase 1/inspect_manifold_and_logic.py`
- `experiments/phase 1/train_phase1.py`

## Verification
.venv\Scripts\python.exe -m py_compile "experiments\phase 1\train_phase1.py" "experiments\phase 1\inspect_manifold_and_logic.py"; .venv\Scripts\python.exe "experiments\phase 1\export_to_notebook.py"; .venv\Scripts\python.exe "experiments\phase 1\inspect_manifold_and_logic.py" --ckpt "experiments\phase 1\checkpoints\planC_Phase 1_checkpoints_phase1_step_5000.pt"
