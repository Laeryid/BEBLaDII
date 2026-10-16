<!-- created: 2026-10-16 -->
<!-- status: Accepted -->
# ADR 105: Phase 6 RoPE in Self-Attention, Geodesic Tangent-Space Noise, Prompt Pass Truncation, and Phase 4 Core Extraction

**Status**: Accepted  
**Date**: 2026-10-16  
**Git Commit**: `2298391` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.3.0.dev3)  

## Context and Problem
1. Внутри CAPromptLayer (Phase 6) Self-Attention по холсту ответа вычислялся без RoPE, создавая позиционную слепоту. 2. В spherical_noise случайный вектор шума имел ненулевую параллельную проекцию на x0, создавая разброс угловых расстояний. 3. Прогон промпта для получения семантического кэша Z_prompt_12 (ADR 099) вызывал все 40 слоев DUS с output_hidden_states=True, создавая лишние вычисления и аллокации для слоев 12-39. 4. Проверенные компоненты диффузии Фаз 3-4 не были вынесены в основной пакет библиотеки src/beb_la_dii.

## Decisions Made
1. Внедрен XLA-friendly RotaryEmbedding в Self-Attention блок CAPromptLayer для холста ответа A. 2. В spherical_noise реализована точная геодезическая траектория через ортогонализацию случайного вектора в касательном пространстве сферы T_{x0}S^{D-1}. 3. Прогон промпта ограничен первыми 12 слоями DUS (0-11) через срез self.dus.layers[:12] в блоке try...finally. 4. Проверенные блоки Фаз 3-4 вынесены в src/beb_la_dii/model/diffusion_blocks.py и экспортированы в src/beb_la_dii/model/__init__.py.

## Consequences
Позитивные: 1. Точное относительное позиционное кодирование токенов на холсте ответа A без добавления обучаемых параметров в state_dict. 2. Устранение углового шума при диффузионном зашумлении благодаря ортогонализации Грама-Шмидта (детерминированная траектория на сфере). 3. Ускорение прогона промпта более чем в 3 раза и экономия памяти HBM за счет отсечения вычислений слоев 12-39. 4. Стандартизация проверенных блоков Фаз 3-4 в src/beb_la_dii/.

## Scope & Affected Files
- `experiments/phase 6/__pycache__/evaluate_phase6_metrics.cpython-313.pyc`
- `experiments/phase 6/evaluate_phase6_metrics.py`
- `experiments/phase 6/tpu kaggle/__pycache__/train_phase6_tpu_notebook.cpython-313.pyc`
- `experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.ipynb`
- `experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.py` (Related KI: `decisions/097_phase6_t_global_orchestrator_and_per_sample_beta_noise.md`)
- `scratch/sync_phase6_notebook.py`
- `src/beb_la_dii/model/__init__.py`
- `src/beb_la_dii/model/__pycache__/__init__.cpython-313.pyc`
- `src/beb_la_dii/utils/__pycache__/loss.cpython-313.pyc`
- `src/beb_la_dii/utils/loss.py` (Related KI: `KI_utilities.md`, `decisions/014_metrics_fix_and_delta_recovery.md`, `decisions/015_centered_cosine_similarity.md`, `decisions/016_decoupling_absolute_space.md`, `decisions/017_new_loss_terms_and_metrics.md`, `decisions/019_decoupling_intermediate_layers.md`, `decisions/022_intermediate_layers_hybrid_regularization.md`, `decisions/023_layer40_raw_topology_and_variance.md`, `decisions/024_spherical_manifold_distillation.md`, `decisions/025_centered_rkd.md`, `decisions/026_gradient_stabilization_rkd_cosine.md`, `decisions/027_huber_loss_variance_covariance.md`, `decisions/028_manual_huber_and_norm_cv_metric.md`, `decisions/029_variance_floor_and_scale_invariant_covariance.md`, `decisions/030_cov_loss_weight_increase_and_gcs_migration.md`, `KI_metrics_reference.md`)
- `src/beb_la_dii/model/__pycache__/diffusion_blocks.cpython-313.pyc`
- `src/beb_la_dii/model/diffusion_blocks.py`
