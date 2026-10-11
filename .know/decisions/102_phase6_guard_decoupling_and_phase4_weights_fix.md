<!-- created: 2026-10-11 -->
<!-- status: Accepted -->
# ADR 102: Phase 6 In-Graph Guard Decoupling, Full Phase 4 Backbone & Diffusion Modules Loading, and GCS Resiliency

**Status**: Accepted  
**Date**: 2026-10-11  
**Git Commit**: `5fc6cab` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.3.0.dev3)  

## Context and Problem
На шагах ~380-400 обучения Phase 6 TPU оптимизатор AdamW полностью прекратил обновлять параметры, модель за 2000 шагов не обучилась, а метрики валидации застыли на 0.6015 (прирост CA строго 0.000). Анализ выявил три причины: (1) в коммите ADR 101 для градиентов модели ошибочно применили условие is_safe = is_finite & (grad_norm <= 5.0) вместо is_finite, из-за чего клипированные до 1.0 градиенты 175M параметров занулялись при естественном превышении нормы 5.0; (2) веса чекпоинта Фазы 4 (phase4_step_85995.pth) не загружались в self.dus из-за отсутствия префикса dus., а модули диффузии (adaLN_attn, adaLN_mlp, t_proj_*) выбрасывались и оставались случайным шумом; (3) падение на вызове gsutil приводило к ошибке при проверке чекпоинтов на Kaggle и сбросу обучения на шаг 0.

## Decisions Made
1. Разделить защиту градиентов оптимизатора и защиту тени EMA: градиенты модели зануляются строго по is_finite (NaN/Inf), так как clip_grad_norm_ уже гарантирует масштаб <= 1.0. Порог аномалий для изоляции EMA/PACE поднят до 50.0.
2. Исправить маппинг весов Phase 4: загружать бэкбон непосредственно в self.dus.load_state_dict, а также полностью восстанавливать все обученные модули диффузии (adaLN_attn_ema, adaLN_mlp_ema, t_proj_global_ema, t_proj_token_ema, t_joint_proj_ema).
3. Добавить проверки shutil.which('gsutil') и вынести экспорт GCP SA credentials в начало main() для бесперебойной работы GCS клиента на Kaggle.

## Consequences
Модель гарантированно выполняет градиентные шаги при нормах градиентов здорового трансформера масштаба 175M параметров, оставаясь на 100% защищенной от NaN/Inf. Бэкбон DUS и все 40 слоев AdaLN стартуют с полноценного фундамента сферической диффузии Фазы 4, а листинг и загрузка чекпоинтов с GCS на Kaggle изолированы от отсутствия gsutil в PATH.

## Scope & Affected Files
- `experiments/phase 6/tpu kaggle/__pycache__/train_phase6_tpu_notebook.cpython-313.pyc`
- `experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.ipynb`
- `experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.py` (Related KI: `decisions/097_phase6_t_global_orchestrator_and_per_sample_beta_noise.md`)
- `scratch/sync_phase6_notebook.py`
