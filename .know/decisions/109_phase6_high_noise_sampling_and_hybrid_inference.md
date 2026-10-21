<!-- created: 2026-10-21 -->
<!-- status: Accepted -->
# ADR 109: Phase 6 High-Noise t_global Mixture, Canvas Self-Attention LR Isolation, Prompt Dropout, and Hybrid Scheduled Inference

**Status**: Accepted  
**Date**: 2026-10-21  
**Git Commit**: `ed07de6` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.3.0.dev3)  

## Context and Problem
На чекпоинте step_49000 (Фаза 6) инструментальная оценка показала стабильный денойзинг (loss=0.0412, cos_sim=0.9588), но качество связной генерации оставалось неудовлетворительным: классический Slerp при старте с фиксированного шума seed(42) удерживал жесткий синтаксический скелет алгебраической задачи из датасета (3: Y, 4: X, **). Анализ выявил три причины: 1) равномерное сэмплирование t_global давало t=1.0 лишь в 4% случаев, из-за чего Cross-Attention почти не обучался генерации от промпта с нуля; 2) внутренний Self-Attention холста (out_proj_sa, qkv_proj_sa) в блоках CA обучался с тем же максимальным LR (2e-4), что и CA, перетягивая градиенты и вырастая по норме до 54.5 против 36.6 у CA; 3) отсутствие Prompt Dropout при обучении лишало CFG истинного безусловного профиля z_uncond.

## Decisions Made
1. Внедрить смесь сэмплирования t_global на CPU с концентрацией на сильном шуме: 20% чистый шум (t=1.00), 40% сильный шум (t in [0.72, 0.96]), 40% средний/малый шум (t in [0.04, 0.68]). 2. Ввести прогрессивное взвешивание функции потерь high_t_boost: w(t) = 0.2 + 0.8 * t, штрафующее потерю контекста на максимальном шуме в 5 раз жестче. 3. Разделить параметры в оптимизаторе AdamW: изолировать Cross-Attention (out_proj с LR=2e-4, qkv/norm_q с LR=1e-4) и прижать внутренний Self-Attention холста (out_proj_sa, qkv_proj_sa) до LR=5e-6 (снижение в 40 раз) и 12 буферов DUS до LR=5e-6. 4. Внедрить XLA-совместимый Prompt Dropout (15% батча с ca_out * 0.0) для обучения безусловного направления z_uncond. 5. В скрипте evaluate_phase6_metrics.py внедрить сэмплер hybrid (Anchor Jump на шаге t=1.00 -> 0.88 с перешумлением по spherical_noise) и Time-Scheduled CFG (спад 4.0 -> 1.0) с поддержкой вариативных seed.

## Consequences
Устраняется доминирование стартового шума epsilon_0 и привязка к синтаксическим скелетам train-выборки. Cross-Attention получает основной поток градиента на высоких шумах без конкуренции со стороны холстового Self-Attention. Модель обучается как условной, так и безусловной траекториям, обеспечивая математически корректный Classifier-Free Guidance. Код блокнота Kaggle TPU и скрипта оценки полностью синхронизирован и валидирован по чеклисту tpu-script-crafting без нарушения статического графа XLA.

## Scope & Affected Files
- `experiments/phase 6/__pycache__/evaluate_phase6_metrics.cpython-313.pyc`
- `experiments/phase 6/evaluate_phase6_metrics.py`
- `experiments/phase 6/tpu kaggle/__pycache__/train_phase6_tpu_notebook.cpython-313.pyc`
- `experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.ipynb`
- `experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.py` (Related KI: `decisions/097_phase6_t_global_orchestrator_and_per_sample_beta_noise.md`)
- `experiments/phase 6/__pycache__/test_advanced_inference.cpython-313.pyc`
- `experiments/phase 6/phase6_generation_demo_49000.html`
- `experiments/phase 6/phase6_generation_demo_49000_cfd.html`
- `experiments/phase 6/test_advanced_inference.py`
