<!-- created: 2026-11-03 -->
<!-- status: Accepted -->
# ADR 118: Phase 6 CAPromptLayer v4: Token-Centered Contrast, Cross-Gated GeGLU, and Zero-Vector Attention Sink

**Status**: Accepted  
**Date**: 2026-11-03  
**Git Commit**: `807db2e` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.3.2.dev2)  

## Context and Problem
Анализ генераций чекпоинтов 111k, 115k и 129k выявил циклический повтор слов промпта (keff ~4.6-6.1) и стирание токена <|void|> при инференсе с CFG=4.0. Инспекция внутренних пространств DUS на слоях 12, 24, 36 выявила сжатие холста в общий конус (81-98.8% амплитуды занимает общий сдвиг mu_A), из-за чего RMSNorm ослеплял внимание к индивидуальному контексту токенов, а RoPE циклически вращал константу. При этом в Softmax Cross-Attention сумма весов всегда равна 1.0, принудительно внедряя слова промпта в хвостовые void-позиции.

## Decisions Made
Внедрена архитектура CAPromptLayer v4 (sa_adapter_version=4): 1) Token-Centered Contrast: вычитание центроида холста mu_A перед RMSNorm для выделения чистого индивидуального контекста токенов; 2) Cross-Gated GeGLU: матрица W_canvas_gate подмешивает состояние холста в вентиль gate, сохраняя val строго от промпта (исключает утечку холста при CFG off); 3) Zero-Vector Attention Sink: обучаемый ключ-сток k_sink с нулевым значением v_sink=0 позволяет голове внимания занулять инъекцию промпта на служебных связках и в void-хвосте, с масштабированием p_active после RMSNorm; 4) Стабилизация оптимизатора: lr_ca_out снижен до 1e-4 с weight_decay=1e-3, ужесточен порог спайков is_safe (grad_norm <= 2.0); 5) Рекомендован возврат к обучению со step_111000.pth (max_resume_step=111000).

## Consequences
Положительные: 1) Устранено ослепление Cross-Attention общим конусом холста (выделен индивидуальный контекст delta_A); 2) Механизм k_sink позволяет отключать инъекцию промпта в хвосте ответа, сохраняя нативный токен void_token при CFG=4.0; 3) Полная обратная совместимость со step_111000.pth (zero-init W_canvas_gate гарантирует идентичный старт); 4) Защита оптимизатора предотвращает деградацию L36 SA Ratio и разгон амплитуд. Компромиссы: незначительное увеличение параметров (+32K на k_sink и W_canvas_gate).

## Scope & Affected Files
- `experiments/phase 6/Demos/blind_judge_key_129k.json`
- `experiments/phase 6/Demos/demo_129k_baseline.html`
- `experiments/phase 6/Demos/demo_129k_smartcrop_soft.html`
- `experiments/phase 6/__pycache__/eval_phase6_134k_suite.cpython-313.pyc`
- `experiments/phase 6/__pycache__/evaluate_phase6_metrics.cpython-313.pyc`
- `experiments/phase 6/evaluate_phase6_metrics.py`
- `experiments/phase 6/kaggle/__pycache__/train_phase6_notebook.cpython-313.pyc`
- `experiments/phase 6/kaggle/train_phase6_notebook.ipynb`
- `experiments/phase 6/kaggle/train_phase6_notebook.py` (Related KI: `decisions/091_global_scheduled_identity_gate.md`, `decisions/096_phase6_gradient_collapse_and_dynamic_void_penalty.md`, `decisions/097_phase6_t_global_orchestrator_and_per_sample_beta_noise.md`)
- `experiments/phase 6/phase6_generation_demo.html`
- `experiments/phase 6/score_generations_129k.json`
- `experiments/phase 6/tpu kaggle/__pycache__/train_phase6_tpu_notebook.cpython-313.pyc`
- `experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.ipynb`
- `experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.py` (Related KI: `decisions/097_phase6_t_global_orchestrator_and_per_sample_beta_noise.md`)
- `experiments/phase 6/Demos/blind_judge_key_111k.json`
