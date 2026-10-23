<!-- created: 2026-10-23 -->
<!-- status: Accepted -->
# ADR 110: Phase 6 True CA-Adapter: Value Payload Isolation, Unified Module Gating, and SA Parameter Reinitialization

**Status**: Accepted  
**Date**: 2026-10-23  
**Git Commit**: `a4ba4be` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.3.0.dev3)  

## Context and Problem
На чекпоинте step_76000 анализ выявил, что внутренний Self-Attention (SA) в блоке CAPromptLayer обучался по входу A_mid = A + warmup_factor * ca_out. Поскольку A_mid на ~90% состоял из базового холста A, SA получал градиент на безусловное денойзирование A (рост нормы out_proj_sa до 55.3 в слое 36). Кроме того, при prompt_dropout в обучении занулялся только ca_out, заставляя SA работать в одиночку без промпта, а при инференсе CFG (_disable=True) отключались оба слоя сразу, что приводило к искажению формулы CFG. В преддверии добавления параллельных модулей (CA_Memory/CLM) требовалось сделать SA строго адаптером/координатором значений CA без паразитного влияния на базовый поток DUS.

## Decisions Made
1. Перевести CAPromptLayer на архитектуру 'Истинного CA-Адаптера':
   - Маршрутизация (Q_sa, K_sa) формируется от контекста холста и инъекции (A + warmup_factor * ca_out) с модуляцией AdaLN(t_emb) и RoPE.
   - Значения (V_sa) формируются ИСКЛЮЧИТЕЛЬНО из ca_out: v_sa = F.linear(norm2(ca_out) * scale_sa, w_v), без участия сырого холста A и аддитивного shift_sa.
2. Объединить CA и SA в единую капсулу:
   - Суммарная добавка delta = ca_out + sa_out.
   - Маска prompt_mask (prompt_dropout) и флаг _disable управляют дельтой целиком.
3. Уравнять скорость обучения (LR) SA с CA:
   - out_proj и out_proj_sa: lr = 2e-4.
   - Внутренние QKV, нормы и AdaLN: lr = 1e-4.
   - Буферные слои DUS: lr = 5e-6.
4. Внедрить маркер sa_adapter_version: 2:
   - При первом возобновлении (Resume) с чекпоинта v1 (включая step_76000) автоматически реинициализировать параметры SA (out_proj_sa = 0.0) в модели и тени EMA/PACE. Сохранить все обученные веса CA и DUS на 100%.
   - В новые чекпоинты сохранять sa_adapter_version: 2.
5. Синхронизировать train_phase6_tpu_notebook.py, train_phase6_tpu_notebook.ipynb и evaluate_phase6_metrics.py.

## Consequences
Положительные:
1. Физическая изоляция: SA физически не способен пропустить в residual stream векторы холста A в обход CA, так как V_sa формируется строго из ca_out.
2. Исключение паразитного обучения: при prompt_dropout (15%) обнуляется весь комбинированный вектор (ca_out + sa_out), благодаря чему SA обучается только в паре с CA.
3. Полная эквивалентность обучения и CFG: _disable=True на инференсе точно соответствует состоянию prompt_mask=0 в обучении.
4. Бесшовный переход при Resume: автоматический сброс устаревших весов SA (v1) предотвращает скачки градиентов и влияние старой нормы 55.3 на шаге 76001.
5. Готовность к параллельным модулям: CAPromptLayer теперь представляет собой полностью капсулированный оператор с нулевым фоновым воздействием при отключении.

Компромиссы:
1. SA-адаптер обучается с нуля (out_proj_sa=0.0), однако начинает сразу в правильной геометрии без сопротивления паразитных весов.
2. LR для внутренних параметров SA и CA теперь объединены (1e-4 для inner, 2e-4 для out_proj).

## Scope & Affected Files
- `.know/doc_state.json`
- `experiments/phase 6/__pycache__/evaluate_phase6_metrics.cpython-313.pyc`
- `experiments/phase 6/evaluate_phase6_metrics.py`
- `experiments/phase 6/tpu kaggle/__pycache__/train_phase6_tpu_notebook.cpython-313.pyc`
- `experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.ipynb`
- `experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.py` (Related KI: `decisions/097_phase6_t_global_orchestrator_and_per_sample_beta_noise.md`)
- `experiments/phase 6/phase6_generation_demo_76000.html`
- `ideas/Generative modeling for time series via Schr/302/250odinger bridge.pdf`
- `ideas/Geometric knowledge spaces, Attention pairing and Architectures.pdf`
