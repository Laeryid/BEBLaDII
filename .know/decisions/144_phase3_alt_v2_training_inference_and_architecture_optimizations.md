<!-- created: 2026-11-25 -->
<!-- status: accepted -->
# ADR 144: Фаза 3 Alt v2 (Plan C): оптимизации обучения (Этап 0), инференса (Этап 1) и архитектуры v2 (Этап 2)

**Status**: accepted  
**Date**: 2026-11-25  
**Git Commit**: `5d064b5` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.4.0.dev3)  

## Context and Problem
В результате глубокого аудита `experiments/phase 3/train_phase3.py` и ветки `experiments/phase 3 alt/` (см. ADR 142 и `BEYOND_DEFAULTS_OPTIMIZATION_IDEAS.md`), дополненного прямыми замерами на реальном `GrassmannianEncoder` (`axis_sigma` и распределение энергии `[sem|syn|pos]` по осям) и калибровкой `NoiseRadarHead`, выявлен ряд системных перекосов и резервов по качеству и скорости:
1. **Обучение (скрытые перекосы):** общий `clip_grad_norm_(..., 1.0)` смешивал `AdamW` и касательные градиенты `OrthMuon`; в `OrthMuon` масштаб `0.2 * sqrt(max(m, n))` давал ~4× разницу углового шага между `128×128` (`GrassSA`) и `1024×2048` (`FullSA`); `EMAModel` без warmup «перекармливал» раннюю траекторию (на шаге 500 вес шага 0 составлял `60.6%`) и использовал `lerp_`, нежелательный на XLA; `L_bary` использовал динамические `torch.where/unique/randperm` и отталкивал стоп-слова; диагональный спектр и внедиагональ уже вычисленной матрицы Грама `Gh` не регуляризовались; константа `SIGMA_DATA = 0.05` вводила в заблуждение, обозначая std шума вместо std данных.
2. **Инференс и скорость:** наивный евклидов CFG смешивал оси без учёта O(8)-выравнивания; линейная сетка `t` и постоянный CFG портили детализацию при `t < 0.25`; проекция промпта (`kv_proj/RoPE/v_gate`) пересчитывалась 40–80 раз за генерацию; `cond`/`uncond` ветки шли двумя последовательными вызовами, причём `GrassCA` вхолостую считался на `uncond` (где выход тождественно `0`); `NoiseRadarHead` (131K-словарь) гонялся по уже `locked` клеткам и `VOID`-хвостам.
3. **Архитектура `v1` vs геометрия `Gr(8, 128)`:** замороженный `GrassmannianEncoder` делит `128d` на `[96 sem | 16 syn | 16 pos]`, а его оси имеют резко убывающий спектр `axis_sigma = [0.08319, 0.06884, 0.05511, 0.04440, 0.03341, 0.02695, 0.02075, 0.01697]` (отношение `4.9×`). В `v1` головы `4×32d` резали границы блоков, `CA` впустую переносил `pos (112:128)`, `OutputHead` использовал единое `t` и скалярную проекцию, `AdaLN` в `FullSA` тратил ~6M параметров с линейным временем (`4` синусоиды), а промпт выравнивался по левому краю (`pos=0`), отдаляясь от `SEP=64` и не позволяя безопасно обрезать `PAD`-хвост.

## Decisions Made
Реализовать и зафиксировать три последовательных этапа оптимизаций (Этапы 0–2 из `BEYOND_DEFAULTS_OPTIMIZATION_IDEAS.md` §9) с сохранением обратной совместимости (`arch_version="v1"` и алиас `SIGMA_DATA = SIGMA_NOISE`):

1. **Этап 0 — Исправления обучения (`train_phase3.py`, `train_phase3_alt.py`):**
   - Переименовано `SIGMA_DATA` → `SIGMA_NOISE = 0.05`; добавлены эмпирический `DEFAULT_AXIS_SIGMA`, поосевой SD3-shift `compute_axis_t(t_mat, delta)` и `add_axis_shifted_noise`.
   - В `OrthMuon` включён сферически-корректный масштаб (`sphere_scale=True`, `sphere_c_base=2.26`), введён раздельный клиппинг `clip_phase3_grads(adamw=1.0, sphere=10.0)`.
   - `EMAModel`: добавлен `warmup=True` (`min(decay, (1+n)/(10+n))`), XLA-безопасное обновление `mul_/add_` (без `lerp_`), CPU-бэкап в `apply_shadow`, сохранение `__num_updates__` в `state_dict` (обратно совместимо).
   - `compute_phase3_loss`: добавлены `loss/spec_diag (0.05·(1-t))`, `loss/orth_offdiag (0.05·(1-t))`, `loss/anchor (0.15, SmoothL1 beta=0.01 на clean не-SEP клетках)`, поосевой `L_basis` (`basis_high_t_mode="sign_only"`), фильтрация стоп-клеток (`is_content_word_cell`, порог доли контента `≥ 0.5`) в `L_bary`/`L_adj` и статический XLA-совместимый `L_bary` (`topk` по `mask*2 + U[0,1)`).
   - Добавлены `LengthBucketBatchSampler` и `crop_batch_for_gpu(full_prob=0.25)` для ускорения на GPU/CPU (автоотключение на XLA).

2. **Этап 1 — Инференс и ускорение (`train_phase3.py`, `train_phase3_alt.py`):**
   - `procrustes_align_basis` (`8×8` SVD в `fp32`, пропуск на XLA) внутри CFG.
   - `build_t_grid` (`mode="sd3"`, `shift=0.55`) и `compute_limited_cfg_scale` (`t_range=[0.25, 0.95]` со спадом к `1.0`).
   - Однократное кэширование KV промпта (`prepare_prompt_cache` / `prepare_prompt_kv`) + единый `2B` батч `forward_cfg_batched` с пропуском `GrassmannianCABlock` для `uncond` половины на инференсе.
   - Во всех 4 сэмплерах (`euler_heun`, `rolling_wave`, `block_ar`, `crystal_radar`) подключены поосевой bridge-шаг, `active_mask=~locked` в `compute_radar_sigma`, ранний выход `if locked.all(): break`, `crystal_lock_max_t=0.15` и early-stop по устойчивому `VOID`-хвосту.
   - В `evaluate_phase3` добавлены метрики `val/orth_err`, `val/spec_l1`, `val/err_sem`, `val/err_syn`, `val/err_pos`.

3. **Этап 2 — Архитектура Phase 3 Alt v2 (`train_phase3.py`, `train_phase3_alt.py`, `matrix_layers.py`, `init_strategies.py`):**
   - Блочно-согласованные головы `sub_head_dim=16`: `8×16d` (`[6 sem | 1 syn | 1 pos]`) в `GrassSA`/`PromptAdapter` и `7×16d=112d` (без `pos`) в `GrassCA`; блочно-диагональная инициализация (`alpha=0.15`, `o_proj[112:128]*0.10`).
   - Правое выравнивание промпта к ответу (`pos=63` у последнего реального токена промпта независимо от Dynamic Crop, `SEP=64`) и `rope_grass_base=500.0`.
   - `FourierTimeEmbedding(64)` (`0.5..16` циклов) + `ConditionEmbedderV2` (`64 + 19` датчиков → `256` с `compute_sensor_importance`) + `AxisBlockAdaLN` (`r=64` с диагностикой `r_eff / k_99 / tail_ratio / dead_frac`).
   - `Energy-Gated SwiGLU` (`rad_feat=[abs_ln, rel_ln]`) в FFN `FullSA`, `GrassSA`, `GrassAlign`, `BilinearGrass`.
   - `GrassmannianOutputHeadV2`: поосевой EDM/Винер по `DEFAULT_AXIS_SIGMA`, `SIGMA_NOISE`, `axis_shift_delta=0.35`, `AxisBlockAdaLN`, `lambda_keep (bias=-3.0)`, `PHMLinear(r=8)`, `output_gain=0.1`, жёсткая защита `SEP`.
   - Адаптированы `matrix_layers.py` (`GrassAlignSABlock` `qk_dim=32 = 2×16d`) и `init_strategies.py` (`all_5` под `112d`/`8×16d`/`OutputHeadV2`).
   - Обновлены Kaggle-блокноты A/B (`export_to_notebook.py`) под `v2` с отдельными GCS-директориями `v2_safe_all5/` (`22.94M`) и `v2_control_dense_idea4/` (`31.81M`).

## Consequences
**Положительные:**
- Размер модели `standard` сокращён ещё на `6.77M` за счёт `AxisBlockAdaLN (r=64)`, `7×16d CA` и `PHMLinear(r=8)` в `OutputHeadV2`: **`22.94M`** (`v2 + safe`, **7.32×** меньше `167.9M` базовой Phase 3 и −22.8% к `v1 safe 29.71M`) и **`31.81M`** (`v2 + dense` против `38.58M` v1).
- Устранена 4× диспропорция угловых шагов `OrthMuon` между `128×128` и `1024×2048` (`ang_ratio = 1.05`), исключён «перекорм» EMA на шагах `1..200`, исключён динамический граф XLA в `L_bary`.
- Инференс ускорен за счёт однократного кэширования KV промпта (1 раз вместо 40–80), батчевого `2B` CFG с пропуском `GrassCA` на `uncond` (−50% стоимости обоих CA-слоёв, численное расхождение `≤ 1.49e-08`), маскированного `NoiseRadar (~locked)` и early-stop по `VOID`-хвосту.
- Обучение на GPU ускоряется на `1.6×–2.2×` благодаря `LengthBucketBatchSampler + GPU Dynamic Crop` при нулевом позиционном дрейфе промпта (`crop_prompt_err = 0.00e+00`) и защите от сдвига длины (`full_prob=0.25`).
- Сохранена полная обратная совместимость: `arch_version="v1"`, алиас `SIGMA_DATA = SIGMA_NOISE`, загрузка старых EMA-чекпоинтов без `__num_updates__`.

**Отрицательные / риски и ограничения:**
- Чекпоинты `v2` структурно несовместимы с `v1` → для v2 выделены отдельные GCS-директории (`v2_safe_all5/`, `v2_control_dense_idea4/`).
- Поосевой SD3-shift (`delta = 0.35`) слегка замедляет очистку младших осей `4..7` на средних `t`; введена страховка `crystal_lock_max_t = 0.15`, но итоговый баланс между `delta ∈ {0.3, 0.35, 0.5}` и порогом кристаллизации должен подтвердиться в 3000-шаговом A/B на Kaggle.
- `GPU Dynamic Crop` и `Procrustes CFG (8×8 SVD)` неприменимы в статическом XLA-графе TPU и автоматически отключаются при `device.type == "xla"`.
- `AxisBlockAdaLN` (`r=64`) может стать узким местом при масштабировании — требуется мониторить `val/adaln_r_eff_mean` и `val/adaln_k99_max` на валидации.

## Alternatives Considered & Rejected
1. **Агрессивный SD3-shift (`delta = 0.7..1.0`):** по результатам калибровки `NoiseRadarHead` при `delta >= 0.7` младшие оси `4..7` оставались зашумлёнными в момент раннего лока радара, жертвуя спроектированными осями ради ~5–10% экономии шагов. Выбран консервативный `delta = 0.35` (рабочий диапазон `0.3..0.5`) со страховкой `crystal_lock_max_t = 0.15`.
2. **Единый `grad_clip = 1.0` на все параметры:** душил касательные обновления `OrthMuon` при всплесках `AdamW` и ломал геометрию шага; заменён на раздельный `clip_phase3_grads`.
3. **Прямая обрезка холста без доли полных батчей (`full_prob = 0`):** вызывала риск сдвига распределения длины при инференсе на полном холсте `N = 128`; зафиксировано `full_prob = 0.25`.
4. **Левое выравнивание промпта (`pos = 0`) в `v2`:** при `GPU Dynamic Crop` меняло относительное расстояние между концом промпта и `SEP`; заменено на правое выравнивание (`pos = 63`).
5. **Ранний выход `if not has_cond` в `GrassCA` во время обучения:** ломал граф DDP (`find_unused_parameters=False`) и XLA; ограничен строго инференсом (плюс `infer_cond_half=True` в `2B` батче).

## Invariants & Rules for AI
- [MUST] Использовать `SIGMA_NOISE = 0.05` для std шума (не путать с поосевым std данных `DEFAULT_AXIS_SIGMA`); оставлять алиас `SIGMA_DATA = SIGMA_NOISE` только для обратной совместимости.
- [MUST] При `axis_shift_delta > 0` использовать `crystal_lock_max_t <= 0.15` в `crystal_radar_sample`, чтобы не фиксировать клетки до очистки младших осей `4..7`.
- [MUST] Применять пропуск `uncond` ветки в `GrassmannianCABlock` и `Procrustes CFG` строго на инференсе (`not training` и не `xla`), а `GPU Dynamic Crop` — строго вне XLA и с `full_prob >= 0.25`.
- [MUST] В `EMAModel` использовать XLA-безопасные `mul_/add_` (без `lerp_`), CPU-бэкап в `apply_shadow` и сохранять `__num_updates__` в `state_dict`.
- [MUST] Хранить чекпоинты `arch_version="v2"` в отдельных GCS-каталогах (`v2_safe_all5/`, `v2_control_dense_idea4/`), не смешивая с `v1`.
- [MUST NOT] Не использовать динамические `torch.where` / `torch.unique` / `torch.randperm` в `L_bary` — только статический `topk`.
- [MUST NOT] Не модифицировать замороженную Систему 1 (`storage/components/encoder`, `storage/components/decoder`) — маскирование `NoiseRadarHead` по `~locked` выполняется на уровне `compute_radar_sigma`.

## Scope & Affected Files
- `experiments/phase 3/train_phase3.py` (Related KI: `KI_experiments.md`, `decisions/034_angular_loss_for_output_projector.md`, `decisions/035_centered_cosine_and_contrastive_neighbor_loss.md`, `decisions/036_replacing_infonce_with_rkd.md`, `decisions/037_phase3_failure_and_plan_b_skip_connections.md`)
- `experiments/phase 3 alt/train_phase3_alt.py`
- `experiments/phase 3 alt/matrix_layers.py`
- `experiments/phase 3 alt/init_strategies.py`
- `experiments/phase 3 alt/test_phase3_alt.py`
- `experiments/phase 3 alt/export_to_notebook.py`
- `experiments/phase 3 alt/BEYOND_DEFAULTS_OPTIMIZATION_IDEAS.md`

## Verification
1. py_compile всех 5 изменённых .py файлов (EXIT 0).
2. .venv\Scripts\python.exe -B "experiments\phase 3 alt\test_phase3_alt.py" — 12/12 PASSED (standard v2 dense=31.81M, safe=22.94M; leak_diff=0.00e+00; t_local_leak=0.00e+00; 0 zero-grad на всех 7 матричных вариантах v2; OrthMuon ang_ratio=1.05; procrustes_err=1.04e-07; 2B+KV_diff<=1.49e-08; crop_prompt_err=0.00e+00; регрессия v1).
3. End-to-end smoke-test: .venv\Scripts\python.exe -B "experiments\phase 3 alt\train_phase3_alt.py" --smoke-test --device cpu --arch-preset tiny --arch-version v2 --matrix-mode safe --init-mode all_5 --gpu-dynamic-crop --ce-weight 0.05 --eval-batches 1 --cfg-dsim-batches 1 --gen-steps 2 (EXIT 0: init sph_err=2.38e-07, cfg_leak=0.00e+00, новые val-метрики orth/spec/blk/adaln r_eff, оба сэмплера и сохранение чекпоинта).
4. Предстоит финальный A/B на Kaggle (3000 шагов): train_phase3_alt_safe_notebook.ipynb (v2 safe 22.94M + all_5) vs train_phase3_alt_control_notebook.ipynb (v2 dense 31.81M + idea4).
