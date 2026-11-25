<!-- created: 2026-11-25 -->
<!-- status: poc -->
# ADR 142: Фаза 3 Alt (Plan C): 2D-Грассманианово ядро (GrassSA/GrassCA/PerAxisGRNv3), матричные оптимизации (safe), умная инициализация (all_5) и A/B-протокол с CFG-ΔSim

**Status**: poc  
**Date**: 2026-11-25  
**Git Commit**: `526bd0e` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.4.0.dev3)  

## Context and Problem
Базовое диффузионное ядро `experiments/phase 3/train_phase3.py` (~167.9M параметров) склеивает 8 ортогональных базисных векторов ячейки Грассманиана $\mathrm{Gr}(8, 128)$ в плоский вектор `1024d` на каждом слое: матрицы `1024 × 1024` и FFN `1024 × 2816` вслепую перемешивают оси, `PromptAdapter` съедает `23.6M` параметров, а 3 тяжёлых `CAPromptBlock` требуются для удержания сигнала промпта. При этом 86% весов компактной версии (`33.1M` из `38.58M` в `standard`) всё равно концентрируется в 3 слоях `FullSABlock`.

Кроме того, на старте обучения без геометрической инициализации модель тратит тысячи шагов на освоение базовых операций (копирование из промпта, ортогонализация осей, поосевая фильтрация шума с учётом убывающего спектра $\sigma_0 > \dots > \sigma_7$), а стандартная метрика `val/cfg_contrast = ||V̂_cond − V̂_uncond||_F` измеряет только расхождение веток, но не гарантирует, что условная ветка ближе к цели `V_0`.

## Decisions Made
Создана изолированная экспериментальная ветка `experiments/phase 3 alt/` (без модификации базовой `experiments/phase 3/` и замороженной Системы 1 `storage/components/`) со следующими решениями:

1. **Гибридное ядро `GrassmannianAltDiffusionCore` (`train_phase3_alt.py`):**
   - Разделение геометрии в $\mathbb{R}^{128}$ (`GrassmannianSABlock` в режимах `GrassFull`/`GrassLocal(r=6)`, 2× `GrassmannianCABlock` с `GrassmannianPromptAdapter`) и глобального синтеза/памяти в $\mathbb{R}^{1024}$ (3× `FullSABlock`).
   - 3-слотовый поосевой DCA-проброс `PerAxisGRNv3` поверх `[x_in, h_ca, x_prev]` с `SiLU`-инициализацией (`w=0, b=[0,0,1]` → строгое тождество на шаге 0 и ненулевой градиент с 1-го шага, без смешивания 8 осей).
2. **Матричные оптимизации (`matrix_layers.py`, дефолт `--matrix-mode safe`):**
   - `PHMLinear` ($\sum_{m=1}^r A_m X F_m^\top$, $r=4$, 2 плотных GEMM без материализации $1024 \times 1024$, нативный блочно-диагональный старт $\alpha=0.25$) применяется **только в проекциях `Q/KV/O` блоков `FullSABlock`**, а `SwiGLU FFN` (`1024 → 2048 → 1024`) сохраняется плотным как память.
   - `GrassAlignSABlock` (точный Грассманиановый скор $\|Q_i K_j^\top\|_F^2 / 8$ на $N \times N = 128 \times 128$ softmax + Procrustes-роутер $\tilde{M}_{ij}$) и `BilinearGrassSABlock` (левые $8 \times 8$ матрицы `axis_mix`, $+128$ параметров/слой).
   - Итоговый размер `standard + safe`: **29.71M** параметров (против `38.58M` `dense` и `167.9M` базовой Phase 3).
3. **Умная инициализация весов (`init_strategies.py`, дефолт `--init-mode all_5 --wiener-damp 0.7`):**
   - 5 стратегий (`idea1_ca_copy`, `idea2_sa_gs`, `idea3_full_blkdiag`, `idea4_wiener_head` с демпфированием `0.7` против стартового скачка `L_energy`, `idea5_kmeans_ffn` для плотных `1024d` и `128d` FFN).
   - Применяется строго **до** создания `EMAModel` и пропускается при `--resume-ckpt`.
4. **Честная валидация и A/B-протокол («3000 шагов правды»):**
   - Введена метрика полезности условия **`CFG-ΔSim = SimGr(V̂_cond, V_0) − SimGr(V̂_uncond, V_0)`** на фиксированном шуме (`t ∈ {0.50, 0.75}`).
   - Скомпилированы два независимых Kaggle GPU блокнота (`export_to_notebook.py` → `Kaggle/train_phase3_alt_{safe,control}_notebook.ipynb`) с раздельными GCS-папками и WandB-именами: боевой претендент `standard + safe + all_5 (damp=0.7)` против консервативного контроля `standard + dense + idea4_wiener_head (damp=0.7)` на 3000 шагов с живой генерацией `t = 1 → 0` каждые 500 шагов.

## Consequences
**Положительные:**
- Параметры ядра сокращены в **5.65×** (`167.9M → 29.71M` в `standard + safe`, ~56 MB в BF16), на CPU (`tiny`, кэш латентов, `batch=4`) ускорение `1.41×` против `ref_dense` (`6.96s → 4.95s/шаг`).
- Сохранены 100% инвариантов Plan C (`sph_err ~ 2e-7`, `cfg_leak = 0.00e+00`, `SEP t=0` тождество, 0 zero-grad параметров на 1-м шаге во всех 7 вариантах).
- `CFG-ΔSim` даёт честный критерий полезности промпта, независимый от шумового разброса веток.
- Раздельные блокноты и GCS-директории позволяют провести чистый параллельный A/B-прогон на Kaggle 2xT4.

**Отрицательные / риски и ограничения (по критическому аудиту):**
- 40-шаговые абляции (`tiny`, 1 seed, 32 val-примера, `t=0.50`) имеют низкую статистическую мощность: отличия `±0.002..0.006` по `Val Loss` и `±0.004` по `Train AUC` находятся на уровне батчевого шума (`0.34–0.63`). Реально подтверждены лишь крупные стартовые эффекты `idea4` (`-0.0227`), `idea5` (`-0.0152`) и сжатие/скорость на CPU.
- Выигрыш `idea4` — это фора аналитического линейного денойзера на старте, а не доказательство лучшей асимптотики; на 40 шагах `TokAcc_low` падает у всех вариантов (`0.78–0.85 → 0.26–0.47`), а `TokAcc_hi = 0.000` (`CE_hi ≈ 12.1 > ln V`).
- Боевая связка `standard + safe + all_5 (damp=0.7)` ещё не проверена на длинной дистанции — статус ADR остаётся `poc` до прохождения 3000-шагового A/B против контроля `dense + idea4_wiener_head`.
- Ускорение PHM на GPU/TPU не измерялось: вторая часть `PHMLinear` (`axis_mix`: `32 → 8`) плохо ложится на тайлы TPU MXU `128×128` и на GPU/TPU даст меньший выигрыш, чем `1.41×` на CPU.

## Alternatives Considered & Rejected
1. **`--matrix-mode ultra` (PHM также в SwiGLU FFN, `10.14M` на `tiny` / `~12M` на `standard`) как дефолт:** отклонён из-за худшего `Val Loss` (`+0.0048`, 6-е место из 7 в `matrix_ablation_tiny`), более медленного спада `L_energy` (`0.990 vs 0.936`) и просадки критичной для генерации с нуля метрики `SimGr_hi (t=0.75)` у всех PHM-вариантов (`0.143–0.145` против `0.149–0.150` у плотной модели). Оставлен опционально для отдельной проверки на длинной дистанции.
2. **Недемпфированный `idea4_wiener_head (damp=1.0)`:** давал резкий стартовый скачок `Val Energy = 2.49` (против `0.76`) и провал `TokAcc_low` на шаге 0 (`0.608 vs 0.788`); демпфирование `wiener_damp = 0.5–0.7` устранило провал (`0.608 → 0.850–0.876` на шаге 0).
3. **Оценка CFG только по норме разности `||V̂_cond − V̂_uncond||_F`:** ввела в заблуждение в `init_ablation_tiny` (`all_5` показал `7.8×` рост нормы при снижении `SimGr_mid = 0.423` ниже `baseline 0.426`). Заменена дополнением `CFG-ΔSim`.
4. **Одиночный блокнот без контрольного прогона:** отклонён — без контрольного `dense + idea4_wiener_head` невозможно отличить вклад оптимизаций от эффекта самой архитектуры на 3000 шагах.

## Invariants & Rules for AI
- [MUST] Сохранять `FullSABlock` SwiGLU FFN плотным в дефолтном режиме `--matrix-mode safe` до доказательства отсутствия потолка памяти у `ultra` на дистанции 3000+ шагов.
- [MUST] Применять `apply_init_strategy` строго ДО инициализации `EMAModel` и пропускать при `--resume-ckpt`; в DDP делать `dist.broadcast` весов с rank 0.
- [MUST] Отправлять `axis_mix` (`PHMLinear` / `BilinearGrassSABlock`) в `AdamW (wd=0)`, `score_temp` — в `scale_1d`, а `SphereLinear.q_weight` — в `OrthMuon (sphere=True)`.
- [MUST] Оценивать силу кондиционирования по `val/cfg_dsim_t50` / `val/cfg_dsim_t75` (`SimGr(cond) − SimGr(uncond) > 0`), а не только по норме разности `val/cfg_contrast`.
- [MUST NOT] Не модифицировать `experiments/phase 3/` и замороженные `storage/components/` при экспериментах в ветке `phase 3 alt`.
- [MUST NOT] Не принимать архитектурные решения по 40-шаговым дельтам `< 0.01` на 1 seed и 32 примерах или по `VOID F1` на ранних шагах без A/B-контроля с живой генерацией.

## Scope & Affected Files
- `experiments/phase 3 alt/train_phase3_alt.py`
- `experiments/phase 3 alt/matrix_layers.py`
- `experiments/phase 3 alt/init_strategies.py`
- `experiments/phase 3 alt/exp_init_ablation.py`
- `experiments/phase 3 alt/exp_matrix_ablation.py`
- `experiments/phase 3 alt/test_phase3_alt.py`
- `experiments/phase 3 alt/export_to_notebook.py`
- `experiments/phase 3 alt/ARCHITECTURE_AND_IDEAS.md`

## Verification
.venv\Scripts\python.exe -B "experiments\phase 3 alt\test_phase3_alt.py" (10/10 PASS: инварианты DCA/CA/SEP, OrthMuon/EMA, 4 сэмплера, все 7 матричных вариантов) и .venv\Scripts\python.exe -B "experiments\phase 3 alt\train_phase3_alt.py" --smoke-test --arch-preset standard --matrix-mode safe --init-mode all_5 --wiener-damp 0.7 (EXIT 0). Финальная верификация (перевод из poc в accepted): A/B-прогон train_phase3_alt_safe_notebook.ipynb vs train_phase3_alt_control_notebook.ipynb на 3000 шагов с живой генерацией t=1→0.
