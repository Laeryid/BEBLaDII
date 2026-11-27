<!-- created: 2026-11-27 -->
<!-- status: accepted -->
# ADR 145: Квантизация замороженной Системы 1 (GrassmannianEncoder и GrassmannianDecoder): пресеты весов, общий INT8-словарь и автопереключение по режиму и ускорителю

**Status**: accepted  
**Date**: 2026-11-27  
**Git Commit**: `783ce50` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.4.0.dev3)  

## Context and Problem
Замороженная Система 1 (`GrassmannianEncoder` и `GrassmannianDecoder`, Plan C, `storage/components/`) в исходном FP32 занимает `817.6 MB` на диске (`encoder/weights.pt` `615.7 MB` + `decoder/weights.pt` `201.9 MB`), из которых `536.9 MB` (~66%) приходится на матрицу словаря `131,072 × 1024`. С учётом копии `lm_head_unit` (`256 MB` в `FastVocabOptimizer`, ADR 143) суммарный объём Системы 1 в памяти при обучении `experiments/phase 3 alt` достигал `~1.07 GB`, ограничивая запас VRAM на Kaggle T4 (16 GB), создавая нагрузку на пропускную способность памяти в `Encoder`/`NoiseRadarHead` и замедляя CPU-инференс.

Анализ реальных весов показал гладкие распределения без критических выбросов: `embeddings.weight ∈ [-0.1754, 0.1828]`, все `SphereLinear.q_weight ∈ [-0.65, 0.70]`, `row_scale ∈ [-0.70, 1.85]` (с отрицательными значениями на `dec_smooth_layers`). При этом архитектурное сжатие Декодера было осознанно исключено, чтобы не переобучать Систему 1. Требовалось реализовать post-training квантизацию без переобучения и внешних зависимостей (`bitsandbytes`, устаревшего `torch.ao.quantization`), с соблюдением FP32-инвариантов Plan C (ADR 125 — FP32 логиты/logsumexp `lm_head`; ADR 126 — FP32 `gram_mlp` и чувствительные модули), с автоматическим переключением под режим (`train` / `inference`) и ускоритель (`CPU` / `CUDA Turing` / `CUDA Ampere+` / `TPU XLA`), а также со 100% обратной совместимостью.

## Decisions Made
Реализована post-training квантизация замороженной Системы 1 с сохранением предквантованных весов в `storage/components/` и автоматическим выбором политики по режиму и ускорителю:

1. **Единый модуль квантизации на чистом PyTorch (`storage/components/quantization.py`, формат `planc_sys1_quant_v1`):**
   - `BakedLinear`: запекание `SphereLinear` (`W = q_weight * row_scale[:, None]`) в `FP16/BF16/FP32` без дублирования `row_scale`.
   - `Int8RowwiseLinear`: симметричная построчная INT8-квантизация запечённых весов (`int8` + `fp32 row_scale`, корректно учитывает отрицательные `row_scale`), с ленивым `oneDNN` prepack (`torch._make_per_channel_quantized_tensor` + `torch.ops.quantized.linear_prepack` / `linear_dynamic`) на CPU при `no_grad` и дифференцируемой on-the-fly деквантизацией на GPU / при включённом autograd.
   - `Int8RowwiseEmbedding`: общий построчный INT8-словарь (`131,072 × 1024`, ~`134 MB`) с `row_scale` и предвычисленным `unit_scale = row_scale * inv_norms`, разделяемый между `encoder.embeddings`, `decoder.lm_head` и `NoiseRadarHead` (`compute_radar_top2_cos`).
   - `preserve_sensitive_fp32`: жёсткая защита от `.half()/.bfloat16()` через `_apply` для чувствительных островков (`BlockDiagonalVAEHead`, `syntax_proj`, `SetCrossAttentionUnpacker.gram_mlp` по ADR 126, `NoiseRadarHead` MLP, `noise_cond_proj`, `RMSNorm`), плюс FP32 для `eigh`, `canonicalize_cell_basis` и логитов/logsumexp `lm_head` (ADR 125).

2. **Предквантованные пресеты в `storage/components/` (`build_quantized_weights.py`, исходные `weights.pt` не изменены):**
   - `encoder/`: `weights_fp16.pt` (`308.1 MB`), `weights_emb_int8_fp16.pt` (`181.6 MB`), `weights_int8.pt` (`156.0 MB`).
   - `decoder/`: `weights_fp16.pt` (`101.7 MB`), `weights_int8.pt` (`51.9 MB`).
   - Карта пресетов добавлена в `encoder/config.json` и `decoder/config.json` (`"quantized_weights"`). При отсутствии готового `.pt` (например, в Kaggle Dataset) `get_encoder` / `get_decoder` автоматически выполняют on-the-fly квантизацию из `weights.pt`.

3. **Автопереключение политики (`System1QuantPolicy` / `resolve_system1_quant_policy`):**
   - `train + CUDA`: `emb_int8` (общий INT8-словарь + запечённые линейные слои в `FP16` на Turing `CC < 8.0` по ADR 126 или `BF16` на Ampere+ `CC >= 8.0`).
   - `train + CPU`: `emb_int8` (INT8-словарь + `oneDNN` NoiseRadar, запечённые FP32 линейные слои).
   - `train/inference + TPU (XLA)`: `half` (запечённые `BF16`, статические формы под MXU).
   - `inference + CPU/CUDA`: `int8` (общий INT8-словарь + все `SphereLinear` в построчном INT8).

4. **Интеграция и обратная совместимость:**
   - В `get_encoder` / `get_decoder` дефолт остаётся `quant_mode="fp32"` (старые пайплайны не затронуты).
   - `decoder.tie_embeddings(...)` принимает `Int8RowwiseEmbedding`, `nn.Embedding` или `Tensor` без дублирования памяти; для INT8 выставляет `decoder.lm_head_inv_norms = unit_scale`, сохраняя совместимость с внешними вызовами (`compute_radar_sigma` в `train_phase3.py`).
   - В `FastVocabOptimizer` (`experiments/phase 3 alt/vocab_optim.py`) при INT8-словаре пропускается выделение 256 MB копии `lm_head_unit`, `NoiseRadar` делегируется в `Int8RowwiseEmbedding.compute_radar_top2_cos`, а `_OnlineChunkedCELinearFn` принимает `lm_head_row_scale` с chunked FP32-деквантизацией (ADR 125).
   - В `train_phase3_alt.py` (и перегенерированных Kaggle-блокнотах) добавлены `--sys1-quant` (дефолт `"auto"`) и `--sys1-mode` (дефолт `"train"`).

## Consequences
**Положительные:**
- Сжатие весов Системы 1 на диске в `3.93×` (`817.6 MB → 207.9 MB` в пресете `int8`), а в RAM/VRAM при обучении — с `~1.07 GB` до `~230 MB` (освобождается `~800+ MB` VRAM на Kaggle T4 за счёт INT8-словаря, FP16/BF16 линейных слоёв и устранения 256 MB дубля `lm_head_unit`).
- Практически нулевая потеря качества на реальных весах (`8` примеров × `24` клетки, `ablation_val_cache_32.pt`):
  - `train_auto (emb_int8)`: Encoder `SimGr = 0.999984`, `cos = 0.999983`; Decoder `TokAcc = 0.9753` (`agree = 1.0000`), `CE = 0.0940` vs `0.0942`, косинус градиента к латенту `grad_cos = 0.999893`, сдвиг NoiseRadar `d_sig = (1.83e-5, 1.54e-5)`.
  - `inf_auto (int8)`: Encoder `SimGr = 0.999892`; Decoder `TokAcc = 0.9766` (`agree = 0.9986`), `CE = 0.0942`, `d_sig = (2.26e-3, 1.17e-3)`, `grad_cos = 0.996825`.
- Ускорение на CPU (idle, `oneDNN`): Encoder **3.18×** (`153.5 → 48.3 ms` в `inf_auto`, `1.31×` в `train_auto`), `NoiseRadar` **3.02×** (`1409 → 467 ms`), полный Decoder + head **1.75–1.90×** (`1934 → 1016..1107 ms`).
- Нулевые внешние зависимости, переносимый `.pt` формат, 100% обратная совместимость (`quant_mode="fp32"` по умолчанию) и автоматический fallback из `weights.pt` на Kaggle.

**Отрицательные / ограничения:**
- Реальный прирост по `ms/step` и VRAM на Kaggle T4 / Ampere / TPU ещё не снимался в боевом прогоне (проверено локально на CPU и юнит-тестами политик).
- На GPU без кастомных INT8 GEMM-ядер (Triton/cuBLASLt) `Int8RowwiseLinear` деквантует веса на лету — на `train + CUDA` это компенсировано выбором `emb_int8` (линейные слои в FP16/BF16 под Tensor Cores).
- Основное время шага обучения `phase 3 alt` на GPU занимает диффузионное ядро и 131K-проекция словаря, поэтому ускорение полного шага обучения на GPU будет умеренным (главный эффект — освобождение VRAM и снижение трафика памяти).
- При будущем переобучении весов Системы 1 потребуется повторно запустить `build_quantized_weights.py` для обновления предквантованных `.pt` пресетов.

## Alternatives Considered & Rejected
1. **Архитектурное сжатие Декодера (уменьшение ширины/глубины/словаря):** отклонено по требованию задачи, так как требует полного переобучения замороженной Системы 1.
2. **Внешние квантизаторы (`bitsandbytes`, `torchao`, `torch.ao.quantization`):** отклонены из-за несовместимости с PyTorch/XLA (TPU), проблем сборки на Windows/CPU и статуса deprecated у `torch.ao.quantization` в PyTorch 2.11.
3. **Наивная INT8-деквантизация на CPU (`q_weight.float() * row_scale` без `oneDNN`):** оказалась медленнее FP32 (`19.3 ms` против `12.8 ms` на слой); заменена ленивым `oneDNN` prepack (`3.8×` на линейных слоях, `6.5×` на чанках словаря).
4. **INT8 для линейных слоёв при обучении на GPU/TPU (`train + CUDA/XLA`):** без кастомных INT8 GEMM-ядер уступает Tensor Cores в FP16/BF16 на backward `L_ce` и мешает MXU на XLA; ограничено режимом `inference`.
5. **Квантизация чувствительных модулей (`BlockDiagonalVAEHead`, `gram_mlp`, `NoiseRadarHead` MLP, `lm_head` logits):** отклонена во избежание нарушения инвариантов ADR 125 и ADR 126.

## Invariants & Rules for AI
- [MUST] Оставлять исходные FP32 файлы `storage/components/encoder/weights.pt` и `storage/components/decoder/weights.pt` нетронутыми и сохранять дефолт `quant_mode="fp32"` в `get_encoder` / `get_decoder` для обратной совместимости.
- [MUST] Сохранять FP32-островки (`BlockDiagonalVAEHead`, `syntax_proj`, `gram_mlp`, `NoiseRadarHead` MLP, `noise_cond_proj`, `RMSNorm`, `eigh`, `canonicalize_cell_basis`, логиты и `logsumexp` `lm_head`) через `preserve_sensitive_fp32` и chunked FP32-деквантизацию (ADR 125, ADR 126).
- [MUST] В режиме `auto` выбирать `emb_int8` для `train + CUDA/CPU`, `half (BF16)` для `TPU (XLA)` и `int8` для `inference + CPU/CUDA`.
- [MUST] При привязке INT8-словаря через `decoder.tie_embeddings` переиспользовать общий буфер `Int8RowwiseEmbedding` без создания 256 MB дубля `lm_head_unit` в `FastVocabOptimizer`.
- [MUST] Сохранять автоматический on-the-fly fallback из `weights.pt`, если предквантованный `.pt` файл отсутствует (например, на Kaggle Dataset).
- [MUST NOT] Не использовать `oneDNN` `linear_dynamic` при включённом autograd (`torch.is_grad_enabled()`) — backward `L_ce` обязан идти через дифференцируемую ветку.
- [MUST NOT] Не вводить внешние зависимости (`bitsandbytes`, deprecated `torch.ao.quantization`), несовместимые с Windows/CPU или PyTorch/XLA.

## Scope & Affected Files
- `storage/components/quantization.py` (Related KI: `KI_model_core.md`)
- `storage/components/build_quantized_weights.py` (Related KI: `KI_model_core.md`)
- `storage/components/encoder/__init__.py` (Related KI: `KI_model_core.md`)
- `storage/components/encoder/config.json` (Related KI: `KI_model_core.md`)
- `storage/components/encoder/README.md` (Related KI: `KI_model_core.md`)
- `storage/components/decoder/__init__.py` (Related KI: `KI_model_core.md`)
- `storage/components/decoder/config.json` (Related KI: `KI_model_core.md`)
- `storage/components/decoder/README.md` (Related KI: `KI_model_core.md`)
- `experiments/phase 3 alt/vocab_optim.py`
- `experiments/phase 3 alt/train_phase3_alt.py`
- `experiments/phase 3 alt/test_phase3_alt.py`
- `experiments/phase 3 alt/Kaggle/train_phase3_alt_safe_notebook.ipynb`
- `experiments/phase 3 alt/Kaggle/train_phase3_alt_control_notebook.ipynb`

## Verification
1. py_compile всех 7 изменённых/созданных .py файлов (EXIT 0).
2. .venv\Scripts\python.exe -B "experiments\phase 3 alt\test_phase3_alt.py" — 13/13 PASSED (тест [13/13]: auto-политики, BakedLinear, Int8RowwiseLinear lin_cos=0.999961, Int8RowwiseEmbedding emb_err=1.66e-03, защита _apply, INT8 FastVocabOptimizer ce_gcos=0.999982, radar_max_err=0.0036).
3. Бенчмарк на реальных весах (8 примеров × 24 клетки, ablation_val_cache_32.pt, idle CPU):
   - train_auto (emb_int8): Encoder 1.31x (153.5->117.3ms, SimGr=0.999984, cos=0.999983), NoiseRadar 3.02x (1409->467ms, d_sig=1.83e-5/1.54e-5), Decoder 1.90x (1934->1016ms, TokAcc=0.9753, agree=1.0000, CE=0.0940 vs 0.0942, grad_cos=0.999893).
   - inf_auto (int8): Encoder 3.18x (153.5->48.3ms, SimGr=0.999892), NoiseRadar 3.02x (466ms), Decoder 1.75x (1107ms, TokAcc=0.9766, agree=0.9986, CE=0.0942, grad_cos=0.996825).
4. Регенерация Kaggle-блокнотов через export_to_notebook.py (safe и control, по 23 ячейки).
