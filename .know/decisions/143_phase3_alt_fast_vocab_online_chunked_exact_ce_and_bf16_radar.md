<!-- created: 2026-11-25 -->
<!-- status: accepted -->
# ADR 143: Оптимизация словаря 131K × 1024 (FastVocabOptimizer): принятие NoiseRadar BF16 и случайного B_ce, отказ от Sampled Softmax (SVD rank=64) в пользу 1-проходного Online Chunked Exact CE

**Status**: accepted  
**Date**: 2026-11-25  
**Git Commit**: `526bd0e` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.4.0.dev3)  

## Context and Problem
В Plan C замороженная матрица словаря `lm_head_weight (131,072 × 1024)` (привязанная к `encoder.embeddings.weight`) создаёт два узких места при обучении Phase 3:
1. `NoiseRadarHead.compute_dict_stats` на каждом шаге переводит словарь в `float32`, умножает на `inv_norms` и ищет Top-2 косинус по `131K` словам.
2. Сквозной токенный CE-лосс (`ADR 140/141`) прогоняет весь батч через 4-слойный `ModernBERT(1024)` (`decoder.decode_hidden`) и затем материализует полную матрицу логитов `(M, 131072)` (`>530 MB` на fwd+bwd при `M=512` в FP32 по ADR 125).

Первоначальная версия `FastVocabOptimizer` (`vocab_optim.py`) использовала:
- преднормированный `lm_head_unit` в `BF16` для `NoiseRadarHead`;
- `Batch-Shared Stratified Sampled Softmax` (`SVD rank=64`, `K_hard=1536`, `K_rand=2048`);
- отбор `B_ce = ceil(B * 0.5)` примеров с максимумом активных токенов (`max_active`) до `decode_hidden`.

Критический аудит выявил, что юнит-тест `[9]` (`grad_cos = 1.0000`) проверял синтетическую матрицу **ранга 32**, а CPU-бенчмарк (`grad_cos = 0.9997`) — равномерные логиты (`CE ≈ ln V`). Специально написанный эксперимент `exp_vocab_fidelity.py` на **реальном полноранговом словаре `131,072 × 1024`** и реальных скрытых состояниях декодера Системы 1 (`16` примеров, `1947` токенов, `t ∈ {0.00, 0.25, 0.50}`) дал критические результаты:
- **`NoiseRadar BF16 vs FP32` прошёл проверку безупречно:** `argmax agree = 1.0000` на всех трёх `t`, потеря `TokAcc = 0.0 п.п.`, `CE` совпал до 3 знаков, средняя ошибка `cos_gap` всего `1.6% / 2.0% / 3.3%` (`Δσ_tok = 9.7e-6 .. 8.3e-5`).
- **`Sampled CE (SVD rank=64, K_hard=1536)` НЕ прошёл критерий приёмки (`grad_cos ≥ 0.98`, `|bias| ≤ 2%`):** на `t = 0.00` `bias = -34.0% / -40.3%`, `grad_cos = 0.717 / 0.691`, `Recall@10 = 0.204`; на рабочем пике веса `w_t` при `t = 0.25` `bias = -30.2% / -32.5%`, `grad_cos = 0.763 / 0.758`, `Recall@10 = 0.263`, инфляция `TokAcc: 0.946 → 0.977`; при `t = 0.50` `bias = -10.7%`, `grad_cos = 0.912`, `TokAcc: 0.530 → 0.607`.
- Отбор `B_ce` по `max_active` вносил систематический перекос, лишая короткие ответы CE-градиента.

## Decisions Made
В `experiments/phase 3 alt/vocab_optim.py` (`FastVocabOptimizer`) и `train_phase3_alt.py` приняты следующие правила работы со словарём `131K × 1024` (100% static shapes, обратимое подключение без модификации `storage/components/decoder`):

1. **Принято: `NoiseRadarHead` на преднормированном `lm_head_unit` в `gemm_dtype` (`4` чанка по `32768`).**
   - При `env.use_amp=True` (GPU/TPU) используется нативный `amp_dtype` (`BF16/FP16`), при `env.use_amp=False` (CPU без AVX-512) — `float32` (избегая 2× замедления от софтовой эмуляции BF16 на CPU).
2. **Принято: случайный статический отбор `B_ce = ceil(B * ce_batch_fraction)` до `decoder.decode_hidden` (`--ce-batch-select random`, по умолчанию).**
   - Любой пример с ≥1 активным токеном (`t < ce_t_max`) получает равную вероятность (`priority = (active > 0) * 2.0 + U[0, 1)`), устраняя перекос против коротких ответов при сохранении статических форм XLA.
3. **Принято (новый дефолт `--ce-mode exact_chunked`): 1-проходный точный `Online Chunked Exact CE` (`_OnlineChunkedCELinearFn`, `chunk=16384`).**
   - За один проход по 8 фиксированным чанкам `16384` строго в `float32` (ADR 125) одновременно вычисляет точный `logsumexp`, точный `argmax` по всем `131,072` словам и точное направление градиента `grad_dir = E_p[W] − W_tgt` (сохраняется один тензор `(M, 1024)` = `2 MB`; пиковая память логитов `~32 MB` вместо `>530 MB`; суммарно 2 GEMM за fwd+bwd без повторного пересчёта в backward).
   - `torch.svd_lowrank` переведён в ленивый режим (`_ensure_svd_sketch()`): в дефолтном `exact_chunked` не вычисляется при старте.
   - В `compute_decoder_ce_loss` добавлен флаг `return_acc_tensor=True` для исключения `tok_acc_t.item()` host-device sync при переносе на PyTorch/XLA (TPU).
4. **Понижено до опционального (`--ce-mode sampled`):** `Batch-Shared Stratified Sampled Softmax (SVD rank=64)` снят с дефолта и оставлен только для отладки/будущих экспериментов с `svd_rank ≥ 256` и увеличенным `K_hard`.

## Consequences
**Положительные:**
- **0.00% смещения CE и `grad_cos = 1.000000`** на полноранговом словаре `131,072 × 1024` в остром режиме (проверено на полноранговой матрице в `test_phase3_alt.py [9/10]`: `max_err = 3.41e-6`, `grad_cos = 1.000000`, 100% совпадение `argmax`).
- Пиковая память на логиты CE снижена с `>530 MB` до **`~32 MB`** без потери FLOPs (2 GEMM на словарь суммарно за fwd+bwd) и без хранения логитов в autograd-графе.
- `NoiseRadarHead` в `BF16` подтверждён на реальном декодере без потери качества (`argmax agree = 1.0000` на всех `t`, `ΔTokAcc = 0.0`, `Δσ_tok < 1e-4`), не требует дополнительных файлов в датасете (`lm_head_unit` строится на лету за `<0.3s` на GPU).
- Случайная фильтрация `B_ce` (`ce_batch_select="random"`) экономит ~2× на 4-слойном `decoder.decode_hidden` без систематического смещения против коротких ответов.
- Дефолтный `exact_chunked` не запускает `torch.svd_lowrank` при старте (мгновенная инициализация, `-16 MB` RAM).

**Отрицательные / ограничения:**
- По чистым FLOPs на CPU `exact_chunked` медленнее `sampled (rank=64)`, который давал `4.0×` на CPU; но на GPU/TPU для `M=512` плотный chunked GEMM по `131K` занимает миллисекунды, а основное время CE-ветки экономится фильтрацией `B_ce` до `decode_hidden` и `NoiseRadar BF16`.
- Ускорения `NoiseRadar BF16` (`4–8×` прогноз) на реальных GPU/TPU ещё не бенчмаркились (измерено только на CPU в FP32: `1.33×`).
- `_OnlineChunkedCELinearFn` рассчитан строго на замороженную `lm_head_weight` (`requires_grad=False`, при попытке обучения весов словаря явно выбрасывает `RuntimeError`).

## Alternatives Considered & Rejected
1. **`Batch-Shared Stratified Sampled Softmax (SVD rank=64, K_hard=1536, K_rand=2048)` как дефолт:** отклонён по результатам `exp_vocab_fidelity.py` — на реальной полноранговой матрице `131K × 1024` и острых распределениях (`t ≤ 0.25`) даёт смещение CE `-30%..-40%` (эффект неравенства Йенсена для `log Z` + нехватка слотов общего пула `1536/512 = 3` слота/токен и низкий `Recall@10 = 0.20..0.28`), разворачивая градиент (`grad_cos = 0.69..0.76`).
2. **Отбор `B_ce` по максимуму активных токенов (`ce_batch_select="max_active"`):** отклонён как дефолт — систематически отсекает короткие ответы и примеры с большим средним `t` от CE-обучения.
3. **2-проходный checkpointed chunked CE:** отклонён в пользу 1-проходного онлайн-накопления `E_p[W]`, так как 2-проходный вариант пересчитывает `h @ W_c^T` на backward (3 GEMM на словарь вместо 2).
4. **Принудительный `BF16` для `NoiseRadar` на CPU (`use_amp=False`):** замедлил CPU в `2×` (`2.31s → 4.55s`) из-за софтовой эмуляции BF16; заменён на `env.amp_dtype if env.use_amp else torch.float32` (`1.33×` ускорение на CPU в FP32, нативный BF16/FP16 на GPU/TPU).
5. **IVF / FAISS / ранний выход `if cos > 0.45` в общем пайплайне:** отклонены из-за несовместимости с XLA/TPU (динамические формы, host-device sync, дорогой gather).

## Invariants & Rules for AI
- [MUST] Использовать `--ce-mode exact_chunked` (или другой 100% точный CE) по умолчанию для боевого обучения с `ce_weight > 0`, сохраняя `float32` вычисление логитов по ADR 125.
- [MUST] Использовать `--ce-batch-select random` при `ce_batch_fraction < 1.0`, чтобы короткие ответы получали CE-градиент с той же вероятностью, что и длинные.
- [MUST] Выбирать `gemm_dtype = env.amp_dtype if env.use_amp else torch.float32` для `FastVocabOptimizer`, не форсируя `BF16` на CPU без аппаратного AVX-512 BF16.
- [MUST] При переносе `compute_decoder_ce_loss` на PyTorch/XLA (TPU) передавать `return_acc_tensor=True` для исключения `.item()` sync в шаге обучения.
- [MUST NOT] Не оценивать корректность аппроксимаций `lm_head` только на синтетических низкоранговых матрицах (`rank=32`) или равномерных логитах (`CE ≈ ln V`) — любая аппроксимация словаря обязана проходить `exp_vocab_fidelity.py` на реальном полноранговом словаре при `t ∈ {0.00, 0.25, 0.50}` с порогом `grad_cos ≥ 0.98`, `|bias| ≤ 2%`.
- [MUST NOT] Не сравнивать `train/ce_tok_acc` из sampled-пула напрямую с точным `val/tok_acc` (инфляция `+2..+8 п.п.`).

## Scope & Affected Files
- `experiments/phase 3 alt/vocab_optim.py`
- `experiments/phase 3 alt/exp_vocab_fidelity.py`
- `experiments/phase 3 alt/train_phase3_alt.py`
- `experiments/phase 3 alt/test_phase3_alt.py`
- `experiments/phase 3 alt/RADICAL_COMPRESSION_IDEAS.md`

## Verification
1) .venv\Scripts\python.exe -B "experiments\phase 3 alt\exp_vocab_fidelity.py" --samples 16 --max-cells 32 --repeats 3 (подтвердил NoiseRadar BF16: agree=1.0000 на t=0.00/0.25/0.50; выявил провал Sampled CE rank=64 при t<=0.25: bias -30..-40%, gcos 0.69..0.76); 2) .venv\Scripts\python.exe -B "experiments\phase 3 alt\test_phase3_alt.py" (10/10 PASS; тест [9/10] на полноранговой матрице в остром режиме для exact_chunked: max_err=3.41e-6, grad_cos=1.000000, 100% argmax match, B_ce=2/4).
