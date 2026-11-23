<!-- created: 2026-11-23 -->
<!-- status: accepted -->
# ADR 141: Фаза 3 Plan C: Ограничение L_CE по уровню шума (t < 0.50), ослабление стоп-слов и отключение CE на Rollout против Unigram Mode Collapse (уточнение ADR 140)

**Status**: accepted  
**Date**: 2026-11-23  
**Git Commit**: `ea5a61e` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.4.0.dev3)  

## Context and Problem
После внедрения сквозного токенного `Cross-Entropy` (`L_CE`, `ce_weight = 0.15`) через замороженный Декодер (ADR 140) на прогоне Фазы 3 (шаги 7500 → 11500) зафиксированы деградация генерации и градиентное плато:

1. **Unigram Mode Collapse (запятые, пробелы, `\n\n`, `и`, `**`):**
   - В `compute_decoder_ce_loss` / `compute_decoder_ce_loss_tpu` маска `t_mat[:, 1:] > 1e-3` включала CE на всём диапазоне шума, включая $t \in [0.75, 1.0]$ и `Two-Step Rollout`.
   - При $t \ge 0.75$ отношение сигнал/шум $\text{SNR}(t) = ((1-t)/t)^2 \le 0.11$, распределение $P(x_0 \mid V_t, Q)$ сильно мультимодально, и минимум Cross-Entropy смещает $\hat{V}_0$ в униграммный центроид корпуса.
   - В окне валидации `hi` ($t \in [0.75, 0.95]$) на шагах 10k → 11.5k `val/ce_loss_hi` снижался (`6.05 → 5.93`), `val/tok_acc_hi` рос (`0.176 → 0.184`), но одновременно `val/sim_gr_hi` **падал** (`0.258 → 0.254`, обвал с `0.446` до CE), общий `Sim_Gr` упал с `0.955` до `0.909`, `radar` ухудшился с `0.032` до `0.077`, а `loss/bary` вырос (`0.005 → 0.007`).
2. **Доминирование CE в градиенте и конфликт весов:**
   - Вклад `0.15 * l_ce ≈ 0.855` составлял ~70% от `loss/total = 1.236`, подавляя геометрические лоссы (`L_gr`, `L_basis`, `L_energy`) при `clip_grad_norm_(1.0)`.
   - В пулинге Энкодера стоп-слова и пунктуация имеют вес `0.02`, тогда как в `L_CE` они имели вес `1.0` наравне со смысловыми токенами.
3. **Замедление шага и плато:**
   - Время шага на 2xT4 выросло с `7–8s` до `18–22s`, а train-лоссы (`loss/ce`, `loss/gr`, `loss/basis`, `train/ce_tok_acc`) вышли на горизонтальное плато.

## Decisions Made
Уточнить расчёт сквозного токенного `L_CE` (ADR 140) синхронно в GPU-версии (`experiments/phase 3/train_phase3.py`) и TPU-версии (`experiments/phase 3/train_phase3_tpu.py`):

1. **Ограничение CE по уровню шума и гладкое затухание веса:**
   - Активировать `L_CE` строго в окне низкого/среднего шума $t_i \in (10^{-3}, \text{ce\_t\_max})$ с `ce_t_max = 0.50` (`--ce-t-max`).
   - Взвешивать по-токенный CE гладкой функцией $w_t(t_i) = \left(\text{clamp}\left(\frac{\text{ce\_t\_max} - t_i}{\text{ce\_t\_max}}, 0, 1\right)\right)^2$ (нулевой вес на границе $t = 0.50$, максимальный при $t \to 0$).
2. **Ослабление стоп-слов и пунктуации в CE:**
   - Передавать `stopword_mask` на устройстве (`sw_dev`) в `compute_decoder_ce_loss` / `compute_decoder_ce_loss_tpu` с понижающим весом `ce_stopword_weight = 0.25` (`--ce-stopword-weight`).
3. **Исключение `Two-Step Rollout` из CE:**
   - На GPU: не вызывать `compute_decoder_ce_loss` при `use_rollout=True` (и возвращать `0` без backward через декодер, если нет активных токенов с $t_i < 0.50$).
   - На TPU: передавать статичную булеву маску `ce_rollout_sample_mask = (torch.arange(batch_size, device=device) >= b_roll)` в `sample_mask`, сохраняя единый XLA-граф без перекомпиляций.
4. **Ребалансировка веса CE:**
   - Снизить дефолтный `ce_weight` с `0.15` до `0.05` (в `parse_args` и `TRAINING_CONFIG` обеих версий).
5. **Откат чекпоинта:**
   - Возобновлять обучение с чистого чекпоинта до включения неограниченного CE (`step_9500`, если не перезаписан, иначе `step_5500`).

## Consequences
**Положительные:**
- Устраняется unigram mode collapse на высоком шуме ($t \in [0.5, 1.0]$): верхняя часть траектории диффузии снова оптимизируется исключительно гладкими геометрическими лоссами (`L_gr`, `L_basis`, `L_energy`), возвращая `Sim_Gr` к уровню `0.95+` и `radar` к `~0.03`.
- CE работает строго по назначению — как хирургическая доводка токенных границ декодера в области уже выбранной моды ($t < 0.50$).
- Понижение веса стоп-слов (`ce_stopword_weight = 0.25`) устраняет конфликт с двухуровневым пулингом Энкодера и не даёт модели снижать CE за счёт пунктуации и пробелов.
- На GPU пропуск backward через декодер при `use_rollout=True` и на high-noise микро-батчах заметно ускоряет шаг обучения.
- На TPU статичная булева маска `ce_rollout_sample_mask` сохраняет единый XLA-граф без перекомпиляций.

**Отрицательные / компромиссы:**
- Чекпоинты, обученные со старым неограниченным CE (step 7500–11500), имеют деградировавшие геометрические метрики и испорченные моменты/EMA; рекомендуется откат на чистый чекпоинт до включения CE (`step_9500`, если не перезаписан, иначе `step_5500`).
- На шагах с преимущественно высоким шумом эффективный размер выборки CE меньше `ce_max_tokens`.

## Alternatives Considered & Rejected
1. **Исключение только точки $t = 1.0$ (оставить CE для $t \in (0, 1.0)$):** отклонено, поскольку при $t \in [0.75, 0.99]$ сигнал составляет 0.3–10% энергии (что подтверждено метриками `val/tok_acc_hi = 0.184` и `val/ce_loss_hi = 5.93` в окне `[0.75, 0.95]` без $t=1.0$) и первые 8–10 шагов сэмплера продолжали бы сваливаться в запятые и `\n\n`.
2. **Продолжение обучения с шага 11500:** отклонено из-за уже деградировавшей геометрии (`Sim_Gr_hi = 0.254`, `radar = 0.077`) и смещённых состояний EMA/оптимизаторов.

## Invariants & Rules for AI
- [MUST] Сквозной `L_CE` через замороженный Декодер применяется только при $t_i < \text{ce\_t\_max}$ ($\le 0.50$) с гладким затуханием веса к `0` на границе $\text{ce\_t\_max}$.
- [MUST NOT] Нельзя включать `L_CE` на высоком шуме ($t > 0.50$) или на примерах `Two-Step Rollout`, где предсказание модели ещё не выбрало моду ответа.
- [MUST] В TPU-версии исключение rollout и фильтрация по $t_i < \text{ce\_t\_max}$ обязаны выполняться через статичные булевы маски и веса без `.item()` и `torch.where(mask)[0]`.
- [MUST] Вес стоп-слов/пунктуации в `L_CE` должен быть понижен (`ce_stopword_weight < 1.0`), чтобы не противоречить пулингу Энкодера и не провоцировать unigram mode collapse.

## Scope & Affected Files
- `experiments/phase 3/train_phase3.py` (Related KI: `KI_experiments.md`, `decisions/034_angular_loss_for_output_projector.md`, `decisions/035_centered_cosine_and_contrastive_neighbor_loss.md`, `decisions/036_replacing_infonce_with_rkd.md`, `decisions/037_phase3_failure_and_plan_b_skip_connections.md`)
- `experiments/phase 3/train_phase3_tpu.py` (Related KI: `KI_experiments.md`)

## Verification
- `py_compile` на `experiments/phase 3/train_phase3.py` и `experiments/phase 3/train_phase3_tpu.py` — PASSED.
- TPU-аудит (`tpu-script-crafting`): Level 1 (Syntax) PASSED, Level 2 (AST Semantic Audit) PASSED, Level 4 (HBM Budget: 2.92 / 15.75 GB) PASS.
- Юнит-тест `compute_decoder_ce_loss` и `compute_decoder_ce_loss_tpu` (с `fake_torch_xla`): нулевой лосс при $t = 0.75 \ge 0.50$, положительный взвешенный лосс при $t = 0.25 < 0.50$, нулевой градиент по `V_0_hat[0]` при `sample_mask[0] = False`.
- CPU smoke-test `train_phase3.py --smoke-test --device cpu`: 2 шага обучения прошли штатно (`ce=0.376 -> 0.301`, `gr=0.6256 -> 0.6981`).
