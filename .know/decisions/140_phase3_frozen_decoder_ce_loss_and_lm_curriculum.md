<!-- created: 2026-11-22 -->
<!-- status: accepted -->
# ADR 140: Фаза 3 Plan C: Сквозной токенный Cross-Entropy (L_CE) через замороженный Декодер, чистый Two-Step Rollout (alpha=1.0) и языковой Curriculum

**Status**: accepted  
**Date**: 2026-11-22  
**Git Commit**: `1f72534` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.4.0.dev3)  

## Context and Problem
Диагностика чекпоинта `phase3_step_9500.pt` (после 4000 шагов с ADR 139: без `t_local`, с `elastic_drop_prob=0.0` и `Two-Step Rollout`) показала, что датчики `s0..s7` заработали корректно (`every_2nd_cell` поднял `tok_acc` шумных ячеек в 8× с 2.2% до 18.4%, а `CFG` в роллауте вырос в 15× до `0.497`), однако генерация с нуля осталась слоговой кашей, а `val/cfg_contrast` к шагу 9500 снова снизился (`0.075 -> 0.019`).

Два контрольных языковых теста на `step_9500` вскрыли корневую проблему самой функции потерь:
1. **Тест 1-cell Cloze (126 чистых GT-ячеек, зашумлена ровно 1 ячейка):** на детерминированных позициях (`'«Аладдина' [???] 'Историю'`, таргет `'», «'`) модель предсказала `' arкto'` (`max_p = 0.13`, `ent = 6.53`, но при этом `sim_gr = 0.302` и `radar = 0.048`).
2. **Тест продолжения после 50% чистого префикса:** после `'«Короля Льва» и «'` (где следующее слово `'Криминальное'` буквально содержится в промпте) модель предсказала `'Sйр'` (`max_p = 0.23`, `sim_gr = 0.309`).

Геометрическая регрессия `L_basis + L_gr` в $\mathbb{R}^{8 \times 128}$ не различает валидное слово и размытый центроид с высокой энтропией, поскольку замороженный `GrassmannianDecoder` не участвовал в функции потерь. Дополнительно подмешивание `(1 - alpha) * V_0` (до 35% точного таргета) в `Two-Step Rollout` и высокая доля легкого шума $t < 0.5$ позволяли модели откатываться к локальной фильтрации белого шума.

## Decisions Made
1. **Добавлен сквозной токенный Cross-Entropy лосс `L_CE` через замороженный `GrassmannianDecoder` (`compute_decoder_ce_loss`):**
   - `PromptAnswerDataset.__getitem__` дополнительно возвращает `ans_targets = a_pack["targets"]` (`-100` на VOID-хвосте, `PAD_TOKEN_ID = 10` на внутренних пустых слотах смешанных ячеек).
   - Веса `GrassmannianDecoder` остаются строго замороженными (`eval()`, `requires_grad=False`), но прямой проход `decoder.decode_hidden(V_0_hat[:, 1:])` выполняется с градиентом по `V_0_hat` через `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`.
   - Для ограничения пиковой VRAM (<= 268 MB на логиты в FP32) логиты `decoder.get_tied_head_logits` и `F.cross_entropy` вычисляются на случайной подвыборке до `ce_max_tokens = 512` активных контентных токенов (`t_i > 1e-3 & ans_targets != -100`) на микро-батч (2048 токенов на шаг оптимизатора).
   - Вес `ce_weight = 0.15` подключается с линейным прогревом `ce_warmup_steps = 300` от `start_step` для мягкого перехода при штатном возобновлении из GCP (`resume_ckpt="auto"`).
2. **Языковая перебалансировка `Two-Step Rollout` и `sample_mixed_noise_schedule`:**
   - В `build_rollout_noisy_canvas` удалено подмешивание чистого таргета (`alpha = 1.0`, чистый самочерновик `V_draft`).
   - В `sample_mixed_noise_schedule` усилены языковые режимы: `25%` High-Noise, `20%` Shifted Flat (`Sigmoid(N(1.0, 1.0))`, медиана `~0.73`), `20%` Causal Prefix / Block-AR (50% с $t = 1.0$ на следующем блоке), `20%` Cloze & Crystal Anchors (50% с $t = 1.0$ на замаскированных клетках), `15%` Rolling Wave.
3. **Прямые токенные метрики валидации (`evaluate_phase3`):**
   - Добавлены и логируются в WandB `val/tok_acc_low`, `val/tok_acc_hi` и `val/ce_loss_hi`.

## Consequences
- **Положительные:**
  - Ядро диффузии впервые получает прямой токенный градиент по всему словарю Tekken (131,072 токенов), штрафующий размытые барицентры с высокой энтропией и выталкивающий предсказания $\hat{V}_0$ на многообразие реальных слов.
  - За счет `torch.utils.checkpoint.checkpoint(decoder.decode_hidden, ...)` и подвыборки до `ce_max_tokens = 512` активных токенов на микро-батч дополнительный расход VRAM на `cuda:0` (T4 15 GB) ограничен ~350–450 MB.
  - В WandB появились прямые языковые метрики (`loss/ce`, `train/ce_tok_acc`, `val/ce_loss_hi`, `val/tok_acc_hi`, `val/tok_acc_low`), исключающие иллюзию прогресса по одному только `Sim_Gr`.
- **Отрицательные / компромиссы:**
  - Дополнительный `decode_hidden` с `gradient checkpointing` и `get_tied_head_logits` на 512 токенах увеличивает время микро-шага примерно на 15–25%.

## Alternatives Considered & Rejected
- **Дальнейшее дообучение только на `L_basis + L_gr` без `L_CE`:** отвергнуто — за 4000 шагов (5500 -> 9500) после устранения `t_local` и `elastic_drop_prob` модель так и не смогла решить даже 1-cell Cloze (`ent = 4.0..6.5`), так как геометрический лосс не штрафует барицентры-кентавры.
- **Вычисление `L_CE` по всем 508 токенам всех 16 примеров одновременно без подвыборки:** отвергнуто — материализация тензора `(16, 508, 131072)` в FP32 требует `4.26 GB` VRAM на `cuda:0`, что создает риск OOM на T4 15 GB в `DataParallel`.

## Invariants & Rules for AI
- [MUST] Сохранять `GrassmannianDecoder` в режиме `eval()` с `requires_grad_(False)` при вычислении сквозного `L_CE` (градиент течет только в `V_0_hat`, `DEC_PARAM_GRADS == 0`).
- [MUST] Оборачивать `decoder.decode_hidden` в `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` и ограничивать число токенов в `get_tied_head_logits` параметром `ce_max_tokens <= 512` для предотвращения OOM на 2xT4 (15 GB).
- [MUST NOT] Подмешивать чистый таргет `V_0` в `build_rollout_noisy_canvas` (`alpha` строго `1.0`), чтобы исключить локальную утечку ответа на шаге 2 роллаута.
- [MUST] Использовать прогрев `ce_warmup_steps` относительно `start_step` при подключении `L_CE` к существующему чекпоинту.

## Scope & Affected Files
- `experiments/phase 3/train_phase3.py` (Related KI: `KI_experiments.md`, `decisions/034_angular_loss_for_output_projector.md`, `decisions/035_centered_cosine_and_contrastive_neighbor_loss.md`, `decisions/036_replacing_infonce_with_rkd.md`, `decisions/037_phase3_failure_and_plan_b_skip_connections.md`)
- `experiments/phase 3/Kaggle/train_phase3_notebook.ipynb` (Related KI: `KI_experiments.md`)

## Verification
1) Проверка дифференцируемости замороженного декодера: GRAD_FINITE=True, GRAD_NORM=18.29, DEC_PARAM_GRADS=0. 2) py_compile + test_phase3_architecture.py (PASSED). 3) Smoke-тест train_phase3.py --smoke-test --device cpu --ce-weight 0.15 --ce-warmup-steps 1 --ce-max-tokens 64 --rollout-prob 1.0 --rollout-start-step 0 (ce: 11.603 -> 9.765, TokAcc/CE_hi логируются). 4) Экспорт train_phase3_notebook.ipynb (23 ячейки).
