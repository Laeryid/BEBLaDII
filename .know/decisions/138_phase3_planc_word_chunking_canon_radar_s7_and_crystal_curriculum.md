<!-- created: 2026-11-21 -->
<!-- status: accepted -->
# ADR 138: Фаза 3 Plan C (старт с нуля): WordAwareElasticChunker, канонизация таргетов V_0, датчик s7 (NoiseRadarHead), Random Crystal Anchors и перебалансировка лосса

**Status**: accepted  
**Date**: 2026-11-21  
**Git Commit**: `09af128` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.4.0.dev2)  

## Context and Problem
После завершения Фазы 2 Plan C (ADR 137, экспорт O(8)-инвариантного GrassmannianDecoder с NoiseRadarHead и WordAwareElasticChunker из phase2_decoder_step_25000.pt) диагностика прежнего чекпоинта диффузии step_7500 (experiments/phase 3/checkpoints/planC_Phase 2_checkpoints_phase3_step_7500.pt) на новом декодере показала: условная генерация и VOID-отсечка живы (Cross-prompt Sim_Gr=0.354 на Rolling Wave / 0.584 на Euler-Heun, sink_c=0.347, adaln/sigma_min=0.138), но декодированный текст распадался на словесно-байтовый салат с битыми UTF-8 (\ufffd), высоким шумом радара (sigma_cell=0.080..0.132, dict cos_max=0.194..0.234) и перегревом энергии (E_content=3.06..3.63 против эталонных 2.42..2.52). Выявлены четыре причины: (1) рассинхрон распределения чанков — диффузия училась на жёсткой нарезке по 4 BPE-токена (slot_pad=4.2..8.0%), а новый декодер обучен на WordAwareElasticChunker; (2) доминирование поэлементного L_basis (вес 1.0 против 0.08 у L_Gr_dir) заставляло ядро угадывать произвольные знаки/вращения осей V_0 несмотря на строгую O(8)-инвариантность декодера; (3) не задействован датчик s7 в ConditionEmbedder под показания NoiseRadarHead; (4) в Mixed Noise Curriculum отсутствовал режим произвольных опорных кристаллов (Random Crystal Anchors) для адаптивной кристаллизации.

## Decisions Made
1. Старт обучения Фазы 3 Plan C с нуля в изолированном пространстве имён (чекпоинты phase3_step_*.pt / phase3_best_model.pt, GCS gs://bebladii-weigths-us/planC/Phase3_Diffusion/checkpoints/, WandB BEBLaDII-Phase3-PlanC, ноутбук experiments/phase 3/Kaggle/train_phase3_notebook.ipynb).
2. Перевести PromptAnswerDataset на WordAwareElasticChunker (целые слова 1..4 токена, elastic_drop_prob=0.0 для промпта, 0.05 для ответа на train и 0.0 на val, готовые двухуровневые веса пулера prompt_weights/ans_weights) и decode_latent_answers на WordAwareElasticChunker.detokenize_non_pad.
3. Применять детерминированную канонизацию базиса canonicalize_cell_basis(V) к выходам замороженного энкодера (V_prompt, V_ans) и якорю sep_concept.pt до добавления шума, чтобы траектория Flow Matching V_t = (1-t)*V_0_canon + t*sigma*xi и вход декодера жили в одном каноническом базисе (lambda_1 >= ... >= lambda_8 + SignFix).
4. Подключить быстрый Выход А декодера (NoiseRadarHead.estimate_noise) через compute_radar_sigma(decoder, V_t) в резервный 8-й датчик s7 модуля ConditionEmbedder при обучении, валидации и инференсе (s7[:, 0] == 0.0 для якоря SEP, один расчёт на V_t для обеих веток CFG без утечки промпта).
5. Перебалансировать гибридный лосс compute_phase3_loss: basis_weight=0.50, gr_weight=0.50 (поднято с 0.08), energy_weight=0.15 (поднято с 0.05), сохранив adj=0.10, bary=0.10, sink=0.10 и w_adaln(s).
6. Добавить в sample_mixed_noise_schedule 5-й режим Random Crystal Anchors (15%: 20-60% случайных контентных клеток с t_i=0.0, остальные в U(0.30, 1.0)); итоговое распределение: 25% High-Noise, 30% Shifted Flat, 15% Causal Prefix/Block-AR, 15% Random Crystal Anchors, 15% Rolling Wave.
7. Реализовать адаптивный сэмплер crystal_radar_sample (фиксация клеток с sigma_cell_hat < 0.04 при t <= 0.65 в чистые якоря t_i=0.0) и добавить логирование val/radar_sigma_cell и val/dict_cos_max в evaluate_phase3.

## Consequences
Положительные: полная совместимость диффузии Фазы 3 с новым O(8)-инвариантным декодером Фазы 2 (ADR 137) по распределению чанков (целые слова 1..4 токена) и каноническому базису; ядро с первого шага обучается опираться на реальный сенсор неуверенности NoiseRadarHead (s7) и на произвольные опорные кристаллы; усиленный градиент по L_Gr_dir и L_energy устраняет перегрев энергии контента и размытие в барицентр; чистый запуск с нуля в изолированном GCS-бакете исключает смешивание весов старой и новой геометрии. Компромиссы: добавлен вызов decoder.estimate_noise (Этап 1 декодера под no_grad, без 4 тяжёлых dec_smooth_layers) на каждом микро-шаге обучения и инференса; обучение диффузии начинается с шага 0 без переноса весов step_7500.

## Alternatives Considered & Rejected
1. Warm-start от чекпоинта step_7500 — отклонён пользователем в пользу чистого старта с нуля, так как меняются сразу распределение чанков (WordAwareElasticChunker), базис таргетов (canonicalize_cell_basis), баланс лоссов и активный входной датчик s7 в ConditionEmbedder.
2. Использование NoiseRadarHead только в сэмплере инференса без подачи в датчик s7 при обучении — отклонено: без обучения на реальных показаниях s7 ядро диффузии не умеет интерпретировать радарную неуверенность при частичной кристаллизации.
3. Полный отказ от L_basis (basis_weight=0.0) — отклонён, так как при линейной интерполяции Flow Matching в R^{8x128} умеренный L_basis=0.50 по каноническому V_0_canon стабилизирует векторное поле денойзинга без конфликта с O(8)-геометрией.

## Invariants & Rules for AI
- [MUST] В Фазе 3 всегда канонизировать V_prompt, V_ans и sep_concept через canonicalize_cell_basis(V) перед добавлением шума, чтобы таргеты Flow Matching совпадали по базису с входом GrassmannianDecoder (ADR 137).
- [MUST] В PromptAnswerDataset и inspect_phase3_generation.py упаковывать промпты и ответы строго через WordAwareElasticChunker и собирать текст через detokenize_non_pad (PAD_TOKEN_ID=10).
- [MUST] Вычислять датчик s7 (radar_sigma = compute_radar_sigma(decoder, V_t)) исключительно под torch.no_grad() по V_t (общий тензор для conditional и unconditional веток CFG, с принудительным s7[:, 0] = 0.0 на SEP), чтобы исключить утечку промпта.
- [MUST NOT] Смешивать чекпоинты новой Фазы 3 (WordAwareElasticChunker + V_0_canon + s7) с чекпоинтами старой Фазы 2/3 на жёсткой 4-токенной нарезке (использовать gs://bebladii-weigths-us/planC/Phase3_Diffusion/checkpoints/).

## Scope & Affected Files
- `experiments/phase 3/train_phase3.py` (Related KI: `KI_experiments.md`, `decisions/034_angular_loss_for_output_projector.md`, `decisions/035_centered_cosine_and_contrastive_neighbor_loss.md`, `decisions/036_replacing_infonce_with_rkd.md`, `decisions/037_phase3_failure_and_plan_b_skip_connections.md`)
- `experiments/phase 3/inspect_phase3_generation.py` (Related KI: `KI_experiments.md`)
- `experiments/phase 3/test_phase3_architecture.py` (Related KI: `KI_experiments.md`)
- `experiments/phase 3/export_to_notebook.py` (Related KI: `KI_experiments.md`)
- `experiments/phase 3/Kaggle/train_phase3_notebook.ipynb` (Related KI: `KI_experiments.md`)

## Verification
.venv\Scripts\python.exe -B "experiments/phase 3/test_phase3_architecture.py" && .venv\Scripts\python.exe -B "experiments/phase 3/train_phase3.py" --smoke-test --device cpu
