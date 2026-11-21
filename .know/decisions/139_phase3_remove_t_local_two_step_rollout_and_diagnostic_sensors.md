<!-- created: 2026-11-21 -->
<!-- status: accepted -->
# ADR 139: Фаза 3 Plan C: Удаление оракульного t_local, детерминированный чанкинг (elastic_drop_prob=0), Two-Step Rollout и диагностическое управление через s0..s7

**Status**: accepted  
**Date**: 2026-11-21  
**Git Commit**: `5f60153` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.4.0.dev3)  

## Context and Problem
Диагностика чекпоинта Фазы 3 Plan C (`phase3_step_4500.pt`) и графиков WandB до шага ~5200 выявила выход обучения на плато около шага 3000–3500 (`loss/total ~ 0.215`, `val/sim_gr_hi ~ 0.43`, монотонный рост `loss/adj`). Пошаговая трассировка диффузии показала, что на первом шаге ($t = 1.0 \to 0.83$) модель уже формирует корректный грубый макро-каркас ответа (точную границу VOID `F1 = 1.0`, язык, домен, ключевые конструкции), но на $t = 0.80 \to 0.0$ сглаженная последовательность барицентров не фокусируется в конкретные слова и застывает в «словах-кентаврах».

Выявлены четыре корневые причины:
1. **Утечка оракульного `t_local` в `ConditionEmbedder`:** синтетический поячеечный тензор `t_mat` подавался в `t_local_emb` (128 из 200 каналов), а также использовался для `s4 = (tf < 1e-3)` и `s6`. На обучении модель опиралась на оракульный `t_local`, игнорируя физические датчики `s0..s7`; на инференсе плоский `t_local = t_cur` конфликтовал с `s7 (radar_sigma)` и `s2 (rk_eff)`.
2. **Фазовый сдвиг таргета от `elastic_drop_prob = 0.05`:** случайный перенос слов в соседние ячейки снижал `Sim_Gr` между двумя проходами одного ответа до `0.12–0.19` (совпадение токенов ячейки `4–16%`), заставляя модель предсказывать размытое скользящее среднее соседних ячеек.
3. **Exposure bias одношагового обучения на $t \in [0.15, 0.75]$:** при зашумлении $V_t = (1-t)V_0^{\text{clean}} + t\sigma\xi$ модель работала как локальный фильтр белого шума (`val/cfg_contrast = 0.026`) и не училась расщеплять семантические барицентры $\hat{V}_0^{\text{draft}}$.
4. **Баг в `crystal_radar_sample`:** радар шума вызывался на зашумленном `V_t` вместо предсказанного `v0` ($\hat{V}_0$), из-за чего порог `crystal_thr = 0.04` не срабатывал на $t \in [0.20, 0.65]$.

## Decisions Made
1. **Удаление оракульного `t_local` из `ConditionEmbedder` и `GrassmannianOutputHead`:**
   - Поячеечный `t_mat` используется строго при генерации зашумленного полотна $V_t$ на обучении и в SNR-весах лосса.
   - `ConditionEmbedder` принимает только глобальный управляющий шум `t_global` (64d) и 8 чисто диагностических датчиков `s0..s7` (вход `Linear(72, 256)` вместо `Linear(200, 256)`). Датчики `s4 = clamp(1.0 - s7 / 0.05, 0, 1)` и `s6 = mean(s4[:, 1:])` вычисляются строго по `NoiseRadarHead` (`s7`), без подглядывания в `t_mat`.
   - `GrassmannianOutputHead` использует `t_global` для ячеек ответа и точное `0.0` только для ячейки 0 (`SEP`).
2. **Бесшовная миграция чекпоинтов (`migrate_cond_embedder_state_dict`):**
   - При штатном возобновлении с последнего чекпоинта в GCP (`resume_ckpt="auto"`) старые веса `cond_embedder.mlp.0.weight` формы `(256, 200)` (в модели, EMA и буферах `exp_avg`/`exp_avg_sq` оптимизатора `AdamW`) автоматически срезаются по `[:, 128:200]`, сохраняя обученные проекции `t_global` и `s0..s7`.
3. **Детерминированная упаковка ответов (`elastic_drop_prob = 0.0`):**
   - Отключаем случайный сдвиг границ слов по умолчанию (`--elastic-drop-prob 0.0`), восстанавливая точную позиционную фазу между соседними ячейками.
4. **Two-Step Rollout (`build_rollout_noisy_canvas`, `--rollout-prob 0.35`, `--rollout-start-step 1000`) и непрерывный профиль чистоты:**
   - В 35% батчей строится 1-шаговый черновик $\hat{V}_0^{\text{draft}}$ из $t_{\text{hi}} \in [0.80, 1.0]$ под `torch.no_grad()`, его неуверенность оценивается `NoiseRadarHead` ($\hat{\sigma}^{\text{draft}}$), после чего смесь $\alpha \hat{V}_0^{\text{draft}} + (1-\alpha)V_0$ перезашумляется к $t_{\text{base}} \in [0.15, 0.70]$ с адаптивным профилем $t_i \propto \hat{\sigma}_i^{\text{draft}}$.
   - В `sample_mixed_noise_schedule` режим `Random Crystal Anchors` расширен мягкими опорными ячейками $t_j \in [0.01, 0.22]$.
5. **Исправление `crystal_radar_sample`:**
   - Радар шума оценивает предсказанное чистое состояние `v0` (`compute_radar_sigma(decoder, v0)`), и каждая ячейка охлаждается со скоростью, обратно пропорциональной `sig_v0`, без принудительного дискретного округления `Decoder -> Encoder` внутри цикла.

## Consequences
- **Положительные:**
  - Полностью устранен разрыв обучения и инференса (exposure bias) по входам AdaLN: и при обучении, и при генерации модель получает только управляющий макро-шаг `t_global` и физические датчики `s0..s7` по `V_t`.
  - Self-Attention вынужден опираться на `s7 (radar_sigma)`, `s1/s2 (rk_eff)` и `s4/s6` для распространения смысла от уже чистых опорных ячеек к размытым соседям.
  - Включено прямое обучение непрерывной Coarse-to-Fine фокусировке (`Two-Step Rollout`: сглаженный барицентр -> точные слова на $t \in [0.15, 0.70]$).
  - Бесшовная миграция весов `(256, 200) -> (256, 72)` позволяет штатно продолжать обучение с последнего чекпоинта в GCP (`resume_ckpt="auto"`) без потери весов и моментов AdamW/OrthMuon.
- **Отрицательные / компромиссы:**
  - `Two-Step Rollout` на 35% батчей добавляет один `no_grad` проход модели и один вызов `NoiseRadarHead`, увеличивая время шага примерно на 15–20%.

## Alternatives Considered & Rejected
- **Перебалансировка весов `L_basis` vs `L_gr`:** отвергнуто как основная мера — тест Прокрустова выравнивания $O(8)$ показал, что на $t \ge 0.85$ относительная ошибка даже после оптимального поворота базиса составляет `0.96–1.10` (причина — усреднение барицентров, а не вращение базиса).
- **Дискретное округление `Decoder -> argmax -> Encoder` внутри цикла диффузии (Manifold/Crystal Rounding):** стабилизирует `radar_sigma` и `rk_eff` на $t=0$, но уводит модель в парадигму Masked/Discrete Diffusion и кристаллизует ранние фрагменты-кентавры вместо непрерывной Coarse-to-Fine фокусировки.

## Invariants & Rules for AI
- [MUST NOT] Передавать поячеечный оракульный `t_local` (`t_mat`) в `ConditionEmbedder` или `GrassmannianOutputHead` (кроме обнуления ячейки 0 `SEP`); модель обязана диагностировать чистоту ячеек строго по `V_t` и `s0..s7`.
- [MUST] Сохранять `elastic_drop_prob = 0.0` в `PromptAnswerDataset` для стабильной фазовой сцепки слов между соседними ячейками.
- [MUST] В `crystal_radar_sample` оценивать чистоту и скорость охлаждения через `compute_radar_sigma(decoder, v0)` по предсказанному `v0`, а не по зашумленному `V_t`.
- [MUST] Поддерживать автоматическую миграцию `(256, 200) -> (256, 72)` в `load_checkpoint` для штатного `resume_ckpt="auto"` из GCP.

## Scope & Affected Files
- `experiments/phase 3/train_phase3.py` (Related KI: `KI_experiments.md`, `decisions/034_angular_loss_for_output_projector.md`, `decisions/035_centered_cosine_and_contrastive_neighbor_loss.md`, `decisions/036_replacing_infonce_with_rkd.md`, `decisions/037_phase3_failure_and_plan_b_skip_connections.md`)
- `experiments/phase 3/test_phase3_architecture.py` (Related KI: `KI_experiments.md`)
- `experiments/phase 3/inspect_phase3_generation.py` (Related KI: `KI_experiments.md`)
- `experiments/phase 3/Kaggle/train_phase3_notebook.ipynb` (Related KI: `KI_experiments.md`)

## Verification
.venv\Scripts\python.exe -B "experiments/phase 3/test_phase3_architecture.py" (подтверждено t_local_leak=0.00e+00, leak_diff=0.00e+00, max_sph_err=1.79e-07 и миграция (256,200)->(256,72)) + smoke-test train_phase3.py --smoke-test --device cpu --rollout-prob 1.0 --rollout-start-step 0 + экспорт ноутбука export_to_notebook.py.
