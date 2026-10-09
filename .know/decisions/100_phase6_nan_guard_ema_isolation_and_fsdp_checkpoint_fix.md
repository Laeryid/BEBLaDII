<!-- created: 2026-10-09 -->
<!-- status: Accepted -->
# ADR 100: Phase 6 In-Graph NaN Guard, EMA Anomaly Isolation, and FSDP Checkpoint Key Normalization

**Status**: Accepted  
**Date**: 2026-10-09  
**Git Commit**: `d702b22` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.3.0.dev3)  

## Context and Problem
На шаге ~1240 обучения Phase 6 TPU (ModernBERT DUS с Cross-Attention слоями 12, 24, 36 и размороженными слоями DUS unfreeze_k=4) произошел градиентный взрыв и мгновенное заражение весов и EMA значением NaN. Валидация на шаге 1400 зафиксировала nan.
Причины:
1. Завершение warmup (шаг 1000) сняло демпфирование поправок CA и SA (warmup_factor=1.0).
2. Агрессивный стартовый LR (ca_out: 5e-4, ca_qkv: 2e-4) без шедулера привел к накоплению амплитуд.
3. Отсутствие NaN-Guard: при первом же не-конечном grad_norm функция clip_grad_norm_ возвращала NaN, все градиенты умножались на NaN, а optimizer_step и ema_tracker.update необратимо разрушали веса.
4. Попытка использовать grad_norm.item() в теле цикла вызывала graph breaks (ADR 079).
5. Несовместимость FSDP-ключей при resume: чекпоинт step 1000 содержал префиксы _orig_module для 120 параметров DUS, из-за чего при сырой загрузке восстанавливались лишь 39 ключей CA, а веса DUS терялись.

## Decisions Made
1. Внедрена тензорная In-Graph защита от NaN/Inf прямо на TPU без вызовов .item() в цикле: при не-конечном grad_norm градиенты принудительно зануляются операцией p.grad.mul_(is_finite.to(p.grad.dtype)).
2. Изоляция EMA и PACE от аномальных всплесков: введен тензорный флаг is_safe = is_finite & (grad_norm <= 5.0). При аномалиях обновление EMA и оттягивание PACE пропускаются (decay=1.0, alpha=0.0).
3. Асинхронное логирование grad_norm в WandB через xm.add_step_closure (0 graph breaks).
4. Калибровка LR для дообучения после прогрева: ca_out снижен до 2e-4, ca_qkv до 8e-5, ca_other и dus до 2.5e-5.
5. Автоматическая нормализация ключей чекпоинта при resume: удаление ._orig_module. и _orig_module., что обеспечивает 100% загрузку всех 159 параметров модели.

## Consequences
- Полная устойчивость процесса оптимизации к градиентным всплескам и NaN без потери производительности XLA TPU (отсутствуют graph breaks).
- Защита скользящего среднего EMA от отравления одиночными стохастическими выбросами.
- Корректное 100% восстановление весов из FSDP/SPMD чекпоинтов (159 из 159 параметров), устранение молчаливого сброса весов DUS при resume.
- Сниженный LR обеспечивает плавную стабилизацию на этапе после прогрева (warmup).

## Alternatives Considered & Rejected
- Синхронная проверка через grad_norm.item() в цикле: отвергнута из-за aten::_local_scalar_dense и катастрофических задержек XLA конвейера (ADR 079).
- Уменьшение unfreeze_k_after_ca: отвергнуто пользователем для сохранения адаптационной емкости DUS слоев.

## Invariants & Rules for AI
- [MUST] Осуществлять защиту оптимизатора и EMA через тензорные in-graph операции без вызова .item() в теле обучающего цикла.
- [MUST] Очищать префиксы _orig_module и module при загрузке чекпоинтов, сохраненных модулями под FSDP/SPMD.
- [MUST NOT] Обновлять тень EMA при градиентных всплесках grad_norm > 5.0 или NaN/Inf.

## Scope & Affected Files
- `experiments/phase 6/tpu kaggle/__pycache__/train_phase6_tpu_notebook.cpython-313.pyc`
- `experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.ipynb`
- `experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.py` (Related KI: `decisions/097_phase6_t_global_orchestrator_and_per_sample_beta_noise.md`)
- `experiments/phase 6/screenshots/2026-10-09 191434 train cos sim.png`
- `scratch/sync_phase6_notebook.py`
- `scratch/test_checkpoint_load.py`

## Verification
.venv/Scripts/python.exe .agents/skills/tpu-script-crafting/scripts/verify_tpu_script.py experiments/phase\ 6/tpu\ kaggle/train_phase6_tpu_notebook.py
.venv/Scripts/python.exe scratch/test_checkpoint_load.py
