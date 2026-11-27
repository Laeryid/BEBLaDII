<!-- created: 2026-11-27 -->
<!-- status: accepted -->
# ADR 146: Итоги A/B-эксперимента Phase 3 Alt v2 на шаге 3000 («3000 шагов правды»): победа safe (22.94M, all_5) над control (31.81M, dense + idea4), закрытие ветки dense и аналитический пайплайн

**Status**: accepted  
**Date**: 2026-11-27  
**Git Commit**: `bb8eb69` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.4.0.dev3)  

## Context and Problem
В рамках A/B-протокола «3000 шагов правды» (ADR 142) и пакета оптимизаций Phase 3 Alt v2 (ADR 144) были обучены до шага 3000 две модели на идентичном пайплайне:
1. `safe` (`planC_Phase3_Alt_Diffusion_v2_safe_all5`, 22.94M параметров, матричный режим `safe` с `PHMLinear`/`BlockDiagLinear` и полным набором инициализаций `all_5`).
2. `control` (`planC_Phase3_Alt_Diffusion_v2_control_dense_idea4`, 31.81M параметров, плотные слои `dense` + только `idea4`).

Для объективного сравнения и принятия решения о дальнейших шагах был создан скрипт комплексной офлайн-аналитики `experiments/phase 3 alt/analyze_phase3_alt.py` и проведён полный парный прогон обеих EMA-моделей на идентичных валидационных батчах (`val_phase6.parquet`, CPU fp32, `seed=42`, `bs=4`, `eval_batches=6`, `cfg_dsim_batches=3`, `gen_steps=20`, `cfg=3.0`, сэмплеры `euler_heun` и `crystal_radar`), дополненный визуальным декодированием 1-шагового денойзинга по сетке `t = [0.10..0.85]`.

## Decisions Made
1. **Признать `safe (22.94M, all_5)` победителем A/B-эксперимента на шаге 3000.** При экономии 27.9% параметров `safe` превосходит `control` (`31.81M, dense`) практически по всем ключевым метрикам:
   - Evaluation Gate: **9/14 PASS** vs **8/14 PASS** (`control` провалил `cfg_dsim_t50`).
   - Высокий шум: `val/sim_gr_hi` **0.3444** vs 0.3321 (`+0.0123`), `val/ce_loss_hi` **8.243** vs 8.289.
   - Следование промпту: `val/cfg_dsim_t50` **+0.0004** vs -0.0007, `t75` **+0.0012** vs +0.0008 (`1.5×`), `t95` **+0.0142** vs +0.0053 (`2.7×`), `val/cfg_contrast_hi` **0.1774** vs 0.1465 (`+21%`).
   - Устойчивость AdaLN: `val/sigma_adaln_min` **0.0278** vs 0.0109 (`2.55×` выше; `control` у порога коллапса `0.01`), `AdaLN r_eff_max` **2.14** vs 1.63.
   - Защита чистых якорей (`Anchor probe`): `lambda_keep(clean/noisy)` **0.323 / 0.240** (`Δ=+0.083`) vs **0.248 / 0.240** (`Δ=+0.007`, контраст в `11×` выше).
   - Геометрия и VOID: `orth_err` **0.0883** vs 0.0908, `spec_l1` **0.0512** vs 0.0549, `radar_sigma_cell` **0.0452** vs 0.0479, `void_f1` **0.2067** vs 0.1717, `void_start_pm1` **0.3750** vs 0.3333.
2. **Закрыть ветку `control (dense)`** и продолжить длинное обучение (resume с `step_3000.pt` до 10K–20K+ шагов) исключительно для **`safe (all_5)`**.
3. **Утвердить `experiments/phase 3 alt/analyze_phase3_alt.py`** как стандартный инструмент офлайн-оценки чекпоинтов Phase 3 Alt v2 (авто-детект архитектуры из `state_dict`, Evaluation Gate, `CFG-ΔSim`, поосевой `t`-профиль, `Anchor probe`, SVD-ранг `AxisBlockAdaLN`, важность 19 датчиков `ConditionEmbedderV2`, генерация `euler_heun`/`crystal_radar`).
4. **Не менять LR для `DCA`/`Cross-Attention` на шаге 3000** — перепроверить их пробуждение на чекпоинтах 6K–10K, так как `s5_m_prompt` уже является датчиком №1 по важности (`0.506`) и `CFG-ΔSim` стабильно положителен на высоком шуме.

## Consequences
Положительные:
- Экономия 27.9% параметров (22.94M / 43.76 MB FP16 vs 31.81M / 60.67 MB FP16) при одновременном росте качества на высоком шуме, устойчивости AdaLN и чувствительности к промпту.
- 1-шаговый денойзинг через декодер Системы 1 на t=0.50 уже восстанавливает текст с TokAcc=0.945–0.964 (включая сложные термины), а на t=0.65 вытягивает осмысленные фразы с TokAcc=0.685–0.743 (safe > control на +6.3% на техническом примере).
- Единый воспроизводимый офлайн-пайплайн аналитики `analyze_phase3_alt.py` для дальнейшего мониторинга на 6K–20K шагах.

Ограничения и компромиссы:
- На 3000 шагах генерация с нуля (`t = 1 -> 0`, 20 шагов, CFG=3.0) у обеих моделей пока не пробивает фазовый барьер `t = 0.75..1.00` (ранний языковой суп без коллапса, `cross_sim = 0.16..0.28`, `void_f1 = 0.207`) — требуется продолжение обучения.
- На CPU без свёртки весов `safe` медленнее `control` (`200.8s` vs `178.0s` на `euler_heun`, `257.1s` vs `169.3s` на `crystal_radar`) из-за мелких незафьюженных `PHM`/`BlockDiag` операций. Для быстрого CPU-инференса в будущем следует добавить `.merge_for_inference()` (материализация `PHMLinear` в плотный вес при eval).

## Alternatives Considered & Rejected
1. Продолжение обучения `control (dense, 31.81M)`: отвергнуто, так как при +38.6% параметров модель уступает `safe` по качеству на высоком шуме, CFG, защите якорей, геометрии и уже приблизилась к порогу коллапса AdaLN (`sigma_adaln_min = 0.0109`).
2. Досрочное повышение LR для `DCA` и `Cross-Attention` на шаге 3000: отвергнуто до проверки на 6K–10K шагах, чтобы не дестабилизировать здоровую динамику раннего этапа обучения.

## Invariants & Rules for AI
- [MUST] Продолжать обучение Phase 3 Alt v2 только из ветки `safe (all_5)` (resume с чекпоинта `step_3000.pt`).
- [MUST] Оценивать последующие контрольные чекпоинты (6K–10K–20K) через `experiments/phase 3 alt/analyze_phase3_alt.py` с фиксированным `seed=42` для сопоставимости динамики Evaluation Gate, `CFG-ΔSim`, `Anchor probe` и `AdaLN r_eff`.
- [MUST NOT] Тратить вычислительный бюджет на дальнейшее обучение закрытой ветки `control (dense)`.
- [MUST NOT] Модифицировать замороженную Систему 1 (`storage/components/`) или базовый `experiments/phase 3/` в рамках дообучения Phase 3 Alt v2.

## Scope & Affected Files
- `experiments/phase 3 alt/analyze_phase3_alt.py`
- `experiments/phase 3 alt/output/analytics_ab_v2_step3000_report.md`
- `experiments/phase 3 alt/output/analytics_safe_v2_step3000_full_report.md`
- `experiments/phase 3 alt/output/analytics_control_v2_step3000_full_report.md`
- `experiments/phase 3 alt/train_phase3_alt.py`

## Verification
1. Синтаксическая проверка: `$env:PYTHONPYCACHEPREFIX = "$env:TEMP\pycache"; .venv\Scripts\python.exe -m py_compile "C:\Experiments\BEBLaDII\experiments\phase 3 alt\analyze_phase3_alt.py"` (EXIT 0).
2. Парный прогон аналитики `analyze_phase3_alt.py` на `safe_step_3000.pt` и `control_step_3000.pt` (`--batch-size 4 --eval-batches 6 --cfg-dsim-batches 3 --gen-steps 20 --val-prompts 1 --samplers euler_heun,crystal_radar`) — результаты зафиксированы в `experiments/phase 3 alt/output/analytics_ab_v2_step3000_report.md`.
3. Визуальная верификация 1-шагового денойзинга через декодер Системы 1 по сетке `t = [0.10, 0.30, 0.50, 0.65, 0.75, 0.85]` на 2 примерах `val_phase6.parquet`.
