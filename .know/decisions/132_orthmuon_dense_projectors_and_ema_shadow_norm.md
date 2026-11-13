<!-- created: 2026-11-13 -->
<!-- status: Accepted -->
# ADR 132: Уточнение ADR 131: возврат скрытых 2D-проекторов в плотный OrthMuon (sphere=False) и ренормализация q_weight в EMAModel.apply_shadow

**Status**: Accepted  
**Date**: 2026-11-13  
**Git Commit**: `e360dda` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.3.3.dev3)  

## Context and Problem
После внедрения ADR 131 (коммит e360dda) аудит кода Phase 1 Plan C (experiments/phase 1/train_phase1.py) выявил два недочёта:
1. Линейное накопление теневых весов q_weight в EMAModel приводит к уходу строк внутрь единичной сферы (||q_i||_2 < 1). Ренормализация выполнялась только при pullback в update(), но отсутствовала в apply_shadow(), из-за чего валидация и сохранение EMA-чекпоинтов работали с ненормированными q_weight, рассогласованными с row_scale.
2. При разделении параметров в create_hybrid_optimizers все плотные 2D-матрицы вне SphereLinear (включая скрытые проекторы syntax_proj.weight 1024->128 и out_proj.weight 256->1024) попали в AdamW (lr=3e-4) вместо Muon, и в EMAModel потеряли режим pullback_muon.

## Decisions Made
1. В create_hybrid_optimizers разделять параметры OrthMuon на две группы:
   - *.q_weight -> OrthMuon (sphere=True, weight_decay=0.0);
   - syntax_proj.weight и out_proj.weight -> плотный OrthMuon (sphere=False, weight_decay=weight_decay), без перевода в SphereLinear, чтобы сохранить масштабную динамику в геометрически чувствительных узлах (B^128 / RoPE и SetCrossAttentionUnpacker) как до ADR 131.
   - Эмбеддинги, vae_head (mu_*, logvar_*), малый gram_mlp (64->128->128 с bias), query_slots остаются в AdamW (weight_decay=weight_decay); row_scale (lr_mult=4.0), *norm* (вкл. q_norm/k_norm), *.bias и 1D-параметры -> AdamW (weight_decay=0.0).
2. В EMAModel.update() применять pullback_muon к *.q_weight, syntax_proj.weight и out_proj.weight, а хордовую ренормализацию после lerp_ выполнять только для *.q_weight.
3. В EMAModel.apply_shadow() при копировании теневых весов для *.q_weight выполнять построчную проекцию на единичную сферу F.normalize(shadow_val.float(), p=2, dim=-1, eps=1e-8), не изменяя линейно усредняемый self.shadow.
4. Синхронизировать Kaggle-ноутбук через experiments/phase 1/export_to_notebook.py.

## Consequences
Положительные:
- Валидация и сохранение EMA-весов получают строго единичные направления строк q_weight (||q_i||_2 = 1), согласованные с row_scale.
- Сохраняется единый режим оптимизации (Muon + pullback_muon) для скрытых 2D-проекторов syntax_proj и out_proj, не нарушая геометрию выходного пространства.
Компромиссы и риски:
- Число групп параметров в opt_muon выросло с 1 до 2; чекпоинты, сохранённые на коммите e360dda (до этого уточнения), при resume сбросят состояния буферов opt_muon через try/except с переинициализацией моментума.

## Alternatives Considered & Rejected
1. Перевод syntax_proj и out_proj в SphereLinear с row_scale (lr_mult=4.0) — отклонён во избежание неконтролируемого сдвига масштабов на границах B^128 / RoPE.
2. Перевод vae_head (96x96, 16x16) и gram_mlp (64x128, 128x128) в OrthMuon — отклонён ввиду малого размера матриц и наличия bias.
3. Принудительная ренормализация самого self.shadow в EMAModel.update() — отклонена, чтобы не искажать линейное скользящее среднее.

## Invariants & Rules for AI
- [MUST] В EMAModel.apply_shadow() всегда проецировать *.q_weight на единичную сферу по строкам в float32 перед копированием в модель.
- [MUST] Не нормализовать сам буфер self.shadow, чтобы сохранить линейность скользящего среднего.
- [MUST] Применять плотный OrthMuon (sphere=False) и pullback_muon к скрытым 2D-проекторам syntax_proj.weight и out_proj.weight.

## Scope & Affected Files
- `experiments/phase 1/train_phase1.py`
- `experiments/phase 1/Kaggle/train_phase1_notebook.ipynb`

## Verification
1. .venv\Scripts\python.exe -m py_compile "C:\Experiments\BEBLaDII\experiments\phase 1\train_phase1.py" (exit code 0).
2. .venv\Scripts\python.exe "C:\Experiments\BEBLaDII\experiments\phase 1\export_to_notebook.py" (22 ячейки экспортированы).
3. Тест в памяти на GrassmannianPhase1Model: muon_groups=[(38, True, 0.0), (2, False, 0.01)], adamw_groups=[(10, 0.0003, 0.01), (40, 0.0003, 0.0), (38, 0.0012, 0.0)], max_sphere_err_after_apply_shadow=1.79e-07.
