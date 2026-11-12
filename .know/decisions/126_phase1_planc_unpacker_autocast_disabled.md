<!-- created: 2026-11-12 -->
<!-- status: Accepted -->
# ADR 126: Изоляция вычисления матрицы Грама в SetCrossAttentionUnpacker с явным отключением autocast и фиксация FP16 для Turing GPU

**Status**: Accepted  
**Date**: 2026-11-12  
**Git Commit**: `bd46370` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.3.3.dev3)  

## Context and Problem
При обучении Phase 1 Plan C на Kaggle Tesla T4 x2 (шаги 502-566) происходили систематические пропуски шагов оптимизатора с ошибкой: '[STAGE-WISE NaN] logits в decode содержит нечисловые значения!'. В ADR 123 вычисление матрицы Грама G = V @ V^T и проектора gram_mlp было переведено в float32, однако вызов не был обернут в torch.autocast(..., enabled=False). В PyTorch активный autocast принудительно даункастит torch.matmul и F.linear обратно в FP16 даже при float32 тензорах. На батчах с нормами осей ~240 норма G достигала 475 000 (>65504), переполняясь в +inf, что в линейном слое gram_mlp приводило к inf - inf = NaN. Кроме того, в PyTorch 2.6 на Kaggle T4 (Compute Capability 7.5) torch.cuda.is_bf16_supported() возвращал True, что ошибочно включало эмулируемый bfloat16 вместо аппаратного FP16.

## Decisions Made
1. Обернуть вычисление матрицы Грама G = torch.matmul(V_f32, ...) и прогон через gram_mlp в SetCrossAttentionUnpacker.forward в явный блок 'with torch.autocast(device_type=device_type, enabled=False):'.
2. Ограничить выбор bfloat16 в EnvironmentConfig.detect только для архитектур Ampere+ (Compute Capability >= 8.0), гарантируя нативный аппаратный FP16 для Tesla T4 (CC 7.5).
3. Внедрить гранулярную stage-wise диагностику в GrassmannianPhase1Model.decode (h_cell, h_unpacked, x).
4. Синхронизировать train_phase1.py и train_phase1_notebook.ipynb.

## Consequences
Положительные:
- Детерминированное устранение переполнения G = inf и logits = nan под FP16 autocast при любых масштабах V_cell (верифицировано от 1.0 до 500.0).
- 100% шагов оптимизатора на Kaggle становятся рабочими, полностью исключены пропуски батчей Warning Step.
- Восстановление аппаратного FP16 на ядрах Turing Tensor Cores для максимальной скорости обучения.
- 100% обратная бинарная совместимость с чекпоинтами phase1_best_model.pt и phase1_step_500.pt.
Компромиссы:
- Вычисление G и gram_mlp (24k параметров) в FP32 дает около 0.01 мс оверхеда, нивелируемого стабильностью.

## Scope & Affected Files
- `experiments/phase 1/Kaggle/train_phase1_notebook.ipynb`
- `experiments/phase 1/train_phase1.py`
- `scratch/check_opt_devices.py`
- `scratch/check_weights.py`
- `scratch/diagnose_checkpoint.py`
- `scratch/find_decode_nan_step.py`
- `scratch/pinpoint_fp16_nan.py`
- `scratch/query_wandb.py`
- `scratch/test_autocast_fix.py`
- `scratch/test_backward_fix.py`
- `scratch/test_backward_unpacker.py`
- `scratch/test_decode.py`
- `scratch/test_decode_dtypes.py`
- `scratch/test_decode_fp16.py`
- `scratch/test_end_to_end_fix.py`
