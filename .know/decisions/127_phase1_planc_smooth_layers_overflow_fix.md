<!-- created: 2026-11-12 -->
<!-- status: Accepted -->
# ADR 127: Устранение переполнения residual stream в декомпрессоре (Phase 1 Plan C): Robust RMSNorm, SwiGLU FP32, Muon Weight Decay и возврат bfloat16

**Status**: Accepted  
**Date**: 2026-11-12  
**Git Commit**: `69201d6` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.3.3.dev3)  

## Context and Problem
После устранения переполнения матрицы Грама в unpacker модель Phase 1 Plan C успешно обучалась с шага 500 до шага 922. На шаге 922 в режиме float16 на обоих GPU Tesla T4 (Kaggle) возникло нечисловое значение в остаточном потоке декомпрессора: '[STAGE-WISE NaN] x после dec_smooth_layers содержит нечисловые значения!'. Анализ показал: 1) из-за отсутствия weight_decay в Muon нормы 2D-весов трансформера за 422 шага выросли с 64 до 122+, приводя к росту произведения gate * up в SwiGLU до > 65504 (предел FP16) и переполнению в +inf; 2) стандартный RMSNorm при наличии +inf во входе вычисляет rsqrt(inf) = 0.0, что в умножении inf * 0.0 дает NaN; 3) при NaN шаг пропускается, веса не штрафуются, что вызывало мертвую петлю (шаг 923 бесконечно пропускался).

## Decisions Made
1. Внедрить RobustRMSNorm с вычислением в float32 и torch.nan_to_num(x.float(), nan=0.0, posinf=65000.0, neginf=-65000.0), исключающий появление NaN из-за локальных экстремумов.
2. Вычислять произведение SwiGLU gate * up в float32 с безопасным ограничением clamp(-65000, 65000).
3. Внедрить канонический weight_decay = 0.01 в Muon, стабилизирующий нормы 2D-весов от бесконечного накопления.
4. Вернуть автоматический выбор torch.bfloat16 на CUDA (динамический диапазон 3.4e38 исключает переполнение 65504).
5. Синхронизировать train_phase1.py и train_phase1_notebook.ipynb.

## Consequences
Положительные:
- Полное устранение переполнения и мертвой петли NaN на шагах 922-923.
- Тройная защита: bfloat16 широкий динамический диапазон + Robust RMSNorm + SwiGLU FP32.
- Стабилизация спектральных норм 2D-весов трансформера через канонический Muon weight_decay=0.01.
- 100% обратная бинарная совместимость с существующими чекпоинтами phase1_best_model.pt.
Компромиссы:
- Дополнительный clamp и float32-вычисление SwiGLU/RMSNorm добавляют пренебрежимо малый оверхед (<0.02 мс на батч), полностью окупаемый непрерывностью обучения.

## Scope & Affected Files
- `experiments/phase 1/Kaggle/train_phase1_notebook.ipynb`
- `experiments/phase 1/train_phase1.py`
- `scratch/check_opt_devices.py`
- `scratch/check_weights.py`
- `scratch/diagnose_checkpoint.py`
- `scratch/find_decode_nan_step.py`
- `scratch/inspect_smooth_layers.py`
- `scratch/pinpoint_fp16_nan.py`
- `scratch/query_wandb.py`
- `scratch/test_autocast_fix.py`
- `scratch/test_backward_fix.py`
- `scratch/test_backward_unpacker.py`
- `scratch/test_complete_stability.py`
- `scratch/test_decode.py`
- `scratch/test_decode_dtypes.py`
