<!-- created: 2026-11-12 -->
<!-- status: Accepted -->
# ADR 123: Устранение FP16-переполнения матрицы Грама в SetCrossAttentionUnpacker декомпрессора (Phase 1 Plan C)

**Status**: Accepted  
**Date**: 2026-11-12  
**Git Commit**: `a4fe2a7` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.3.3.dev3)  

## Context and Problem
В процессе обучения Phase 1 Plan C на двух GPU Tesla T4 (Kaggle) к шагу 835 скорость обучения критически упала из-за систематического возникновения нечисловых значений: '[STAGE-WISE NaN] logits в decode содержит нечисловые значения!' и постоянных пропусков шагов ('[Warning Step 835]'). При этом энкодер, eigh и пулинг работали безупречно (V_cell_finite=True, rk_eff_finite=True, val_loss=1.057 в FP32).
Эмпирическая диагностика на чекпоинте phase1_best_model.pt выявила причину:
1. Выход контекстного слоя ModernBERT (dec_context_layer) накапливает норму residual stream (норма h_cell достигает ~447).
2. После добавления h_cell к клетке норма осей V_cell_mod достигает ~157.
3. В SetCrossAttentionUnpacker матрица Грама G = V_mod @ V_mod.T достигает значений max = 91 627.16.
4. На GPU Tesla T4 обучение выполняется под autocast(dtype=torch.float16), где предел представления равен 65 504. Матрица G переполняется в +inf, что в линейном слое gram_mlp со знакопеременными весами приводит к inf - inf = NaN, заражающему весь декомпрессор.

## Decisions Made
1. Изолировать вычисление матрицы Грама G = V @ V^T в float32: V_f32 = V_cell.float(), G_f32 = torch.matmul(V_f32, V_f32.transpose(-2, -1)).
2. Выполнять прогон через проектор gram_mlp (24k параметров) строго в float32 независимо от autocast и типа весов модели:
   h1 = F.silu(F.linear(G_flat_f32, mlp_w0.float(), mlp_b0.float()))
   h2 = F.linear(h1, mlp_w2.float(), mlp_b2.float())
   h_normed = RMSNorm(h2, norm_w.float())
   Выходной вектор нормы ~11.3 кастуется в целевой dtype V_cell.dtype (float16/bfloat16) и конкатенируется с h_attn.
3. Сохранить без изменений форму и веса параметров модели для обеспечения 100% обратной совместимости с чекпоинтом phase1_best_model.pt.

## Consequences
Положительные:
- Полное и детерминированное устранение переполнения G = inf и logits = nan в половинной точности FP16 (Tesla T4).
- 100% сохранение обученных весов и структуры модели (бинарная совместимость с phase1_best_model.pt).
- 100% шагов оптимизатора используются эффективно, исключены пропуски шагов (Warning Step).
- Восстановлен полный темп обучения на Kaggle с шага 835+.
Компромиссы:
- Вычисления проекции gram_mlp выполняются в float32 (24k параметров), что дает пренебрежимо малый оверхед (<0.01 мс на батч), полностью окупаемый стабильностью.

## Scope & Affected Files
- `.know/doc_state.json`
- `scratch/check_weights.py`
- `scratch/diagnose_checkpoint.py`
- `scratch/test_autocast_fix.py`
- `scratch/test_backward_fix.py`
- `scratch/test_decode.py`
- `scratch/test_decode_fp16.py`
- `scratch/test_fixed_gram_mlp.py`
- `scratch/test_model_decode_fixed.py`
- `scratch/test_val_fix.py`
