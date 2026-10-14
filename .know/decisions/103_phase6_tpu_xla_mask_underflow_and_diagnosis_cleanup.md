<!-- created: 2026-10-14 -->
<!-- status: Accepted -->
# ADR 103: Phase 6 TPU XLA Attention Mask Stabilization (finfo.min Underflow Fix) and Multi-Stage Diagnosis Memory Cleanup

**Status**: Accepted  
**Date**: 2026-10-14  
**Git Commit**: `79a6133` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.3.0.dev3)  

## Context and Problem
В процессе запуска Phase 6 на Kaggle TPU (скрипт train_phase6_tpu_notebook.py) выполнение завершилось двумя критическими сбоями:
1. Генерация NaN в градиентах (139 из 156 параметров содержали NaN ДО all-reduce). Диагностика показала, что ровно 17 параметров слоя 39 (Global Attention) и верхней части слоя 38 (MLP) оставались finite, а начиная со слоя 38 градиент взрывался в NaN. Причина: ModernBERT использует скользящее окно (local_attention: 128) на не-глобальных слоях (38, 37 и др.), где маска sliding_window_mask заполняется значением torch.finfo(self.dtype).min (-3.4028e38 в FP32). На TPU XLA при дифференцировании Softmax в SDPA аппаратные умножители при операциях с такими экстремальными отрицательными константами вызывают floating point underflow/overflow, порождая NaN (0 * inf = NaN), который через residual stream заражает все 139 параметров нижележащих слоев. Кроме того, в CAPromptLayer использовалась булева маска attention_mask_q.bool(), которая в PyTorch/XLA также понижается в -inf.
2. OOM в памяти HBM (17.85G при емкости 15.75G) в блоке run_nan_diagnosis: запуск двух последовательных backward-проходов на полном батче B=64 (сначала TEST 1 с CA, затем TEST 2 с disable_ca=True) привел к компиляции двух конкурирующих HLO-графов, где матрицы внимания f32[1024, 513, 513] весили по 1.27 ГБ, переполнив HBM на 2.10 ГБ.

## Decisions Made
1. Внедрить monkey-patch patch_modernbert_xla_safe_mask для ModernBertModel._update_attention_mask, заменяющий torch.finfo(self.dtype).min на безопасный сентинел -10000.0 (через torch.where для global_mask и sliding_mask при значениях < -1e20).
2. В CAPromptLayer заменить булеву маску mask_Q.bool() на явную 4D аддитивную маску со значениями 0.0 для активных токенов и -10000.0 для паддинга.
3. В блок run_nan_diagnosis внедрить изолированную очистку памяти TPU (clear_tpu_memory: gc.collect() + torch_xla.sync(wait=True, reset_scope=True) / xm.mark_step()), вызываемую после каждого этапа (TEST 1, TEST 2, TEST 3), а также перевести stop_after_diagnosis в False для бесшовного перехода к тренировке после pre-flight проверки.

## Consequences
1. Полное устранение первопричины NaN в backward SDPA на всех 40 слоях ModernBERT DUS и блоках Cross-Attention при сохранении оригинальной семантики скользящего окна.
2. Безопасное прохождение многоэтапной диагностики без риска переполнения памяти HBM TPU благодаря явному сбросу области видимости XLA-графа (reset_scope=True) между изолированными тестами.
3. Гарантированный запуск обучения после диагностики без дополнительных ручных перезапусков.

## Scope & Affected Files
- `experiments/phase 6/tpu kaggle/__pycache__/train_phase6_tpu_notebook.cpython-313.pyc`
- `experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.ipynb`
- `experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.py` (Related KI: `decisions/097_phase6_t_global_orchestrator_and_per_sample_beta_noise.md`)
- `scratch/sync_phase6_notebook.py`
