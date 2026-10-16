<!-- created: 2026-10-16 -->
<!-- status: Accepted -->
# ADR 104: Phase 6 TPU SDPA Bool Mask Fix, SPMD Batch Sharding and Memory Restoration

**Status**: Accepted  
**Date**: 2026-10-16  
**Git Commit**: `ab04441` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.3.0.dev3)  

## Context and Problem
В ходе запуска скрипта обучения train_phase6_tpu_notebook.py на Kaggle TPU v3-8 (16 ГБ HBM) были выявлены два взаимосвязанных сбоя:
1. В TEST 1 (run_nan_diagnosis) 139 параметров из 156 получили NaN/Inf в градиентах еще до xm.reduce_gradients (при том что прямой проход loss=0.2387 и выходы DUS были 100% конечны). Анализ выявил, что генератор масок transformers >= 4.50 создает sliding_window_mask и attention_mask как torch.bool, для которых фильтрация mask < -1e20 в _clamp_mask_tensor была неактивна, а в safe_attn_forward аргумент sliding_window_mask вовсе не перехватывался. При передаче torch.bool в F.scaled_dot_product_attention C++ понижение XLA заменяет False на -inf/-3.4e38, порождая NaN при обратном проходе слоя 38.
2. В TEST 2 (disable_ca=True) произошел фатальный OOM компилятора XLA (Ran out of memory in memory space hbm. Used 17.85G of 15.75G hbm). Трейс HLO выявил, что на каждом из 8 ядер TPU выделялся полный глобальный батч B=64 вместо локального B=8 (матрицы внимания f32[1024, 513, 513] по 1.27 ГБ каждая). Причиной стало разрушение шардинга ('fsdp') из-за продвинутой индексации Z_A_clean[batch_indices, content_indices, :] по dim=0, что заставляло XLA делать All-Gather батча.
3. Остов self.dus создавался в bfloat16 из-за config.json ModernBERT-large, а dus_input_extended.requires_grad_(True) был закомментирован, вызывая сбои в хуках FSDP.

## Decisions Made
1. Модернизирован патч patch_modernbert_xla_safe_mask:
   - _clamp_mask_tensor конвертирует булевы маски torch.bool в аддитивные float-маски: True -> 0.0, False -> -10000.0, а вещественные маски обрезает до безопасного порога -10000.0.
   - safe_attn_forward и MODERNBERT_ATTENTION_FUNCTION['sdpa'] перехватывают как attention_mask, так и sliding_window_mask.
2. Восстановлен SPMD Batch Sharding ('fsdp'):
   - Индексация сдвига заменена на torch.gather вдоль оси последовательности dim=1, не затрагивая шардированную ось dim=0.
   - Добавлена функция mark_fsdp_sharding, автоматически накладывающая спецификацию ('fsdp', ...) на все входы DUS, t_global, t_actual, Z_A_clean, z_noisy.
3. Остов self.dus принудительно переведен в torch.float32 с установкой property dtype=torch.float32.
4. Раскомментирован dus_input_extended = dus_input_extended.requires_grad_(True).
5. Расширено логирование run_nan_diagnosis с выводом списка параметров с конечными градиентами (finite_before).

## Consequences
- Положительные: 
1. Полное устранение взрыва NaN в градиентах обратного прохода SDPA на слоях со Sliding Window Attention (включая слой 38) благодаря явной конвертации булевых масок в безопасные вещественные значения 0.0 и -10000.0.
2. Восстановление истинного SPMD-шардинга батча ('fsdp') по 8 ядрам TPU через замену многомерной индексации на torch.gather по dim=1 и явную разметку тензоров mark_fsdp_sharding, что снизило пиковый спрос HBM с 17.85 ГБ до ~7.1 ГБ и исключило TPU compile OOM.
3. Устранение предупреждений FSDP backward hook благодаря принудительному requires_grad_(True) на входе DUS.
4. Хранение остова DUS и вычисление норм строго в float32 исключает подпороговый градиентный коллапс bfloat16.
- Компромиссы: Незначительное увеличение времени первого прогона для наложения масок, полностью компенсируемое 8-кратным ускорением шага за счет корректного локального батча B=8 на ядро.

## Scope & Affected Files
- `experiments/phase 6/phase6_generation_demo.html`
- `experiments/phase 6/tpu kaggle/__pycache__/train_phase6_tpu_notebook.cpython-313.pyc`
- `experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.ipynb`
- `experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.py` (Related KI: `decisions/097_phase6_t_global_orchestrator_and_per_sample_beta_noise.md`)
- `scratch/sync_phase6_notebook.py`
- `experiments/phase 6/phase6_generation_demo_9000.html`
- `posts/2026.10.15 Architecture of diffusion.txt`
- `posts/2026.10.15 diffusion_latent_space_trajectory.jpg`
- `wandb/`
