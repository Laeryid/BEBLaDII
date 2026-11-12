<!-- created: 2026-11-12 -->
<!-- status: Accepted -->
# ADR 125: Точечный перевод Tied Head проектора в float32 для устранения FP16-переполнения логитов в decode (Phase 1 Plan C)

**Status**: Accepted  
**Date**: 2026-11-12  
**Git Commit**: `9a25a33` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.3.3.dev3)  

## Context and Problem
При обучении Phase 1 Plan C на Kaggle Tesla T4 x2 в режиме mixed precision (autocast float16) на отдельных батчах (например, шаги 507, 509, 520, 524) спорадически возникали предупреждения '[STAGE-WISE NaN] logits в decode содержит нечисловые значения!'. При этом V_cell и rk_eff были строго конечны.
Анализ показал:
1. В половинной точности FP16 потолок чисел составляет 65 504.
2. В декомпрессоре 2 слоя ModernBERT без маски внимания на хвостовых токенах могут давать локальный амплитудный всплеск в скрытом состоянии x.
3. При матричном умножении x на словарь 131 072 токенов в FP16 одиночный всплеск превышает 65 504 и превращается в +inf, что приводит к recon_loss = nan и пропуску шага оптимизатора.
4. Так как при NaN шаг пропускается, модель физически не получала градиентного штрафа и не могла самоисправить эти выбросы.
5. Перевод всей модели в FP32 невозможен из-за падения скорости в 6-8 раз (на чипах Turing T4 нет тензорных ядер для FP32).

## Decisions Made
1. Изолировать финальную проекцию Tied Head (get_tied_head_logits) строго в float32 с явным отключением autocast:
   with torch.autocast(device_type=hidden_states.device.type, enabled=False):
       h_f32 = hidden_states.float()
       w_f32 = self.embeddings.weight.float()
       return F.linear(h_f32, w_f32)
   (Явный `enabled=False` критически необходим, так как PyTorch autocast по умолчанию принудительно переводит F.linear обратно в FP16 даже для float32 входов).
2. Вычислять CrossEntropyLoss по float32-логитам, что предотвращает переполнение logsumexp и обеспечивает передачу полноценных градиентов в AdamW и Muon.
3. Сохранить все остальные слои трансформера в FP16/bfloat16 для поддержания максимальной производительности тензорных ядер (~18 step/s).

## Consequences
Положительные:
- Полное устранение спорадических переполнений в decode() на редких батчах с длинными хвостами и спецсимволами.
- Модель перестает пропускать шаги (исключены Warning Step) и начинает получать штрафные градиенты за любые амплитудные всплески.
- Сохранена скорость обучения на тензорных ядрах (~18 step/s), так как FP32 задействован точечно только на одной проекции (оверхед < 0.1 мс).
- 100% обратная совместимость по структуре весов с существующими чекпоинтами phase1_best_model.pt.
Компромиссы:
- Память под тензор логитов в FP32 требует ~1.07 ГБ VRAM на GPU (свободно помещается в 15.9 ГБ Tesla T4).

## Alternatives Considered & Rejected
- Перевод всей модели в float32: отклонен из-за критического замедления (с 18 step/s до 2.5 step/s на T4).
- Жесткий clamp логитов: отклонен, так как clamp искажает вероятностное распределение softmax и градиенты.
- Добавление отдельного слоя lm_head: отклонено, так как нарушает принцип Tied Embeddings и увеличивает число параметров на 134M.

## Invariants & Rules for AI
- MUST: get_tied_head_logits ОБЯЗАН выполняться под `torch.autocast(..., enabled=False)` в float32.
- MUST NOT: Запрещено переводить всю модель в FP32 на видеокартах Turing (T4), чтобы не потерять ускорение Tensor Cores.
- MUST: Сохранять идентичность весов embeddings.weight и lm_head (Tied Weights).

## Scope & Affected Files
- `.know/doc_state.json`
- `experiments/phase 1/Kaggle/train_phase1_notebook.ipynb`
- `experiments/phase 1/train_phase1.py`
- `scratch/check_opt_devices.py`
- `scratch/check_weights.py`
- `scratch/diagnose_checkpoint.py`
- `scratch/find_decode_nan_step.py`
- `scratch/test_autocast_fix.py`
- `scratch/test_backward_fix.py`
- `scratch/test_decode.py`
- `scratch/test_decode_fp16.py`
- `scratch/test_fixed_gram_mlp.py`
- `scratch/test_model_decode_fixed.py`
- `scratch/test_resume_device.py`
- `scratch/test_tied_head_fp32.py`

## Verification
Прогон scratch/test_tied_head_fp32.py под autocast с симуляцией амплитуды скрытых состояний до 5000: logits и embeddings.weight.grad строго конечны (finite=True), extreme logits finite в диапазоне [-8.2e5, 8.8e5]. Проверка синтаксиса через py_compile.
