# Implementation Plan: Оптимизация Reasoning-стадии на TPU

- **Affected layers**: TPU Training Script (`experiments/train_phase1_TPU_fsdp.py`).
- **Read KIs**: 
  - `c:\Experiments\BEBLaDII\.know\knowledge\KI_model_core.md`
  - `c:\Experiments\BEBLaDII\.know\knowledge\KI_architecture.md`
  - `c:\Experiments\BEBLaDII\.know\knowledge\KI_experiments.md`
- **KIs Constraints**: 
  - Загрузка весов должна сопоставлять ключи (smart loading), игнорируя префиксы `teacher.*` и адаптируясь под изменение вложенности.
  - Необходимо предотвратить коллапс латентного пространства через отжиг KL (beta).
  - Учитель заморожен (requires_grad=False).
  - Скрипт поддерживает возобновление (Resume), следовательно параметры планировщиков и прогрева должны опираться на `global_step`.

## Шаги реализации

### 1. Smart Weights Loading
Вместо примитивного `load_state_dict(..., strict=False)` будет внедрена локальная функция `smart_load_weights(model, sd)`.
Она будет перебирать обучаемые параметры `distiller` (исключая `teacher`), сопоставлять их по суффиксам с загруженным state_dict и переносить значения. Также будет добавлен вывод в лог статистики (сколько ключей успешно загрузилось), чтобы гарантировать прозрачность инициализации.

### 2. Поддержка Targeted Loss Masking
Скрипт будет модифицирован для принятия `loss_mask` из батча:
```python
actual_mask = batch['loss_mask'] if 'loss_mask' in batch else batch['attention_mask']
```
Маска будет корректно шардироваться через `xs.mark_sharding` и передаваться в `DistillationLoss`. Это позволит датасету в будущем управлять тем, на каких токенах вычисляется штраф (исключив промпты).

### 3. Global KL-Annealing (Resume-safe)
Перед передачей в функцию потерь `beta` будет рассчитываться динамически в зависимости от `global_step`:
```python
warmup_steps = 1000.0
current_beta = min(0.0001, 0.0001 * (global_step / warmup_steps))
```
Это защитит VAE на старте и автоматически подхватит максимальное значение при Resume после 1000 шага.

### 4. LR Scheduler & Warmup Tuning
1. Параметры `CosineAnnealingWarmRestarts` будут изменены: `T_0=2000`, `eta_min=1e-6`.
2. После вызова `scheduler.step()` будет применяться однократный ручной warmup (до 1000 шага):
```python
if global_step <= 1000:
    warmup_factor = max(0.01, global_step / 1000.0)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * warmup_factor
```
При Resume (если `global_step > 1000`) этот блок будет проигнорирован, и будет использоваться чистый выход косинусного планировщика.
### 5. KL Precision Fix
В `src/beb_la_dii/utils/loss.py` расчет `kl_loss_raw` будет переведен в `float32` через принудительное приведение `.float()`. Это устранит эффект дискретизации (ступеньки по 2.0 единицы), вызванный потерей точности bfloat16 при накоплении сумм в диапазоне >256.
