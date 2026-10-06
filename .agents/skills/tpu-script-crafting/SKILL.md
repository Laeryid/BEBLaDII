---
name: tpu-script-crafting
description: >-
  Руководство, чеклист и валидатор для создания, рефакторинга и проверки обучающих скриптов под TPU (PyTorch/XLA, Google Cloud TPU v6e, Kaggle TPU v5e/v3-8). Активируй этот скилл при разработке, адаптации или аудите любых TPU-скриптов и блокнотов.
---

# TPU Script Crafting & Verification Skill

Этот скилл содержит концентрированный опыт и проверенный чеклист для разработки скриптов обучения на Google Cloud и Kaggle TPU (PyTorch/XLA). Он создан на базе анализа 246 коммитов и 134 исправлений багов в проекте BEBLaDII.

---

## 1. Трехуровневая система проверки (Verification Pipeline)

Перед отправкой любого скрипта на TPU запустите единый трехуровневый верификатор:

```powershell
.venv\Scripts\python.exe .agents/skills/tpu-script-crafting/scripts/run_tpu_smoke_test.py <path_to_tpu_script.py>
```

Команда автоматически выполняет 3 уровня проверок:

1. **Level 1 (Синтаксис & Байткод)**: [`verify_tpu_syntax.py`](./scripts/verify_tpu_syntax.py)
   Проверяет скрипт на `SyntaxError`, `IndentationError`, `TabError` через `py_compile`.
2. **Level 2 (Семантический AST-аудит)**: [`verify_tpu_script.py`](./scripts/verify_tpu_script.py)
   Парсит AST-дерево Python, проверяет контекст циклов, аргументы DataLoader, условия ветвлений, корректность `torch.save`, блокировку `XLA_USE_BF16` и валидацию EMA.
3. **Level 3 (CPU Dry-Run / Smoke Test)**: [`run_tpu_smoke_test.py`](./scripts/run_tpu_smoke_test.py)
   Запускает скрипт в изолированном CPU-моке XLA ([`fake_torch_xla`](./scripts/fake_torch_xla)) на 1–2 шага. Проверяет импорты, сборку графа autograd, forward/backward pass, расчет функций потерь, шаг оптимизатора, шаг EMA и сохранение чекпоинта без реального чипа TPU.

---

## 2. Критический чеклист для TPU (Pre-Flight Checklist)

Сверьте код построчно с этим чеклистом перед запуском обучения:

### Блок A: Окружение и Точность (Precision & Environment)
- [ ] **НЕТ флагу `XLA_USE_BF16 = 1`**: Переменная окружения `os.environ["XLA_USE_BF16"] = "1"` полностью удалена. В противном случае 7-битная мантисса BF16 зануляет градиентные поправки параметров при малых LR (ADR 057, 082).
- [ ] **Мастер-веса в Float32**: Обучаемые параметры модели, модули AdaLN, проекторы и буферы оптимизатора AdamW хранятся строго в `torch.float32`.
- [ ] **Замороженные модули в BF16**: Замороженные энкодеры/декодеры (Qwen, VAE) явно приведены к `bfloat16` (`.to(torch.bfloat16)`) для экономии памяти HBM.
- [ ] **Отключение Dynamo**: `torch._dynamo.disable()` вызван в самом начале, чтобы избежать конфликта TorchDynamo с компилятором XLA.

### Блок B: Граф XLA и Компиляция (Zero-Recompilation Rule)
- [ ] **НЕТ `.item()` внутри шага обучения**: Никаких вызовов `loss.item()`, `grad_norm.item()` или `metric.item()` на каждом шаге! Синхронизация CPU-TPU (`aten::_local_scalar_dense`) замораживает конвейер (ADR 079).
- [ ] **Батчинг скаляров для логов**: Все логируемые скаляры упаковываются в один тензор: `torch.stack([...]).cpu().tolist()` или передаются через `xm.add_step_closure(...)`.
- [ ] **Константный Learning Rate**: Значение `param_group['lr']` не переназначается на каждом шаге из Python float. Используется PACE с постоянным LR либо обновление LR раз в 50+ шагов (`if step % 50 == 1:`).
- [ ] **Тензорные гиперпараметры (Pullback Alpha)**: Любые динамические коэффициенты (например, `pullback_alpha`) создаются как `torch.Tensor(..., device=dev)` без `requires_grad` и обновляются тензорными операциями (ADR 081).
- [ ] **Детерминированное ветвление в графе**: НЕТ условиям вроде `if torch.rand(1).item() < 0.5:`. Вместо них используется фиксированная маска по батчу (например, `sc_mask` зануляет ровно 50% батча) — это удерживает строго 1 постоянный граф в кеше XLA.

### Блок C: DataLoader и Форма Тензоров
- [ ] **`drop_last=True`**: Все DataLoader'ы (train и val) обязаны иметь `drop_last=True`. Неполный финальный батч инвалидирует граф XLA и вызывает долгую перекомпиляцию.
- [ ] **Статичные формы тензоров**: Никаких булевых масок переменной длины внутри графа (`tensor[mask]` где `mask.sum()` меняется от батча к батчу). Все padding-маски обрабатываются через умножение на 0 или `torch.where`.
- [ ] **Генерация шума/распределений на CPU**: Сложные генераторы случайных величин (например, `torch.distributions.Beta`) вызываются на стороне CPU, а полученные тензоры передаются на TPU единым трансфером `.to(device)` (ADR 097).
- [ ] **Оптимизация DataLoader**: Включены `persistent_workers=True` и `prefetch_factor=4` (или 8) для устранения простоя TPU в ожидании хоста.

### Блок D: Память и Архитектура (OOM Prevention)
- [ ] **Gradient Checkpointing выключен для ModernBERT**: На XLA Sliding Window Attention в ModernBERT конфликтует с GC и вызывает утечку 90+ ГБ HBM (ADR 005). Шардирование весов выполняется через SPMD/FSDP.
- [ ] **Своевременный `xm.mark_step()`**:
  - Вызывается в конце каждого шага обучения (`optimizer.step()`, затем `xm.mark_step()`).
  - Вызывается в конце каждого шага валидации внутри цикла, чтобы очистить временный граф батча.
  - **НЕ вызывается** между прямым и обратным проходами (`loss.backward()`), чтобы не дробить граф.
- [ ] **Очистка ссылок на тензоры**: Явный `del loss, out, metrics` в конце шага, чтобы сборщик мусора Python не удерживал графы autograd.
- [ ] **НЕТ `lerp_` со скаляром**: Не использовать `tensor.lerp_(..., weight=scalar)` из-за бага PyTorch/XLA, аллоцирующего CPU-тензор. Заменять на `tensor.sub_(diff * alpha_tensor)`.

### Блок E: FSDP, SPMD и Multi-Processing
- [ ] **Послойная обертка модулей**: Не оборачивать динамические контейнеры (словари `model.ca_layers`) целиком. Оборачивать каждый слой отдельно:
  ```python
  for key in model.ca_layers.keys():
      model.ca_layers[key] = SpmdFullyShardedDataParallel(model.ca_layers[key], mesh=mesh)
  ```
- [ ] **Загрузка весов ДО FSDP**: Чекпоинт базовой модели загружается ДО оборачивания в `SpmdFullyShardedDataParallel`. Загрузка весов после FSDP разрушает шардинг `xs.mark_sharding` (ADR 006).
- [ ] **Уникальность групп оптимизатора**: При разделении параметров по разным LR убедитесь, что множества параметров не пересекаются (`if p.requires_grad and "ca_layer" not in name:`).

### Блок F: EMA, Валидация и Чекпоинты
- [ ] **In-place обновление теней EMA при Resume**:
  ```python
  # ПРАВИЛЬНО:
  if target_key in ema.shadow:
      ema.shadow[target_key].copy_(v.to(device=ema.shadow[target_key].device, dtype=torch.float32))
  # НЕПРАВИЛЬНО (ломает девайс и шардинг):
  ema.shadow.update(ema_update)
  ```
- [ ] **Валидация по EMA-весам**:
  ```python
  model.eval()
  ema.apply(actual_model)
  try:
      with torch.no_grad():
          # Валидация
  finally:
      ema.restore(actual_model)
      model.train()
  ```
- [ ] **Сохранение через `xm.save`**: Для сохранения чекпоинта использовать `xm.save(state_dict, path, master_only=True)`. Обычный `torch.save` без проверки ранга вызовет гонку записи всеми 8 процессами.

### Блок G: Дуальные пути (Kaggle vs Local Resolution)
- [ ] **Никакого жесткого хардкода `/kaggle/`**: Скрипты должны поддерживать как запуск в облаке Kaggle, так и локальный Smoke Test на CPU.
- [ ] **Паттерн `resolve_resource_path`**: Использовать относительные локальные пути (`PROJECT_ROOT / "storage/components/sep_token.pt"`) с поиском в `/kaggle/input` при работе на Kaggle:
  ```python
  def resolve_resource_path(filename: str, local_rel_path: str = "") -> str:
      if Path("/kaggle/input").exists():
          for match in Path("/kaggle/input").rglob(filename): return str(match)
      return str(Path(os.environ.get("PROJECT_ROOT", "C:/Experiments/BEBLaDII")) / local_rel_path)
  ```
- [ ] **Read-only vs Writable**: Всегда сохранять чекпоинты и временные файлы строго в `/kaggle/working/` (или локальный каталог), но **никогда** в read-only `/kaggle/input`.

---

## 3. Референсный шаблон шага обучения (Golden Pattern)

```python
# Очищаем градиенты
optimizer.zero_grad()

# Прямой проход (in-graph детерминированный)
outputs = model(inputs, sc_mask=deterministic_sc_mask)
loss, metrics_dict = compute_loss(outputs)

# Обратный проход
loss.backward()

# Клиппинг градиентов (XLA-safe)
torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

# Шаг оптимизатора (с барьером XLA)
xm.optimizer_step(optimizer)

# Тензорный EMA Pullback (без перекомпиляции)
ema_tracker.step_tensor(actual_model, alpha_tensor=pullback_alpha_tensor)

# Асинхронное логирование (1 разрез графа)
if global_step % log_steps == 0:
    stacked_metrics = torch.stack([loss.detach(), *metrics_dict.values()])
    def async_log(step_val, metrics_tensor):
        values = metrics_tensor.cpu().tolist()
        wandb.log({"loss": values[0], ...}, step=step_val)
    xm.add_step_closure(async_log, args=(global_step, stacked_metrics))

# Фиксация границы шага
xm.mark_step()
```

---

## 4. Дополнительные материалы
- Карта сопоставления путей Kaggle и локального репозитория: [kaggle_paths_mapping.md](./references/kaggle_paths_mapping.md)
- Детальный разбор исторических кейсов: [tpu_pitfalls_case_studies.md](./references/tpu_pitfalls_case_studies.md)
- Скрипт синтаксической проверки (Level 1): [verify_tpu_syntax.py](./scripts/verify_tpu_syntax.py)
- Скрипт семантического AST-анализа (Level 2): [verify_tpu_script.py](./scripts/verify_tpu_script.py)
- Раннер полного 3-уровневого тестирования (Level 3): [run_tpu_smoke_test.py](./scripts/run_tpu_smoke_test.py)
