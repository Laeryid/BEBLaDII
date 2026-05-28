<!-- created: 2026-05-28 -->
# ADR 028: Ручная реализация Huber Loss и метрика сферичности Norm CV

## Context and Problem

В ADR-027 была зафиксирована замена MSE на Huber Loss для `cov_loss` и `v_state` с целью устранения взрывного O(S³) градиента. Реализация использовала стандартные функции PyTorch:

```python
# Старый вариант (ADR-027, первая версия)
cov_loss = 2.0 * F.huber_loss(cov_off_diag, torch.zeros_like(cov_off_diag), delta=1.0, reduction='sum') / D
prior_loss = m_state.pow(2).mean() + 2.0 * F.huber_loss(v_state, torch.ones_like(v_state), delta=1.0) + 0.1 * cov_loss
soft_var_penalty = 2.0 * F.huber_loss(F.relu(v_state - 1.5), torch.zeros_like(v_state), delta=1.0)
```

**Проблема:** `F.huber_loss` с `reduction='sum'` требует построения вспомогательного тензора `zeros_like(cov_off_diag)` (или `ones_like`) — матрицы размером **(D × D)** для ковариационной матрицы. При D=1024 это **~4 MB дополнительной памяти на слой** только под целевой тензор. На TPU/XLA это также мешает фьюзингу операций в единый XLA-kernel, так как `F.huber_loss` внутри строит граф с дополнительными аллокациями.

Дополнительный контекст: по ходу анализа архитектуры возник вопрос о том, насколько текущее обучение гарантирует, что векторы лежат **на сфере**, а не в шаровом слое ненулевой толщины. Все активные лоссы (centered cosine, RKD, cov_loss) инвариантны к норме вектора — явного сигнала, фиксирующего `‖x‖ = r`, нет.

## Decisions Made and Lessons Learned

### 1. Ручная реализация Huber (оптимизация, коммит `e8b0ed4`)

**What was tried and didn't work:** Стандартный `F.huber_loss` с `zeros_like` / `ones_like` — лишние аллокации, блокировка XLA-фьюзинга.

**Successful Solution (Best Practice):** Заменить `F.huber_loss(x, target, delta=1.0)` на `torch.where`-реализацию без создания вспомогательного тензора:

```python
# 2*Huber(x, 0, δ=1): x^2 при |x|<1, 2*(|x|-0.5) при |x|≥1
_abs = x.abs()
result = 2.0 * torch.where(_abs < 1.0, 0.5 * x.pow(2), _abs - 0.5)
```

Для `v_state - 1.0` и `relu(v_state - 1.5)` — аналогично, вычисляется разница напрямую без `ones_like`. XLA компилирует это в единый kernel (`where` + арифметика — чистый element-wise граф).

**Три места изменений в `loss.py`:**
- `cov_loss` (все слои): `zeros_like(cov_off_diag)` → ручной `torch.where`
- `prior_loss` (слой 40): `ones_like(v_state)` → `_v_diff = v_state - 1.0`, ручной `torch.where`
- `soft_var_penalty` (слои 20, 30): `zeros_like(v_state)` → `_svp = relu(v_state-1.5)`, ручной `torch.where` (relu гарантирует ≥0, поэтому `abs` = `identity`)

### 2. Метрика Norm CV — диагностика сферичности (текущее изменение)

**Проблема:** Неизвестно, насколько «тонкой» является сфера, на которой лежат векторы слоя 40. Это критично для Phase 8 (диффузия): если слой толстый (CV > 0.1), slerp не будет адекватным и потребуется адаптация noise schedule.

**Successful Solution (Best Practice):** Добавить в `metrics` безградиентную метрику **Norm CV** (коэффициент вариации норм токенов):

```python
# Вычисляется на сырых BERT-векторах слоя 40, только для активных (non-padding) токенов
active_norms = s_norms[attention_mask.bool()]
_norm_mean = active_norms.mean().clamp(min=1e-6)
_norm_std  = active_norms.std(unbiased=False)
metrics["norm_cv_l40_raw"] = (_norm_std / _norm_mean).detach()
```

**Интерпретация значений:**
- `CV < 0.05` → практически идеальная сфера; slerp в Phase 8 корректен без изменений
- `CV 0.05–0.10` → тонкий слой; незначительная зависимость от нормы, slerp приемлем
- `CV > 0.10` → слой толстый; перед Phase 8 необходима явная нормализация (`F.normalize`) или адаптация noise schedule диффузии

Метрика **не влияет на лосс** — только наблюдательная.

## Impact

- **Positive (Huber):** Экономия ~4 MB памяти на слой (устранение `zeros_like` / `ones_like` для D×D матрицы). XLA может компилировать `torch.where + pow` в один element-wise kernel без промежуточных аллокаций. Математически идентично предыдущей реализации при δ=1.
- **Positive (Norm CV):** Впервые появляется возможность наблюдать геометрию латентного пространства в терминах «сферичности». Это напрямую влияет на выбор стратегии диффузии в Phase 8 (slerp vs. Euclidean noise schedule vs. явная нормализация).
- **Negative:** Ручная реализация Huber менее читаема, чем `F.huber_loss`. Компенсируется комментариями в коде с явной записью математической формулы.
- **Dependency:** ADR-027 (решение о переходе на Huber) → ADR-028 (оптимизация реализации Huber + добавление диагностики).
