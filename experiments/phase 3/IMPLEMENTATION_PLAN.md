# Phase 3: Output Projector — Implementation Plan

> **Статус:** Ожидает одобрения  
> **Дата:** 2026-06-11  
> **Основан на:** `ideas/phase3_output_projector.md`, `reports/phase2_reasoning_and_topology_report.md`

---

## Контекст и цель

**Задача:** обучить `OutputProjector` (OP) — компонент, замыкающий диффузионный цикл:

```
L40_ctx (1024-dim, l40-пространство)
    → OutputProjector
        → Z_target (1024-dim, l0-пространство)
```

**Ключевые факты о пространствах (из Phase 2 отчёта):**

| Метрика | l40 | l0 | Задача OP |
|---|---|---|---|
| `rank1_ratio` | 0.654 | 0.353 | Снизить (~2×) |
| `norm_cv` | 0.089 (min: 0.035) | 0.016 | Схлопнуть |
| `isotropy` | 0.2028 | 0.2205 | Сохранить |
| `mu_norm` | — | ~20.4 | Воспроизвести |

Вариация нормы l40 (norm_cv=0.089) — **устойчивое равновесие** между Prior Loss и RKD,
несёт реальный сигнал от учителя. OP должен перекодировать его в угловую компоненту l0.

---

## Архитектурные решения

### OutputProjector

```python
# Финальная архитектура (стартовая; 2 слоя, без LayerNorm на входе)
Linear(1024 → 2048) → GELU → Linear(2048 → 1024)
```

**Обоснование:**
- **Без LayerNorm на входе:** норма l40 несёт реальный учительский сигнал
  (norm_cv=0.089 — равновесие, а не артефакт). OP видит сырую норму и
  перераспределяет её в направление l0-вектора.
- **Расширение 2048:** даёт пространство для манёвра при перекодировании
  топологии; прямое `1024→1024→1024` вырождается при малых данных.
- **Без Centered Cosine в loss:** таргет Z_hat_target является суперпозицией
  l0-векторов с near-uniform нормами. Huber к правильному таргету несёт
  всю необходимую геометрическую информацию.
- **2 слоя вместо 3:** train/val gap — главный критерий; 3-й слой добавляется
  только если геометрические метрики l0 не достигаются с 2 слоями.

### Loss

```python
L = Huber(OP(L40_ctx), Z_hat_target)
```

---

## Стратегия данных: Soft Dictionary Matching

### Шаг 1. Предвычисление словарей (локально, CPU/GPU)

Два словаря по всему словарю Qwen (~152 064 токенов):

**Оба словаря строятся за один forward pass** (D_X0 — вход в DUS, D_L40 — выход):

```python
# Один прогон на батч токенов:
h = input_projector(qwen_embeddings)   # → D_X0[i]  (позиция 1, после InputProjector)
out = dus_model(h)                      # → D_L40[i] (позиция 1, слой 40)
```

Реализуется через forward hook на выходе InputProjector — отдельного прогона для D_X0 не нужно.

| Словарь | Источник | Shape | Размер FP16 |
|---|---|---|---|
| D_X0 | Выход InputProjector (mu_head) | [152064, 1024] | ~300 MB |
| D_L40 | Выход DUS слоя 40 | [152064, 1024] | ~300 MB |

**Важно:** оба словаря строятся из **одного чекпоинта** (финальный Phase 2, ~36k шагов).
DUS и InputProjector заморожены на весь Phase 3.

Хранить отдельно `D_L40_norm = normalize(D_L40)` для cosine-поиска.

**Оценка времени предвычисления:**

> Железо: Intel Iris Xe (интегрированная GPU, CPU-режим PyTorch, MKL).
> seq_len=2 — ключевое преимущество: attention тривиален, FFN хорошо батчируется.

| Batch size | Оценка |
|---|---|
| 64 | ~3–5 часов |
| 128 | ~2–4 часа (16 GB RAM позволяет) |

Пиковое потребление RAM: DUS FP16 (~1.1 GB) + D_X0 + D_L40 (~600 MB) ≈ **~1.7 GB** — комфортно вписывается в 16 GB.

Словари строятся **один раз**, переиспользуются для всех экспериментов с tau и архитектурой.

**После построения — загрузка на GCS:**
```bash
gsutil cp storage/dictionaries/D_X0.pt  gs://bebladii-weigths/phase3/dictionaries/
gsutil cp storage/dictionaries/D_L40.pt gs://bebladii-weigths/phase3/dictionaries/
gsutil cp storage/dictionaries/D_L40_norm.pt gs://bebladii-weigths/phase3/dictionaries/
```
TPU-скрипт загружает словари из GCS в начале обучения (единоразово).

### Шаг 2. Soft-разложение (on-the-fly на TPU)

```python
# 1. Cosine similarity (не raw dot product — norm_cv=0.089 в D_L40 искажает dot product)
scores = normalize(L40_ctx) @ D_L40_norm.T / tau      # [seq_len, 152064]

# 2. Top-k отбор + softmax
topk_idx, topk_scores = topk(scores, k=k)
alpha = softmax(topk_scores)                           # [seq_len, k]

# 3. Перенос коэффициентов на D_X0
Z_hat_raw = alpha @ D_X0[topk_idx]                    # [seq_len, 1024]

# 4. Восстановление длины (адаптивное, без фиксированной сферы)
L_expected = alpha @ norm(D_X0[topk_idx])             # [seq_len]
Z_hat_target = normalize(Z_hat_raw) * L_expected      # [seq_len, 1024]
```

### Шаг 3. Данные предложений

- **Источник:** Magpie-Reasoning-V2 + OpenThoughts-114k + CulturaX (RU + CS)
- **Длина:** до 512 токенов включая якорь `<|thought|>` (chunk_size = 512 − len(anchor_ids))
- **Объём:** ~150k чанков (50k Magpie + 50k OT + 25k CulturaX RU + 25k CulturaX CS)
- **Формат:** Parquet, колонка `input_ids: list[int]`

> [!IMPORTANT]
> **Датасет pre-tokenized.** Файлы уже содержат токены (`input_ids`), а не сырой текст.
> Токенизация выполнена в `prepare_phase3_data.py` на этапе подготовки.
>
> **Следствие:** тренировочный скрипт (`train_phase3_TPU.py`) **не должен**
> загружать токенизатор и не должен вызывать `tokenizer.encode()`.
> Батч подаётся напрямую в embedding-таблицу Qwen через `qwen_embed_weight[input_ids]`.

**GCS путь к датасету:** `gs://bebladii-datasets-us/phase 3/train_data/`  
**GCS путь к словарям:** `gs://bebladii-datasets-us/phase 3/dictionaries/`

---

## Экспериментальный протокол: выбор tau и архитектуры

### Метрики для диагностики

**Группа A — качество alpha (диагностика tau):**

| Метрика | Формула | Целевой диапазон |
|---|---|---|
| `alpha_k_eff` | `exp(H(alpha))`, H = энтропия | k/6 < k_eff < k/2 |
| `alpha_top1_self` | top-1 alpha = входной токен (одиночные токены) | 0.70 – 0.95 |
| `alpha_entropy_std` | std(H(alpha)) по батчу | > 0.3 |

**Правила выбора tau:**
```
k_eff < k/6           → tau слишком мал (вырождение в 1-NN)
k_eff > k/2           → tau слишком велик (каша)
alpha_top1_self < 0.70 → tau увеличить
alpha_top1_self > 0.95 → tau уменьшить
```

**Группа B — геометрия OP (диагностика архитектуры):**

| Метрика | Цель | Источник истины |
|---|---|---|
| `val_cosine_sim` | max | Качество проекции |
| `val_huber` | убывает без плато | Сходимость |
| `train_val_gap` | < 0.05 | Переобучение |
| `op_norm_cv` | → 0.016 | l0-отчёт |
| `op_rank1_ratio` | → 0.353 | l0-отчёт |
| `op_isotropy` | → 0.220 | l0-отчёт |

**Правила выбора числа слоёв:**
```
op_norm_cv >> 0.016  И  train_val_gap мал  → добавить 3-й слой
op_norm_cv ≈ 0.016   ИЛИ train_val_gap растёт → 2 слоя достаточно
```

### Этапы эксперимента

**Этап 1 — выбор tau** (фиксируем: 2 слоя OP, k=28):

| Запуск | tau | Ожидаемый k_eff |
|---|---|---|
| tau_005 | 0.05 | ~2–4 (скорее всего мал) |
| tau_010 | 0.10 | ~5–8 |
| tau_020 | 0.20 | ~10–14 |
| tau_050 | 0.50 | ~15–20 (скорее всего велик) |

Длина: ~2 000 шагов на TPU v6e. Смотрим Группу A.

**Этап 2 — выбор архитектуры** (фиксируем tau из Этапа 1):

| Запуск | Архитектура | Параметры |
|---|---|---|
| op_2layer | `1024 → 2048 → 1024` | ~4.2M |
| op_3layer | `1024 → 2048 → 1024 → 1024` | ~6.3M |

Длина: ~10 000 шагов. Смотрим Группу B.

---

## Инфраструктура

### Замороженные компоненты
- `DUSModel` (40 layers): заморожен, `.eval()`
- `InputProjector`: заморожен, `.eval()`
- `DeepSeek-R1` Teacher: **не нужен в Phase 3** (нет дистилляции)

### Обучаемый компонент
- `OutputProjector` только

### TPU
- v6e-4, SPMD FSDP (аналогично Phase 2)
- Статические XLA-формы: `[batch_size, seq_len, 1024]`
- GCS sync каждые 500 шагов → `gs://bebladii-weigths/checkpoints/phase3/`

### Чекпоинт Phase 2
- Загружать веса DUS + InputProjector из финального чекпоинта (~36k шагов)
- OP инициализируется случайно

---

## Структура файлов

```
experiments/phase 3/
├── IMPLEMENTATION_PLAN.md      ← этот файл
├── build_dictionaries.py       ← предвычисление D_L40 и D_X0 (CPU/GPU)
├── train_phase3_TPU.py         ← основной скрипт обучения
└── eval_op_geometry.py         ← замер op_norm_cv, op_rank1_ratio, op_isotropy
```

---

## Открытые вопросы (решаются в экспериментах)

- [ ] Оптимальное значение tau — Этап 1
- [ ] 2 vs 3 слоя OP — Этап 2
- [ ] Финальное k (28 vs 56) — пробовать в Этапе 1 параллельно с tau

---

## Связанные артефакты

- [`ideas/phase3_output_projector.md`](file:///C:/Experiments/BEBLaDII/ideas/phase3_output_projector.md)
- [`reports/phase2_reasoning_and_topology_report.md`](file:///C:/Experiments/BEBLaDII/reports/phase2_reasoning_and_topology_report.md)
- [`experiments/phase 2/train_phase1_TPU_fsdp.py`](file:///C:/Experiments/BEBLaDII/experiments/phase%202/train_phase1_TPU_fsdp.py)
- ADR-015, ADR-016, ADR-025, ADR-029, ADR-030
