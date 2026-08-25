# Phase 4: Per-Token Diffusion с Иерархическим Шумом

**Дата:** 2026-08-25  
**Статус:** Планирование  
**Предшествует:** Phase 3 (каноническая сферическая диффузия, ADR-060)  
**Связанные файлы:** per_token_diffusion_vs_uniform.md

---

## Концепция

Phase 4 обучает модель работать в **двумерном пространстве состояния** каждого токена:

| Сигнал | Семантика | При обучении | При инференсе |
|---|---|---|---|
| 	_global | Фаза диффузии / «температура системы» | Сэмплируется равномерно из U(0, 1) | Задаётся Оркестратором (номер шага) |
| 	_reported[i] | Заявленная уверенность в токене | Честная или ложная (см. ниже) | Выход ConfidenceHead: 1 - confidence_i |

**Ключевая семантика 	_global:**
- **Ранняя фаза** (	_global → 1): допустимо менять даже формально «уверенные» токены, если контекст требует. Система находится в режиме глобального поиска.
- **Поздняя фаза** (	_global → 0): опорные векторы меняются неохотно. Трогаем только то, что само считает себя ненадёжным.

Модель должна **знать** 	_global — именно она принимает решение, стоит ли переписывать токен с низким 	_reported, опираясь на контекст соседей.

---

## Механизм генерации обучающих сэмплов

### Два типа токенов в батче

Для каждого токена i при заданном 	_global принимается решение:

`
is_false_confident[i] ~ Bernoulli(p_false(t_global))
`

#### Обычный токен (is_false_confident = False)
`
t_actual[i]   ~ U(t_min(t_global), t_max(t_global))
t_reported[i] = t_actual[i]          # честный сигнал
z_noisy[i]    = noise(z_clean[i], t_actual[i])
`

#### Ложно-уверенный токен (is_false_confident = True)
`
t_actual[i]   ~ U(t_global * 0.6, t_global)   # реально сильно зашумлён
t_reported[i] ~ U(0.02, 0.15)                  # притворяется чистым
z_noisy[i]    = noise(z_clean[i], t_actual[i]) # зашумление по реальному t
`

Модель видит 	_reported[i] и 	_global, но лосс считается относительно z_clean[i].

### Вероятность ложной уверенности

`python
p_false(t_global) = t_global ** alpha    # alpha in [1.5, 2.0]

# Примеры:
# t_global = 0.9  ->  p_false ~ 0.77   (ранняя фаза: много ловушек)
# t_global = 0.5  ->  p_false ~ 0.35   (средняя фаза)
# t_global = 0.2  ->  p_false ~ 0.06   (поздняя фаза: почти нет ловушек)
`

### Диапазон зашумления обычных токенов

`python
t_min(t_global) = max(0.0, t_global - delta)
t_max(t_global) = min(1.0, t_global + delta)
# delta ~ 0.3  ->  широкий диапазон вокруг t_global, не выходящий за границы
`

---

## Что модель учится делать

`
Поздняя фаза (t_global ~ 0):
  token[i] сообщает t=0.05 -> скорее всего правда -> доверяй, не трогай

Ранняя фаза (t_global ~ 0.9):
  token[i] сообщает t=0.05 -> может лгать -> проверяй через контекст соседей
  если соседи семантически несогласны -> перепиши, невзирая на уверенность
`

Таким образом формируется **скептицизм, управляемый фазой**: модель учится не слепо
доверять локальным сигналам уверенности, а взвешивать их относительно глобального
контекста итерации.

---

## Архитектура: изменения относительно Phase 3

### AdaLN: два входа вместо одного

**Phase 3 (текущий):**
`
t_emb [B, 1, dim]  ->  shift/scale [B, 1, D]  (broadcast на весь T)
`

**Phase 3.5 (per-token, без t_global):**
`
t_emb [B, T, dim]  ->  shift/scale [B, T, D]
`

**Phase 4 (иерархический шум):**
`python
t_token_emb  = embed(t_reported)   # [B, T, dim]
t_global_emb = embed(t_global)     # [B, 1, dim]  (broadcast)

# Конкатенация, а не сложение — модель сама учится их взвешивать
cond = proj(torch.cat([t_token_emb, t_global_emb.expand_as(t_token_emb)], dim=-1))
# cond: [B, T, D]  ->  per-token shift/scale
`

> **Важно:** 	_global и 	_reported[i] — разные входы, не суммируются.
> Это позволяет модели улавливать противоречие:
> «t_global = 0.9, а t_reported[i] = 0.05 -> подозрительно, проверь по контексту».

### Совместимость с весами Phase 3

Веса Phase 3 загружаются как инициализация. Слои proj для нового AdaLN инициализируются
с нулевым выходом (zero-init), чтобы на старте модель вела себя идентично Phase 3.

---

## Функция потерь

### Ключевое правило: лосс по 	_actual, не по 	_reported

`python
loss[i] = mse(predicted[i], z_clean[i])

# Веса лосса:
# НЕ Min-SNR по t_reported — он может лгать!
# Взвешивание по t_actual[i]:
w[i] = min_snr_weight(t_actual[i])   # или uniform (1.0) для простоты
`

Использование 	_reported[i] для весов дало бы ложно-лёгкий лосс на «уверенных» токенах,
которые на самом деле сильно зашумлены — модель научилась бы их игнорировать.

### Min-SNR в Phase 4

В Phase 3 Min-SNR был необходим (downweight высокого t — там нет сигнала при uniform t).
В Phase 4 ситуация меняется: токены при высоком 	_actual теперь информативны через контекст.
Рекомендуется убрать Min-SNR полностью или взвешивать равномерно (w=1.0).

---

## ConfidenceHead: обучение после DUS (Phase 4b)

ConfidenceHead обучается **отдельно** после Phase 4a, на уже обученном DUS.
Задача: предсказывать качество восстановления без доступа к z_clean.

`python
target[i]    = cos_sim(DUS_output[i], z_clean[i])
predicted[i] = ConfidenceHead(DUS_output[i])
loss         = mse(predicted, target)
`

При инференсе: 	_reported[i] = 1 - ConfidenceHead(output[i])

---

## Предлагаемый порядок фаз

`
Phase 3    Базовый денойзинг (uniform t, одинаковый для всей фразы)
              Критерий перехода: cos_h39_t_low > 0.80
              |
Phase 3.5a Per-token AdaLN + random t_i на токен (дообучение DUS)
              Модель учится использовать соседей-якорей
              |
Phase 4a   Иерархический шум: t_global в AdaLN + ложно-уверенные токены
              Модель учится скептицизму, управляемому фазой
              |
Phase 4b   ConfidenceHead: предсказание cos_sim(output, z_clean) per-token
              Голова умеет измерять неопределённость без знания z_clean
              |
Phase 5    CA_Prompt: поверх модели, понимающей per-token уверенность
`

---

## Сравнение фаз

| | Phase 3 | Phase 3.5a | Phase 4a |
|---|---|---|---|
| Задача | P(z_clean given z_noisy, t) | P(z_clean[i] given z_noisy[i], context) | Скептицизм к t_reported при высоком t_global |
| AdaLN вход | t [B,1,D] | t_per_token [B,T,D] | t_reported [B,T,D] + t_global [B,1,D] |
| Ложные уверенности | Нет | Нет | p_false(t_global) = t_global^alpha |
| Лосс-веса | Min-SNR по t | Uniform | По t_actual, игнор t_reported |
| Min-SNR | Да | Убрать | Убрать |

---

## Поведение Оркестратора при инференсе

`python
confidence  = ConfidenceHead(DUS_output)   # [B, T] in [0, 1]
t_per_token = 1 - confidence

# t_global задаётся шагом итерации (cosine schedule или линейный)
t_global = scheduler.get_t(step)

# Оркестратор решает, какие токены перезашумить
if t_global > THRESHOLD_EARLY:
    # Ранняя фаза: модели разрешено переписывать даже уверенные токены
    allow_resample = torch.ones_like(confidence, dtype=torch.bool)
else:
    # Поздняя фаза: трогаем только то, что само считает себя ненадёжным
    allow_resample = (confidence < CONFIDENCE_THRESHOLD)

# Перезашумляем только разрешённые токены
z_next = resample(z_current, t_per_token, mask=allow_resample)
`

---

## Гиперпараметры (начальные значения для поиска)

| Параметр | Значение | Комментарий |
|---|---|---|
| lpha | 1.5–2.0 | Крутизна p_false(t_global) |
| t_actual для ложных | U(t_global*0.6, t_global) | Реальный шум подставного токена |
| t_reported для ложных | U(0.02, 0.15) | Ложная уверенность |
| delta для обычных | 0.3 | Ширина диапазона t_actual вокруг t_global |
| LR Phase 4a | ~0.1x Phase 3 LR | Fine-tune поверх Phase 3 весов |
| Zero-init proj | Да | Совместимость с Phase 3 на старте |

---

## Связанные файлы

- [per_token_diffusion_vs_uniform.md](per_token_diffusion_vs_uniform.md) — базовая концепция per-token диффузии (Phase 3.5)
- 	ool_calling_via_latent_inpainting.md — применение per-token t для JSON-инпейнтинга
- ADR-058: Canvas Format и якорь \<|thoughts|>- ADR-060: Каноническая диффузия на сфере (Phase 3)
- ADR-074: Удаление костылей AdaLN после фикса бага с данными
