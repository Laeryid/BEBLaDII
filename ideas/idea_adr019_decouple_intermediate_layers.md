# IDEA: ADR-019 — Decoupling Intermediate Layers from Teacher Representations

> Статус: ИДЕЯ (не одобрена к реализации)
> Дата: 2026-05-23

## Суть

Полный отказ от привязки промежуточных слоев студента (l20, l30) к промежуточным слоям учителя (Qwen l14, l21). Полное удаление `feat_proj_20` и `feat_proj_30`.

## Обоснование

В Conditional Latent Diffusion один прогон сети = один шаг денойзинга, а не синтаксический парсинг токена. Промежуточные слои учителя — артефакт LLM-архитектуры без семантического аналога в диффузионной сети. Привязка l20/l30:
- Расходует ёмкость сети на невыполнимую задачу
- Создаёт конкурирующий градиентный сигнал с L_delta на l40
- Ведёт к росту cov_loss и падению изотропии (наблюдается на графиках, шаги 6k-9k)

## Схема изменений

```
До:
  layer_mapping: {20: 14, 30: 21, 40: 28}
  feat_proj_20: BERT(1024) → Qwen(3584)  ← L_state + L_delta
  feat_proj_30: BERT(1024) → Qwen(3584)  ← L_state + L_delta
  feat_proj_40: BERT(1024) → Qwen(3584)  ← L_state + L_delta

После:
  layer_mapping: {40: 28}
  feat_proj_20: УДАЛЁН
  feat_proj_30: УДАЛЁН
  feat_proj_40: BERT(1024) → Qwen(3584)  ← L_state + L_delta (без изменений)
  raw_states[20]: собираем (1024d)        ← для cov_loss / isotropy без проектора
  raw_states[30]: собираем (1024d)        ← то же
  raw_states[40]: собираем (1024d)        ← уже используется (prior loss)
```

## Затронутые файлы

| Файл | Изменение |
|------|-----------|
| `src/beb_la_dii/model/distiller.py` | `layer_mapping={40:28}`, `regularized_layers=[20,30,40]`, сбор raw_states без проекции |
| `src/beb_la_dii/model/assembler.py` | Удалить `feat_proj_20`, `feat_proj_30` |
| `src/beb_la_dii/utils/loss.py` | `layer_weights={40:1.0}`, cov_loss для l20/l30 с малым весом (0.02) |
| `experiments/train_phase1_TPU_fsdp.py` | Убрать логирование scale l20/l30, добавить `val/l20_cov_loss`, `val/l30_cov_loss` |
| `.know/decisions/019_*.md` | Новый ADR |

## Совместимость с чекпоинтами

`smart_load_weights` проигнорирует ключи `feat_proj_20/30` при `strict=False`. Риска для `feat_proj_40` и студента нет.

## Когда применять

Рекомендуется как следующий шаг после завершения текущего запуска (или при его рестарте). Особенно актуально, если `val/l40_cov_loss` продолжит расти.
