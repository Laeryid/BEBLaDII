<!-- last_verified: 2026-05-13 -->
# KI: Model Inspection & Loading

## Overview
Инструменты для глубокого анализа структуры весов, сверки ключей и обеспечения стабильной загрузки компонентов системы (Vect, Proj, Student).

## Key Components

### Weight Analysis
| Script | Purpose | Details |
|---|---|---|
| `deep_inspect_weights.py` | Анализ структуры. | Распаковка вложенных `dict` (например, в проекторах) и вывод статистик по слоям. |
| `check_model_keys.py` | Сверка ключей. | Сравнение `state_dict` живой модели с файлом на диске для отладки `smart_load`. |
| `find_weights.py` | Поиск тензоров. | Утилита для быстрого поиска конкретных весов в больших чекпоинтах. |

### Validation
| Script | Purpose | Details |
|---|---|---|
| `verify_loading.py` | Тест загрузки. | Эмуляция процесса `smart_load` для проверки корректности маппинга слоев. |
| `test_missing_keys.py` | Проверка целостности. | Поиск отсутствующих или лишних ключей при миграции между версиями архитектуры. |



## Related KIs

## Non-obvious Details
- **Nested State Dicts**: В Phase 1 Reasoning проекторы сохраняются как вложенные словари. Обычный `torch.load` может не показать их структуру без глубокого обхода.
- **Fuzzy Matching**: Логика загрузки часто опирается на сопоставление суффиксов имен тензоров (например, `weight` или `bias`), что позволяет игнорировать изменения в префиксах (например, `student.model.` vs `model.`).

## Common Pitfalls
- **Shape Mismatch**: Загрузка весов от ModernBERT-base в ModernBERT-large. Скрипт `verify_loading.py` должен поймать это до начала обучения.
- **Teacher Weights Leak**: Случайная загрузка весов учителя в обучаемые слои студента. Всегда проверяйте логи `smart_load` на предмет исключенных ключей `teacher.*`.
