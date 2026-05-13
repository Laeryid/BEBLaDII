<!-- last_verified: 2026-05-13 -->
# KI: Kaggle Operations

## Overview
Инструменты и процедуры для работы в среде Kaggle: синхронизация кода, управление датасетами и экстренное сохранение состояния.

## Key Components

### Synchronization & Deployment
| Script | Platform | Purpose |
|---|---|---|
| `sync_to_kaggle.ps1` | Local (PS) | Синхронизация локальных изменений в `src` и `scripts` с Kaggle Dataset через Kaggle CLI. |
| `kaggle_emergency_save.py` | Kaggle | Сохранение текущих чекпоинтов в `/kaggle/working` при угрозе прерывания сессии. |

### Debugging & Safety
| Script | Purpose | Details |
|---|---|---|
| `debug_nan_kaggle.py` | Поиск NaN. | Специализированный трекер градиентов для условий ограниченного доступа Kaggle. |
| `check_tpu_nprocs.py` | Проверка TPU. | Валидация доступности ядер TPU и настройки топологии перед запуском. |


## Related KIs
- [[KI_experiments.md]] (via `experiments/train_phase1_kaggle.py`)

## Non-obvious Details
- **Kaggle API Limits**: Скрипт `sync_to_kaggle.ps1` может вызвать ошибку 429 при слишком частом запуске. Рекомендуется группировать изменения.
- **Dataset Update**: Обновление датасета через CLI занимает время (до нескольких минут). Ноутбук увидит изменения только после перезапуска сессии или переподключения датасета.

## Common Pitfalls
- **Read-only Filesystem**: Попытка записи в `/kaggle/input` приведет к ошибке. Всегда используйте `/kaggle/working` или `/tmp` для временных файлов.
- **Lost Artifacts**: Если сессия прервана без вызова `kaggle_emergency_save.py`, все промежуточные веса будут потеряны.
