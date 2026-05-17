<!-- last_verified: 2026-05-13 -->
# KI: Data Preparation

## Overview
Процессы и скрипты для подготовки данных, инициализации компонентов и формирования датасетов для обучения Фазы 1 и Фазы 4.

## Key Components

### Dataset Engineering
| Script | Purpose | Details |
|---|---|---|
| `prepare_phase1_data_v2.py` | Токенизация CulturaX. | Сохраняет в Parquet. Требует >200GB диска для кэша HF. |
| `prepare_kaggle_data.py` | Формирование Kaggle Dataset. | Конвертация Parquet -> Arrow, создание метаданных и схем. |
| `downsample_magpie.py` | Фильтрация Magpie. | Уменьшение объема данных для быстрой отладки Reasoning. |
| `download_datasets.py` | Загрузка данных. | HF/Kaggle CLI интеграция для автоматизации. |

### Infrastructure Initialization
| Script | Purpose | Details |
|---|---|---|
| `build_prebuilt_latentbert.py` | Сборка базового ModernBERT. | Извлекает веса и сохраняет в `storage/prebuilt`. Обязателен перед первым запуском. |
| `init_components.py` | Инициализация структуры. | Создает папки в `storage/` и веса проекторов по умолчанию. |
| `download_teacher.py` | Загрузка учителя. | Скачивание Qwen-7B и его токенизатора. |



## Related KIs

## Non-obvious Details
- **Parquet vs Arrow**: Система использует Parquet для промежуточного хранения, но `prepare_kaggle_data.py` конвертирует всё в Arrow для оптимального чтения через библиотеку `datasets` в Kaggle.
- **Disk Requirements**: При работе токенизатора CulturaX убедитесь в наличии свободного места в `~/.cache/huggingface`.

## Common Pitfalls
- **Out of Disk Space**: Ошибка `No space left on device` при токенизации. Решение: очистить кэш или использовать внешний диск.
- **Prebuilt Missing**: Запуск обучения без `build_prebuilt_latentbert.py` приведет к `FileNotFoundError` при загрузке студента.
