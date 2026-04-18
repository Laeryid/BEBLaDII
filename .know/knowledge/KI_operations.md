<!-- last_verified: 2026-04-18 -->
# KI: Data & Ops Scripts

## Overview
Технические и отладочные скрипты для подготовки данных, инициализации компонентов, синхронизации с Kaggle и проведения Smoke-тестов.

## Key Components

### Data Preparation
| Class / Function / Script | File | Purpose |
|---|---|---|
| `prepare_phase1_data_v2.py` | `scripts/` | Токенизация CulturaX и сохранение в Parquet для Фазы 1. |
| `prepare_kaggle_data.py` | `scripts/` | Формирование структуры датасета для Kaggle (arrow schema, metadata). |
| `downsample_magpie.py` | `scripts/` | Фильтрация и уменьшение объема данных Magpie для ускорения итераций. |
| `download_datasets.py` | `scripts/` | Автоматическая загрузка необходимых датасетов из HuggingFace/Kaggle. |
| `download_teacher.py` | `scripts/` | Загрузка весов учителя (Qwen) и токенизатора. |

### Kaggle Operations
| Class / Function / Script | File | Purpose |
|---|---|---|
| `kaggle_emergency_save.py` | `scripts/` | Экстренное сохранение чекпоинтов при угрозе остановки сессии Kaggle. |
| `sync_to_kaggle.ps1` | `scripts/` | PowerShell скрипт для синхронизации локальных правок с Kaggle Dataset. |
| `debug_nan_kaggle.py` | `scripts/` | Специализированный отладчик для поиска причин появления NaN в условиях Kaggle. |

### Component Initialization
| Class / Function / Script | File | Purpose |
|---|---|---|
| `build_prebuilt_latentbert.py` | `scripts/` | Сборка локального LatentBERT из весов HF (BERT) и сохранение в `storage/prebuilt`. |
| `fix_model_save.py` | `root` | Хак для исправления структуры сохраненных моделей при миграции версий. |

### Smoke Tests
| Class / Function / Script | File | Purpose |
|---|---|---|
| `smoke_test_qwen.py` | `root` | Быстрая проверка работоспособности базовой модели Qwen. |
| `smoke_test_forward.py` | `scripts/` | Полный тест прямого прохода всей системы (Vect, Proj, Model). |
| `nan_debug.py` | `root` | Инструмент для перехвата и локализации градиентных взрывов. |

### Loading and Weight Matching
| Class / Function / Script | File | Purpose |
|---|---|---|
| `smart_load_weights` | `experiments/train_phase1_kaggle.py` | Универсальная загрузка с поддержкой многокомпонентных файлов, фильтрацией учителя и нечетким матчингом по суффиксам. |

## Non-obvious Details
- **Fuzzy Weight Matching**: `smart_load_weights` использует сопоставление по "хвостам" (suffixes) имен тензоров. Это позволяет загружать веса даже при изменении глубины вложенности модели (например, `student.model.layers` vs `student.model.model.layers`).
- **Strict Teacher Exclusion**: Все тензоры, начинающиеся на `teacher.*`, принудительно исключаются из процесса загрузки для предотвращения порчи замороженных весов.
- **Parquet vs Arrow**: `prepare_phase1_data_v2.py` использует Parquet для промежуточного хранения, но `prepare_kaggle_data.py` конвертирует всё в Arrow для оптимального чтения через `datasets`.
- **Prebuilt LatentBERT**: Скрипт `build_prebuilt_latentbert.py` обязателен перед первым запуском Фазы 1, так как система ожидает наличие готовых тензоров в `storage/`.

## Common Pitfalls
- **Out of Disk Space**: При работе `prepare_phase1_data_v2.py` требуется минимум 200GB свободного места из-за кэширования HuggingFace.
- **Kaggle API Limit**: Частые запуски `sync_to_kaggle.ps1` могут привести к блокировке API (429 Too Many Requests). Используйте батчинг изменений.
