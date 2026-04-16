<!-- last_verified: 2026-04-14 -->
# KI: Utilities

## Что это
Вспомогательные модули для подготовки данных, расчета специфических функций потерь (losses), управления экспериментами и интеграции токенизатора в проекте BEBLaDII.

## Ключевые компоненты
| Класс / Функция | Файл | Назначение |
|---|---|---|
| `get_tokenizer` | [tokenizer.py](file:///c:/Experiments/BEBLaDII/src/beb_la_dii/utils/tokenizer.py) | Загрузка Qwen2.5 токенизатора с добавлением спецтокенов `<|thought|>`, `<|im_start|>`, `<|im_end|>`. |
| `DistillationLoss` | [loss.py](file:///c:/Experiments/BEBLaDII/src/beb_la_dii/utils/loss.py) | Расчет суммарного Mask-Aware лосса (MSE + Cosine) с исключением padding-токенов. |
| `DistillationDataset` | [data.py](file:///c:/Experiments/BEBLaDII/src/beb_la_dii/utils/data.py) | Универсальный датасет с кастомными форматтерами текстов (`raw`, `magpie`, `sharegpt`). |
| `get_dataloader` | [data.py](file:///c:/Experiments/BEBLaDII/src/beb_la_dii/utils/data.py) | Формирование батчей. Поддерживает динамическое сканирование папок для стадий через `indexed_parquet`. |
| `ExperimentManager` | [experiment_manager.py](file:///c:/Experiments/BEBLaDII/src/beb_la_dii/utils/experiment_manager.py) | Управление жизненным циклом экспериментов и снимками состояний (Snapshots). |
| `ExperimentTracker` | [experiment_tracker.py](file:///c:/Experiments/BEBLaDII/src/beb_la_dii/utils/experiment_tracker.py) | Отслеживание хода обучения, запись метрик в JSONL и управление чекпоинтами. |
| `download_dataset` | [download_datasets.py](file:///c:/Experiments/BEBLaDII/scripts/download_datasets.py) | Скрипт подготовки данных: загрузка Parquet-файлов с HuggingFace. |

## Неочевидные детали
- **Схема взвешивания (Weighted Loss)**: По умолчанию слои имеют разные веса: слой 20 (0.5), слой 30 (0.7), слой 40 (1.0).
- **Thought Injection**: В формате `magpie` тег `<|thought|>` добавляется всегда. В `sharegpt` — только ко второму сообщению (первому ответу ассистента), чтобы имитировать процесс рассуждения.
- **Indexed Access**: `DistillationDataset` не использует стриминг. Вместо этого используется `IndexedParquetDataset` для быстрого произвольного доступа к примерам, что критично при перемешивании (shuffle).
- **Kaggle & Windows Integration**: `ExperimentTracker` собирает данные об окружении. Скрипт загрузки данных отключает симлинки (`local_dir_use_symlinks=False`) для предотвращения ошибок прав доступа в Windows.
- **Стадия Awakening**: На этой стадии используется смесь `CulturaX` (90%) и `Magpie` (10%) для одновременного выравнивания и сохранения общих знаний.

## Типичные ошибки
- **OOM (Out of Memory)**: Возникает при больших `max_length`. Рекомендуемое значение для отладки — 512.
- **Loss Explosion**: Ошибка при несовпадении размерностей `hidden_states` учителя и ученика.
- **PermissionError (Windows)**: Возникает при работе с `history.jsonl` или при попытке создать симлинки во время скачивания датасетов.
