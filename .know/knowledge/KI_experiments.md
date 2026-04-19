<!-- last_verified: 2026-04-19 -->
# KI: Experiments

## Что это
Песочница для отработки гипотез по дистилляции и основной хаб для запуска тренировочных скриптов проекта BEBLaDII. Основной фокус сейчас: Phase 1 (Awakening & Reasoning).

## Ключевые компоненты
| Класс / Функция | Файл | Назначение |
|---|---|---|
| `Phase 1 Training` | [train_phase1_kaggle.ipynb](file:///c:/Experiments/BEBLaDII/experiments/train_phase1_kaggle.ipynb) | Основной инструмент обучения. Поддерживает стадии Awakening и Reasoning. |
| `smart_load_weights` | [train_phase1_kaggle.ipynb](file:///c:/Experiments/BEBLaDII/experiments/train_phase1_kaggle.ipynb) | Ультра-надежный загрузчик весов. Распаковывает вложенные структуры проекторов и матчит ключи по суффиксам. |
| **Logic Benchmarks** | experiments/ | Оценка сохранения цепочек рассуждений (CoT) на элитных примерах (планируется). |

## Детали обучения (Phase 1)
### Стадия 2: Reasoning
- **Целевая архитектура**: latentBERT (ModernBERT-large, 40 слоев).
- **Компоненты**: Включает μ-VAE голову для работы в латентном пространстве.
- **Loss Function**: `DistillationLoss` с весом косинусного сходства (cos_weight=20.0) и **KL-дивергенцией** для регуляризации латентного пространства (BETA отжиг до 0.0001).
- **Оптимизации**: Используется `Mixed Precision (torch.amp)` и `Gradient Checkpointing`.

### Конфигурация (Kaggle T4x2)
- `MAX_LENGTH`: 4096.
- `BATCH_SIZE`: 1 (с `GRAD_ACCUM_STEPS`: 8–16).
- **Зеркалирование данных**: Скрипт автоматически настраивает симлинки из `/kaggle/input` в локальную структуру `data/` и `storage/`.

## Неочевидные детали
- **Smart Loading**: При загрузке весов Awakening в Reasoning сессию, `smart_load_weights` игнорирует префиксы `teacher.*` и корректно сопоставляет веса студента, даже если структура `state_dict` изменилась (например, при добавлении VAE).
- **Emergency Checkpoints**: В блокноте предусмотрена ячейка "Emergency Resume Checkpoint", сохраняющая полное состояние (модель + оптимизатор + скейлер) для восстановления после прерывания Kaggle.
- **Normalization**: Лосс нормализуется на `GRAD_ACCUM_STEPS` перед `backward()`.

## Типичные ошибки
- **CUDA OOM**: При выбеге памяти добавлен отлов исключения: очистка `empty_cache`, вызов `gc.collect()`, обнуление градиентов и пропуск батча.
- **NaN Loss**: В режиме `autocast` возможны NaN. Скрипт проверяет `torch.isnan(loss)` и пропускает такие итерации без обновления весов.
- **Device Mixup**: Обязательно использовать `distiller.student_device` для перемещения батчей, так как учитель и студент могут находиться на разных GPU в режиме `device_map`.
