<!-- last_verified: 2026-04-14 -->
# KI: Experiments

## Что это
Песочница для отработки гипотез по дистилляции и основной хаб для запуска тренировочных скриптов проекта BEBLaDII.

## Ключевые компоненты
| Класс / Функция | Файл | Назначение |
|---|---|---|
| `Phase 1 Training` | [train_phase1_kaggle.py](file:///c:/Experiments/BEBLaDII/experiments/train_phase1_kaggle.py) | Скрипт для латентного выравнивания (Distillation Phase 1). Использует `ReasoningDistiller` для обучения проекторов и latentBERT. |
| **Logic Benchmarks** | experiments/ | Оценка сохранения цепочек рассуждений (CoT) на элитных примерах (планируется). |
| **Teacher Comparison** | experiments/ | Сравнение влияния различных учителей на финальные метрики latentBERT (ученика). |

## Детали обучения (Phase 1)
- **Целевая архитектура**: latentBERT (база `ModernBERT-large`, 28 -> 40 слоев через DUS), Учитель `DeepSeek-R1-Distill-Qwen-7B`.
- **Обучаемые параметры**: Градиенты включены только для `latentBERT`, `input_projector` и `feature_projectors`. Веса учителя (`teacher`) заморожены (`requires_grad=False`).
- **Конфигурация (Kaggle T4x2)**:
    - `MAX_LENGTH`: 4096 (Покрывает ~65-70% CoT цепочек в Reasoning датасетах).
    - `BATCH_SIZE`: 1 (с `GRAD_ACCUM_STEPS`: 8–16).
    - **Оптимизации**: Обязателен `Gradient Checkpointing` для студента и `os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"`. Снижает нагрузку, позволяет загружать длинные тексты.
    - `WandB`: Интеграция через `wandb.init` для отслеживания динамики лосса в реальном времени.

## Неочевидные детали
- **Manual Grad Control**: В скрипте `train_phase1_kaggle.py` цикл вручную включает `requires_grad` для компонентов. Это критично, так как по умолчанию после инициализации дистиллятора некоторые проекторы могут быть заморожены.
- **Normalization**: Лосс нормализуется на `GRAD_ACCUM_STEPS` перед `backward()`, что обеспечивает корректное усреднение градиентов при накоплении.
- **Save Format**: Итоговые веса сохраняются в один `.pt` файл, содержащий `state_dict` трех ключевых подсистем: latentBERT и обоих типов проекторов. (Поддерживается загрузка через `CUSTOM_STUDENT_WEIGHTS_PATH`).
- **Device Synchronization**: Крайне важно следить за тем, чтобы все тензоры (`input_ids`, `attention_mask`) перемещались на корректный девайс (`student_device`) перед передачей в дистиллятор во избежание ошибки `Expected all tensors to be on the same device`.

## Типичные ошибки
- **Missing API Key**: Если переменная окружения `WANDB_API_KEY` не задана, логирование в облако будет пропущено.
- **CUDA OOM**: В тренировочном цикле добавлен механизм отлова исключения выбега за пределы памяти: при падении очищается кэш (`empty_cache`), вызывается `gc.collect()`, обнуляются градиенты и батч пропускается. 
- **Path Bias**: Скрипт использует `from src.beb_la_dii...`, что требует запуска из корня проекта.
