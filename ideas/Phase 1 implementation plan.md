# План реализации Фазы 1: Дистилляция latentBERT

Этот план описывает шаги по созданию инфраструктуры для переноса способностей к рассуждению из `DeepSeek-R1-Distill-Qwen-7B` в новую модель **`latentBERT`** (на базе `ModernBERT-large`, расширенного до 40 слоев).

## Затронутые слои:
- **Model Layer**: Архитектура DUS (для latentBERT), Проекторы, Дистиллятор.
- **Utility Layer**: Токенизатор, Лосс-функции, Загрузка данных.
- **Experiment Layer**: Скрипт обучения для Kaggle/Colab.

## Прочитанные KI:
- `.know/knowledge/KI_model_core.md`
- `.know/knowledge/KI_utilities.md`

## Ограничения из KI:
- **Запрещено**: Каскадное принуждение синхронных редьюсеров к асинхронности (Async/Await).
- **Критическая зона**: Стык слоев 20-21 в DUS модели.
- **Обязательно**: Использование `.venv\Scripts\python.exe`.

---

## Предложенные изменения

### 1. Архитектура и Модели (Model Layer)

#### [NEW] [dus.py](file:///c:/Experiments/BEBLaDII/src/beb_la_dii/model/dus.py)
Реализация Depth Up-Scaling для latentBERT.
- Функция `create_latentbert`: берет `ModernBERT-large` (28 слоев), создает 40 слоев через overlap (Блок 1: 1-20, Блок 2: 9-28).
- Настройка `gradient_checkpointing` для глубокой сети.

#### [NEW] [projectors.py](file:///c:/Experiments/BEBLaDII/src/beb_la_dii/model/projectors.py)
Система MLP-адаптеров.
- `InputProjector`: 4096 (Qwen Embedding) -> 768 (latentBERT Hidden Size). 
    - **Роль**: Полностью заменяет нативный слой эмбеддингов ModernBERT.
- `FeatureProjector`: 768 (latentBERT) -> 4096 (Qwen) с Residual Connections и LayerNorm.

#### [NEW] [distiller.py](file:///c:/Experiments/BEBLaDII/src/beb_la_dii/model/distiller.py)
Основной класс-обертка `ReasoningDistiller`.
- Инициализация Teacher (DeepSeek-R1-7B, 4-bit, frozen).
- Инициализация Student (ModernBERT-DUS).
- Форвард-пасс с извлечением Hidden States из слоев 20, 30, 40 ученика и соответствующих слоев учителя.

---

### 2. Утилиты (Utility Layer)

#### [NEW] [tokenizer.py](file:///c:/Experiments/BEBLaDII/src/beb_la_dii/utils/tokenizer.py)
Мост для токенизатора.
- **Полное удаление** нативного токенизатора ModernBERT.
- Внедрение последовательной цепочки: `Qwen 2.5 Tokenizer` -> `InputProjector`.
- Настройка специальных токенов Qwen (включая `<|im_start|>`, `<|im_end|>`, `<|thought|>`) для корректной передачи CoT-структур.

#### [NEW] [loss.py](file:///c:/Experiments/BEBLaDII/src/beb_la_dii/utils/loss.py)
Расчет `L_total`.
- **Чистый Feature-based Alignment**: `MSELoss` + `CosineSimilarityLoss`. 
- *(KL Divergence исключен на Фазе 1, так как предсказание следующего токена напрямую не дистиллируется из CausalLM учителя в MLM ученика)*.
- Взвешивание слоев: 0.5 (20), 0.7 (30), 1.0 (40).

#### [NEW] [data.py](file:///c:/Experiments/BEBLaDII/src/beb_la_dii/utils/data.py)
Загрузка и фильтрация 100k примеров.
- Поддержка Magpie (Reasoning), Cosmopedia (Academic), CulturX (RU/CS).
- Логика "Parallel Bridge" (CS-RU, CS-EN).

---

### 3. Эксперименты и Обучение (Execution)

#### [NEW] [train_phase1_kaggle.py](file:///c:/Experiments/BEBLaDII/experiments/train_phase1_kaggle.py)
Финальный скрипт для запуска в Kaggle/Colab (с учетом лимитов 2x T4 16GB).
- Интеграция с `unsloth` для максимального ускорения и `bitsandbytes` (4-bit) для учителя.
- Ограничение памяти: жесткий `max_length <= 512` или `1024`, `micro_batch_size = 1` с `gradient_accumulation_steps`.
- 2-стадийное обучение: Alignment (замороженные кости) -> Full Distillation.
- Логирование в WandB/Tensorboard.

## Открытые вопросы
1. **WandB**: Нужно ли настраивать логирование в Weights & Biases сразу в этом скрипте (рекомендуется для Kaggle)?
2. **Checkpointing**: Достаточно ли сохранять только ученика + проекторы (экономия места в Kaggle)?

## План верификации

### Автоматические тесты
- Проверка синтаксиса: `.venv\Scripts\python.exe -m py_compile src/beb_la_dii/**/*.py`
- Тест DUS: загрузка модели и проверка количества слоев (40).
- Тест Проекторов: проверка размерностей векторов на выходе.

### Ручная верификация
- Запуск тренировочного цикла на 10 итерациях локально (на CPU/маленькой модели), чтобы убедиться в отсутствии RuntimeError перед загрузкой в Kaggle.
