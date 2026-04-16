<!-- last_verified: 2026-04-15 -->
# KI: Model Core

## Что это
Ядро системы, отвечающее за архитектуру моделей и процесс дистилляции рассуждений (Reasoning) из крупного "учителя" в компактный latentBERT. Реализует Depth Up-Scaling (DUS) и проекцию латентных пространств.

## Ключевые компоненты
| Класс / Функция | Файл | Назначение |
|---|---|---|
| `BEComponent` | [base.py](file:///c:/Experiments/BEBLaDII/src/beb_la_dii/model/base.py) | Базовый класс компонентов. Содержит `load_weights(path)` — основной метод загрузки весов. |
| `ComponentRegistry` | [component_registry.py](file:///c:/Experiments/BEBLaDII/src/beb_la_dii/model/component_registry.py) | **Только сохранение** весов после обучения. Загрузка через `weights_map`. |
| `ModelAssembler` | [assembler.py](file:///c:/Experiments/BEBLaDII/src/beb_la_dii/model/assembler.py) | Принимает `weights_map: dict` и собирает дистиллятор. Главная точка инициализации системы. |
| `InputProjector` | [projectors.py](file:///c:/Experiments/BEBLaDII/src/beb_la_dii/model/projectors.py) | MLP-адаптер для входа: Qwen (3584) -> ModernBERT (1024). |
| `FeatureProjector` | [projectors.py](file:///c:/Experiments/BEBLaDII/src/beb_la_dii/model/projectors.py) | Адаптер скрытых состояний: BERT (1024) -> Qwen (3584) с Residual-связью. |
| `DUSModel` | [dus.py](file:///c:/Experiments/BEBLaDII/src/beb_la_dii/model/dus.py) | Обертка над ModernBERT, расширенным до 40 слоев. Переопределяет `load_weights()`. |
| `create_latentbert` | [dus.py](file:///c:/Experiments/BEBLaDII/src/beb_la_dii/model/dus.py) | Функция реализации Depth Up-Scaling через копирование блоков. |
| `ReasoningDistiller` | [distiller.py](file:///c:/Experiments/BEBLaDII/src/beb_la_dii/model/distiller.py) | Основной класс связки "Учитель-Ученик" для процесса дистилляции. |

## Неочевидные детали
- **DUS Overlap (40 слоев)**: Используется схема дублирования слоев из 28-слойной базы ModernBERT-large:
    - Блок 1: Слои 0-19.
    - Блок 2: Слои 8-27.
    - Итоговая структура включает 40 слоев, где слои 8-19 исходной модели продублированы для сохранения семантической непрерывности. В эту структуру дополнительно интегрируются 6 Cross-Attention слоев (3 для промпта, 3 для RAG), привязанных к full-attention блокам.
- **Residual Projection**: `FeatureProjector` использует `nn.Linear` для подгонки размерности residual-связи, что критично для стабильности градиентов при дистилляции.
- **Layer Mapping**: Дистиллятор сопоставляет слои latentBERT [20, 30, 40] со слоями учителя [14, 21, 28] для поэтапного извлечения признаков.
- **Frozen Teacher**: Teacher-модель (DeepSeek-R1-7B) всегда работает в 4-bit режиме (`nf4`) и полностью заморожена.
- **Правильные размерности (Qwen2.5)**: DeepSeek-R1-Distill-Qwen-7B базируется на Qwen2.5 с `hidden_size=3584` (не 4096 как Qwen1). Размерности всех проекторов: InputProjector `3584->1024`, FeatureProjector `1024->3584`.

## Подход к весам компонентов (weights_map)

**Основной паттерн для всех компонентов:**
```python
component = MyComponent.from_scratch(
    component_id="...", version="v1.0",
    weights_path="/path/to/weights.pt"  # опционально; None = случайная инициализация
)
```

- **Скелет** (архитектура) всегда создаётся заново из кода — никакой зависимости от сохранённых конфигов.
- **DUSModel**: скелет строится через DUS из `ModernBERT-large`. Готовые веса поверх — опционально.
- **Проекторы**: инициализируются случайно; при наличии `weights_path` грузят обученные веса.
- **`ModelAssembler.assemble_phase1_distiller(weights_map=...)`**: принимает `dict[component_id -> weights_path]`. Пустой/None = всё случайно.
- **`ComponentRegistry`**: переведён только в режим сохранения (`save_component`). Загрузка — только через `weights_map`.
- **Kaggle-интеграция**: `build_weights_map()` в `train_phase1_kaggle.py` автоматически выбирает пути из Kaggle-датасета или локального `storage/components/`.

## Типичные ошибки
- **Weights Path Mismatch**: `build_weights_map()` выводит статус `[found]`/`[random init]` для каждого компонента — проверяй логи перед запуском обучения.
- **DUSModel.load_weights()**: переопределён, т.к. веса хранятся в `self.model` (HF-модуль), а не в `self`. При ручном сохранении — используй `registry.save_component()`, а не `torch.save(dus_model.state_dict())`.
- **Gradient Checkpointing**: В `create_latentbert` принудительно включается Checkpointing через `base_model.gradient_checkpointing_enable()`. Если его отключить при 40 слоях, возникнет `OutOfMemory` на GPU с <24GB VRAM.
