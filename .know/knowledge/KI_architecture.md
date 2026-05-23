<!-- last_verified: 2026-05-23 -->
# KI: Architecture & Training Phases

## Что это
Высокоуровневое описание архитектуры BEBLaDII (Reasoning Latent Diffusion) версии 2.0 (согласно ADR-020) и стратегии итеративного обучения компонентов (10-фазный пайплайн). Система реализует принцип разделения процессов логического анализа (System 2, латентная диффузия) и лингвистической генерации (System 1, декодер).

## Ключевые компоненты
| Компонент | Класс / Путь | Назначение |
|---|---|---|
| **System 2 Engine** | `DUSModel` (40 layers) | Ядро диффузии на базе ModernBERT (Diffusion Backbone), выполняющее итеративное уточнение смыслов. |
| **Reasoning Distiller**| `ReasoningDistiller` | Класс-оркестратор дистилляции знаний из DeepSeek-R1 в Diffusion Backbone. |
| **Latent Encoder** | `distiller.input_projector` | Проектирует эмбеддинги Qwen (3584) в базовое пространство диффузии (1024). |
| **Feature Projectors**| `nn.ModuleDict` | Набор проекторов для выравнивания промежуточных слоев [14, 21, 28] учителя и [20, 30, 40] Diffusion Backbone. |
| **Output Projector** | | Возвращает вектор из внутреннего многообразия слоя 40 обратно во входное базовое пространство диффузии (замыкает цикл). |
| **CA_Prompt Layers**| `ModuleList` в составе DUS | 3 слоя Cross-Attention для внедрения пользовательского промпта. |
| **CA_Memory Layers** | `ModuleList` в составе DUS | 3 слоя Cross-Attention для внедрения фактов базы знаний (CLM). |
| **CA_Context Layers** | `ModuleList` в составе DUS | 3 слоя Cross-Attention для внедрения операционного контекста (Context Register). |
| **System 1 (Voice)** | `DeepSeek-R1-Distill-Qwen`| Замороженный декодер, определяющий целевое семантическое пространство. |

## Реализация фаз обучения (10-Phase Pipeline)

1. **Фаза 1: Awakening**: Инициализация Latent Encoder (выполнена).
2. **Фаза 2: Reasoning**: DUS Distillation (в процессе).
3. **Фаза 3: Output Projection**: Обучение компонента замыкания цикла диффузии.
4. **Фаза 4: Decoder Bridge**: Обучение Latent Decoder.
5. **Фаза 5: CLM Pooling**: Сжатие латентных чанков в единый запрос.
6. **Фаза 6: CLM Index Mapper**: Проекция запроса под векторный ANN-поиск в FAISS.
7. **Фаза 7: Confidence Sensor**: Обучение Confidence Head геометрической метрике кристаллизации (прогресс диффузии).
8. **Фаза 8: Prompt Conditioning**: Обучение слоёв CA_Prompt интеграции жёстких критериев качества.
9. **Фаза 9: Memory Integration**: Обучение слоёв CA_Memory (подача фактов) и Relevance Gate (нейронная оценка релевантности).
10. **Фаза 10: Tool Use & Context**: Обучение слоёв CA_Context работе с жестко структурированным операционным реестром (Context Register).

## Механизм обусловливания (Conditioning)
Мы используем парадигму **Conditional Latent Diffusion**. Вместо конкатенации входного промпта со стартовым шумом, система внедряет внешние сигналы через разреженные слои Cross-Attention (CA):
- **CA_Prompt**: подается через 3 CA-слоя (начало, середина, конец).
- **CA_Memory**: подается через 3 отдельных CA-слоя. Используется обучаемый весовой параметр `gamma`.
- **CA_Context**: операционный реестр, подключенный аналогичным образом.

## Неочевидные детали
- **DUS (Depth Up-Scaling)**: Расширение до 40 слоев через дублирование блоков (слои 0-19 и 8-27 с перекрытием на слоях 8-19).
- **Оптимизация памяти**: Gradient Checkpointing, `use_cache=False`, очистка кэша CUDA.
- **Лосс (Фаза 2)**: `DistillationLoss` (MSE + Centered Cosine Similarity) на слоях 20, 30, 40 (без промежуточных, ADR-016).

## Типичные ошибки
- **Mismatch в размерностях**: Токены Qwen2.5 имеют dim 3584, ModernBERT — 1024. Прямое сравнение невозможно без Latent Encoder.
- **NaN Loss**: Часто возникает в начале Фазы 2, если пропустить Фазу 1. Отрабатывается скриптом через `NaN loss detected... Skipping batch`.
- **Заморозка учителя**: Учитель должен быть в режиме `.eval()` и с `requires_grad=False` для экономии VRAM.
- **weights_map**: При запуске обучения проверяй логи `build_weights_map()` — статус `[found]`/`[random init]` покажет, загружены ли сохранённые веса или инициализированы заново.

## Related KIs
- `decisions/020_architecture_v2_10_phases.md`
