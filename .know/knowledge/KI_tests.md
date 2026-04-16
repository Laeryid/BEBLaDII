<!-- last_verified: 2026-04-10 -->
# KI: Tests

## Что это
Система автоматизированного контроля качества, охватывающая базовую инфраструктуру, логику масштабирования моделей (DUS) и корректность работы проекторов.

## Ключевые компоненты
| Класс / Функция | Файл | Назначение |
|---|---|---|
| **Infrastructure Tests** | [test_infrastructure.py](file:///c:/Experiments/BEBLaDII/tests/test_infrastructure.py) | Проверка `BEComponent`, `ComponentRegistry` и `ExperimentManager`. Валидация сохранения/загрузки компонентов и изоляции версий. |
| **DUS Stability** | [test_dus_stability.py](file:///c:/Experiments/BEBLaDII/tests/test_dus_stability.py) | Тестирование логики расширения слоев (Depth Up-Scaling). Проверка корректности итогового количества слоев в `ModernBERT`. |
| **Projector & Wrapper Tests** | [test_constructor_logic.py](file:///c:/Experiments/BEBLaDII/tests/test_constructor_logic.py) | Проверка размерностей тензоров в `InputProjector`/`FeatureProjector` и инициализации `DUSModel`. |
| **Inference Smoke Test** | [test_inference.py](file:///c:/Experiments/BEBLaDII/tests/test_inference.py) | Сквозная проверка (Smoke Test) процесса инференса через `MockDistiller`. Верификация 1024-мерных тензоров. |
| **temp_storage** (fixture) | [test_infrastructure.py](file:///c:/Experiments/BEBLaDII/tests/test_infrastructure.py) | Изолированная временная директория для тестов реестра и экспериментов. Очищается автоматически после завершения тестов. |

## Неочевидные детали
- **Мокирование (Mocking)**: Тесты DUS используют `unittest.mock.patch` для имитации загрузки весов `transformers`. Это позволяет проверять архитектурную логику без наличия GPU.
- **MockDistiller**: В `test_inference.py` реализована мини-версия дистиллятора. Она имитирует выходы Teacher-модели (Qwen), что позволяет тестировать проекторы и latentBERT (ученика) на обычном CPU.
- **1024-dim Verification**: Тест инференса принудительно проверяет, что ModernBERT-large (1024 скрытых признака) корректно стыкуется с MLP-проекторами, возвращающими 4096 признаков.
- **Изоляция реестра**: Тесты версии в `ComponentRegistry` проверяют, что загрузка `v1.0` не возвращает данные от `v2.0`, даже если `component_id` совпадает.

## Типичные ошибки
- **Storage Cleanup Failure**: Если тесты прерываются аварийно, папка `storage/pytest_temp` может остаться. Фикстура `temp_storage` старается минимизировать этот риск через `yield`.
- **Import Conflicts**: При добавлении новых тестов важно импортировать компоненты через `beb_la_dii`, чтобы избежать проблем с путями в среде `.venv`.
- **Mock Misuse**: Неверное мокирование `from_pretrained` может привести к попытке реального сетевого запроса, что замедлит или обвалит тесты в CI.
