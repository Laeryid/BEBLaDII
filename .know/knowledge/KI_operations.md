<!-- last_verified: 2026-05-13 -->
# KI: Operations & Smoke Tests

## Overview
Общие операционные скрипты и Smoke-тесты для быстрой проверки работоспособности системы BEBLaDII.

## Key Components

### Smoke Tests
| Script | Purpose | Details |
|---|---|---|
| `smoke_test_forward.py` | Тест полного прохода. | Проверка цепочки: InputProjector -> latentBERT -> FeatureProjectors. Не требует GPU. |
| `verify_new_tpu.py` | Тест XLA/TPU. | Проверка инициализации распределенного обучения на TPU v6e. |

### System Diagnostics
| Script | Purpose | Details |
|---|---|---|
| `inspect_system.py` | Инспекция окружения. | Вывод версий библиотек (torch, xla, transformers) и параметров CPU/GPU/TPU. |
| `evaluate_lengths.py` | Анализ токенов. | Статистический анализ длин последовательностей в датасетах. |

## Non-obvious Details
- **Mocking Teacher**: `smoke_test_forward.py` использует `MagicMock` для имитации учителя Qwen, что позволяет запускать тест без загрузки 28GB весов.
- **Root Invocation**: Большинство скриптов в `scripts/` следует запускать из корня проекта: `.venv/Scripts/python.exe -m scripts.<script_name>`.

## Common Pitfalls
- **Python Path**: Ошибка `ModuleNotFoundError: No module named 'src'` возникает при запуске напрямую из папки `scripts/`. Используйте флаг `-m` из корня.
- **TPU Device Mesh**: При запуске на TPU через `scripts/run_tpu.sh` убедитесь, что переменная `TPU_CHIPS` соответствует физической топологии.




## Related KIs

