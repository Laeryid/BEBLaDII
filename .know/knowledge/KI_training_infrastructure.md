<!-- last_verified: 2026-05-09 -->
# KI: Training Infrastructure

## Overview
Инфраструктура для обучения моделей в распределенной среде. Основной упор сделан на работу с TPU v6e через PyTorch XLA, использование SPMD и FSDP для эффективного шардирования весов.

## Ключевые компоненты
| Класс / Функция | Файл | Назначение |
|---|---|---|
| `train_phase1_TPU_fsdp.py` | `experiments/` | Основной скрипт обучения Phase 1 на TPU. |
| `SpmdFullyShardedDataParallel` | `torch_xla` | Реализация FSDP через SPMD для шардирования модели. |
| `load_awakening_weights` | `train_phase1_TPU_fsdp.py` | Загрузка весов стадии Awakening (сложная распаковка вложенных словарей). |
| `smart_load_weights` | `train_phase1_TPU_fsdp.py` | Умное сопоставление ключей `state_dict` по суффиксам. |
| `Anti-Phase Beta` | `train_phase1_TPU_fsdp.py` | Планировщик коэффициента Beta в противофазе с Learning Rate (ADR-011). |

## Неочевидные детали
- **TPU v6e Config**: Скрипт жестко настроен на топологию v6e-4 (`TPU_CHIPS_PER_HOST_BOUNDS="2,2,1"`).
- **SPMD Mesh**: Используется сетка `(num_devices, 1)` с осями `fsdp` и `model`. 
- **GCS Sync**: Rank 0 автоматически синхронизирует чекпоинты и файлы истории (`history.jsonl`) с бакетом `gs://bebladii-weigths/checkpoints/` каждые 500 шагов.
- **Anti-Phase Beta Scheduler**: 
    - Цикл: 2000 шагов (совпадает с `T_0` в `CosineAnnealingWarmRestarts`).
    - Логика: Максимум `BETA_MAX=0.1` достигается на 1000-м шаге, когда LR минимален. Это заставляет модель сильнее фокусироваться на регуляризации скрытых состояний, когда шаг обучения замедляется.
- **Targeted Loss Masking**: Если в батче есть `loss_mask`, она имеет приоритет над `attention_mask`. Это позволяет исключать из лосса префиксы или системные промпты.

## Типичные ошибки
- **XLA Graph Breaks**: Операции `print` или обращение к тензорам вне `xm.mark_step()` замедляют TPU.
- **GCS Permission**: Ошибки `gsutil` возникают, если на инстансе TPU не настроен `gcloud auth` или права доступа к бакету.
- **Weights Mismatch**: Если `matched params` при загрузке значительно меньше общего числа параметров — проверьте маппинг имен в `smart_load_weights`.



## Related KIs
- [[KI_model_core.md]] (via `src/beb_la_dii/model/component_registry.py`)

