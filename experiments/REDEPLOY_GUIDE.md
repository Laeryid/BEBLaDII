# REDEPLOY GUIDE (BEBLaDII)

## Запуск на TPU v6e-4

Для стабильного обучения используйте многопроцессорный запуск через `torchrun`. Это разделяет нагрузку на управляющую память XLA (sflag) между ядрами.

### Команда запуска:
```bash
# Использовать 4 процесса (по одному на ядро v6e-4)
torchrun --nproc_per_node=4 experiments/train_phase1_TPU_fsdp.py
```

### Решение проблем (Troubleshooting)

#### 1. Ошибка "Internal error when accessing libtpu multi-process lockfile"
**Причина:** Предыдущий запуск завершился некорректно и оставил файл блокировки.
**Решение:**
```bash
sudo rm -f /tmp/libtpu_lockfile
# Если не помогло, убить процессы, захватившие TPU:
sudo fuser -k /dev/vfio/*
```

#### 2. Ошибка "RESOURCE_EXHAUSTED: Ran out of memory in memory space sflag"
**Причина:** Слишком сложный граф компиляции (большой батч или длинные последовательности).
**Решение:**
- Уменьшить `batch_size` до 1 на ядро.
- Увеличить `accumulation_steps` (например, до 4).
- Уменьшить `max_length` до 2048.

### Конфигурация окружения
В коде должны быть установлены следующие переменные для PJRT:
- `PJRT_DEVICE=TPU`
- `TPU_CHIPS_PER_HOST_BOUNDS=2,2,1`
- `TPU_NUM_DEVICES=4`
