# Cloud TPU v6e (Trillium) Redeploy Guide

Это руководство содержит шаги для быстрой настройки свежего инстанса TPU v6e. 
**ВАЖНО:** Для автоматизации на Spot-инстансах рекомендуется обернуть эти команды в bash-скрипт `setup_tpu.sh`.

## 1. Рекомендуемые параметры инстанса
- **Image:** `tpu-ubuntu2204-base` (или `v2-alpha-tpuv6e`)
- **Software Stack:** PyTorch 2.4 / Python 3.10+

## 2. Команды для установки ПО
Выполните эти команды сразу после создания инстанса и активации `.venv`:

```bash
# 1. Очистка старых конфликтующих пакетов
pip uninstall -y libtpu-nightly jax jaxlib torch-xla torch

# 2. Установка "Золотой пары" PyTorch 2.4
pip install torch==2.4.0 torchvision torchaudio --no-cache-dir
pip install torch_xla[tpu]==2.4.0 -f https://storage.googleapis.com/tpu-pytorch/releases/tpuvm/release-2.4.html --no-cache-dir

# 3. Установка совместимого JAX и библиотеки libtpu (PJRT 0.54)
pip install jax==0.4.31 jaxlib==0.4.31 --no-cache-dir
pip install libtpu-nightly==0.1.dev20240912+nightly -f https://storage.googleapis.com/jax-releases/libtpu_releases.html --no-cache-dir
```

## 3. Системные правки (КРИТИЧНО)
Без этих правок XLA не увидит библиотеку или не сможет выделить память.

```bash
# Создание системной ссылки на библиотеку (решает ошибку "libtpu.so not found")
sudo ln -sf $(pwd)/.venv/lib/python3.10/site-packages/libtpu/libtpu.so /usr/lib/libtpu.so
sudo ldconfig

# Поднятие лимитов заблокированной памяти (memlock)
echo "* soft memlock unlimited" | sudo tee -a /etc/security/limits.conf
echo "* hard memlock unlimited" | sudo tee -a /etc/security/limits.conf
# Примечание: лимиты вступают в силу после перезахода в SSH или рестарта
```

## 4. Переменные окружения для запуска
Добавляйте их в начало вашего тренировочного скрипта или экспортируйте в bash:

```python
import os
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "2,2,1" # Сетка для v6e-4
```

## 5. Проверка готовности
Запустите скрипт верификации:
`python scripts/verify_new_tpu.py`

Если JAX видит 4 устройства, а PyTorch проходит тест умножения матриц — сервер готов к работе.
