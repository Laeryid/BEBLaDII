# Cloud TPU v6e (Trillium) Redeploy Guide

Это руководство содержит шаги для быстрой настройки свежего инстанса TPU v6e и подключения к нему.

## 1. Подключение и передача файлов

### А. Копирование скриптов сетапа
Выполните с локальной машины (из корня проекта), чтобы перебросить необходимые файлы на TPU VM:
```powershell
# Копирование скрипта настройки
gcloud compute tpus tpu-vm scp experiments\setup_tpu_vm.sh bebladii-v6e-4:~/ --zone europe-west4-a

# Копирование исходного кода (если нужно быстро обновить без git)
gcloud compute tpus tpu-vm scp --recurse src bebladii-v6e-4:~/BEBLaDII/ --zone europe-west4-a
```

### Б. Подключение по SSH с пробросом порта
Для работы с Jupyter/Colab обязательно пробросьте порт **8888**:
```powershell
gcloud compute tpus tpu-vm ssh bebladii-v6e-4 --zone europe-west4-a -- -L 8888:localhost:8888
```

---

## 2. Настройка ОС и Лимитов (КРИТИЧНО)

Без этих правок XLA не сможет выделить память. Выполните на TPU VM:

```bash
# Поднятие лимитов заблокированной памяти (memlock)
echo "* soft memlock unlimited" | sudo tee -a /etc/security/limits.conf
echo "* hard memlock unlimited" | sudo tee -a /etc/security/limits.conf

# Примечание: лимиты вступают в силу после перезахода в SSH или рестарта сессии.
```

---

## 3. Установка ПО ("Золотая пара" v6e)

Выполните эти команды сразу после создания инстанса и активации `.venv`:

```bash
# 1. Очистка старых конфликтующих пакетов
pip uninstall -y libtpu-nightly jax jaxlib torch-xla torch

# 2. Установка PyTorch 2.4
pip install torch==2.4.0 torchvision torchaudio --no-cache-dir
pip install torch_xla[tpu]==2.4.0 -f https://storage.googleapis.com/tpu-pytorch/releases/tpuvm/release-2.4.html --no-cache-dir

# 3. Установка совместимого JAX и библиотеки libtpu (PJRT 0.54)
pip install jax==0.4.31 jaxlib==0.4.31 --no-cache-dir
pip install libtpu-nightly==0.1.dev20240912+nightly -f https://storage.googleapis.com/jax-releases/libtpu_releases.html --no-cache-dir

# 4. Создание системной ссылки на библиотеку (решает ошибку "libtpu.so not found")
sudo ln -sf $(pwd)/.venv/lib/python3.10/site-packages/libtpu/libtpu.so /usr/lib/libtpu.so
sudo ldconfig
```

---

## 4. Переменные окружения для запуска

Добавляйте их в начало вашего тренировочного скрипта или экспортируйте в bash:

```python
import os
os.environ["PJRT_DEVICE"] = "TPU"
# Сетка для v6e-4 (4 чипа на хосте)
os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "2,2,1" 
```

---

## 5. Запуск Jupyter для Colab

Если окружение настроено, запустите сервер для удаленного подключения:
```bash
source .venv/bin/activate
jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --port=8888 \
  --NotebookApp.port_retries=0 \
  --no-browser
```

---

## 6. Подключение в Google Colab

1. В Colab нажмите на стрелочку у кнопки **Connect**.
2. Выберите **Connect to a local runtime**.
3. Вставьте URL с токеном из консоли TPU (например, `http://localhost:8888/?token=...`).

---

## 7. Проверка готовности
Запустите скрипт верификации:
`python scripts/verify_new_tpu.py`

Если JAX видит 4 устройства, а PyTorch проходит тест умножения матриц — сервер готов.
