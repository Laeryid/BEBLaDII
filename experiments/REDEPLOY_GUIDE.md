# Cloud TPU v6e (Trillium) Redeploy Guide

Это руководство содержит шаги для быстрой настройки свежего инстанса TPU v6e и подключения к нему.

## 1. Подключение и передача файлов

### А. Копирование скриптов сетапа
Выполните с локальной машины (из корня проекта), чтобы перебросить необходимые файлы на TPU VM:
```powershell
# Копирование скрипта настройки
gcloud compute tpus tpu-vm scp C:\Experiments\BEBLaDII\experiments\setup_tpu_vm.sh bebladii-v6e-4:/home/hp/  --zone europe-west4-a

# Копирование исходного кода (если нужно быстро обновить без git)
gcloud compute tpus tpu-vm scp --recurse src bebladii-v6e-4:~/BEBLaDII/ --zone europe-west4-a
```

### Б. Подключение по SSH с пробросом порта и SSH-агента
Для работы с Jupyter и автоматической авторизации в Git (без генерации ключей на каждом новом Spot-инстансе) используйте **SSH Agent Forwarding**:

1. **Локально (Windows)**: запустите **PowerShell от имени администратора** и убедитесь, что ваш ключ добавлен в агент:
```powershell
# В PowerShell (Администратор):
Set-Service -Name ssh-agent -StartupType Automatic
Start-Service ssh-agent
ssh-add $HOME\.ssh\id_ed25519
```
2. **Подключение**: добавьте флаг `-A` (проброс агента):
```powershell
gcloud compute tpus tpu-vm ssh bebladii-v6e-4 --zone europe-west4-a -- -A -L 8888:localhost:8888
```

### В. Клонирование репозитория
Самый простой способ (не требует ключей, если проект публичный):
```bash
mkdir -p ~/projects && cd ~/projects
git clone https://github.com/Laeryid/BEBLaDII.git
cd BEBLaDII
```

> **Примечание**: Если проект приватный, используйте SSH (см. ниже) или HTTPS с токеном: `https://<TOKEN>@github.com/...`

---

### Г. Альтернатива: Использование SSH (для приватных репо)
Если вам нужно пушить код или проект приватный:
1. **Проброс агента** (локально `-A` при подключении).
2. **Ручное копирование**:
   - На TPU: `mkdir -p ~/.ssh`
   - Локально: `gcloud compute tpus tpu-vm scp "$env:USERPROFILE\.ssh\id_ed25519" bebladii-v6e-4:/home/hp/.ssh/id_ed25519 --zone europe-west4-a`
   - На TPU (Активация): `eval $(ssh-agent -s) && ssh-add ~/.ssh/id_ed25519`

---

## 2. Запуск скрипта настройки на TPU

После того как вы скопировали файл и подключились по SSH, выполните **на удаленной машине**:

```bash
# Сделать скрипт исполняемым
chmod +x ~/setup_tpu_vm.sh

# Запустить процесс настройки
~/setup_tpu_vm.sh
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

**ВАЖНО**: Всегда работайте внутри `.venv`:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Выполните установку:
```bash
# 1. Очистка старых конфликтующих пакетов
pip uninstall -y libtpu-nightly jax jaxlib torch-xla torch

# 2. Установка PyTorch 2.4 и XLA (без автоматических зависимостей TPU)
pip install torch==2.4.0 torchvision torchaudio --no-cache-dir
pip install torch_xla==2.4.0 -f https://storage.googleapis.com/tpu-pytorch/releases/tpuvm/release-2.4.html --no-cache-dir

# 3. Установка зависимостей TPU и JAX вручную (решает ошибку libtpu-nightly)
pip install cloud-tpu-client>=0.10.0 indexed-parquet-dataset
pip install jax==0.4.31 jaxlib==0.4.31 --no-cache-dir
pip install libtpu-nightly==0.1.dev20240912+nightly -f https://storage.googleapis.com/jax-releases/libtpu_releases.html --no-cache-dir

# 4. Создание системной ссылки на библиотеку (решает ошибку "libtpu.so not found")
sudo ln -sf $(pwd)/.venv/lib/python3.10/site-packages/libtpu/libtpu.so /usr/lib/libtpu.so
sudo ldconfig

# 5. Исправление возможных ошибок wandb (ImportError: cannot import name 'Imports')
pip install --force-reinstall wandb
```

---

## 4. Переменные окружения для запуска

Добавляйте их в начало вашего тренировочного скрипта или экспортируйте в bash:

```python
import os
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["XLA_USE_BF16"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
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

## 8. Запуск тренировки (Phase 1)

**ВАЖНО**: Не используйте Jupyter для запуска основной тренировки на TPU. Разделение кода по ячейкам ломает механизмы инициализации XLA/PJRT. Используйте чистый Python скрипт.


### Команда запуска (Standard DDP):
```bash
python experiments/train_phase1_TPU.py
```

### Команда запуска (FSDP - Рекомендуется для v6e):
```bash
# Позволяет использовать batch_size=4 и seq_len=4096 без OOM
python experiments/train_phase1_TPU_fsdp.py
```

Если нужно запустить в фоне (рекомендуется для Spot):
```bash
nohup python experiments/train_phase1_TPU_fsdp.py > training.log 2>&1 &
tail -f training.log
```

---

## 9. Проверка готовности
Запустите скрипт верификации:
`python scripts/verify_new_tpu.py`

Если JAX видит 4 устройства, а PyTorch проходит тест умножения матриц — сервер готов.
