# Инструкция по переразвертыванию на Cloud TPU VM (Spot)

Эта инструкция поможет быстро восстановить рабочее окружение при получении нового Spot-инстанса.

---

## 1. С локального компьютера (Windows/PowerShell)

Выполняйте эти команды из корня проекта `c:\Experiments\BEBLaDII`.

### А. Подключение с пробросом порта для Colab
Замените `bebladii-v6-8` на актуальное имя инстанса, если оно изменится.
```powershell
gcloud compute tpus tpu-vm ssh bebladii-v6e-4 --zone europe-west4-a -- -L 8888:localhost:8888
```

### Б. Отправка скрипта настройки
Отправка только установочного скрипта (код будет склонирован позже через Git/Jupyter):
```powershell
gcloud compute tpus tpu-vm scp C:\Experiments\BEBLaDII\experiments\setup_tpu_vm.sh bebladii-v6e-4:/home/hp/setup_tpu_vm.sh --zone europe-west4-a
```

---

## 2. На удаленном сервере (TPU VM)

После подключения по SSH выполните настройку.

### А. Первый запуск (на чистой машине)
Запустите скрипт для настройки окружения:
```bash
bash setup_tpu_vm.sh
```
Скрипт установит Python venv, PyTorch XLA и Jupyter.
```bash
mkdir ~/BEBLaDII
cd ~/BEBLaDII
chmod +x experiments/setup_tpu_vm.sh
./experiments/setup_tpu_vm.sh
```

### Б. Запуск Jupyter для Colab
Если окружение уже настроено, просто запустите сервер:
```bash
source .venv/bin/activate
jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --port=8888 \
  --NotebookApp.port_retries=0 \
  --no-browser
```

---

## 3. В Google Colab

1. Нажмите на стрелочку у кнопки **Connect** (Подключиться).
2. Выберите **Connect to a local runtime** (Подключиться к локальной среде выполнения).
3. Вставьте URL с токеном из консоли сервера (вида `http://localhost:8888/?token=...`).

---
**Заметка:** Если инстанс был просто остановлен, а не удален, все файлы в `~/BEBLaDII` и окружение `.venv` сохранятся на диске. В этом случае Шаг 2А выполнять не нужно.
