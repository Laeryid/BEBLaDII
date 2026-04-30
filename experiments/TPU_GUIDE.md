# Инструкция по работе с Cloud TPU (BEBLaDII)

Этот файл содержит технические детали для управления вашими ресурсами TPU.

## Оптимальный выбор: TPU v4-32 (On-demand)
Этот вариант наиболее стабилен, так как не является Spot-инстансом.

### 1. Создание через терминал
```cmd
gcloud compute tpus tpu-vm create bebladii-tpu-v4 `
    --zone=us-central2-b `
    --accelerator-type=v4-32 `
    --version=tpu-vm-base
```
*(Примечание: Используйте `^` вместо `` ` `` в CMD или пишите одной строкой)*

### 2. Подключение
```powershell
gcloud compute tpus tpu-vm ssh bebladii-tpu-v4 --zone=us-central2-b
```

---

## Альтернатива: TPU v6e-64 (Spot)
Мощнее, но может быть прерван в любой момент.

### 1. Подключение (Multi-host)
Для v6e-64 команды нужно запускать на всех 8 воркерах.
```powershell
gcloud compute tpus tpu-vm ssh bebladii-tpu-v6e --zone=us-east1-d --worker=all --command="ls"
```

---

## Общие инструкции для всех типов TPU

### 1. Проверка чипов (Python)
Выполните внутри TPU VM:
```python
import jax
# Для v4-32 должно быть 32
# Для v6e-64 должно быть 64
print(f"Total devices: {jax.device_count()}") 
```

### 2. Монтирование Cloud Storage (GCS)
На всех воркерах:
```bash
mkdir -p ~/data/datasets
mkdir -p ~/data/weights
gcsfuse bebladii-datasets ~/data/datasets
gcsfuse bebladii-weigths ~/data/weights
```

### 3. Синхронизация кода с Windows
```powershell
# Пример для v4-32 в us-central2-b
gcloud compute tpus tpu-vm scp --recurse src bebladii-tpu-v4:~/ --zone=us-central2-b
```

### 4. Настройка окружения
```bash
# Внутри TPU VM
sudo apt-get update && sudo apt-get install -p python3-venv gcsfuse
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# Или для PyTorch:
pip install torch~=2.3.0 torch_xla[tpu]~=2.3.0 -f https://storage.googleapis.com/tpu-pytorch/wheels
```
