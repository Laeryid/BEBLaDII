import os
import sys

def check_tpu():
    print("--- Диагностика TPU / XLA ---")
    if "PJRT_DEVICE" not in os.environ:
        os.environ["PJRT_DEVICE"] = "TPU"
        print("[INFO] PJRT_DEVICE не был задан, устанавливаем 'TPU'...")
        
    try:
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
    except ImportError:
        print("ОШИБКА: torch_xla не установлен в текущем окружении.")
        return

    devices = xm.get_xla_supported_devices()
    print(f"Доступные XLA устройства: {devices}")
    num_devices = len(devices)
    print(f"Количество ядер: {num_devices}")
    
    print("\n--- Рекомендации по nprocs ---")
    if num_devices == 8:
        print("TPU v3-8/v4-8 обнаружен. Используйте nprocs=8.")
    elif num_devices == 1:
        print("Обнаружено только 1 ядро (возможно, TPU VM Single Core или неверная инициализация).")
        print("Использование nprocs=8 может вызвать ошибку 'Unsupported nprocs'. Попробуйте nprocs=1.")
    else:
        print(f"Обнаружено {num_devices} ядер. Используйте nprocs={num_devices}.")

    print("\n--- Окружение PJRT ---")
    pjrt_device = os.environ.get("PJRT_DEVICE", "NOT_SET")
    print(f"PJRT_DEVICE: {pjrt_device}")
    if pjrt_device == "CPU" and num_devices > 0:
        print("ВНИМАНИЕ: PJRT_DEVICE установлен в CPU, но TPU ядра доступны. Проверьте настройки среды.")

if __name__ == "__main__":
    check_tpu()
