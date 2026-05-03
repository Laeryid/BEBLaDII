import os
import sys

def check_tpu():
    print("--- Диагностика TPU / XLA ---")
    # Пытаемся инициализировать TPU, если не вышло - откатываемся на CPU
    try:
        if "PJRT_DEVICE" not in os.environ:
            os.environ["PJRT_DEVICE"] = "TPU"
        import torch_xla.core.xla_model as xm
        devices = xm.get_xla_supported_devices()
        print(f"[SUCCESS] PJRT_DEVICE={os.environ['PJRT_DEVICE']} инициализирован.")
    except Exception as e:
        print(f"[WARN] Инициализация TPU не удалась: {e}")
        print("[INFO] Переключаемся на PJRT_DEVICE=CPU для работы через XLA на процессоре.")
        os.environ["PJRT_DEVICE"] = "CPU"
        import torch_xla.core.xla_model as xm

    try:
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
