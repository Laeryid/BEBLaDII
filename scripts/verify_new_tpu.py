import os
import torch
import sys

def verify():
    print("="*40)
    print("   TPU VM READINESS CHECK   ")
    print("="*40)
    
    # 1. Force PJRT for TPU
    os.environ["PJRT_DEVICE"] = "TPU"
    
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        print(f"torch_xla version: {torch_xla.__version__}")
    except ImportError:
        print("[ERROR] torch_xla is not installed!")
        sys.exit(1)
    
    # 2. Check for physical devices
    print("\n[1/3] Checking /dev/accel* devices...")
    import subprocess
    try:
        accels = subprocess.check_output("ls -l /dev/accel*", shell=True).decode()
        print(f"Found devices:\n{accels.strip()}")
    except:
        print("[FAIL] No /dev/accel* devices found. TPU is not accessible.")
        return

    # 3. Check XLA supported devices
    print("\n[2/3] Checking XLA logical cores...")
    try:
        devices = xm.get_xla_supported_devices()
        print(f"Detected {len(devices)} XLA cores: {devices}")
    except Exception as e:
        print(f"[FAIL] XLA failed to see devices: {e}")
        return

    # 4. Simple computation test
    print("\n[3/3] Running test computation on XLA...")
    try:
        device = xm.xla_device()
        a = torch.randn(100, 100).to(device)
        b = torch.randn(100, 100).to(device)
        c = torch.matmul(a, b)
        xm.mark_step() # Essential for XLA
        print(f"[SUCCESS] Matrix multiplication on {device} completed.")
    except Exception as e:
        print(f"[FAIL] Computation failed: {e}")

if __name__ == "__main__":
    verify()
