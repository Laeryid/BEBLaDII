import os
import subprocess
import platform

def run(cmd):
    try:
        res = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode()
        return res.strip()
    except Exception as e:
        return f"Недоступно ({str(e).split(':')[-1].strip()})"

def inspect():
    print("="*30)
    print("   SYSTEM INSPECTION   ")
    print("="*30)
    print(f"Platform: {platform.platform()}")
    print(f"Python:   {platform.python_version()}")
    
    print("\n[CPU & Memory]")
    cpu_model = run("lscpu | grep 'Model name' | cut -d ':' -f 2")
    print(f"CPU Model: {cpu_model}")
    print(f"Memory:\n{run('free -h')}")
    
    print("\n[TPU Hardware Check]")
    # Проверка наличия ускорителей в системе
    accel_devs = run('ls -l /dev/accel*')
    print(f"Devices in /dev/accel*: \n{accel_devs}")
    
    print("\n[Environment Variables]")
    for var in ["PJRT_DEVICE", "XRT_TPU_CONFIG", "TPU_NAME", "TPU_CHIPS_PER_HOST_BOUNDS"]:
        print(f"{var: <25}: {os.environ.get(var, 'NOT_SET')}")
        
    print("\n[Software Versions]")
    print(run("pip list | grep -E 'torch-xla|libtpu|jax'"))
    
    print("\n[Disk Usage]")
    print(run("df -h ."))

if __name__ == "__main__":
    inspect()
