import os
from pathlib import Path

def list_pt_files(root_dir="/kaggle"):
    print(f"{'File Path':<80} | {'Size (MB)':>10}")
    print("-" * 93)
    
    found_any = False
    # Обходим все директории
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".pt"):
                found_any = True
                full_path = Path(root) / file
                try:
                    size_mb = full_path.stat().st_size / (1024 * 1024)
                    print(f"{str(full_path):<80} | {size_mb:>10.2f}")
                except Exception as e:
                    print(f"{str(full_path):<80} | Error: {e}")
    
    if not found_any:
        print("Ни одного .pt файла не найдено.")

if __name__ == "__main__":
    # Можно вызвать для всего /kaggle или конкретно для /kaggle/working / /kaggle/input
    target = "/kaggle" 
    print(f"Поиск .pt файлов в {target}...")
    list_pt_files(target)
