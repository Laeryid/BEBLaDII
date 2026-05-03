import os
from pathlib import Path

def find_weights(root="storage"):
    print(f"--- Поиск весов и конфигов в '{root}/' ---")
    extensions = ["*.pt", "*.safetensors", "*.bin", "config.json"]
    found = False
    
    # Превращаем в абсолютный путь для удобства копирования
    root_path = Path(root).absolute()
    
    for ext in extensions:
        for path in root_path.rglob(ext):
            # Пропускаем временные файлы и кэши
            if ".git" in str(path) or ".ipynb_checkpoints" in str(path):
                continue
            
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"[{path.name}] ({size_mb:.2f} MB)")
            print(f"  Путь: {path}")
            found = True
            
    if not found:
        print("Ничего не найдено. Проверьте структуру папок или наличие симлинков.")

if __name__ == "__main__":
    # Запускаем поиск в storage и data (на случай если веса там)
    find_weights("storage")
    print("-" * 30)
    find_weights("data")
