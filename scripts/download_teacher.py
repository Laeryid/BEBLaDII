import os
from huggingface_hub import snapshot_download

def download_teacher():
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    local_dir = os.path.join("storage", "prebuilt", "deepseek-7b")
    
    print(f"--- Скачивание учителя: {model_id} ---")
    print(f"Путь назначения: {os.path.abspath(local_dir)}")
    
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
    
    # Скачиваем только нужные файлы для инференса и квантования
    # Исключаем лишние чекпоинты, если они есть
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False, # Копируем физически, чтобы потом загрузить в Kaggle
        ignore_patterns=["*.msgpack", "*.h5", "*.bin"], # Обычно нам нужны safetensors
    )
    print("\nГотово! Учитель скачан в локальную папку проекта.")

if __name__ == "__main__":
    download_teacher()
