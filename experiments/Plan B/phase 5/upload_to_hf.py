import os
from huggingface_hub import HfApi, login

def upload_weights():
    # token = "YOUR_HF_TOKEN"
    # login(token)
    
    api = HfApi()
    repo_id = "your_username/BEBLaDII-Phase5-ConfidenceHead" # Замените на ваш username
    
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True, private=True)
        print(f"Репозиторий {repo_id} готов.")
    except Exception as e:
        print(f"Ошибка при создании репозитория: {e}")
        return

    # Путь к весам. Укажите реальный файл, когда Голова будет обучена, 
    # или используйте базовые веса Фазы 4/5 для инициализации.
    weights_path = r"C:\Experiments\BEBLaDII\experiments\phase 4\local_checkpoints\phase4_step_85995.pth"
    
    if not os.path.exists(weights_path):
        print(f"Файл {weights_path} не найден.")
        return
        
    print(f"Начинается загрузка {weights_path}...")
    try:
        api.upload_file(
            path_or_fileobj=weights_path,
            path_in_repo="confidence_head_base.pth",
            repo_id=repo_id,
            commit_message="Initial commit of Phase 5 Confidence Head base weights"
        )
        print("Загрузка успешно завершена!")
    except Exception as e:
        print(f"Ошибка загрузки: {e}")

if __name__ == "__main__":
    print("ВНИМАНИЕ: Скрипт требует HF_TOKEN. Раскомментируйте код и вставьте токен.")
    # upload_weights()