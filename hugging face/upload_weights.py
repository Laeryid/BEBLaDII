import os
from huggingface_hub import HfApi, create_repo

# Массив с файлами для загрузки.
# Ключ local_path - исходный путь, repo_name - желаемое имя в репозитории на Hugging Face.
FILES_TO_UPLOAD = [
    {
        "local_path": r"C:\Experiments\BEBLaDII\experiments\phase 1\planB_phase1_checkpoints_phase1_vae_step_20000.pth",
        "repo_name": "phase1_vae_step_20000.pth"
    },
    {
        "local_path": r"C:\Experiments\BEBLaDII\experiments\phase 2\planB_phase2_checkpoints_decoder_step_9000.pth",
        "repo_name": "phase2_decoder_step_9000.pth"
    },
    {
        "local_path": r"C:\Experiments\BEBLaDII\experiments\phase 3\local_checkpoints\phase3_step_17995.pth",
        "repo_name": "phase3_diffusion_step_17995.pth"
    },
    {
        "local_path": r"C:\Experiments\BEBLaDII\hugging face\README.md",
        "repo_name": "README.md"
    },
    {
        "local_path": r"C:\Experiments\BEBLaDII\storage\components\sep_token.pt",
        "repo_name": "sep_token.pt"
    }
]

def main():
    # TODO: Замените 'YOUR_USERNAME' на ваше имя пользователя на Hugging Face
    # и 'bebladii-foundation-weights' на желаемое имя репозитория
    repo_id = "bulyakovbr/bebladii-foundation-weights"

    # Инициализация API
    # Примечание: предполагается, что вы уже залогинены через 'huggingface-cli login'
    # или переменная окружения HF_TOKEN установлена.
    api = HfApi()

    print(f"Проверка/создание репозитория {repo_id}...")
    try:
        # Создаем публичный репозиторий, если его еще нет
        create_repo(repo_id, private=False, exist_ok=True)
    except Exception as e:
        print(f"Ошибка при создании репозитория: {e}")
        print("Убедитесь, что вы авторизованы (huggingface-cli login) или HF_TOKEN задан.")
        return

    for file_info in FILES_TO_UPLOAD:
        local_path = file_info["local_path"]
        repo_name = file_info["repo_name"]

        if not os.path.exists(local_path):
            print(f"❌ Файл не найден: {local_path}. Пропускаем.")
            continue

        print(f"⏳ Загрузка {local_path} -> {repo_name}...")
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_name,
                repo_id=repo_id,
                repo_type="model"
            )
            print(f"✅ Успешно загружен: {repo_name}")
        except Exception as e:
            print(f"❌ Ошибка при загрузке {repo_name}: {e}")

    print("🎉 Загрузка завершена!")

if __name__ == "__main__":
    main()
