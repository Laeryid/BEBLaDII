"""
Скрипт синхронизации весов из GCP Cloud Storage (GCS) на Google Drive с поддержкой докачки (rsync).

Использование в Google Colab:
    1. Примонтировать Google Drive
    2. Выполнить auth.authenticate_user()
    3. Запустить скрипт через python scripts/download_gcs_to_gdrive.py
"""

import os
import subprocess
import sys


def check_colab_environment():
    """Проверка работы в окружении Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def setup_and_rsync(
    gcs_src_path: str = "gs://bebladii-weigths-us/planB/phase3/checkpoints/",
    gdrive_dest_dir: str = "/content/drive/MyDrive/BEBLaDII_weights_phase3_attempt20260722/",
):
    """
    Выполняет подготовку директорий и синхронизацию файлов (rsync) из GCS на Google Drive.
    """
    if check_colab_environment():
        from google.colab import auth, drive

        print("[1/4] Монтирование Google Drive...")
        drive.mount('/content/drive')

        print("[2/4] Авторизация в GCP...")
        auth.authenticate_user()

    print(f"[3/4] Создание целевой директории: {gdrive_dest_dir}")
    os.makedirs(gdrive_dest_dir, exist_ok=True)

    print(f"[4/4] Запуск синхронизации (rsync): {gcs_src_path} -> {gdrive_dest_dir}")
    cmd = ["gcloud", "storage", "rsync", "-r", gcs_src_path, gdrive_dest_dir]

    try:
        subprocess.run(cmd, check=True)
        print("Синхронизация успешно завершена!")
    except subprocess.CalledProcessError as err:
        print(f"Ошибка при выполнении gcloud storage rsync: {err}")
        print("Пробуем резервный вариант с gsutil -m rsync...")
        backup_cmd = ["gsutil", "-m", "rsync", "-r", gcs_src_path, gdrive_dest_dir]
        subprocess.run(backup_cmd, check=True)

    print("\nФайлы на Google Drive:")
    if os.path.exists(gdrive_dest_dir):
        for item in os.listdir(gdrive_dest_dir):
            item_path = os.path.join(gdrive_dest_dir, item)
            size_mb = os.path.getsize(item_path) / (1024 * 1024) if os.path.isfile(item_path) else 0
            print(f" - {item} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    src_path = sys.argv[1] if len(sys.argv) > 1 else "gs://bebladii-weigths/checkpoints/"
    dest_dir = sys.argv[2] if len(sys.argv) > 2 else "/content/drive/MyDrive/BEBLaDII_weights/"
    setup_and_rsync(src_path, dest_dir)
