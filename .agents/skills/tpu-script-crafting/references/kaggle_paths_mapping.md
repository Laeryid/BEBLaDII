# Kaggle vs Local Repository: File & Path Mapping Reference

При разработке скриптов под Kaggle TPU (v5e-8, v3-8) критически важно понимать дуальную структуру путей: в Kaggle среда строго изолирована и монтирует входные данные в `/kaggle/input` (read-only), а рабочую директорию — в `/kaggle/working` (writable). В локальном окружении разработка ведётся в корневом каталоге `C:\Experiments\BEBLaDII`.

---

## 1. Сопоставительная таблица путей (Mapping Table)

| Ресурс / Компонент | Путь на Kaggle (`/kaggle/...`) | Локальный путь (`PROJECT_ROOT`) | Описание / Назначение |
|---|---|---|---|
| **Корень репозитория** | `/kaggle/working/BEBLaDII` | `C:/Experiments/BEBLaDII` | Исходный код `src/beb_la_dii`, конфиги, компоненты |
| **Базовая модель Qwen (HF)** | `/kaggle/input/datasets/ragnar123/qwen2-5-1-5b`<br>или `/kaggle/input/qwen2-5-1-5b` | `Qwen/Qwen2.5-1.5B`<br>(HF Hub / local cache) | Входной токенизатор и эмбеддинги промпта |
| **Бэкбон ModernBERT (HF)** | `/kaggle/input/models/answer-ai/modernbert/transformers/large/2`<br>или `/kaggle/input/modernbert-large` | `answerdotai/ModernBERT-large`<br>(HF Hub / local cache) | Базовая архитектура латентного остова |
| **Веса VAE LatentEncoder** | `/kaggle/input/**/planB_phase1_checkpoints_phase1_vae_step_20000.pth` | `experiments/phase 1/planB_phase1_checkpoints_phase1_vae_step_20000.pth` | Проектор эмбеддингов Qwen в латентное пространство (1024-d) |
| **Чекпоинт DUS Phase 4** | `/kaggle/input/**/phase4_step_85995.pth` | `experiments/phase 4/local_checkpoints/phase4_step_85995.pth` | Обученный латентный остов DUS (Hierarchical Diffusion) |
| **Токен-разделитель (SEP)** | `/kaggle/working/BEBLaDII/storage/components/sep_token.pt` | `storage/components/sep_token.pt` | Токен `<\|thoughts\|>` (якорь границы промпта и ответа) |
| **Токен пустоты (VOID)** | `/kaggle/working/BEBLaDII/storage/components/void_token.pt` | `storage/components/void_token.pt` | Токен `<\|thought_void\|>` (маскирование контекста) |
| **Словарь латентов (Dict)** | `/kaggle/input/**/latent_dict.pt` | `experiments/phase 5/local/latent_dict.pt` | Кодовая книга латентов для декодирования и метрик DProx |
| **Датасет Train (Parquet)** | `/kaggle/input/**/train_phase6.parquet` | `BEBLaDII-planB-Phase6-Data/phase 6/data/train_phase6.parquet` | Обучающая выборка (Q, A с масками) |
| **Датасет Val (Parquet)** | `/kaggle/input/**/val_phase6.parquet` | `BEBLaDII-planB-Phase6-Data/phase 6/data/val_phase6.parquet` | Валидационная выборка |
| **Директория сохранения** | `/kaggle/working/checkpoints/...` | `experiments/phase 6/checkpoints/...` | Локальный сброс перед отправкой в GCS |
| **Облачное хранилище (GCS)**| `gs://bebladii-weigths-us/planB/...` | `gs://bebladii-weigths-us/planB/...` | Бакет постоянного хранения чекпоинтов |

---

## 2. Золотой паттерн: Dual-Environment Path Resolver

Чтобы скрипт мог без модификаций запускаться **и локально (для Level 3 Dry-Run), и на Kaggle TPU**, используйте этот универсальный резолвер:

```python
import os
import pathlib
from pathlib import Path

def get_project_root() -> Path:
    """Определяет корень проекта в зависимости от среды."""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle").exists():
        return Path("/kaggle/working/BEBLaDII")
    # Локально: берем переменную окружения PROJECT_ROOT или fallback на C:/Experiments/BEBLaDII
    return Path(os.environ.get("PROJECT_ROOT", "C:/Experiments/BEBLaDII"))

def resolve_resource_path(filename: str, local_rel_path: str = "") -> str:
    """
    Разрешает путь к файлу:
    1. Ищет в локальном репозитории по относительному пути (local_rel_path).
    2. Если запущено на Kaggle, ищет в /kaggle/input через rglob.
    """
    root = get_project_root()
    
    # 1. Проверяем локальный путь репозитория
    if local_rel_path:
        local_file = root / local_rel_path
        if local_file.exists():
            return str(local_file)
            
    # 2. Проверяем /kaggle/input (для Kaggle окружения)
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for match in kaggle_input.rglob(filename):
            return str(match)

    # 3. Fallback на имя файла
    fallback = root / local_rel_path if local_rel_path else Path(filename)
    return str(fallback)

def resolve_model_path(hf_model_id: str, kaggle_input_dir: str) -> str:
    """
    Возвращает локальный путь к весам из датасета Kaggle, 
    либо HuggingFace hub id при локальном запуске.
    """
    kaggle_p = Path(kaggle_input_dir)
    if kaggle_p.exists():
        # Ищем папку, содержащую config.json
        for cfg in kaggle_p.rglob("config.json"):
            return str(cfg.parent)
    return hf_model_id
```

---

## 3. Критические особенности Kaggle Filesystem

1. **`/kaggle/input` строго Read-Only**: Попытка сохранить туда чекпоинт или временный лог вызовет `PermissionError` или `OSError: Read-only file system`. Сохранение чекпоинтов строго в `/kaggle/working/` или `/tmp/`.
2. **Лимит диска `/kaggle/working` (20 GB)**: Диск быстро переполняется весами. Для долгих тренировок чекпоинты необходимо отправлять в GCS через `gsutil` и удалять локально:
   ```python
   def sync_to_gcs_and_delete(local_path: str, gcs_dir: str):
       gcs_path = gcs_dir.rstrip("/") + "/" + os.path.basename(local_path)
       subprocess.run(["gsutil", "-q", "cp", local_path, gcs_path], check=True)
       os.remove(local_path)
   ```
3. **Обновление репозитория в ячейке 1**:
   На Kaggle репозиторий клонируется в `/kaggle/working/BEBLaDII`. Всегда используйте безопасное обновление:
   ```python
   PROJECT_ROOT = Path("/kaggle/working/BEBLaDII")
   if not PROJECT_ROOT.exists():
       subprocess.run(["git", "clone", "https://github.com/Laeryid/BEBLaDII.git", str(PROJECT_ROOT)], check=True)
   else:
       subprocess.run(["git", "-C", str(PROJECT_ROOT), "pull"], check=True)
   ```
