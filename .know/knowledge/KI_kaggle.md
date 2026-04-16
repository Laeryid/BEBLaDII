<!-- last_verified: 2026-04-14 -->
# KI: Kaggle Integration

## Что это
Инструкции по использованию Kaggle CLI для загрузки и обновления ресурсов проекта (веса моделей, подготовленные датасеты).

## Настройка API
1. Получите `kaggle.json` в настройках профиля Kaggle (`Settings -> API -> Create New Token`).
2. Разместите файл по пути: `%USERPROFILE%\.kaggle\kaggle.json` (обычно `C:\Users\<Name>\.kaggle\kaggle.json`).

## Основные команды (через .venv)
Библиотеку следует запускать через `python -m kaggle`, чтобы гарантировать использование окружения:

| Команда | Описание |
|---|---|
| `.venv\Scripts\python.exe -m kaggle datasets create -p kaggle_upload --dir-mode tar` | Создать новый датасет из папки (с сохранением структуры). |
| `.venv\Scripts\python.exe -m kaggle datasets version -p kaggle_upload -m "msg" --dir-mode tar` | Создать новую версию датасета (с сохранением структуры). |
| `.venv\Scripts\python.exe -m kaggle datasets status <id>` | Проверить статус загрузки. |

## Неочевидные детали
- **Структура папок**: Всегда используйте флаг `--dir-mode tar`, иначе Kaggle пропустит вложенные директории (например, `data/`). Режим `tar` предпочтительнее для уже сжатых файлов (parquet/pt).
- **Метаданные**: Файл `dataset-metadata.json` в папке загрузки ОБЯЗАТЕЛЕН. Он содержит `id` и `title`.
- **Windows**: Если команда `kaggle` не найдена в PATH, всегда используйте префикс `.venv\Scripts\python.exe -m`.
- **Подготовка**: Перед загрузкой всегда запускайте `scripts/prepare_kaggle_data.py` для формирования структуры папок.

## Аварийное сохранение (Rescue)
Если обучение в Kaggle Notebook зависло или сессия истекает, а чекпоинты не сохранились на диск, используйте скрипт `scripts/kaggle_emergency_save.py`.

**Как использовать:**
1. Остановите ячейку с обучением (Interrupt).
2. Скопируйте содержимое `scripts/kaggle_emergency_save.py` в новую ячейку.
3. Запустите её. Скрипт попытается найти объект модели в памяти (даже через `gc`) и принудительно сохранить веса в `/kaggle/working/`.
4. Используйте появившуюся ссылку `FileLink` для немедленного скачивания.

## Типичные ошибки
- **WinError 3 (FileNotFound)**: Возникает, если в `kaggle_upload` отсутствуют промежуточные папки (исправлено в `prepare_kaggle_data.py`).
- **401 Unauthorized**: Проверьте наличие и содержимое `kaggle.json`.
- **Slug conflict**: Если `id` в метаданных уже занят другим пользователем или удаленным датасетом.
- **404 на скачивание**: Если `FileLink` выдает 404, попробуйте `Save Version -> Quick Save` и скачайте файл из вкладки Output после завершения.
