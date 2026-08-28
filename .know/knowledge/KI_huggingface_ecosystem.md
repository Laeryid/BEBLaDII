<!-- last_verified: 2026-08-25 -->
# KI: Hugging Face Ecosystem

## Что это
Описание работы со скриптами в папке `hugging face/`, правила загрузки весов проекта в Hub и маппинг названий файлов.

## Основные операции
Скрипт `upload_weights.py` используется для загрузки фундаментальных чекпоинтов на Hugging Face (в публичный репозиторий `bulyakovbr/bebladii-foundation-weights`).

### Важные нюансы
*   **Авторизация**: Перед запуском скрипта необходимо убедиться, что выполнен `huggingface-cli login` или установлена переменная окружения `HF_TOKEN`.
*   **Маппинг файлов**: При загрузке локальные названия файлов заменяются на более понятные `repo_name` для репозитория. Это помогает избежать путаницы при использовании весов другими исследователями.

## Маппинг весов (Reference)
Ниже представлен текущий маппинг файлов (Local Path -> Repo Name), используемый в скрипте:

| Фаза | Локальный путь (пример) | Имя в HF Repo |
| :--- | :--- | :--- |
| **Phase 1 (VAE)** | `experiments\phase 1\planB_phase1_checkpoints_phase1_vae_step_20000.pth` | `phase1_vae_step_20000.pth` |
| **Phase 2 (Decoder)** | `experiments\phase 2\planB_phase2_checkpoints_decoder_step_9000.pth` | `phase2_decoder_step_9000.pth` |
| **Phase 3 (Diffusion)** | `experiments\phase 3\local_checkpoints\phase3_step_17995.pth` | `phase3_diffusion_step_17995.pth` |
| **Tokens** | `storage\components\sep_token.pt` | `sep_token.pt` |

> [!WARNING]
> Если структура папок или названия чекпоинтов меняются в ходе новых экспериментов, необходимо обязательно обновлять массив `FILES_TO_UPLOAD` в файле `hugging face/upload_weights.py`!

## Related KIs
* `KI_model_storage.md`
