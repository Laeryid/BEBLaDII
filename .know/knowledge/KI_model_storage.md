<!-- last_verified: 2026-11-15 -->
# KI: Model Storage & Lifecycle

## Что это
Рекомендации и правила хранения весов моделей (внешних и собственных), управления кэшем и стратегии работы в разных окружениях (Local vs Kaggle).

## Локальное хранение (Windows)
* **Путь к кэшу HuggingFace**: `C:\Users\hp\.cache\huggingface\hub` (стандартное поведение библиотек `transformers` и `huggingface_hub`).
* **Управление кэшем**:
  * Регулируется переменной окружения `HF_HOME`.
  * Анализ места: `huggingface-cli cache scan`.
* **Правило**: Не хранить внешние (базовые) модели в папке проекта.

## Структура `storage/components/` (Plan C)
* `storage/components/tokenizer/` — автономный токенизатор Tekken (`vocab_size=131072`, Plan C).
* `storage/components/plan_c_layers.py` — общие слои Plan C (`SphereLinear`, `ModernBERTLayer`, `RMSNorm`, `RotaryEmbedding128`, `LorentzianEigh`, `clamp_to_unit_ball`).
* `storage/components/encoder/` — автономная компонента `GrassmannianEncoder` (`__init__.py`, `config.json`, `README.md`, `weights.pt`).
* `storage/components/decoder/` — автономная компонента `GrassmannianDecoder` (`__init__.py`, `config.json`, `README.md`, `weights.pt`).
* `storage/components/sep_concept.pt`, `storage/components/void_concept.pt` — компактные детерминированные якорные концепты Plan C `(8, 128)`.
* `storage/components/plan B/` — архивные веса и конфиги Plan B (`model/`, `projector/`, `sep_token.pt`, `void_token.pt`).

## Стратегия версионирования (Git)
* **Крупные веса (`.pt`, `.pth`, `storage/**/weights.pt`)**: исключены через `.gitignore`, хранятся локально, в GCS или в Private Kaggle Dataset / Hugging Face.
* **Мелкие якорные тензоры (~4 КБ)**: `sep_concept.pt`, `void_concept.pt` и архивные `plan B/{sep_token.pt, void_token.pt}` разрешены явными исключениями в `.gitignore`.
* **Конфиги и обвязка**: `.json`, `.md` и `.py` внутри `storage/components/` обязательно хранятся в Git.

## Работа на Kaggle
| Ресурс | Стратегия | Механизм |
|---|---|---|
| **Base Models** (ModernBERT, Mistral SVD) | Динамическая загрузка / препроцессинг | HF или заранее собранный Kaggle Dataset |
| **Custom Weights** (`encoder/weights.pt`, `decoder/weights.pt`) | Постоянное хранение | Private Kaggle Dataset / GCS |

## Юридические аспекты и Лицензии
* **Apache 2.0**: Основная лицензия проекта. Позволяет публичное распространение кода и весов.
* **Атрибуция**: При публикации весов, инициализированных из моделей Mistral/Tekken или DeepSeek, в README указывается первоисточник.
