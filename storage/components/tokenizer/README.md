# Компонента: Токенизатор Tekken (Mistral AI)

Компонента содержит автономный токенизатор **Tekken** (семейство Mistral NeMo 12B / Mistral Large 2), принятый в качестве базового для архитектуры **Plan C** (см. [ADR 120](file:///C:/Experiments/BEBLaDII/.know/decisions/120_plan_c_tekken_tokenizer_for_cyrillic.md) и [plan C.md](file:///C:/Experiments/BEBLaDII/experiments/plan%20C.md)).

---

## 1. Назначение и характеристики

- **Размер словаря:** ровно **$131\,072$** токена (`vocab_size = 131072`).
- **Алгоритм:** Byte-level BPE с регулярными выражениями под мультиязычный текст и код.
- **Преимущества для кириллицы:** значительно превосходит Qwen2.5 по коэффициенту сжатия русских слов (в среднем 3.0–3.5 символа на токен), не дробит окончания на мелкие субтокены и байты.
- **Автономность:** Не требует скачивания весов моделей Mistral (общий размер словаря на диске $\approx 17$ МБ). При этом сохраняется полная OOD-генерализация на произвольной лексике вне локальных датасетов.

---

## 2. Структура файлов директории

| Файл | Размер | Назначение |
| :--- | :---: | :--- |
| `tokenizer.json` | $\approx 17.0$ МБ | Полный граф BPE-слияний и словарь в формате Hugging Face `tokenizers` |
| `tekken.json` | $\approx 14.8$ МБ | Нативный формат токенизатора `mistral-common` (Mistral AI) |
| `tokenizer_config.json` | $\approx 185$ КБ | Метаданные, спецтокены (`<s>`, `</s>`, `<unk>`) и настройки парсера |
| `special_tokens_map.json` | $\approx 437$ Б | Карта специальных токенов |
| `__init__.py` | $\approx 1$ КБ | Вызываемый Python-модуль с хелпером `get_tokenizer()` |

---

## 3. Способы использования в коде

### Вариант А: Через встроенный модуль компонента (Рекомендуется)

```python
from storage.components.tokenizer import get_tokenizer

# Загрузка строго из локальной папки (без обращений к сети)
tokenizer = get_tokenizer()

text = "Спектральное латентное пространство Грассманиана"
token_ids = tokenizer.encode(text)
decoded_text = tokenizer.decode(token_ids)

print("IDs:", token_ids)
print("Decoded:", decoded_text)
```

### Вариант Б: Через стандартный Hugging Face `transformers`

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    r"C:\Experiments\BEBLaDII\storage\components\tokenizer",
    local_files_only=True,
    fix_mistral_regex=True,
)
```

### Вариант В: Через сверхбыстрый Rust-движок `tokenizers`

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(
    r"C:\Experiments\BEBLaDII\storage\components\tokenizer\tokenizer.json"
)
output = tokenizer.encode("Привет, мир!")
print(output.ids)
```

---

## 4. Специфика интеграции в Plan C

1. **Клетка холста:** $8 \times 128 = 1024$ чисел.
2. **Выходная матрица словаря декомпрессора (`LatentDecompressorDecoder`):**
   - Проекция: `nn.Linear(1024, 131072, bias=False)`.
   - Вес в памяти: ровно **256 МБ** в `bfloat16` (134.2M параметров).
   - Обучается с нуля по `CrossEntropyLoss` на текстовом корпусе без необходимости переноса 4096-мерных весов Mistral.
3. **Совместимость токенов:** ID от $0$ до $131\,071$ остаются неизменными (без переиндексации).
