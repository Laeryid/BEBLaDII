from pathlib import Path
from typing import Optional
from transformers import AutoTokenizer, PreTrainedTokenizerFast

TOKENIZER_DIR = Path(__file__).resolve().parent


def get_tokenizer(
    tokenizer_path: Optional[str | Path] = None,
    local_files_only: bool = True,
    fix_mistral_regex: bool = True,
) -> PreTrainedTokenizerFast:
    """Загружает токенизатор Tekken (Mistral NeMo 12B, vocab=131072) из локальной директории.

    Args:
        tokenizer_path: Путь к директории токенизатора (по умолчанию: локальная папка компонента).
        local_files_only: Загрузка строго с локального диска без обращения к HuggingFace Hub.
        fix_mistral_regex: Применить исправление regex-паттерна для токенизаторов Mistral/Tekken.

    Returns:
        Экземпляр PreTrainedTokenizerFast.
    """
    path = str(tokenizer_path or TOKENIZER_DIR)
    kwargs = {
        "local_files_only": local_files_only,
    }
    if fix_mistral_regex:
        kwargs["fix_mistral_regex"] = True

    return AutoTokenizer.from_pretrained(path, **kwargs)


__all__ = ["get_tokenizer", "TOKENIZER_DIR"]
