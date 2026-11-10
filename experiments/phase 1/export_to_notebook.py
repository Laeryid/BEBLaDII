"""
Скрипт экспорта py-percent скрипта train_phase1.py в Jupyter Notebook для Kaggle.
Создает C:\\Experiments\\BEBLaDII\\experiments\\phase 1\\Kaggle\\train_phase1_notebook.ipynb
"""

import os
import sys
import json
import re


def py_percent_to_notebook(py_path: str, ipynb_path: str):
    with open(py_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cells = []
    current_cell_type = None
    current_cell_lines = []

    # Добавляем в начало блокнота служебную ячейку для Kaggle с установкой пакетов и клонированием при необходимости
    setup_code = [
        "# Установка необходимых пакетов на Kaggle\n",
        "!pip install -q einops wandb\n",
    ]
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": setup_code
    })

    def flush_cell():
        nonlocal current_cell_type, current_cell_lines
        if current_cell_type is not None and current_cell_lines:
            # Убираем лишние пустые строки в начале и конце ячейки
            while current_cell_lines and current_cell_lines[0].strip() == "":
                current_cell_lines.pop(0)
            while current_cell_lines and current_cell_lines[-1].strip() == "":
                current_cell_lines.pop()

            if current_cell_lines:
                # Если markdown, убираем лидирующие '# ' из строк
                if current_cell_type == "markdown":
                    cleaned_source = []
                    for line in current_cell_lines:
                        if line.startswith("# "):
                            cleaned_source.append(line[2:])
                        elif line.startswith("#"):
                            cleaned_source.append(line[1:])
                        else:
                            cleaned_source.append(line)
                    source_lines = cleaned_source
                else:
                    source_lines = current_cell_lines

                cell = {
                    "cell_type": current_cell_type,
                    "metadata": {},
                    "source": source_lines
                }
                if current_cell_type == "code":
                    cell["execution_count"] = None
                    cell["outputs"] = []
                cells.append(cell)

        current_cell_type = None
        current_cell_lines = []

    for line in lines:
        if line.startswith("# %% [markdown]"):
            flush_cell()
            current_cell_type = "markdown"
        elif line.startswith("# %%"):
            flush_cell()
            current_cell_type = "code"
        else:
            if current_cell_type is not None:
                current_cell_lines.append(line)

    flush_cell()

    # Формируем структуру Jupyter Notebook (nbformat 4.5)
    notebook = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.10"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    os.makedirs(os.path.dirname(ipynb_path), exist_ok=True)
    with open(ipynb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    print(f"[Export] Блокнот успешно экспортирован ({len(cells)} ячеек) в:\n{ipynb_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_py = os.path.join(base_dir, "train_phase1.py")
    out_ipynb = os.path.join(base_dir, "Kaggle", "train_phase1_notebook.ipynb")
    py_percent_to_notebook(src_py, out_ipynb)
