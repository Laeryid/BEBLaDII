"""
Скрипт экспорта py-percent скрипта train_phase2.py в Jupyter Notebook для Kaggle.
Создает experiments/phase 2/Kaggle/train_phase2_notebook.ipynb.
"""

import argparse
import json
import os


def py_percent_to_notebook(py_path: str, ipynb_path: str) -> None:
    with open(py_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cells = []
    current_cell_type = None
    current_cell_lines = []

    setup_code = [
        "# 1. Установка необходимых пакетов на Kaggle\n",
        "!pip install -q einops wandb\n",
        "\n",
        "# 2. Синхронизация репозитория BEBLaDII (git clone / git pull)\n",
        "import os, sys, subprocess\n",
        "REPO_URL = 'https://github.com/Laeryid/BEBLaDII.git'\n",
        "PROJECT_ROOT = '/kaggle/working/BEBLaDII'\n",
        "\n",
        "if os.path.exists('/kaggle'):\n",
        "    if not os.path.exists(PROJECT_ROOT):\n",
        "        print(f'Клонирование репозитория из {REPO_URL}...')\n",
        "        subprocess.run(['git', 'clone', REPO_URL, PROJECT_ROOT], check=False)\n",
        "    else:\n",
        "        print('Репозиторий уже существует. Выполняю git pull...')\n",
        "        subprocess.run(['git', '-C', PROJECT_ROOT, 'pull'], check=False)\n",
        "    if PROJECT_ROOT not in sys.path:\n",
        "        sys.path.insert(0, PROJECT_ROOT)\n",
        "    print('[Repo Sync] Репозиторий актуализирован и добавлен в sys.path.')\n",
    ]
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": setup_code,
        }
    )

    def flush_cell() -> None:
        nonlocal current_cell_type, current_cell_lines
        if current_cell_type is not None and current_cell_lines:
            while current_cell_lines and current_cell_lines[0].strip() == "":
                current_cell_lines.pop(0)
            while current_cell_lines and current_cell_lines[-1].strip() == "":
                current_cell_lines.pop()

            if current_cell_lines:
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
                    "source": source_lines,
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

    notebook = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.10"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    parent = os.path.dirname(ipynb_path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    with open(ipynb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    print(f"[Export] Блокнот Phase 2 (Decoder & Radar) экспортирован ({len(cells)} ячеек) в:\n{ipynb_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Export Phase 2 Decoder train_phase2.py to Kaggle notebook")
    parser.add_argument("--src", type=str, default=os.path.join(base_dir, "train_phase2.py"))
    parser.add_argument("--dst", type=str, default=os.path.join(base_dir, "Kaggle", "train_phase2_notebook.ipynb"))
    args = parser.parse_args()
    py_percent_to_notebook(args.src, args.dst)
