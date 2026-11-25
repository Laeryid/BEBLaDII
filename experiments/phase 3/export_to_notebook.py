"""
Скрипт экспорта py-percent скрипта train_phase2.py в Jupyter Notebook для Kaggle.
Создает experiments/phase 2/Kaggle/train_phase2_notebook.ipynb.
"""

import argparse
import json
import os


def py_percent_to_notebook(py_path: str, ipynb_path: str, is_tpu: bool = False) -> None:
    with open(py_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cells = []
    current_cell_type = None
    current_cell_lines = []

    if is_tpu:
        setup_code = [
            "# 1. Установка необходимых пакетов на Kaggle TPU (с проверкой каждого пакета)\n",
            "import os, sys, subprocess\n",
            "try:\n",
            "    import einops\n",
            "    import wandb\n",
            "    from google.cloud import storage  # noqa: F401\n",
            "except ImportError:\n",
            "    subprocess.run(\n",
            "        [sys.executable, '-m', 'pip', 'install', '-q', '--no-cache-dir',\n",
            "         'einops', 'wandb', 'google-cloud-storage'],\n",
            "        check=False,\n",
            "    )\n",
            "\n",
            "# 2. Синхронизация репозитория BEBLaDII (git clone / git pull)\n",
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
            "\n",
            "# 3. Диагностика наличия TorchTPU (пассивная, без захвата TPU-чипов до xmp.spawn)\n",
            "try:\n",
            "    import importlib.metadata as _im\n",
            "    import importlib.util as _iu\n",
            "    _tpu_pkgs = sorted(\n",
            "        f\"{d.metadata['Name']}=={d.version}\"\n",
            "        for d in _im.distributions()\n",
            "        if d.metadata.get('Name') and any(k in d.metadata['Name'].lower() for k in ('tpu', 'xla', 'torch'))\n",
            "    )\n",
            "    _has_torch_tpu = _iu.find_spec('torch_tpu') is not None\n",
            "    _has_torchtpu = _iu.find_spec('torchtpu') is not None\n",
            "    _has_xla = _iu.find_spec('torch_xla') is not None\n",
            "    import torch as _t\n",
            "    _has_attr = hasattr(_t, 'tpu')\n",
            "    _priv1 = getattr(getattr(_t, '_C', None), '_get_privateuse1_backend_name', lambda: None)()\n",
            "    _tpu_env = {\n",
            "        k: v for k, v in sorted(os.environ.items())\n",
            "        if any(k.startswith(p) for p in ('TPU_', 'PJRT_', 'LIBTPU_', 'XLA_', 'TORCH_TPU'))\n",
            "    }\n",
            "    _installed = _has_torch_tpu or _has_torchtpu or _has_attr or (_priv1 == 'tpu')\n",
            "    print('=' * 64, flush=True)\n",
            "    print(\n",
            "        f'[TorchTPU Probe] INSTALLED={_installed} | spec(torch_tpu)={_has_torch_tpu} '\n",
            "        f'| spec(torchtpu)={_has_torchtpu} | spec(torch_xla)={_has_xla}',\n",
            "        flush=True,\n",
            "    )\n",
            "    print(\n",
            "        f\"[TorchTPU Probe] torch=={_t.__version__} | hasattr(torch,'tpu')={_has_attr} | privateuse1={_priv1!r}\",\n",
            "        flush=True,\n",
            "    )\n",
            "    print(f'[TorchTPU Probe] packages={_tpu_pkgs}', flush=True)\n",
            "    print(f'[TorchTPU Probe] env={_tpu_env}', flush=True)\n",
            "    print('=' * 64, flush=True)\n",
            "except Exception as _e:\n",
            "    print(f'[TorchTPU Probe] check failed (non-fatal): {_e}', flush=True)\n",
        ]
    else:
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
            "accelerator": "TPU" if is_tpu else "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    parent = os.path.dirname(ipynb_path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    with open(ipynb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    tag = "Phase 3 TPU" if is_tpu else "Phase 3"
    print(f"[Export] Блокнот {tag} успешно экспортирован ({len(cells)} ячеек) в:\n{ipynb_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Export train_phase3*.py to Kaggle notebook")
    parser.add_argument("--tpu", action="store_true", help="Экспортировать TPU-версию (train_phase3_tpu.py)")
    parser.add_argument("--src", type=str, default="")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()
    default_src = os.path.join(base_dir, "train_phase3_tpu.py" if args.tpu else "train_phase3.py")
    default_out = os.path.join(
        base_dir,
        "Kaggle",
        "train_phase3_tpu_notebook.ipynb" if args.tpu else "train_phase3_notebook.ipynb",
    )
    py_percent_to_notebook(args.src or default_src, args.out or default_out, is_tpu=args.tpu)
