"""
Скрипт экспорта train_phase3_alt.py в два готовых Kaggle GPU блокнота (A/B «3000 шагов правды», v2):
  1. experiments/phase 3 alt/Kaggle/train_phase3_alt_safe_notebook.ipynb
     — Основная боевая связка: standard + v2 + safe (22.94M) + all_5 (damp=0.7) + GPU Dynamic Crop + Online Chunked Exact CE + NoiseRadar BF16/FP16.
  2. experiments/phase 3 alt/Kaggle/train_phase3_alt_control_notebook.ipynb
     — Консервативный контроль: standard + v2 + dense (31.81M) + idea4_wiener_head (damp=0.7) + GPU Dynamic Crop + Online Chunked Exact CE + NoiseRadar BF16/FP16.

Запуск:
  .venv\\Scripts\\python.exe -B "experiments\\phase 3 alt\\export_to_notebook.py"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List


PROFILES: Dict[str, Dict[str, Any]] = {
    "safe": {
        "filename": "train_phase3_alt_safe_notebook.ipynb",
        "title": "BEBLaDII Phase 3 Alt v2 (Plan C) — Боевой прогон `standard + v2 + safe (22.94M) + all_5`",
        "subtitle": (
            "- **Архитектура v2:** `standard` + `arch_version=v2` + `matrix_mode=safe` (`22.94M`, PHM `r=4` в Q/KV/O + плотный `1024d` Energy-Gated SwiGLU FFN + `GrassAlign (2×16d)` + `L·X·R`, `8×16d` SA / `7×16d` CA без `pos`, `AxisBlockAdaLN (r=64)`, `ConditionEmbedderV2 (Fourier 64 + 19 датчиков)`, `GrassmannianOutputHeadV2 (EDM/Винер + PHM r=8 + lambda_keep)`, правое выравнивание промпта `pos=63`, `rope_grass_base=500`)\n"
            "- **Обучение (Этап 0 + v2):** `OrthMuon (sphere_scale, c_base=2.26)`, раздельный `grad_clip (adamw=1.0, sphere=10.0)`, `EMAModel (warmup=True)`, SD3-shift по осям (`axis_shift_delta=0.35`), лоссы `spec_diag=0.05`, `orth_offdiag=0.05`, `anchor=0.15`, `basis_high_t_mode=sign_only`, статический XLA `L_bary` + фильтр стоп-клеток, `GPU Dynamic Crop (full_prob=0.25)` + `LengthBucketBatchSampler`\n"
            "- **Инференс (Этап 1):** Procrustes CFG (`8×8 SVD fp32`), `t_grid=sd3 (shift=0.55)`, `limited_cfg=[0.25, 0.95]`, кэш KV промпта + батчевый `2B` CFG с пропуском `uncond` в `GrassCA`, маскированный `NoiseRadar (~locked)` + `crystal_lock_max_t=0.15` + early-stop по `VOID`-хвосту\n"
            "- **Инициализация:** `all_5` (`wiener_damp=0.7`, `init_samples=128`)\n"
            "- **Словарь:** `NoiseRadar` в нативном `FP16/BF16` + 1-проходный точный `Online Chunked Exact CE` (`chunk=16384`, `0%` смещения) + случайный отбор `B_ce` (`ce_batch_select=random`, `0.5`)\n"
            "- **Валидация:** `SimGr(low/hi)`, `CFG-ΔSim(t50/t75)`, `TokAcc/CE`, `orth_err / spec_l1`, `blk(sem/syn/pos)`, `adaln r_eff/k99`, живая генерация `t=1→0` каждые 500 шагов\n"
            "- **GCS чекпоинты:** `gs://bebladii-weigths-us/planC/Phase3_Alt_Diffusion/v2_safe_all5/checkpoints/`\n"
            "- **Подключите в Kaggle Input:** `bogdanbuliakov/bebladii-planc-phase1-data` и `bogdanbuliakov/bebladii-planc-phase2-data` (ничего нового по словарю добавлять не нужно!)\n"
            "- **Секреты Kaggle:** `WANDB_API_KEY`, `GCP_SA_JSON`"
        ),
        "config_overrides": {
            "arch_preset": "standard",
            "arch_version": "v2",
            "axis_shift_delta": 0.35,
            "gpu_dynamic_crop": True,
            "gpu_dynamic_crop_full_prob": 0.25,
            "matrix_mode": "safe",
            "init_mode": "all_5",
            "wiener_damp": 0.7,
            "max_steps": 3000,
            "gcs_dir": "gs://bebladii-weigths-us/planC/Phase3_Alt_Diffusion/v2_safe_all5/checkpoints/",
            "wandb_name": "phase3-alt-v2-standard-safe-all5",
        },
    },
    "control": {
        "filename": "train_phase3_alt_control_notebook.ipynb",
        "title": "BEBLaDII Phase 3 Alt v2 (Plan C) — Консервативный контроль A/B `standard + v2 + dense (31.81M) + idea4`",
        "subtitle": (
            "- **Назначение:** консервативный эталон v2 для A/B-сравнения с `v2 + safe + all_5` на одинаковых данных, расписании, `GPU Dynamic Crop`, `Online Chunked Exact CE` и `CFG-ΔSim`\n"
            "- **Архитектура v2:** `standard` + `arch_version=v2` + `matrix_mode=dense` (`31.81M`, без PHM в Q/KV/O / GrassAlign / L·X·R, со всеми улучшениями v2: `8×16d`/`7×16d`, `AxisBlockAdaLN`, `ConditionEmbedderV2`, `OutputHeadV2`, `Energy-Gated SwiGLU`)\n"
            "- **Обучение (Этап 0 + v2):** `OrthMuon (sphere_scale, c_base=2.26)`, раздельный `grad_clip`, `EMAModel (warmup=True)`, `axis_shift_delta=0.35`, `spec_diag=0.05`, `orth_offdiag=0.05`, `anchor=0.15`, `GPU Dynamic Crop (full_prob=0.25)`\n"
            "- **Инференс (Этап 1):** Procrustes CFG, `t_grid=sd3`, `limited_cfg=[0.25, 0.95]`, кэш KV + батчевый `2B` CFG с пропуском `uncond` в `GrassCA`, маскированный `NoiseRadar (~locked)`\n"
            "- **Инициализация:** `idea4_wiener_head` (`wiener_damp=0.7`, `init_samples=128` — чистая Винеровская инициализация без `idea1/2/3/5`)\n"
            "- **Словарь:** `NoiseRadar` в нативном `FP16/BF16` + 1-проходный точный `Online Chunked Exact CE` (`chunk=16384`) + случайный отбор `B_ce` (`random`, `0.5`)\n"
            "- **GCS чекпоинты:** `gs://bebladii-weigths-us/planC/Phase3_Alt_Diffusion/v2_control_dense_idea4/checkpoints/`\n"
            "- **Подключите в Kaggle Input:** `bogdanbuliakov/bebladii-planc-phase1-data` и `bogdanbuliakov/bebladii-planc-phase2-data`\n"
            "- **Секреты Kaggle:** `WANDB_API_KEY`, `GCP_SA_JSON`"
        ),
        "config_overrides": {
            "arch_preset": "standard",
            "arch_version": "v2",
            "axis_shift_delta": 0.35,
            "gpu_dynamic_crop": True,
            "gpu_dynamic_crop_full_prob": 0.25,
            "matrix_mode": "dense",
            "init_mode": "idea4_wiener_head",
            "wiener_damp": 0.7,
            "max_steps": 3000,
            "gcs_dir": "gs://bebladii-weigths-us/planC/Phase3_Alt_Diffusion/v2_control_dense_idea4/checkpoints/",
            "wandb_name": "phase3-alt-v2-standard-dense-idea4-control",
        },
    },
}


def _apply_config_overrides(cell_lines: List[str], overrides: Dict[str, Any]) -> List[str]:
    """Подменяет значения ключей в словаре TRAINING_CONFIG в последней ячейке."""
    out: List[str] = []
    for line in cell_lines:
        replaced = False
        stripped = line.lstrip()
        for k, v in overrides.items():
            prefix = f'"{k}":'
            if stripped.startswith(prefix):
                indent = line[: len(line) - len(stripped)]
                out.append(f'{indent}"{k}": {v!r},\n')
                replaced = True
                break
        if not replaced:
            out.append(line)
    return out


def build_notebook(py_path: Path, ipynb_path: Path, profile_key: str) -> None:
    prof = PROFILES[profile_key]
    with open(py_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cells: List[Dict[str, Any]] = []

    # Ячейка 0: карточка запуска (Markdown)
    header_md = [f"# {prof['title']}\n", "\n"] + [ln + "\n" for ln in prof["subtitle"].splitlines()]
    cells.append({"cell_type": "markdown", "metadata": {}, "source": header_md})

    # Ячейка 1: зависимости, git clone/pull, sys.path и проверка наличия вспомогательных модулей
    setup_code = [
        "# 1. Установка необходимых пакетов на Kaggle\n",
        "!pip install -q einops wandb\n",
        "\n",
        "# 2. Синхронизация репозитория BEBLaDII (git clone / git pull)\n",
        "import os, sys, subprocess\n",
        "from pathlib import Path\n",
        "\n",
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
        "    for _p in [\n",
        "        PROJECT_ROOT,\n",
        "        os.path.join(PROJECT_ROOT, 'experiments', 'phase 3'),\n",
        "        os.path.join(PROJECT_ROOT, 'experiments', 'phase 3 alt'),\n",
        "    ]:\n",
        "        if os.path.isdir(_p) and _p not in sys.path:\n",
        "            sys.path.insert(0, _p)\n",
        "\n",
        "    # Fallback: если свежие модули phase 3 alt залиты через Kaggle Dataset (/kaggle/input), подхватим их тоже\n",
        "    required_mods = ['vocab_optim.py', 'matrix_layers.py', 'init_strategies.py']\n",
        "    alt_dir = Path(PROJECT_ROOT) / 'experiments' / 'phase 3 alt'\n",
        "    missing = [m for m in required_mods if not (alt_dir / m).exists()]\n",
        "    if missing and os.path.isdir('/kaggle/input'):\n",
        "        for mod_name in list(missing):\n",
        "            hits = list(Path('/kaggle/input').rglob(mod_name))\n",
        "            if hits:\n",
        "                mod_dir = str(hits[0].parent)\n",
        "                if mod_dir not in sys.path:\n",
        "                    sys.path.insert(0, mod_dir)\n",
        "                print(f'[Fallback Input] Найден {mod_name} в {mod_dir}')\n",
        "                missing.remove(mod_name)\n",
        "    if missing:\n",
        "        raise FileNotFoundError(\n",
        "            f'Не найдены модули Phase 3 Alt: {missing}. '\n",
        "            'Сделайте git push ветки в GitHub (Laeryid/BEBLaDII) или приложите эти .py файлы в Kaggle Dataset.'\n",
        "        )\n",
        "    print('[Setup OK] Репозиторий и все модули Phase 3 Alt (vocab_optim, matrix_layers, init_strategies) готовы.')\n",
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

    current_cell_type = "code"
    current_cell_lines: List[str] = []

    def flush_cell() -> None:
        nonlocal current_cell_type, current_cell_lines
        if current_cell_type is not None and current_cell_lines:
            while current_cell_lines and current_cell_lines[0].strip() == "":
                current_cell_lines.pop(0)
            while current_cell_lines and current_cell_lines[-1].strip() == "":
                current_cell_lines.pop()

            if current_cell_lines:
                if current_cell_type == "markdown":
                    cleaned = []
                    for ln in current_cell_lines:
                        if ln.startswith("# "):
                            cleaned.append(ln[2:])
                        elif ln.startswith("#"):
                            cleaned.append(ln[1:])
                        else:
                            cleaned.append(ln)
                    source_lines = cleaned
                else:
                    source_lines = current_cell_lines
                    if any("TRAINING_CONFIG" in ln for ln in source_lines):
                        source_lines = _apply_config_overrides(source_lines, prof["config_overrides"])

                cell: Dict[str, Any] = {
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
            title = line[4:].strip()
            if title:
                cells.append(
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": [f"## {title}\n"],
                    }
                )
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

    ipynb_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ipynb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    print(f"[Export] Профиль '{profile_key}' экспортирован ({len(cells)} ячеек): {ipynb_path}")


def main() -> None:
    alt_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Export train_phase3_alt.py to Kaggle GPU notebooks")
    parser.add_argument(
        "--profiles",
        type=str,
        default="safe,control",
        help="Какие профили собрать (через запятую): safe, control",
    )
    parser.add_argument("--src", type=str, default=str(alt_dir / "train_phase3_alt.py"))
    parser.add_argument("--out-dir", type=str, default=str(alt_dir / "Kaggle"))
    args = parser.parse_args()

    src_path = Path(args.src)
    out_dir = Path(args.out_dir)
    for key in [k.strip() for k in args.profiles.split(",") if k.strip()]:
        if key not in PROFILES:
            raise ValueError(f"Неизвестный профиль '{key}'. Доступны: {list(PROFILES.keys())}")
        build_notebook(src_path, out_dir / PROFILES[key]["filename"], key)


if __name__ == "__main__":
    main()
