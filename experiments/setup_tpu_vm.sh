#!/bin/bash
# ==============================================================================
# SETUP SCRIPT FOR CLOUD TPU VM (Spot)
# Project: BEBLaDII
# ==============================================================================

# 1. Системные зависимости
echo "[1/5] Installing system packages..."
sudo apt update && sudo apt install -y python3.10-venv unzip htop

# 2. Создание и настройка .venv
echo "[2/5] Setting up virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install -U pip wheel

# 3. Установка PyTorch XLA (специфично для TPU)
echo "[3/5] Installing PyTorch and XLA..."
pip install torch~=2.3.0 torchvision torch_xla[tpu]~=2.3.0 -f https://storage.googleapis.com/libtpu-releases/index.html

# 4. Установка зависимостей проекта и Jupyter
echo "[4/5] Installing project dependencies & Jupyter..."
if [ -f "pyproject.toml" ]; then
    pip install -e .
else
    # Фоллбэк если установка через -e невозможна
    pip install transformers datasets accelerate wandb indexed-parquet-dataset
fi
pip install jupyter

# 5. Финальные инструкции
echo "=============================================================================="
echo "[5/5] SETUP COMPLETE!"
echo "=============================================================================="
echo "Чтобы запустить Jupyter для Colab, используй:"
echo ""
echo "source .venv/bin/activate"
echo "jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --NotebookApp.port_retries=0 --no-browser"
echo "=============================================================================="
