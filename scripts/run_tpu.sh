#!/bin/bash
# Скрипт для запуска дистилляции Phase 1 на TPU v6e-4 (4 ядра)

echo "--- [LAUNCHER] Запуск Phase 1 на 4 ядрах TPU v6e ---"
torchrun --nproc_per_node=4 experiments/train_phase1_TPU_fsdp.py
