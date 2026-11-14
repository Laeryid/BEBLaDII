<!-- created: 2026-11-14 -->
<!-- status: Accepted -->
# ADR 133: Калибровка OrthMuon, LR warmup и усиление Uniformity Loss для чистого перезапуска Phase 1 Plan C с шага 0

**Status**: Accepted  
**Date**: 2026-11-14  
**Git Commit**: `e939615` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.3.3.dev3)  

## Context and Problem
После внедрения OrthMuon на сфере (SphereLinear), нового Uniformity Loss и корректировки конфигурации Phase 1 Plan C наблюдалась просадка при продолжении со старых чекпоинтов и избыточный стартовый сдвиг весов embeddings.weight на шаге 0 (плотные градиенты привязанной LM head по всем 131,072 токенам словаря давали gnorm~665..668 при 9.73% относительном сдвиге эмбеддингов за первые 5 шагов). Кроме того, масштаб шага OrthMuon (max(1, m/n)^0.5) при muon_lr=0.02 не был согласован с Algorithm 2 (0.2*sqrt(max(m, n))), а разброс max_steps между CLI (5000) и TRAINING_CONFIG (500000) искажал профиль косинусного спада LR. Принято решение перезапустить обучение Phase 1 Plan C с 0 шага на Kaggle с откалиброванным оптимизатором и функцией потерь.

## Decisions Made
1. В OrthMuon.step масштабирование шага переведено на канонический Algorithm 2: scale = 0.2 * math.sqrt(max(p.size(0), p.size(1))).
2. Откалиброван LR schedule: muon_lr снижен с 0.02 до 0.002 (muon_min_lr: 0.008 -> 1e-4), adamw_min_lr снижен с 1e-4 до 3e-5 (adamw_lr=3e-4 сохранен), max_steps унифицирован до 100000 в train(), parse_args() и TRAINING_CONFIG.
3. В get_cosine_lr() добавлен линейный LR warmup (warmup_steps=1000 в train/parse_args/TRAINING_CONFIG, warmup_start_factor=0.05) перед косинусным спадом на отрезке [warmup_steps, max_steps].
4. Вес Uniformity Loss (eta / uniformity_weight) увеличен с 0.01 до 0.04 по умолчанию и явно проброшен в compute_phase1_loss, evaluate и train.
5. В режиме --smoke_test в main() установлен args.lr_warmup_steps = 1; все изменения экспортированы в experiments/phase 1/Kaggle/train_phase1_notebook.ipynb (22 ячейки).
6. gcs_dir в TRAINING_CONFIG оставлен пустым (''), чтобы использовать стандартный путь при условии ручной архивации старых чекпоинтов пользователем.

## Consequences
Положительные: устранение стартового ударного сдвига embeddings.weight и стабильная калибровка обновлений OrthMuon под полный цикл обучения 100k шагов с нуля на Kaggle (с усиленной регуляризацией изотропности V_cell). Компромиссы: более осторожный темп изменений в первые 1000 шагов (разогрев от 5% до 100% LR); для чистого старта с шага 0 старые чекпоинты в GCS (gs://bebladii-weigths-us/planC/Phase 1/checkpoints/) должны быть заархивированы вручную перед запуском при сохранении resume_ckpt='auto'.

## Alternatives Considered & Rejected
1. Возобновление обучения со старого чекпоинта step_28500 с добавлением только LR warmup (отклонено ради достижения более высокого потолка качества латентного пространства при чистом старте с шага 0 с новой конфигурацией потерь и OrthMuon).
2. Изменение gcs_dir на отдельный подкаталог checkpoints_v2 (отклонено пользователем: старая папка GCS будет заархивирована вручную).

## Invariants & Rules for AI
- MUST сохранять согласованность значений по умолчанию между TRAINING_CONFIG, parse_args() и сигнатурами train()/evaluate()/create_hybrid_optimizers() в train_phase1.py.
- MUST после любых правок train_phase1.py перегенерировать Kaggle/train_phase1_notebook.ipynb через export_to_notebook.py.
- MUST использовать scale = 0.2 * math.sqrt(max(p.size(0), p.size(1))) при базовом muon_lr ~ 0.002 для OrthMuon и сохранять линейный warmup LR при старте с шага 0.

## Scope & Affected Files
- `experiments/phase 1/train_phase1.py`
- `experiments/phase 1/Kaggle/train_phase1_notebook.ipynb`

## Verification
1. Syntax check (.venv py_compile): exit code 0.
2. CPU-симуляция шагов 0..4 (val.parquet, batch=4, seq_len=128): относительный сдвиг embeddings.weight за 5 шагов снизился с 9.73% до 0.50% (loss 11.818 -> 11.716, frob 1.621 -> 1.619).
3. Полный --smoke_test на CPU (--no_gcs --wandb disabled --resume_ckpt=""): шаги 1..3, две валидации (шаги 2 и 3) и сохранение чекпоинта завершены с exit code 0.
4. Перегенерация Kaggle/train_phase1_notebook.ipynb (22 ячейки): exit code 0.
