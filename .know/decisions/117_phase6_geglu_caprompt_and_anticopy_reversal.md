<!-- created: 2026-11-02 -->
<!-- status: Accepted -->
# ADR 117: Phase 6 Non-Linear CAPromptLayer (GeGLU + AdaLN), Prompt Copy Penalty Reversal, and GPU-TPU Synchronization

**Status**: Accepted  
**Date**: 2026-11-02  
**Git Commit**: `29d602a` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.3.2.dev2)  

## Context and Problem
Анализ чекпоинта step_110000 (после отката к 107000 с добавлением Prompt Anti-Copy Penalty из ADR 116) выявил рассогласование: метрика DProx просела до 0.670, а слепой судья и скоринг score_generations.py показали, что декодер ModernLatentDecoder выдавал несвязный текст, в то время как LatentDict NN дословно воспроизводил слова промпта.

Аудит архитектуры выявил фундаментальное ограничение: субблоки Cross-Attention и Self-Attention внутри CAPromptLayer являлись строго линейными операторами (V -> out_proj без какой-либо функции активации). Вся нелинейная обработка перекладывалась на последующие слои DUS. На высоких уровнях диффузионного шума (t -> 1.0) холст не содержит связной информации, и линейный адаптер физически не способен выполнить синтез новых логических сущностей из промпта. Введение же адверсариального лосса prompt_copy_penalty создавало градиентное отталкивание от промпта, приводя к наклону векторов и падению DProx, но не давало модели инструмента для нелинейной генерации ответа.

## Decisions Made
1. Внедрить в CAPromptLayer нелинейные субблоки GeGLUMLP по архитектурному образцу ModernBertMLP из DUS:
   - Wi: 1024 -> 4096 (Xavier uniform), нелинейность GeGLU (F.gelu(u) * g), Wo: 2048 -> 1024 (Zero-init).
   - Субблок mlp_ca (с RMSNorm и AdaLNModulation по времени t) устанавливается после Cross-Attention.
   - Субблок mlp_sa (с RMSNorm и AdaLNModulation по времени t) устанавливается после Self-Attention над объединенным сигналом delta = ca_out + sa_out.
   - Zero-init матриц Wo гарантирует строго тождественный нулевой вклад новых нелинейных операторов на старте со step_107000 (mlp(A) == 0.0), исключая разрушение латентного пространства.
2. Полностью удалить Prompt Anti-Copy Penalty (prompt_copy_weight и prompt_copy_margin) из лосса compute_phase6_loss и Config.
3. Ввести sa_adapter_version: 3:
   - При Resume с sa_adapter_version < 3 (чекпоинт 107k v2) разрешать отсутствие весов mlp_ca и mlp_sa в load_info.missing_keys.
   - При sa_adapter_version >= 3 (новые чекпоинты GPU/TPU) требовать 100% совпадения всех обучаемых весов и EMA-теней.
4. Синхронизировать GPU и TPU скрипты обучения:
   - Выравнивание high_t_sampling = True и prompt_dropout = 0.15 на GPU.
   - Чанкованный расчет RawDProx в bfloat16 на GPU (экономия ~2.8 ГБ VRAM).
   - Исправление off-by-one ошибки global_step при Resume на GPU.
   - Устранение ложного подавления architectural warmup_factor до 0.01 при сессионном прогреве оптимизатора на TPU.
   - Добавление мониторинга амплитуд mlp_ca.Wo и mlp_sa.Wo в WandB.
5. Расширить score_generations.py извлечением финального DProx и метрики совпадения промпта для LatentDict NN (nn_prompt_share).

## Consequences
Положительные:
1. CAPromptLayer получает полноценную нелинейную выразительность: GeGLU позволяет синтезировать концептуально новые семантические векторы ответа из промпта и холста вместо чисто линейного проецирования.
2. Нулевая инициализация Wo гарантирует точное сохранение геометрии чекпоинта 107k на 1-м шаге без стресса для модели.
3. Устранение prompt_copy_penalty снимает искусственный адверсариальный разрыв между скрытым состоянием DUS и декодером, восстанавливая естественный DProx.
4. Полная бинарная и весовая совместимость чекпоинтов: обучение можно свободно чередовать между локальным GPU и кластером TPU.
5. Экономия ~2.8 ГБ VRAM на GPU за счет чанкования RawDProx в bfloat16.

Компромиссы:
1. Увеличение числа обучаемых параметров в каждом CAPromptLayer на 8.39M параметров (суммарно +25.17M параметров на 3 слоя: 12, 24, 36).
2. Замедление шага обучения на TPU/GPU не более чем на 3-4% за счет двух дополнительных GeGLU блоков.

## Alternatives Considered & Rejected
- Сохранение линейного CAPromptLayer с увеличением веса prompt_copy_penalty: приводит к полному коллапсу DProx и деградации генерации в шум.
- Внедрение однослойного MLP (Linear + GELU) без гейтинга: уступает GeGLU по плотности представления признаков и не соответствует единому архитектурному стилю ModernBERT / DUS.
- Внедрение GeGLU без временной модуляции AdaLN: лишает нелинейный оператор адаптивности к уровню диффузионного шума t.

## Invariants & Rules for AI
- [MUST] Выходные проекции Wo в GeGLUMLP обязаны инициализироваться строго нулями (zero-init), обеспечивая тождественный старт без сдвига распределения.
- [MUST] Параметры mlp_ca.Wo и mlp_sa.Wo должны направляться в группу ca_out_params (lr = 2e-4, weight_decay = 0.0), а входные матрицы Wi, RMSNorm и AdaLNModulation — в ca_inner_params (lr = 1e-4, weight_decay = 0.0).
- [MUST] Буфер latent_dict на GPU должен храниться в bfloat16 с чанкованием по 16 блокам для предотвращения OOM.
- [MUST NOT] Не использовать искусственные адверсариальные штрафы за близость к промпту (prompt_copy_penalty), нарушающие сферическую топологию латентного пространства.

## Scope & Affected Files
- `experiments/phase 6/__pycache__/evaluate_phase6_metrics.cpython-313.pyc`
- `experiments/phase 6/__pycache__/score_generations.cpython-313.pyc`
- `experiments/phase 6/evaluate_phase6_metrics.py`
- `experiments/phase 6/kaggle/__pycache__/train_phase6_notebook.cpython-313.pyc`
- `experiments/phase 6/kaggle/train_phase6_notebook.ipynb`
- `experiments/phase 6/kaggle/train_phase6_notebook.py` (Related KI: `decisions/091_global_scheduled_identity_gate.md`, `decisions/096_phase6_gradient_collapse_and_dynamic_void_penalty.md`, `decisions/097_phase6_t_global_orchestrator_and_per_sample_beta_noise.md`)
- `experiments/phase 6/phase6_generation_demo.html`
- `experiments/phase 6/score_generations.py`
- `experiments/phase 6/tpu kaggle/__pycache__/train_phase6_tpu_notebook.cpython-313.pyc`
- `experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.ipynb`
- `experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.py` (Related KI: `decisions/097_phase6_t_global_orchestrator_and_per_sample_beta_noise.md`)
- `experiments/phase 6/blind_judge_key_110k.json`
- `experiments/phase 6/demo_110k_baseline.html`
- `experiments/phase 6/demo_110k_smartcrop_hard_buffered16.html`
- `experiments/phase 6/demo_110k_smartcrop_hard_v2.html`

## Verification
Запуск автономного набора тестов scratch/test_caprompt_compat.py:
1. Проверка прямого прохода CAPromptLayer с новыми GeGLU блоками (B=2, T_a=64, D=1024) -> shape torch.Size([2, 64, 1024]).
2. Проверка zero-init Wo: mlp(A) == 0.0.
3. Проверка загрузки чекпоинта 107k (v2): ровно 10 недостающих ключей GeGLU, 0 неожиданных ключей.
4. Проверка полного цикла обучения на GPU (1 шаг оптимизатора, Wo != 0), сохранение v3 и строгая загрузка на TPU (v3): 0 missing keys, 100% совпадение весов.
5. Проверка синтаксиса через py_compile для всех затронутых скриптов.
