<!-- created: 2026-10-28 -->
<!-- status: Superseded by ADR 116 -->
# ADR 115: Синтез современных методов переменной длины и теории Exposure Bias для усиления Phase 6 (SmartCrop, dLLM-Var, LR-DLLM, Epsilon Scaling)

> [!WARNING]
> **This ADR is superseded by [ADR 116: Phase 6 Coherence Restoration, Boundary Void Weighting, and Adaptive Inference Cropping](./116_phase6_coherence_boundary_adaptive_crop.md).**

**Status**: Superseded by [ADR 116](./116_phase6_coherence_boundary_adaptive_crop.md)
**Date**: 2026-10-28  
**Git Commit**: `82bfdeb` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.3.0.dev3)  

## Context and Problem
В ходе разработки Phase 6 (генерация завершенных ответов переменной длины) и анализа причин копирования вопроса в хвост холста на шагах 107k-115k (ADR 114) проведен глубокий анализ четырех ключевых препринтов из папки ideas/:
1. 'ideas/Diffusion Language Models Are Natively Length-Aware.pdf' (Rossi et al., Mar 2026) — доказательство того, что DLM на шаге t=1.0 уже кодируют распределение длины ответа; показано, что избыточный паддинг ухудшает качество генерации и провоцирует галлюцинации в хвосте.
2. 'ideas/Diffusion LLM with Native Variable Generation Lengths.pdf' (Yang et al., Oct 2025, dLLM-Var) — анализ дуального коллапса EOS (неспособность остановиться vs преждевременное схлопывание всего ответа в EOS) и решение через детерминированное зашумление терминаторов границы ответа.
3. 'ideas/Elucidating the Exposure Bias in Diffusion Models.pdf' (Ning et al., ICLR 2024) — аналитическое доказательство расширения дисперсии распределения при сэмплинге (Var(x_hat_t) > Var(x_t)) из-за ошибки предсказания и метод устранения дрейфа через Input Perturbation на обучении и Epsilon Scaling на инференсе.
4. 'ideas/Improving Variable-Length Generation in Diffusion Language Models via Length Regularization.pdf' (Cheng et al., Feb 2026, LR-DLLM) — статистико-физическое обоснование логарифмического роста энтропии токенов (S ~ a*log(L) + b) по мере удаления от якоря промпта, объясняющее скатывание модели к тривиальным лексическим повторам на длинных пустых участках.

Сопоставление с кодовой базой выявило ключевой дефект расчета t_reported = (1.0 - RawDProx): близость считалась только к latent_dict (словарю слов). Для чистого void_embed сходство со словами низкое (~0.35), из-за чего t_reported в хвосте никогда не опускался ниже 0.65, удерживая AdaLN в режиме высокого шума.

## Decisions Made
Принять синтетическую стратегию поэтапной интеграции находок из статей в Phase 6:
1. Инспекция и исправление расчета RawDProx (Вариант 3): включить void_embed в вычисление максимального сходства (torch.maximum(sims_dict, sim_void)), гарантируя t_reported -> 0 при схождении хвоста к состоянию покоя.
2. Training-Free SmartCrop на инференсе (Вариант 1, Rossi et al., 2026): реализовать в evaluate_phase6_metrics.py расчет функции выживания F(l) = 1 - Prod(1 - phi_j) на шаге t=1.0 и поддержать режимы 'hard' (физическая обрезка холста до L_hat) и 'soft' (якорение позиций j >= L_hat в void_embed).
3. Граничное зашумление и вес границы (Yang et al., 2025, dLLM-Var): зафиксировать в fixes_plan.md в качестве будущей Итерации 4 правило зашумления зоны границы ответа (t >= 0.75) и повышенного веса лосса (1.20) для первых 16 токенов пустоты.
4. Сферический Epsilon Scaling (Ning et al., 2024): поддержать масштабирование шага Slerp (Delta_theta / lambda_t, lambda_t in [1.02, 1.05]) для подавления дрейфа дисперсии ошибки при денойзинге.
5. Изоляция тестирования: провести замеры влияния каждого инференс-улучшения в отдельности на фиксированном промпте из обучающего датасета.

## Consequences
1. Обеспечена строгая изоляция улучшений: каждое расширение валидируется независимо через отдельный CLI-флаг скрипта оценки. 2. Устраняется критическая слепая зона модели: токенный AdaLN теперь может видеть низкое расчетное время t_reported -> 0 на позициях <|void|>, когда z сходится к void_embed. 3. SmartCrop позволяет исключить паразитное рассеивание внимания на неинформативный хвост холста, снижая задержку генерации и устраняя наведенный Exposure Bias. 4. Обучающие модификации (высокий шум и вес границы) четко спланированы в fixes_plan.md без преждевременного усложнения запущенной Итерации 1.

## Results of Inference Comparison (2026-10-28)
Условия: чекпоинт `step_107000.pth`, --max_len 500, --steps 25 (сетка шума обучения dt=0.04, иное число шагов дает OOD для AdaLN), sampler hybrid, seed 42, один промпт из train_phase6.parquet: 'How does your generation mechanism contrast with traditional next-token prediction models?'. Отчеты: `experiments/phase 6/demo_baseline_len500_s25.html`, `demo_void_dprox.html`, `demo_smartcrop_soft.html`, `demo_smartcrop_hard_v2.html`, `demo_smartcrop_hard_buffered16.html`. Метрики: `experiments/phase 6/score_generations.py` -> `score_generations.json`; слепая оценка: `blind_judge.md`, `blind_judge_key.json`.

| Режим | Холст | Время CPU | Слов (не-void) | PPL GPT-2 | distinct-2 | distinct-3 | Слепая оценка (1-5) |
|---|---|---|---|---|---|---|---|
| baseline | 500 | 6.5 мин | 384 | 31.2 | 0.405 | 0.686 | 1 |
| void_dprox (Вариант 3) | 500 | 6.5 мин | 379 | 32.6 | 0.402 | 0.698 | 1 |
| soft SmartCrop (1B) | 500 | 6.5 мин | 222 | 67.6 | 0.633 | 0.895 | 2.5 |
| hard SmartCrop v2 (срез по L_hat=294) | 294 | 4.0 мин | 218 | 58.2 | 0.590 | 0.810 | 1.5 |
| hard SmartCrop + 16 void buffer (1A) | 306 | 4.5 мин | 218 | 55.4 | 0.599 | 0.819 | 2.5 |

Выводы:
1. Без обрезки хвоста baseline зацикливается (эхо промпта, distinct-2 ~0.40); любой вариант SmartCrop поднимает distinct-2 до 0.59-0.63 и дает более тематичный текст. Теория Rossi et al. (длина кодируется уже при t=1.0) подтверждена: детектор находит ~290 токенов.
2. Void-Aware DProx (Вариант 3) исправляет только метрику DProx, но не генерацию (void_dprox ~ baseline).
3. PPL GPT-2 как метрика связности вводит в заблуждение: у зацикленного baseline PPL минимальна (повторы предсказуемы). Читать только совместно с distinct-n.
4. Soft и hard+buffer16 неразличимы в пределах шума; hard+buffer16 на ~35% быстрее. Жесткий срез без буфера (hard v2) обрывает фразу, буфер из 16 void устраняет эффект.
5. Ограничение: один промпт, один seed, шумный чекпоинт. Вывод про превосходство SmartCrop над baseline устойчив, ранжирование soft vs hard+buffer16 не установлено.
6. Рекомендованный режим инференса по умолчанию для дальнейших тестов: `--smartcrop hard` с буфером 16 void.

## Alternatives Considered & Rejected
- Одновременное включение всех теоретических улучшений в обучение TPU: отклонено во избежание запутывания эффектов и срыва текущего обучения Итерации 1.
- Полный отказ от генерации токенов пустоты в пользу динамической остановки без void_embed: отклонено, так как сферическое латентное пространство и RoPE-архитектура DUS опираются на непрерывное векторное состояние холста.

## Invariants & Rules for AI
- [MUST] Расчет t_reported обязан учитывать как словарные векторы (latent_dict), так и системные токены (void_embed), чтобы исключить смещение времени при инференсе и обучении.
- [MUST] Режимы SmartCrop (hard/soft) и Epsilon Scaling на инференсе должны активироваться строго через явные CLI-аргументы, сохраняя стандартный режим по умолчанию нетронутым.
- [MUST] Все новые гипотезы для скрипта обучения TPU должны предварительно фиксироваться в fixes_plan.md без нарушения текущей Итерации 1.

## Scope & Affected Files
- `experiments/phase 6/fixes_plan.md`
- `experiments/phase 6/evaluate_phase6_metrics.py`
- `experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.py` (Related KI: `decisions/097_phase6_t_global_orchestrator_and_per_sample_beta_noise.md`)

## Verification
.venv\Scripts\python.exe -m py_compile "experiments/phase 6/evaluate_phase6_metrics.py"
.venv\Scripts\python.exe "experiments/phase 6/evaluate_phase6_metrics.py" --check-files
