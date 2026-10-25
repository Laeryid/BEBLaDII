<!-- created: 2026-10-25 -->
<!-- status: Accepted -->
# ADR 111: Phase 6 Coherence Fix: Uniform/Span Noise, RoPE Coordinate Stabilization, Transition Bigram Loss, and 512 Canvas Synchronization

**Status**: Accepted  
**Date**: 2026-10-25  
**Git Commit**: `a6e8330` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.3.0.dev3)  

## Context and Problem
На чекпоинте step_107000 инструментальный аудит выявил качественный потолок: метрика чистого вклада Cross-Attention на шуме t=1.0 (val_content_gain_CA_tg1.0) за последние 13 000 шагов прибавила лишь +0.0014 (застыв на отметке +0.1047 / cos=0.1437), а на средних шумах (t <= 0.5) вклад CA упал почти до нуля (+0.0024), приводя к генерации несвязного 'мешка тематических слов' вместо грамматически выстроенных предложений.
Анализ выявил 4 коренные причины:
1) По-токенное независимое Beta-сэмплирование шума создавало 'чистые якорные токены' из таргета прямо внутри зашумленного ответа, из-за чего DUS учился предсказывать слова по чистым соседям в обход промпта (Shortcut Learning);
2) Случайный префиксный сдвиг shift_k = randint(0, 11) в 100% батча размывал абсолютную координатную сетку RoPE, усредняя структуру начала предложения в позиционный барицентр и порождая артефакт '|' на 0-м токене;
3) Отсутствовал штраф за нарушение межсловных связей (биграмм) при по-токенном косинусном лоссе;
4) В функции потерь PAD-позиции void_embed (занимающие до 85% холста в 512 токенов) забирали на себя львиную долю градиента;
5) Тестовый холст длиной 64 токена физически не вмещал ответы датасета (93.65% ответов длиннее 64 токенов, медиана 271) и нарушал геометрию внимания Attention Sink относительно обучающего холста 512.

## Decisions Made
1. Реформировать расписание шума в sample_token_noise_levels:
   - 60% батча: Flat Uniform Noise (t_i == t_global по всему холсту), ликвидирующий чистые эталонные якоря и полностью совпадающий с реальным инференсом;
   - 40% батча: Span-Masking Noise (непрерывными блоками по 16 токенов), исключающий по-токенную утечку локальных биграмм.
2. Стабилизировать позиции RoPE и начало ответа:
   - В 90% примеров зафиксировать shift_k = 0 (ответ начинается строго с позиции 0, устраняя артефакт '|' и стабилизируя синтаксическое дерево);
   - Снизить вероятность внутренних mid-voids с 5% до 1.5%.
3. Внедрить Transition / Bigram Cosine Loss (L_trans) в compute_phase6_loss:
   - L_trans = 1 - cos(norm(z_{i+1} - z_i), norm(z_{tgt, i+1} - z_{tgt, i})) по смежным контентным позициям с весовым коэффициентом lambda = 0.25.
4. Ребалансировать вес потерь на VOID-токенах:
   - Ввести множитель token_importance = content_mask + 0.20 * void_mask, снижая вес пустых позиций холста в 5 раз.
5. Синхронизировать скрипт оценки evaluate_phase6_metrics.py и ноутбук train_phase6_tpu_notebook.ipynb с полноразмерным холстом max_len = 512.

## Consequences
Положительные:
1. Устранение Shortcut Learning: 60% плоского шума и 40% блочного маскирования заставляют Cross-Attention и буферные слои DUS извлекать семантику промпта на всех уровнях t, устраняя плато CA_gain на средних шумах.
2. Стабилизация синтаксиса: привязка начала ответа к позиции 0 устраняет размытие позиций в RoPE, предотвращает сваливание в 'мешок слов' и полностью ликвидирует артефакт '|' на 0-м токене.
3. Обучение грамматическим биграммам: Transition Cosine Loss заставляет модель учить согласованные переходы между словами, формируя синтаксически корректные фразы.
4. Фокусировка градиента на контенте: ребалансировка весов снижает вклад пустых PAD/VOID слотов с 85% до управляемых 15-20%, отдавая основной градиентный поток смысловым токенам ответа.
5. Полное совпадение геометрии внимания: выравнивание холста инференса и датасетной валидации с обучающим холстом 512 обеспечивает корректную работу Attention Sink и предсказуемую энтропию внимания.

Компромиссы:
1. Потребуется продолжить обучение модели с чекпоинта step_107000 в новом режиме, чтобы градиенты Transition Loss и Uniform Noise перенастроили веса CA и DUS-буферов.
2. Процедура Span-Masking на CPU добавляет незначительную вычислительную операцию repeat_interleave перед шагом TPU.

## Scope & Affected Files
- `experiments/phase 6/__pycache__/evaluate_phase6_metrics.cpython-313.pyc`
- `experiments/phase 6/evaluate_phase6_metrics.py`
- `experiments/phase 6/tpu kaggle/__pycache__/train_phase6_tpu_notebook.cpython-313.pyc`
- `experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.ipynb`
- `experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.py` (Related KI: `decisions/097_phase6_t_global_orchestrator_and_per_sample_beta_noise.md`)
- `experiments/phase 6/phase6_generation_demo_107000.html`
- `ideas/CONVERGENCE OF MUON WITH NEWTON/342/200/223SCHULZ.pdf`
- `ideas/Capturing In-Context Learning Dynamics with Task Operators.pdf`
- `ideas/Capturing In-Context Learning Dynamics withTask Operators.html`
- `ideas/Correct-Token Retention in Uniform-State Diffusion Language Models.pdf`
- `ideas/Vector-Quantized Geodesics as.pdf`
- `ideas//320/230/320/264/320/265/321/217 /320/264/320/273/321/217 /320/273/320/260/321/202/320/265/320/275/321/202/320/275/320/276/320/263/320/276 /320/277/321/200/320/276/321/201/321/202/321/200/320/260/320/275/321/201/321/202/320/262/320/260 /320/277/321/200/320/270 /320/277/320/265/321/200/320/265/320/267/320/260/320/277/321/203/321/201/320/272/320/265.pdf`
