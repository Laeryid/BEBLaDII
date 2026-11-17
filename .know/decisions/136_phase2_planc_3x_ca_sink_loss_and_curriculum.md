<!-- created: 2026-11-17 -->
<!-- status: accepted -->
# ADR 136: Фаза 2 Plan C: 10-слойное ядро с 3x CA_Prompt, штраф L_sink, декомпозиция Грассманова лосса и полу-AR curriculum

**Status**: accepted  
**Date**: 2026-11-17  
**Git Commit**: `aed0da5` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.4.0.dev2)  

## Context and Problem
На чекпоинтах до step_24500 (8-слойное ядро с единственным блоком CAPromptBlock на слое 4) даже после сдвига расписания шума к высоким t (коммит aed0da5) наблюдался срыв условной генерации: sink_content оставался на уровне 0.63-0.88 (контентные ячейки уходили в нулевой k_sink), CFG_contrast падал до 0.0227 (hi=0.1963), при t=1.0 энергия предсказания ||V_0_hat||_F^2 сжималась до 0.35 (против 2.52 у чистых данных) из-за минимума ненормированного ||G_h - G_0||_F^2 и SmoothL1 при низкой определённости, а 65% батча содержали чистые/слабозашумлённые ячейки в V_t, позволяя Self-Attention обходить промпт. В то же время замороженный GrassmannianDecoder (Система 1) при гауссовом возмущении std=0.030 (Sim_Gr=0.8176) сохранял точность восстановления токенов tok_acc=0.9716. В качестве дополнения к ADR 135 потребовалась перестройка стека CA, регуляризации sink, функции потерь, ConditionEmbedder, расписания шума и сэмплеров.

## Decisions Made
1. 10-слойный стек ядра GrassmannianDiffusionCore (167.89M параметров; дополнение к ADR 135): 0:Full, 1:Local(r=8), 2:CA_Prompt, 3:Full, 4:CA_Prompt, 5:Full, 6:CA_Prompt, 7:Full, 8:Local(r=8), 9:Full. Единый общий 2-слойный PromptAdapter (23.60M) кэшируется на промпте, при этом каждый из 3 блоков CAPromptBlock имеет независимые веса.
2. Дифференцируемый контроль sink и однопроходное внимание в CAPromptBlock: softmax по [K_p || k_sink] считается однократно в float32 (исключен дублирующий вызов SDPA, контекст вычисляется как probs[..., :N_p] @ v_p с учетом v_sink=0). Тензор sink_weights (B, 3, 128) передаётся в compute_phase2_loss для явного штрафа sink_weight=0.10 в коридоре [0.08, 0.35] строго по активным контентным ячейкам (1 <= i <= i_last, t_i > 1e-3) при m_prompt == 1 (VOID и CFG-uncond не штрафуются).
3. Декомпозиция Грассманова лосса в compute_phase2_loss: L_Gr (weight=0.08) переведён на масштабно-инвариантный угловой лосс (1 - Sim_Gr), не сжимающий норму при высоком шуме, и дополнен явным L_energy (weight=0.05) по SmoothL1(Tr(G_h), Tr(G_0)) и 0.5*SmoothL1(||G_h||_F, ||G_0||_F).
4. Перебалансированный Mixed Noise Curriculum + ConditionEmbedder: 30% High-Noise/Pure-Start (1/3 ровно t=1.0, 2/3 U(0.80, 1.0)), 35% Shifted Flat Sigmoid(N(0.75, 1.0)), 20% Causal Prefix / Block-AR (префикс 1..cut при t=0 с учётом i_last, хвост >= 0.30/0.55), 15% Rolling Wave [0.0..0.25] -> [0.75..1.0]. В ConditionEmbedder t_bar считается строго по активным зашумленным ячейкам (t_i > 1e-3), а доля чистого префикса передаётся в датчик s6.
5. Сэмплеры с CFG Rescale и полу-AR режимы: добавлен _apply_cfg_and_clamp (cfg_rescale=0.7) в euler_heun_sample, rolling_wave_sample (window=56.0, num_steps=40, early-stop по >=8 VOID подряд) и block_ar_sample (2-й порядок Эйлера-Хьюна внутри блока).

## Consequences
Положительные: трёхкратное усиление пропускной способности кросс-внимания на промпт (3 блока CA_Prompt) и прямой градиентный контроль k_sink устраняют коллапс в word-salad и возвращают CFG-контраст; декомпозированный угловой лосс + L_energy предотвращают радиальное сжатие нормы предсказания при t -> 1; однопроходный расчёт внимания в float32 в CAPromptBlock исключает повторный вызов SDPA и FP16 underflow. Компромиссы: число обучаемых параметров ядра увеличилось с ~148.50M до 167.89M; добавление Causal Prefix и Rolling Wave в curriculum снижает долю полностью однородного денойзинга в батче, но открывает полу-AR генерацию.

## Alternatives Considered & Rejected
1. Донастройка только расписания шума (aed0da5) при одном CAPromptBlock на слое 4 — не смогла преодолеть порог CFG=0.0227 и доминирование k_sink (0.63-0.88).
2. Использование ненормированного ||G_h - G_0||_F^2 в качестве единственного Грассманова лосса — математически смещает минимум масштаба ||V_0_hat||_F^2 вниз при малом Sim_Gr на высоком шуме.
3. Rolling Wave с window=32 и 24 шагами — давал всего ~4.8 шага денойзинга на ячейку при 1-м порядке Эйлера и без early-stop.

## Invariants & Rules for AI
- [MUST] В CAPromptBlock сохранять однократный расчёт вероятностей внимания в float32 по [K_p || k_sink] без повторного вызова SDPA и учитывать нулевой вклад v_sink == 0 при умножении probs[..., :N_p] @ v_p.
- [MUST] Применять штраф L_sink строго к зашумленным контентным ячейкам (1 <= i <= i_last, t_i > 1e-3) при m_prompt > 0.5; VOID-хвост и unconditional-ветка (m_prompt == 0) не должны штрафоваться за уход в k_sink.
- [MUST] В ConditionEmbedder вычислять глобальный t_bar только по активным зашумленным ячейкам ответа (t_i > 1e-3), чтобы чистый префикс в Block-AR/Rolling Wave не занижал глобальный уровень шума.
- [MUST] Синхронизировать любые изменения train_phase2.py в Kaggle/train_phase2_notebook.ipynb через export_to_notebook.py и перепроверять 10-слойный стек через test_phase2_architecture.py.
- [MUST NOT] Сжимать L_Gr до ненормированной разности ||G_h - G_0||_F^2 без отдельного углового и энергетического термов (это вызывает радиальную просадку ||V_0_hat||_F^2 к порогу VOID_ENERGY_THR=0.35 на старте t=1.0).

## Scope & Affected Files
- `experiments/phase 2/train_phase2.py`
- `experiments/phase 2/test_phase2_architecture.py`
- `experiments/phase 2/inspect_phase2_generation.py`
- `experiments/phase 2/experiment_plan.md`
- `experiments/phase 2/Kaggle/train_phase2_notebook.ipynb`

## Verification
.venv\Scripts\python.exe -B "experiments/phase 2/test_phase2_architecture.py" && .venv\Scripts\python.exe -B "experiments/phase 2/train_phase2.py" --smoke-test --device cpu
