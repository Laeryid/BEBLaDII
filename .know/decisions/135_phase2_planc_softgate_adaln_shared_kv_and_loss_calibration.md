<!-- created: 2026-11-15 -->
<!-- status: accepted -->
# ADR 135: Спецификация Фазы 2 Plan C: Soft-Gate AdaLN, Shared KV, Unit-RMS Residual Stream и калибровка лосса

**Status**: accepted  
**Date**: 2026-11-15  
**Git Commit**: `f1a6e6f` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.4.0.dev2)  

## Context and Problem
Для Фазы 2 Plan C (условная латентная диффузия на холсте Gr(8, 128) с замороженной Системой 1) требовалось согласовать масштаб V_t (sigma_data=0.05) с остаточным потоком трансформера O(1), обеспечить ненулевой градиент на 1-м шаге без шумового коллапса AdaLN (уточнение механики после Plan B ADR 064/074), сократить параметры KV без интерференции RoPE и откалибровать веса гибридного лосса по эмпирическим замерам Q1-Q7.

## Decisions Made
1. Unit-RMS Residual Stream: входные тензоры делятся на sigma_data=0.05, выходная поправка в GrassmannianOutputHead умножается на 0.05 * g(t_i), где g(t_i)=min(t_i/0.02, 1.0), с последующим построчным clamp_to_unit_ball.
2. Soft-Gate AdaLN + ConditionEmbedder: проекция Linear(256, 6*1024) инициализируется N(0, 1e-8) (std=1e-4) с нулевыми shift/scale и gate_bias=0.02 (2% вклад блока на старте); пол разнообразия L_adaln = mean_l ReLU(1e-2 - sigma_adaln^(l)) включается с линейным warm-up w_adaln(s)=0.05*min(1, s/500).
3. Shared KV Projection + Split K/V Channel Gating во всех 11 блоках внимания (2 PromptAdapter + 8 SA + 1 CAPromptBlock): разветвление U=kv_proj(X) строго до k_norm и RoPE; K=RoPE(k_norm(U)), V=v_gate*U (без RoPE); в CAPromptBlock центрирование Token-Centered Contrast применяется только к K_p.
4. Формат холста: энкодер кодирует 508 токенов (N_ans=127), sep_concept.pt подставляется на pos 0 после энкодера с t_0=0.0 и сбрасывается перед декодером; VOID-ячейки детектируются в латенте (||V||_F^2 < 0.35 or Sim_Gr > 0.80).
5. Калибровка лосса в float32 снаружи DataParallel/DDP: L_total = L_basis + 0.02*L_Gr + 0.1*L_adj + 0.1*L_bary + w_adaln(s)*L_adaln (tau_adj=0.85, tau_bary=0.50).

## Consequences
Стартовое sigma_adaln ~= 8.15e-5 не взрывает градиенты, плавный warm-up L_adaln предотвращает преждевременное раскачивание модуляций; экономия -11.53M параметров от Shared KV до ~148.50M обучаемого ядра; DataParallel корректно считает L_bary и L_adaln снаружи на полном микро-батче.

## Alternatives Considered & Rejected
1. Zero Init AdaLN (ADR 074) — обнуляет градиенты на старте или задерживает выход из плато; Micro-Noise std=1e-3 — избыточная стартовая дисперсия без warm-up.
2. Вес L_Gr=0.5 — доминирование в 23.7-39.2x над L_basis по значению и градиенту (Q4).
3. Прямое сложение V_t + Delta_X без нормировки 0.05 — затопление полезного сигнала масштабом слоёв O(1).

## Invariants & Rules for AI
- [MUST] Не использовать ADR 105 в supersedes (ADR 105 относится к Plan B RoPE/geodesic noise).
- [MUST] В Shared KV ветка V никогда не проходит через RoPE и центрирование Token-Centered Contrast.
- [MUST] Вычислять всю функцию потерь (включая L_bary и статистику L_adaln) в float32 снаружи DataParallel на полном микро-батче.
- [MUST] Замороженная Система 1 (GrassmannianEncoder, GrassmannianDecoder) остаётся в eval() с requires_grad_(False).
- [MUST NOT] Подавать sep_concept.pt в GrassmannianDecoder при инференсе/валидации — ячейка 0 всегда отбрасывается (V_hat[:, 1:]).

## Scope & Affected Files
- `experiments/phase 2/experiment_plan.md`
- `experiments/phase 2/train_phase2.py`
- `experiments/phase 2/inspect_phase2_generation.py`

## Verification
.venv\Scripts\python.exe "experiments/phase 2/test_phase2_open_questions.py" && .venv\Scripts\python.exe "experiments/phase 2/train_phase2.py" --smoke-test
