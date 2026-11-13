<!-- created: 2026-11-13 -->
<!-- status: Accepted -->
# ADR 131: Переход на сферическую оптимизацию весов OrthMuon (d=1) и QK-norm в трансформере (по статье Hu et al., 2026)

**Status**: Accepted  
**Date**: 2026-11-13  
**Git Commit**: `8932724` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.3.3.dev3)  
**Reference**: Kai Hu, Li Zhang, Weichen Yu, Matt Fredrikson, Haoqi Hu. *Do Orthogonality Constraints Help Language-Model Training Beyond Sphere Optimization?* (CMU, arXiv:2611.00593v1, Oct 2026) — [PDF](../../ideas/Do%20Orthogonality%20Constraints%20Help%20Language-Model%20Training%20Beyond%20Sphere%20Optimization.pdf)

## Context and Problem
В текущей реализации Phase 1 Plan C (`experiments/phase 1/train_phase1.py`) 2D-веса трансформера (`ModernBERTLayer` и декодер) обучаются гибридом `Muon + AdamW`, причём в `ModernBERTLayer` отсутствует нормализация Query и Key (`q_norm`/`k_norm` установлены только в `SetCrossAttentionUnpacker`). В ADR 127 была выявлена проблема неконтролируемого роста норм 2D-весов в `Muon` (с 64 до 122+), приводящая к переполнению в SwiGLU.

Анализ статьи Hu et al. (2026) показал следующие факты, применимые немедленно:
1. **Сферическая оптимизация (`OrthMuon d=1`) стабильно превосходит тюнингованный `Muon`**: при представлении матрицы $W = \operatorname{diag}(a) Q$, где каждая строка $q_i$ матрицы $Q$ лежит на единичной сфере $\|q_i\|_2 = 1$, а вектор масштабов $a$ обучается `AdamW`, validation loss в LLM (360M–3.1B, C4 и FPO-mix) улучшается на **12.0–17.7 milli-loss**, а в диффузионном трансформере `SiT-B/2` (`DiT+AdaLN`) FID-50K снижается с `21.55` до `20.81` (при ~98% пропускной способности `Muon`).
2. **Около половины выигрыша даёт единичная сфера на строках, вторую половину — обучаемые per-row масштабы $a$** (особенно при повышенном LR на $a$, в ~4× выше базового `AdamW` LR). При этом простая нормализация строк в `forward()` (`WeightNorm` / Forward-normalized Muon) даёт меньше половины эффекта; обновлять $Q$ нужно в сферических координатах (касательный шаг + ретракция/прямая ренормализация строки).
3. **Обязательность `QK-norm`:** в диффузионной модели `SiT-B/2` без `QK-norm` как `AdamW`, так и `OrthMuon d=1` / `OrthAdam d=1` периодически расходятся на позднем этапе затухания LR из-за роста произведения $W_q^{h\top} W_k^h$. Добавление per-head RMSNorm на `Q` и `K` полностью устраняет эти расходимости (`0/41` падений).
4. **Отрицательный результат по Штифелевой ортогональности весов (`d > 1`):** блочная ортогональность строк (`d = 8, 32, ...`), `Manifold Muon` и `Projected DGram` не улучшают качество сверх `d=1`, а при `d >= 32` стабильно его ухудшают.

## Decisions Made
1. **Заменить обычный `Muon` для 2D-весов внимания и MLP (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`) на сферическую оптимизацию `OrthMuon d=1`:**
   - Параметризовать каждый такой линейный слой через строчно-нормированный фактор $Q$ ($\|q_i\|_2 = 1$) и обучаемый вектор per-row масштабов $a$ ($W = a[:, \text{None}] \cdot Q$).
   - Обновлять $Q$ в оптимизаторе сферическим шагом: вычисление направления `Muon` $Z$ (Nesterov + 5 итераций Ньютона–Шульца), касательная проекция по строкам $Z_\perp = Z - \langle Z, Q \rangle_{\text{row}} Q$, шаг $A = Q - \eta s Z_\perp$ и прямая ренормализация строк $Q^+ = \operatorname{norm\_rows}(A)$ в `float32`, с **`weight_decay = 0.0`** на $Q$.
   - Обновлять per-row вектор масштабов $a$ оптимизатором `AdamW` с **`weight_decay = 0.0`** и повышенным learning rate (~4× от базового `AdamW` LR согласно Appendix R статьи).
   - Эмбеддинги, 1D-параметры, `RMSNorm`, `vae_*` и вспомогательные головы остаются в `AdamW`.
2. **Внедрить обязательный per-head `QK-norm` (`RMSNorm` по `head_dim` на `q` и `k` с обучаемым scale и `weight_decay = 0.0`)** в `ModernBERTLayer` (`experiments/phase 1/train_phase1.py` и синхронном Kaggle-ноутбуке) и во всех последующих блоках внимания Plan C.
3. **Не внедрять ортогонализацию между строками весов (`d > 1`, `Manifold Muon`, `Projected DGram`)**, сохраняя при этом ортонормированность и диагональную регуляризацию Грама в пространстве латентных активаций-клеток $\operatorname{Gr}(8, 128)$ (ADR 128).

## Consequences
Положительные:
- Более быстрая и качественная сходимость по сравнению с обычным `Muon` при минимальном оверхеде (<2%, т.к. при `d=1` не требуются $m\times m$ полиномиальные ретракции).
- Структурное устранение неконтролируемого роста норм 2D-весов (причина переполнения из ADR 127): норма направлений строк жёстко равна 1, а масштаб явно вынесен в вектор $a$.
- Защита от поздних расходимостей и взрыва attention logits за счёт `QK-norm`.

Компромиссы и риски:
- При возобновлении со старых чекпоинтов требуется однократная конвертация плотных матриц $W \to (a, Q)$ по строчным нормам, а при экспорте или переходе к SFT веса следует смержить обратно в $W = \operatorname{diag}(a) Q$.
- Необходимо синхронизировать устройства состояний новых параметров `(Q, a)` при resume согласно ADR 124.

## Alternatives Considered & Rejected
- **WeightNorm / Forward-normalized Muon** (нормализация строк $W$ только в `forward()`): отклонено, так как в статье (Section 4.5, Table 4) восстанавливает менее половины выигрыша `OrthMuon d=1`.
- **Stiefel (`d > 1`), `Manifold Muon`, `Projected DGram` на матрицах весов**: отклонено, так как ухудшают validation loss относительно `d=1`.
- **Q/K SPD-факторы (`W = S Q`)**: отклонено, так как на масштабах эффект исчезает, а при `d=128` провоцирует взрыв attention logits.

## Invariants & Rules for AI
- [MUST] Обновлять направления строк $Q$ касательным шагом и сферической ренормализацией внутри оптимизатора (в `float32`), а не через дифференцируемую нормировку весов в `forward()`.
- [MUST] Выставлять `weight_decay = 0.0` для сферических факторов $Q$, per-row масштабов $a$ и весов `QK-norm`.
- [MUST] Использовать per-head `QK-norm` в слоях внимания трансформера при обучении с `OrthMuon d=1` и в будущей диффузионной фазе System 2.
- [MUST] Хранить мастер-веса и состояния оптимизаторов (`AdamW` и `OrthMuon`) в `float32` при `bfloat16` `autocast`.
- [MUST NOT] Не добавлять Штифелевы ограничения (`d > 1`) или `Manifold Muon` на матрицы весов трансформера.
- [MUST NOT] Не путать отказ от ортогональности строк весов с геометрией латентных клеток $\operatorname{Gr}(8, 128)$ из ADR 128.

## Verification
1. `.venv\Scripts\python.exe -m py_compile "experiments\phase 1\train_phase1.py"`
2. Проверка инвариантов шага оптимизатора: после шага $\|q_i\|_2 = 1 \pm 10^{-6}$, `weight_decay = 0` на $(Q, a, \text{q\_norm}, \text{k\_norm})$, корректная работа `resume` на multi-GPU (ADR 124).
3. Короткий сравнительный прогон Phase 1 Plan C с контролем validation loss и отсутствия роста норм весов/логитов относительно базового `Muon`.

## Scope & Affected Files
- `experiments/phase 1/train_phase1.py`
- `experiments/phase 1/Kaggle/train_phase1_notebook.ipynb`
- `ideas/Do Orthogonality Constraints Help Language-Model Training Beyond Sphere Optimization.pdf`
- `ideas/articles index.md`
