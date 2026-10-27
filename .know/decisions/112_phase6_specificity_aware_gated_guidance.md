<!-- created: 2026-10-27 -->
<!-- status: Accepted -->
# ADR 112: Phase 6 Specificity-Aware Likelihood-Ratio Gated Guidance (arXiv:2610.00395)

**Status**: Accepted  
**Date**: 2026-10-27  
**Git Commit**: `3f6bfd2` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.3.0.dev3)  

## Context and Problem
При оценке диффузионной генерации Фазы 6 (ADR 109) сэмплирование с классическим линейным Classifier-Free Guidance (CFG scale 4.0 -> 1.0) приводило к паразитной экстраполяции шума на нейтральных токенах холста (пунктуация, грамматический клей, служебные слова). Линейное вычитание безусловного предсказания (z_uncond) выбивало векторы нейтральных токенов из гладкого многообразия естественного языка в разреженные периферийные углы 1024-мерной сферы, порождая синтаксический мусор (обрывки редких токенов, иероглифы) и ограничивая метрику попадания в словарь DProx на уровне ~0.75.

## Decisions Made
На основе теоретического аппарата статьи Wang, Ma, Stärk, Barzilay (MIT CSAIL, arXiv:2610.00395) внедрить в evaluate_phase6_metrics.py режим сэмплирования Specificity-Aware Likelihood-Ratio Gated Guidance (--sampler specificity, specificity_hybrid). 
Для каждого токена холста вычисляется степень расхождения d_i = (1 - cos(z_pred, z_uncond)) * 0.5 и логистический сигмоидный гейт отношения правдоподобия g_i = sigmoid((d_i - tau) / beta). 
Эффективная сила CFG модулируется индивидуально для каждого токена: gamma_{eff, i} = 1.0 + (cfg_t - 1.0) * g_i, после чего результирующий вектор проецируется на единичную сферу S^{D-1} через safe_normalize. Для токенов общего синтаксиса (d_i < tau) CFG мягко сбрасывается до 1.0, сохраняя естественную траекторию денойзинга, а усиленный CFG фокусируется исключительно на семантически специфичных токенах (~5% холста).

## Consequences
Позитивные:
1. Полное устранение синтаксического мусора, редких BPE-обрывков и иноязычных токенов в генерациях холста 250 токенов.
2. Рост метрики попадания в центроиды латентного словаря DProx с 0.752 до 0.851 (+0.10).
3. Сохранение полной пластичности холста System 2: при необходимости переосмысления токена под промпт расхождение d_i возрастает, открывая гейт и форсируя перезапись токена.
4. Полная обратная совместимость со стандартными сэмплерами (standard, anchor_jump, hybrid).

Компромиссы:
1. Добавление двух новых гиперпараметров: порога специфичности (--spec_tau, default 0.12) и температуры сигмоиды (--spec_temp, default 0.04).

## Invariants & Rules for AI
- [MUST] Сохранять строгую единичную норму латентных векторов на сфере S^{D-1} через safe_normalize после применения модулированного CFG.
- [MUST] Для токенов общего синтаксиса с d_i < tau обеспечивать gamma_{eff} -> 1.0 без искажения траектории денойзинга.
- [MUST NOT] Применять жесткую дискретную заморозку токенов, сохраняя непрерывную динамическую реакцию гейта на расхождение условного и безусловного полей.

## Scope & Affected Files
- `experiments/phase 6/__pycache__/evaluate_phase6_metrics.cpython-313.pyc`
- `experiments/phase 6/evaluate_phase6_metrics.py`
- `experiments/phase 6/phase6_generation_demo_107000.html`
- `experiments/phase 6/phase6_generation_demo_107000_orig.html`
- `experiments/phase 6/phase6_generation_demo_107000_spec.html`
- `ideas/Specificity-Aware Diffusion Steering via.pdf`
- `posts/2026.10.26 Phase 3 Diffusion.jpg`
- `posts/2026.10.26 Phase 3 Diffusion.jpg~`
- `posts/2026.10.26 Phase 3 Diffusion.txt`
- `posts/2026.10.26 Phase 3 Diffusion_sketch.jpg`
- `posts/Gemini_Generated_Image_5vru2b5vru2b5vru.jfif`
- `posts/Gemini_Generated_Image_5vru2b5vru2b5vru.jfif~`

## Verification
Инструментальная генерация в evaluate_phase6_metrics.py на чекпоинте step_107000 с параметрами --sampler specificity_hybrid --max_len 250 показала рост DProx до 0.851, подтвердила подавление 95% нейтрального шума (spec_gate=0.05-0.06 на t < 0.8) и устранила синтаксический мусор в HTML-отчете.
