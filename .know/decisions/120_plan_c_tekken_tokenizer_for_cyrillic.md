<!-- created: 2026-11-10 -->
<!-- status: Accepted -->
# ADR 120: Выбор токенизатора Tekken (Mistral AI) для кириллицы и Plan C

**Status**: Accepted  
**Date**: 2026-11-10  
**Git Commit**: `a5ae0a6` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.3.3.dev3)  

## Context and Problem
В рамках подготовки к Плану С (спектральное латентное пространство Грассманиана, сжатие концептов и декомпрессор) возникла необходимость пересмотра базового токенизатора. Ранее использовавшийся токенизатор Qwen2.5 (151.6k токенов) спроектирован с приоритетом под китайский язык (CJK), английский и код. На кириллице Qwen2.5 показывает избыточное дробление: окончания, морфологические формы и редкие слова разбиваются на цепочки мелких 2-3 символьных субтокенов или байтовый фолбэк, что создает паразитную нагрузку на диффузионное ядро и раздувает длину последовательности. С другой стороны, обучение собственного компактного токенизатора (32k-64k) нежелательно из-за риска потери генерализации на словах вне обучающих датасетов проекта (out-of-distribution лексика, редкие термины, сленг, многоязычные заимствования).

## Decisions Made
1. Принять в качестве базового токенизатора для Plan C токенизатор Tekken от Mistral AI (Mistral NeMo 12B / Mistral Large 2, размер словаря ~131 072 токена, BPE).
2. Отказаться от обучения кастомного специализированного словаря для сохранения открытого лексикона и устойчивой работы на любых текстах вне датасетов проекта.
3. Использовать словарь Tekken для VAE Фазы 1 (разложение токенов в спектральные подпространства Грассманиана) и последующего декомпрессора LatentDecompressorDecoder (восстановление текста из матриц плотности).
4. Зафиксировать архитектурное решение в experiments/plan C.md.

## Consequences
Положительные:
- Существенно более высокая плотность токенизации кириллицы по сравнению с Qwen2.5 при меньшем словаре (131k против 151.6k у Qwen2.5 и 256k у Gemma 2).
- Экономия VRAM на матрицах эмбеддингов энкодера и проекций выходного словаря декомпрессора.
- Сохранение качественной токенизации кода, математики и английского языка из коробки без затрат на обучение собственного BPE.
- Высокая обобщающая способность (OOD) на редких словах, неологизмах и терминах вне локальных датасетов.

Компромиссы/Отрицательные:
- Необходимость интеграции mistral-common / Hugging Face fast tokenizer для Tekken в пайплайн данных.
- Словарь (131k) крупнее чисто русскоязычных словарей (например, ruGPT-3 на 50k), но обеспечивает универсальность без катастрофического раздувания (как у 256k).

## Alternatives Considered & Rejected
1. Qwen2.5 Tokenizer (151.6k): Отклонен из-за посредственного сжатия кириллицы (слишком много CJK токенов в словаре).
2. Gemma 2 / Command R Tokenizers (256k): Отклонены из-за чрезмерного размера словаря (256k), ведущего к перерасходу VRAM на матрицах эмбеддингов.
3. Обучение собственного BPE на 32k-64k: Отклонено из-за ухудшения OOD-обобщения на лексике вне обучающих датасетов проекта.

## Invariants & Rules for AI
- [MUST] Использовать токенизатор Tekken (Mistral) во всех компонентах Plan C, взаимодействующих с текстом (Phase 1 VAE, LatentDecompressorDecoder).
- [MUST NOT] Обучать суррогатные кастомные словари, ухудшающие OOD-генерализацию на произвольных текстах.

## Scope & Affected Files
- `.know/doc_config.json` (Related KI: `KI_knowledge_infrastructure.md`)
- `experiments/Plan B/phase 1/check_checkpoint.py`
- `experiments/Plan B/phase 1/contrastive_loss.csv`
- `experiments/Plan B/phase 1/contrastive_loss.png`
- `experiments/Plan B/phase 1/cov_loss.csv`
- `experiments/Plan B/phase 1/cov_loss.png`
- `experiments/Plan B/phase 1/effective_dim.csv`
- `experiments/Plan B/phase 1/effective_dim.png`
- `experiments/Plan B/phase 1/sanity test.txt`
- `experiments/Plan B/phase 1/screenshots/18062026 night 1 val.jpg`
- `experiments/Plan B/phase 1/screenshots/18062026 night 1.jpg`
- `experiments/Plan B/phase 1/screenshots/19062026 day 2 kl.jpg`
- `experiments/Plan B/phase 1/screenshots/19062026 day 2.jpg`
- `experiments/Plan B/phase 1/screenshots/19062026 day val 2.jpg`
- `experiments/Plan B/phase 1/screenshots/19062026 day val.jpg`

## Verification
Сравнительный бенчмарк fertility rate (число символов на токен) на русскоязычном корпусе между Tekken и Qwen2.5.
