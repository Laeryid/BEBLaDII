<!-- created: 2026-11-09 -->
<!-- status: Accepted -->
# ADR 119: Ретроспектива и завершение Плана Б: Пределы поточечной BPE-диффузии на холсте (Demos 107k–134k) и переход к Плану С

**Status**: Accepted  
**Date**: 2026-11-09  
**Git Commit**: `5d5bff8` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.3.3.dev3)  

## Context and Problem
В рамках реализации Плана Б (Phase 1 VAE -> Phase 2 Decoder -> Phase 4 DUS backbone -> Phase 6 CA-слои) была разработана сквозная архитектура неавторегрессионной диффузии текста на гиперсфере R^1024.
Обучение Фазы 6 на TPU/GPU достигло 134 000 шагов. В директории `experiments/phase 6/Demos/` была проведена всесторонняя слепая оценка чекпоинтов (107k, 110k, 111k, 114k, 119k, 122k, 129k, 134k) с замером метрик (PPL GPT-2, Distinct-1/2/3, DProx, Sink-Ratio) и детальным анализом артефактов (blind_judge_*.md, score_generations_*.json).
Анализ генераций выявил неустранимый барьер:
1. Проблема суперпозиции и 'слова-кентавры': при обучении на длинном холсте (T=256..512) и высоком шуме (t in [0.7, 1.0]) DUS не знает точной пословной фазы и усредняет вероятные BPE-токены, порождая химеры (напр., 'choizrediz', 'FantT', 'nextDigitovan', 'yovykiiti').
2. Зацикливание и эхо промпта: все варианты сэмплеров (SmartCrop hard/soft/adaptive, Specificity LR-Gated Guidance, импульсный CFG, Void-DProx) дают одинаковый синтаксический каркас со спамом ключевых слов вопроса.
3. Деградация поздних шагов: к 134k шагам модель начинает сваливаться в коллапс частотных технических токенов ('the', 'convolution', 'V').
4. Изолированная диагностика чекпоинта 119k (Тесты 1-5) доказала: траектория застывает уже к 25 шагу (25, 50 и 100 шагов посимвольно идентичны); Self-Conditioning не лечит зацикливание; но сжатие по 4 токена (Mean+Norm, Тест 5) сохраняет 100% смыслового скелета без потерь.

## Decisions Made
1. Официально завершить План Б и признать исчерпанным потенциал прямого поточечного моделирования BPE-токенов на длинном холсте (T=256..512).
2. Зафиксировать фундаментальную причину неудачи Плана Б: непрерывное латентное диффузионное пространство на сфере для дискретных субсловных BPE-токенов страдает от Phase Uncertainty (фазовой неопределенности позиции слова). Без авторегрессионного сдвига или смысловой агрегации модель усредняет траектории.
3. Утвердить переход к Плану С (Иерархическая декомпрессия):
   - Разделение обязанностей: Диффузия (System 2) отвечает исключительно за макро-смысловые концепт-патчи (N=32..64).
   - Декомпрессор (System 1, LatentDecompressorDecoder N -> 4N) отвечает за микро-синтаксис, BPE-склейку и восстановление стоп-слов/пунктуации.
   - Подавление и зашумление стоп-слов (StopwordAwareChunkPooler) на этапе пулинга, чтобы DUS не тратил ёмкость на выравнивание связок.

## Consequences
1. Чекпоинты Фазы 6 (107k–134k) официально замораживаются в статусе исследовательского архива. Дальнейшее обучение Phase 6 на TPU прекращается.
2. Проект полностью переключается на План С (Concept-Patch Diffusion & Decompressor Decoder):
   - Диффузионное ядро DUS освобождается от микро-орфографии и обучается на компактном холсте N=32..64.
   - Декомпрессор (LatentDecompressorDecoder) берет на себя восстановление синтаксиса, предлогов и BPE-согласования из смысловых корней.
3. Время диффузионного инференса сокращается в 4-8 раз благодаря короткому холсту N=64.
4. Зафиксированы строгие отрицательные результаты по эвристическим сэмплерам (SmartCrop, Specificity Guidance, ступенчатый CFG), что экономит исследовательские ресурсы.

## Alternatives Considered & Rejected
- Продолжение обучения Phase 6 до 200k+ шагов (начиная со 120k наблюдается регрессия и коллапс в технические токены).
- Попытки решить проблему зацикливания чисто эвристическими сэмплерами инференса (SmartCrop, Specificity LR-gating, Dynamic Void-DProx, перебор CFG).
- Увеличение числа шагов диффузии (шаги 25, 50 и 100 выдают посимвольно идентичный результат).

## Invariants & Rules for AI
- Запрещено использовать полноразмерный холст BPE-токенов (T >= 256) в DUS без предварительного смыслового сжатия/пулинга.
- Диффузионное ядро должно оперировать на уровне абстрактных смысловых концептов (N <= 64).
- Декодирование в текст должно осуществляться через обученный декомпрессор (N -> 4N), восстанавливающий синтаксические связки.

## Scope & Affected Files
- `experiments/phase 1/__pycache__/test_phase1_sanity.cpython-313.pyc`
- `experiments/phase 1/__pycache__/train_phase1_vae_fsdp.cpython-313.pyc`
- `experiments/phase 1/check_checkpoint.py`
- `experiments/phase 1/contrastive_loss.csv`
- `experiments/phase 1/contrastive_loss.png`
- `experiments/phase 1/cov_loss.csv`
- `experiments/phase 1/cov_loss.png`
- `experiments/phase 1/effective_dim.csv`
- `experiments/phase 1/effective_dim.png`
- `experiments/phase 1/sanity test.txt`
- `experiments/phase 1/screenshots/18062026 night 1 val.jpg`
- `experiments/phase 1/screenshots/18062026 night 1.jpg`
- `experiments/phase 1/screenshots/19062026 day 2 kl.jpg`
- `experiments/phase 1/screenshots/19062026 day 2.jpg`
- `experiments/phase 1/screenshots/19062026 day val 2.jpg`

## Verification
- Слепые судейские протоколы: `experiments/phase 6/Demos/blind_judge_*.md` (шаги 107k..134k).
- Количественные замеры сэмплеров: `experiments/phase 6/Demos/score_generations_*.json`.
- Отчет об изолированной диагностике: `experiments/phase 6/phase6_isolated_diagnostics_report.md`.
- Успешный запуск и верификация PoC декомпрессора Plan C: `experiments/phase 2 plan C/test_decompressor_poc.py`.
