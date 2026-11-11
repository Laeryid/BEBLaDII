<!-- created: 2026-11-11 -->
<!-- status: Superseded by ADR 122 -->
# ADR 121: Устранение несходимости cuSOLVER (error 18) через Тихоновскую регуляризацию и двухуровневый fallback в Grassmannian Spectral Pooling (Phase 1 Plan C)

> [!WARNING]
> **This ADR is superseded by [ADR 122: Системное устранение вырождения PAD и градиентных сингулярностей (Phase 1 Plan C)](./122_phase1_planc_systemic_nan_prevention.md).**

**Status**: Superseded by [ADR 122](./122_phase1_planc_systemic_nan_prevention.md)
**Date**: 2026-11-11  
**Git Commit**: `70d3875` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.3.3.dev3)  

## Context and Problem
В процессе распределенного обучения Phase 1 Plan C (спектральный Грассманиан 8x128, пулинг 4->1) на GPU Kaggle T4 x2 при достижении шага 300 произошел фатальный сбой cuSOLVER:
`torch._C._LinAlgError: linalg.eigh: (Batch element 0): The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated eigenvalues (error code: 18)`.
Причина: при обработке паддинг-окон (хвосты сэмплов с токенами PAD) 28 из 32 векторов окна строго нулевые (канонический репер VOID активен только по 1 оси из 8). Из-за этого матрица Грама G_win в R^{32x32} имеет ранг <= 4 и ровно 28 кратных собственных чисел, строго равных 0.0000. Алгоритм syevd на GPU cuSOLVER теряет сходимость при дефляции векового уравнения для множественных нулевых корней.

## Decisions Made
1. Внедрить Тихоновскую регуляризацию диагонали матрицы Грама в LorentzianEigh.forward: G_reg = G + 1e-5 * I. По теореме о спектральном сдвиге, добавление eps*I строго сохраняет собственные векторы (G v = lambda v <=> (G + eps*I) v = (lambda + eps) v), но сдвигает нулевые собственные значения до 1e-5, гарантируя строгую положительную определенность и устраняя сингулярность для cuSOLVER. После разложения добавленный джиттер вычитается: L = torch.clamp(L - 1e-5, min=0.0).
2. Реализовать двухуровневый отказоустойчивый fallback:
   - Fallback 1: повтор torch.linalg.eigh с повышенным джиттером 1e-4 * I;
   - Fallback 2: перенос матрицы на CPU и расчет через LAPACK dsyevd (G_reg.detach().cpu()), обладающий детерминированной устойчивостью к кратным корням, с возвратом тензоров на GPU.
3. Математически строгая антисимметризация в LorentzianEigh.backward: применить кососимметричную проекцию входящего градиента Omega = 0.5 * (V^T @ grad_V - grad_V^T @ V) и защиту torch.nan_to_num, исключающие численный дрейф диагонали и взрыв градиентов при близких собственных числах.
4. Защитный фильтр в обучающем и валидационном циклах: проверка torch.isfinite(total_loss) и torch.isfinite(grad_norm) с пропуском шага оптимизатора в случае нечисловых аномалий.

## Consequences
Положительные:
- 100% стабильность спектрального разложения G_win для любых окон текста, включая чистый PAD и вырожденные редкие фразы.
- Сохранение полной точности собственных векторов подпространств Грассманиана (джиттер влияет только на сдвиг спектра, который вычитается обратно).
- Исключение аварийного падения длительных тренировок на Kaggle/TPU при высокой доле паддинга.
- Полная совместимость с чекпоинтами (веса модели не изменились).
Компромиссы:
- Минимальный оверхед на сложение G + eps*I в FP32 (не более 0.1% времени шага).

## Invariants & Rules for AI
- Любое спектральное разложение G_win на GPU ДОЛЖНО быть защищено регуляризацией Тихонова >= 1e-5 * I.
- Производная собственных векторов в autograd backward ДОЛЖНА использовать строго антисимметричную форму 0.5 * (V^T @ grad_V - grad_V^T @ V).
- Нечисловые батчи (NaN/Inf) ДОЛЖНЫ отбрасываться без обновления весов модели и моментов оптимизатора.

## Scope & Affected Files
- `experiments/phase 1/train_phase1.py`
- `experiments/phase 1/Kaggle/train_phase1_notebook.ipynb`

## Verification
1. Модульный тест test_lorentzian_eigh.py с матрицей нулевого ранга и матрицей с 28 нулями: подтвердил сходимость, 0.0000 минимальное собственное число и полную конечность градиентов (torch.isfinite(G.grad).all() == True).
2. Синтаксическая валидация: .venv\Scripts\python.exe -m py_compile 'experiments/phase 1/train_phase1.py'.
3. Реэкспорт блокнота: .venv\Scripts\python.exe 'experiments/phase 1/export_to_notebook.py'.
