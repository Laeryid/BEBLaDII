<!-- created: 2026-11-17 -->
<!-- status: accepted -->
# ADR 137: Выделение Фазы 2 Plan C (Генеративный Декодер и Радар Шума), перенос диффузии в Фазу 3, детерминированная O(8)-канонизация и словный чанкинг 1..4 токена

**Status**: accepted  
**Date**: 2026-11-17  
**Git Commit**: `0090108` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.4.0.dev2)  

## Context and Problem
После диагностики чекпоинта диффузионного ядра step_7500 (experiments/phase 3/checkpoints/planC_Phase 2_checkpoints_phase2_step_7500.pt) выявлены четыре структурных разрыва между автоэнкодером Фазы 1 и диффузией: (1) GrassmannianDecoder не являлся строго O(8)-инвариантным из-за разворачивания (8, 128) -> 1024 в dec_context_layer и диагональной матрицы Грама в SetCrossAttentionUnpacker (при случайном повороте R in O(8) точность падала с 0.9942 до 0.7689); (2) декодер обучался только на слабом джиттере std=0.035 (Sim_Gr=0.996) и распадался на word-salad при интерполяции гипотез MixOther (Sim_Gr=0.6766 -> tok_acc=0.4913); (3) жёсткая сетка по 4 BPE-токена разрезала слова между клетками, нарушая условие когерентного сложения BPE-осколков (Теорема 9 Эккарта-Янга в plan_C_math_foundations.md) и вызывая сдвиг всех последующих ячеек при добавлении одного слова; (4) диффузия оперировала единым скалярным расписанием t без оценки поклеточного/потокенного шума sigma для кристаллизации опорных клеток.

## Decisions Made
1. Разделить обучение декодера и диффузии: текущая директория диффузии перенесена из experiments/phase 2/ в experiments/phase 3/, а experiments/phase 2/ выделена под обучение Генеративного Декодера-Распаковщика и Радара Шума на замороженном энкодере Фазы 1.
2. Внедрить детерминированную O(8)-канонизацию базиса canonicalize_cell_basis(V) в storage/components/plan_c_layers.py (матрица Грама 8x8 в float32 с autocast(enabled=False) по ADR 126, safe_lorentzian_eigh по ADR 122, сортировка по убыванию собственных значений и SignFix в R^128) на входе GrassmannianDecoder (под флагом canonicalize_basis=True с сохранением обратной совместимости с весами Фазы 1).
3. Ввести единый словно-ориентированный эластичный чанкер WordAwareElasticChunker (1..4 BPE-токена без разрыва слов длины <= 4, elastic_drop_prob=0.20 при обучении, двухуровневые веса с eps_pad=0.001, target=pad_token_id=10 во внутренних пустых слотах смешанных клеток, target=-100 в хвостовых VOID-клетках, сборка через detokenize_non_pad).
4. Реализовать двухвыходной GrassmannianDecoder в storage/components/decoder/__init__.py: Выход А (NoiseRadarHead, метод estimate_noise) оценивает поклеточный и потокенный шум (sigma_cell_hat, sigma_tok_hat) по спектру, признакам Этапа 1 и chunked top-2 косинусной близости к словарю под no_grad (compute_dict_stats); Выход Б инжектирует sigma_tok_hat через zero-init noise_cond_proj перед расширенными слоями dec_smooth_layers (num_smooth_layers=4, доп. слои с row_scale=1e-3).
5. Создать пайплайн experiments/phase 2/train_phase2.py (аугментации clean_jitter, gaussian, mix_other, crystal_mask + случайные R in O(8), OrthMuon d=1 + AdamW, отдельный GCS-бакет gs://bebladii-weigths-us/planC/Phase2_Decoder/checkpoints/), тесты test_phase2_decoder.py и экспорт Kaggle-блокнота.

## Consequences
Положительные: достигнута детерминированная O(8)-инвариантность (max_V_diff=3.87e-06, Sim_Gr=1.000000, совпадение argmax 100%); слова <= 4 токенов больше не разрываются между клетками и обратимо собираются через detokenize_non_pad; декодер разделён на быстрый радар шума (Выход А) и генеративный контур с учётом потокенной неуверенности (Выход Б) при полной заморозке латентного пространства Фазы 1 (GrassmannianEncoder и embeddings.weight). Компромиссы: добавлен обязательный промежуточный этап обучения новой Фазы 2 перед модернизацией и перезапуском диффузии в Фазе 3; в смешанных клетках (1..3 токена) внутренние пустые слоты явно предсказываются как pad_token_id=10 и требуют удаления через detokenize_non_pad.

## Alternatives Considered & Rejected
1. Удаление внутреннего RoPE (112:128) для борьбы с жёсткой сеткой 4 токенов — отклонено, так как оператор плотности rho = sum w_i v_i v_i^T коммутативен и превратил бы клетку в неупорядоченный bag-of-words без устранения межъячеечного сдвига.
2. Обучение O(8)-инвариантности только через аугментацию случайными поворотами без canonicalize_cell_basis — отклонено (прямое следствие Определения 3 из plan_C_math_foundations.md и п. 2.4.3 plan C.md требует детерминированной канонизации).
3. Поддержание двух отдельных нейросетей декодера (для оценки шума и генерации текста) — отклонено во избежание расхождения критериев уверенности; выбран единый модуль с быстрым выходом estimate_noise на базе Этапа 1.

## Invariants & Rules for AI
- [MUST] В GrassmannianDecoder (Фаза 2 и Фаза 3) перед передачей клеток в dec_context_layer выполнять canonicalize_cell_basis(V) в float32 с отключенным autocast (ADR 122, ADR 126).
- [MUST] На Фазе 2 держать GrassmannianEncoder и общую матрицу embeddings.weight (131072 x 1024) полностью замороженными (eval(), requires_grad=False) во избежание смещения латентного пространства Фазы 1.
- [MUST] Различать в WordAwareElasticChunker внутренние закрывающие слоты смешанных клеток (target=pad_token_id=10, вес CE 0.25) и хвостовые VOID-клетки (target=-100, ignore_index); при декодировании текста всегда фильтровать pad_token_id=10 через detokenize_non_pad.
- [MUST NOT] Удалять гармонический RoPE (112:128) внутри ячеек пулера — без него сумма внешних произведений теряет порядок токенов внутри клетки.
- [MUST NOT] Размещать чекпоинты Фазы 2 Декодера в одном GCS-бакете с диффузией Фазы 3 (использовать gs://bebladii-weigths-us/planC/Phase2_Decoder/checkpoints/).

## Scope & Affected Files
- `experiments/phase 2/Kaggle/.gitkeep`
- `experiments/phase 2/Kaggle/train_phase2_notebook.ipynb`
- `experiments/phase 2/export_to_notebook.py`
- `experiments/phase 2/inspect_phase2_generation.py`
- `experiments/phase 2/prepare_phase2_data.py`
- `experiments/phase 2/test_phase2_architecture.py`
- `experiments/phase 2/test_phase2_open_questions.py`
- `experiments/phase 2/train_phase2.py`
- `storage/components/decoder/__init__.py`
- `storage/components/encoder/__init__.py`
- `storage/components/plan_c_layers.py`
- `experiments/phase 2/test_phase2_decoder.py`
- `experiments/phase 3/`

## Verification
.venv\Scripts\python.exe -B "experiments/phase 2/test_phase2_decoder.py" && .venv\Scripts\python.exe -B "experiments/phase 2/train_phase2.py" --smoke_test --device cpu
