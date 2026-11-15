<!-- created: 2026-11-15 -->
<!-- status: accepted -->
# ADR 134: Финализация Фазы 1 Plan C (step_19500 EMA), модульная сборка System 1 и архивация Plan B

**Status**: accepted  
**Date**: 2026-11-15  
**Git Commit**: `237f2f6` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.4.0.dev2)  

## Context and Problem
Фаза 1 Plan C (обучение автоэнкодера Gr(8, 128) и иерархического декомпрессора) достигла рабочих метрик на шаге 19500 (val/recon=0.001367, Token Accuracy=100%, PPL=1.00-1.02, val/effective_dim=64.39/128, квантово-булева алгебра с дистрибутивностью Sim_Gr=0.8735 и аннигиляцией противоречия до 0.000001). При этом выяснилось, что в сыром ema_state_dict строки *.q_weight смещались внутрь сферы (до 0.3095), инспекторы inspect_phase1_latent.py и inspect_manifold_and_logic.py не применяли сферическую проекцию из ADR 132, а storage/components/ и README.md всё ещё отражали архивный Plan B.

## Decisions Made
1. Зафиксировать чекпоинт шага 19500 как финальный базовый артефакт Фазы 1 Plan C и сохранить чистый EMA-чекпоинт experiments/phase 1/checkpoints/phase1_step_19500_ema.pt (только нормализованный model_state_dict и метрики, без оптимизаторов; исходные чекпоинты сохранены).
2. Вынести общие слои Plan C (SphereLinear, ModernBERTLayer, RMSNorm, RotaryEmbedding128, LorentzianEigh, clamp_to_unit_ball) в storage/components/plan_c_layers.py и создать автономные модули storage/components/encoder/ (GrassmannianEncoder, 61 тензор) и storage/components/decoder/ (GrassmannianDecoder с разделяемым tied_lm_head, 67 тензоров).
3. Скомпилировать детерминированные якорные концепты storage/components/sep_concept.pt ([' think', ' reason', ' logic', ' canvas'], IDs [3648, 5972, 13951, 24661]) и storage/components/void_concept.pt (4x pad_token_id=10) с добавлением исключений в .gitignore.
4. Перенести отчёты Plan B в reports/plan_B/, старые компоненты в storage/components/plan B/, обновить README.md (раздел Results & Reports под Plan C, Plan B отмечен как failed), experiments/Plan B/README.md и experiments/phase 1/experiment_plan.md.
5. Добавить построчную проекцию EMA *.q_weight в inspect_phase1_latent.py и inspect_manifold_and_logic.py, и обновить .know/knowledge/KI_model_assembly.md и KI_model_storage.md под Plan C.

## Consequences
1. Размер финального чекпоинта сокращён с 2.86 ГБ до 772.8 МБ без потери точности; автономные компоненты storage/components/{tokenizer, encoder, decoder} готовы к использованию на Фазе 2 и на Kaggle с нулевым расхождением выходов (max diff = 0.00e+00).
2. Архивные веса и отчёты Plan B изолированы в storage/components/plan B/ и reports/plan_B/, корневой README.md актуализирован под Plan C.
3. Обязательная проекция EMA *.q_weight устраняет скрытое сжатие строк внутрь сферы (ранее max |1 - ||q||| = 0.3095) и восстанавливает корректную геометрию и устойчивость к шуму при инспекции.

## Alternatives Considered & Rejected
1. Слияние SphereLinear в плотный nn.Linear при экспорте — отклонено пользователем ради переиспользования SphereLinear и OrthMuon на последующих фазах.
2. Повторный прогон обучения ради 80-90 effective_dim — отклонено, так как все 128 каналов уже активно задействованы (diag128 в [0.651, 2.446], PR синтаксиса 11.70/16), а искусственное выбеливание разрушит семантическую кластеризацию и булеву логику.

## Invariants & Rules for AI
- [MUST] При извлечении EMA-весов из тренировочных чекпоинтов Plan C проецировать *.q_weight на единичную сферу в float32 (ADR 132).
- [MUST] При совместной сборке System 1 передавать encoder в get_decoder(encoder=enc) во избежание дублирования матрицы embeddings (131072 x 1024).
- [MUST] Сохранять SphereLinear (W = diag(row_scale) @ q_weight) в компонентах Plan C для совместимости с OrthMuon (d=1) на последующих фазах.
- [MUST NOT] Не удалять архивные отчёты и компоненты Plan B — хранить их в reports/plan_B/ и storage/components/plan B/.

## Scope & Affected Files
- `.gitignore`
- `.know/doc_config.json` (Related KI: `KI_knowledge_infrastructure.md`)
- `.know/doc_state.json`
- `.know/ki_config.json` (Related KI: `KI_knowledge_infrastructure.md`)
- `experiments/phase 1/inspect_manifold_and_logic.py`
- `experiments/phase 1/inspect_phase1_latent.py`
- `storage/components/model/teacher_embedder/config.json`
- `storage/components/projector/feat_proj_20/v1.0/config.json`
- `storage/components/projector/feat_proj_30/v1.0/config.json`
- `storage/components/projector/feat_proj_40/v1.0/config.json`
- `storage/components/projector/qwen_to_bert_input/v1.0/config.json`
- `storage/components/sep_token.pt` (Related KI: `decisions/058_phase4_canvas_format_and_sep_token.md`)
- `storage/components/void_token.pt`
- `reports/plan_B/`
- `storage/components/decoder/`

## Verification
py_compile OK для inspect_phase1_latent.py, inspect_manifold_and_logic.py, plan_c_layers.py, encoder/__init__.py, decoder/__init__.py; max |V_comp - V_ref| = 0.00e+00, max |logits_comp - logits_ref| = 0.00e+00; sep_concept.pt и void_concept.pt скомпилированы.
