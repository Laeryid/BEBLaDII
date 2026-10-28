<!-- created: 2026-10-28 -->
<!-- status: Accepted -->
# ADR 114: Ребалансировка весов <|void|>, ликвидация Exposure Bias через Spurious-Tail Erasing и фиксация отката к step_107000

**Status**: Accepted  
**Date**: 2026-10-28  
**Git Commit**: `16194ed` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.3.0.dev3)  

## Context and Problem
На шагах 107 000 -> 115 000 в Phase 6 модель перестала заполнять конец холста токенами <|void|> и перешла к копированию слов вопроса. Эмпирический аудит выявил две связанные причины: 1) Байесовский перекос лосса в пользу контента против войда (вес void_mask был снижен до 0.05-0.20, а на контенте сумма весов составляла ~2.0 + действовал жесткий dynamic void penalty с порогом ~0.20); 2) Exposure Bias: модель никогда не обучалась затирать слова в хвосте в <|void|>, поэтому любые намёки на слова на средних шумах t<=0.5 денойзились обратно в слова (в Probe B cos со словами = 0.95 при 0% войда). Чекпоинты после 107k признаны деградировавшими и удалены.

## Decisions Made
1. Откатить обучение к проверенному чекпоинту step_107000 (установить max_resume_step = 107000). 2. Поднять вес void_mask в token_importance с 0.20 до 0.75 в compute_phase6_loss. 3. Смягчить dynamic void penalty: установить безопасный фиксированный порог отталкивания 0.50 с весом 0.25 (void_penalty = F.relu(cos_sim_to_void - 0.50) * content_mask). 4. Внедрить аугментацию Spurious-Tail Erasing: в 25% примеров батча на TPU входной z_noisy на позициях хвоста формируется из зашумленного текста ответа, в то время как целевой таргет Z_A_clean строго содержит чистый void_embed. 5. Модернизировать evaluate_phase6_metrics.py: объединять непрерывные спаны пустоты в токены <|void xN|>, выводить метрику Void% и встраивать полный пошаговый лог генерации под спойлер в HTML.

## Consequences
Восстановлен честный байесовский баланс выбора между продолжением текста и переходом в состояние покоя (~50% порог). Устранен фундаментальный Exposure Bias: модель учится затирать случайный контент в хвосте в <|void|>. В валидатор добавлена прозрачная визуализация токенов пустоты <|void xN|> и сохранение полного пошагового лога генерации под спойлером в HTML.

## Alternatives Considered & Rejected
- Оставление веса void_mask=0.05..0.20: приводит к полному отказу модели ставить <|void|> при неопределенной длине ответа.
- Коррекция только параметров инференса (отключение CFG оверрайда строки 826 или подстановка void_embed в словарь): не решает проблему, так как на весах 107k/115k модель при стандартном сэмплере дает <0.4% войда из-за Exposure Bias.
- Полная одновременная смена 6 гиперпараметров (Prompt Anti-Copy, Causal Ramp, Boundary Weights): отклонена в пользу минималистичной изолированной Итерации 1 для чистоты эксперимента.

## Invariants & Rules for AI
- [MUST] Целевой таргет z_clean для хвостовых позиций всегда строго содержит чистый void_embed.
- [MUST] Входной источник z_noisy_source при аугментации Spurious-Tail Erasing формируется строго через torch.where без динамических форм XLA.
- [MUST NOT] Не допускать снижения веса void_mask в token_importance ниже 0.50.
- [MUST NOT] Не использовать динамический порог dynamic_margin < 0.40 для void_penalty.

## Scope & Affected Files
- `experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.py` (Related KI: `decisions/097_phase6_t_global_orchestrator_and_per_sample_beta_noise.md`)
- `experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.ipynb`
- `experiments/phase 6/evaluate_phase6_metrics.py`
- `experiments/phase 6/fixes_plan.md`

## Verification
.venv\Scripts\python.exe -m py_compile "experiments/phase 6/evaluate_phase6_metrics.py"
.venv\Scripts\python.exe -m py_compile "experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.py"
.venv\Scripts\python.exe -m jupytext --to notebook --output "experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.ipynb" "experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.py"
