<!-- last_verified: 2026-11-21 -->
# KI: Сборка модели для инференса (Model Assembly Guide — Plan C)

## Overview
Этот документ описывает модульную сборку и ключевые инварианты компонентов **BEBLaDII Plan C (System 1)** на базе финализированного энкодера Фазы 1 (`step_19500` EMA) и генеративного декодера с радаром шума Фазы 2 (`phase2_decoder_step_25000.pt` EMA, ADR 137) для инференса и последующего обучения диффузионного ядра Фазы 3.

*(Архивные компоненты предыдущего Plan B перенесены в `storage/components/plan B/` и `experiments/Plan B/`.)*

---

## 1. Компоненты архитектуры и их происхождение

Пайплайн System 1 состоит из автономных модулей в `storage/components/`:

1. **Токенизатор (`storage/components/tokenizer`)**:
   - *Архитектура*: Tekken BPE (Mistral AI, `vocab_size = 131072`, `pad_token_id = 10`).
   - *Загрузка*: `from storage.components.tokenizer import get_tokenizer`.
2. **Общие слои Plan C (`storage/components/plan_c_layers.py`)**:
   - *Содержимое*: `SphereLinear` ($W = \operatorname{diag}(\text{row\_scale}) \cdot Q$ для `OrthMuon d=1`, ADR 131), `RMSNorm` (в `float32`), `ModernBERTLayer` (с per-head `QK-norm` и `SwiGLU`), `RotaryEmbedding128`, `LorentzianEigh`, `safe_lorentzian_eigh`, `apply_rope_16d`, `clamp_to_unit_ball`, `canonicalize_cell_basis` (детерминированная $\mathcal{O}(8)$-канонизация, ADR 137), `WordAwareElasticChunker` (словный чанкинг `1..4` токена и `detokenize_non_pad`, ADR 137).
3. **Энкодер и спектральный пулер (`storage/components/encoder`)**:
   - *Архитектура*: `GrassmannianEncoder` (`embeddings` $131072 \times 1024$, `embed_norm`, `LexicalCoreMLP` $\to [0:96]$ + $[112:128]$, `ContextSyntaxEncoder` 2 слоя ModernBERT $\to [96:112]$, `BlockDiagonalVAEHead`, `GrassmannianTokenNormalizer`, `GrassmannianSpectralPooler` $4 \to 1$ с Partial RoPE и Lorentzian Eigh).
   - *Выход*: последовательность концепт-клеток $\operatorname{Gr}(8, 128)$ формы `(B, N, 8, 128)` (61 тензор в `weights.pt`, EMA на единичной сфере, `phase1_step_19500_ema.pt`).
   - *Загрузка*: `from storage.components.encoder import get_encoder`.
4. **Генеративный декодер и Радар Шума (`storage/components/decoder`)**:
   - *Архитектура*: `GrassmannianDecoder` Фазы 2 (`canonicalize_basis=True`, `enable_noise_radar=True`, `num_smooth_layers=4`): входная детерминированная $\mathcal{O}(8)$-канонизация `canonicalize_cell_basis(V)` в `float32`, `cell_in_norm`, 1 контекстный слой `ModernBERTLayer` по холсту $N$, $\mathcal{O}(8)$-инвариантный `SetCrossAttentionUnpacker` $1 \to 4$ с инъекцией матрицы Грама $G = V V^T$, **Выход А** `NoiseRadarHead` (`estimate_noise` $\to$ `sigma_cell_hat`, `sigma_tok_hat`), инъекция `sigma_tok_hat` через `noise_cond_proj`, **Выход Б** — 4 сглаживающих/генерирующих слоя `ModernBERTLayer` по длине $4N$, `dec_final_norm`, привязанный выходной проектор `lm_head_weight`.
   - *Выход*: логиты словаря Tekken `(B, 4N, 131072)` и опционально словарь радара `radar` при `return_radar=True` (122 собственных тензора в `weights.pt`, нормализованные EMA-веса `phase2_decoder_step_25000.pt` + разделяемая матрица `embeddings.weight` от энкодера).
   - *Загрузка*: `from storage.components.decoder import get_decoder` (также поддерживает передачу `weights_path` напрямую на тренировочный чекпоинт Phase 2 с `ema_state_dict` / `decoder_state_dict`).
5. **Якорные концепты (`sep_concept.pt`, `void_concept.pt`)**:
   - `storage/components/sep_concept.pt` — детерминированный репер разделителя формы `(8, 128)` (`[" think", " reason", " logic", " canvas"]`, IDs `[3648, 5972, 13951, 24661]`, Sign Fixing).
   - `storage/components/void_concept.pt` — канонический репер пустоты VOID формы `(8, 128)` из 4 `pad_token_id = 10`.

---

## 2. Локальные пути

```python
TOKENIZER_DIR     = "C:/Experiments/BEBLaDII/storage/components/tokenizer"
ENCODER_DIR       = "C:/Experiments/BEBLaDII/storage/components/encoder"
DECODER_DIR       = "C:/Experiments/BEBLaDII/storage/components/decoder"
SEP_CONCEPT_PATH  = "C:/Experiments/BEBLaDII/storage/components/sep_concept.pt"
VOID_CONCEPT_PATH = "C:/Experiments/BEBLaDII/storage/components/void_concept.pt"
PHASE1_EMA_CKPT   = "C:/Experiments/BEBLaDII/experiments/phase 1/checkpoints/phase1_step_19500_ema.pt"
PHASE2_DEC_CKPT   = "C:/Experiments/BEBLaDII/experiments/phase 2/checkpoints/phase2_decoder_step_25000.pt"
```

---

## 3. Критические нюансы реализации (Ловушки)

### Ловушка 1: Ненормированные `*.q_weight` в сыром `ema_state_dict` (ADR 132, ADR 134)
В процессе обучения линейное усреднение в `EMAModel.shadow` смещает строки `*.q_weight` внутрь единичной сферы (на `step_19500` отклонение достигало `0.309`).
**Решение:** Для инференса и Фазы 3 использовать готовые `storage/components/{encoder, decoder}/weights.pt` (где `*.q_weight` уже спроецированы на единичную сферу в `float32`). При загрузке сырых тренировочных чекпоинтов обязательно применять `F.normalize(v.float(), p=2, dim=-1, eps=1e-8)` (`get_decoder(weights_path=...)` делает это автоматически для `ema_state_dict`).

### Ловушка 2: Переполнение `float16`/`bfloat16` в канонизации, `SetCrossAttentionUnpacker` и `lm_head` (ADR 125, ADR 126, ADR 137)
Проекция в 131 072 логита, `canonicalize_cell_basis(V)` и расчёт матрицы Грама $G = V V^T$ с последующим `gram_mlp` чувствительны к половинной точности под `torch.autocast`.
**Решение:** В `canonicalize_cell_basis`, `SetCrossAttentionUnpacker.forward`, `NoiseRadarHead.forward` и `GrassmannianDecoder.get_tied_head_logits` явно отключать `autocast(enabled=False)` и выполнять расчёты в `float32`.

### Ловушка 3: Дублирование матрицы `embeddings.weight` (537 МБ) в памяти
`GrassmannianDecoder` использует `embeddings.weight` (`131072 x 1024`) как привязанную выходную голову (`tie_word_embeddings = True`).
**Решение:** При совместной сборке всегда передавать энкодер в декодер: `dec = get_decoder(encoder=enc, device=device)`. При автономном вызове `get_decoder()` без аргумента `encoder` матрица `embeddings.weight` подхватывается автоматически из `../encoder/weights.pt`.

### Ловушка 4: Словный чанкинг `1..4` токена и фильтрация `SLOT_PAD` (`id=10`) (ADR 128, ADR 137)
Декодер Фазы 2 обучен на словно-ориентированных клетках `WordAwareElasticChunker` (`1..4` BPE-токена без разрыва слов длины `<= 4`), где внутренние пустые слоты смешанных клеток имеют вес `eps_pad = 0.001` и предсказываются как `pad_token_id = 10`.
**Решение:** Для упаковки текста использовать `WordAwareElasticChunker.chunk_tokens(...)`, а при сборке текста из предсказаний декодера всегда удалять `pad_token_id = 10` через `decoder.decode_text(V_cell, tokenizer)` или `WordAwareElasticChunker.detokenize_non_pad(...)`.

### Ловушка 5: Возмущения клеток внутри шара $\mathbb{B}^{128}$ (ADR 129)
Для возврата зашумлённых или алгебраически преобразованных клеток в шар $\mathbb{B}^{128}$ использовать только `clamp_to_unit_ball(V)` (не `safe_squash`, который сжимает нормы внутри шара в ~11.3 раза).

---

## 4. Эталонный код сборки (Reference Implementation)

```python
import torch
from storage.components.tokenizer import get_tokenizer
from storage.components.encoder import get_encoder
from storage.components.decoder import get_decoder
from storage.components.plan_c_layers import WordAwareElasticChunker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = get_tokenizer()
encoder = get_encoder(device=device)
decoder = get_decoder(encoder=encoder, device=device)

sep_concept = torch.load("storage/components/sep_concept.pt", map_location=device, weights_only=False)
void_concept = torch.load("storage/components/void_concept.pt", map_location=device, weights_only=False)

chunker = WordAwareElasticChunker(tokenizer=tokenizer, chunk_size=4, pad_token_id=10)
text = "Спектральное латентное пространство Грассманиана."
ids = tokenizer.encode(text, add_special_tokens=False)
packed = chunker.chunk_tokens(ids, max_cells=16, elastic_drop_prob=0.0)

input_ids = packed["input_ids"].unsqueeze(0).to(device)
weights = packed["weights"].unsqueeze(0).to(device)

with torch.no_grad():
    V_cell = encoder(input_ids, weights=weights)                 # (1, 16, 8, 128)
    sigma_cell, sigma_tok, _ = decoder.estimate_noise(V_cell)    # Выход А: Радар Шума
    logits, radar = decoder(V_cell, return_radar=True)           # Выход Б: (1, 64, 131072) + радар
    reconstructed = decoder.decode_text(V_cell, tokenizer=tokenizer)[0]
```
