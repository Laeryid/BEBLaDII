<!-- last_verified: 2026-11-15 -->
# KI: Сборка модели для инференса (Model Assembly Guide — Plan C)

## Overview
Этот документ описывает модульную сборку и ключевые инварианты компонентов **BEBLaDII Plan C (System 1)** на базе финализированного чекпоинта Фазы 1 (`step_19500` EMA) для инференса и последующего обучения диффузионного ядра Фазы 2.

*(Архивные компоненты предыдущего Plan B перенесены в `storage/components/plan B/` и `experiments/Plan B/`.)*

---

## 1. Компоненты архитектуры и их происхождение

Пайплайн System 1 состоит из автономных модулей в `storage/components/`:

1. **Токенизатор (`storage/components/tokenizer`)**:
   - *Архитектура*: Tekken BPE (Mistral AI, `vocab_size = 131072`, `pad_token_id = 10`).
   - *Загрузка*: `from storage.components.tokenizer import get_tokenizer`.
2. **Общие слои Plan C (`storage/components/plan_c_layers.py`)**:
   - *Содержимое*: `SphereLinear` ($W = \operatorname{diag}(\text{row\_scale}) \cdot Q$ для `OrthMuon d=1`, ADR 131), `RMSNorm` (в `float32`), `ModernBERTLayer` (с per-head `QK-norm` и `SwiGLU`), `RotaryEmbedding128`, `LorentzianEigh`, `apply_rope_16d`, `clamp_to_unit_ball`.
3. **Энкодер и спектральный пулер (`storage/components/encoder`)**:
   - *Архитектура*: `GrassmannianEncoder` (`embeddings` $131072 \times 1024$, `embed_norm`, `LexicalCoreMLP` $\to [0:96]$ + $[112:128]$, `ContextSyntaxEncoder` 2 слоя ModernBERT $\to [96:112]$, `BlockDiagonalVAEHead`, `GrassmannianTokenNormalizer`, `GrassmannianSpectralPooler` $4 \to 1$ с Partial RoPE и Lorentzian Eigh).
   - *Выход*: последовательность концепт-клеток $\operatorname{Gr}(8, 128)$ формы `(B, N, 8, 128)` (61 тензор в `weights.pt`, EMA на единичной сфере).
   - *Загрузка*: `from storage.components.encoder import get_encoder`.
4. **Декомпрессор-декодер (`storage/components/decoder`)**:
   - *Архитектура*: `GrassmannianDecoder` (`cell_in_norm`, 1 контекстный слой `ModernBERTLayer` по холсту $N$, $\mathcal{O}(8)$-инвариантный `SetCrossAttentionUnpacker` $1 \to 4$ с инъекцией матрицы Грама $G = V V^T$, 2 сглаживающих слоя `ModernBERTLayer` по длине $4N$, `dec_final_norm`, привязанный выходной проектор `lm_head_weight`).
   - *Выход*: логиты словаря Tekken `(B, 4N, 131072)` (67 собственных тензоров в `weights.pt` + разделяемая матрица `embeddings.weight` от энкодера).
   - *Загрузка*: `from storage.components.decoder import get_decoder`.
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
```

---

## 3. Критические нюансы реализации (Ловушки)

### Ловушка 1: Ненормированные `*.q_weight` в сыром `ema_state_dict` (ADR 132)
В процессе обучения линейное усреднение в `EMAModel.shadow` смещает строки `*.q_weight` внутрь единичной сферы (на `step_19500` отклонение достигало `0.309`).
**Решение:** Для инференса и Фазы 2 использовать готовые `storage/components/{encoder, decoder}/weights.pt` или `phase1_step_19500_ema.pt`, где `*.q_weight` уже спроецированы на единичную сферу в `float32`. При загрузке сырых тренировочных чекпоинтов обязательно применять `F.normalize(v.float(), p=2, dim=-1, eps=1e-8)`.

### Ловушка 2: Переполнение `float16`/`bfloat16` в `SetCrossAttentionUnpacker` и `lm_head` (ADR 125, ADR 126)
Проекция в 131 072 логита и расчёт матрицы Грама $G = V V^T$ с последующим `gram_mlp` чувствительны к половинной точности под `torch.autocast`.
**Решение:** В `GrassmannianDecoder.get_tied_head_logits` и `SetCrossAttentionUnpacker.forward` явно отключать `autocast(enabled=False)` и выполнять расчёты в `float32`.

### Ловушка 3: Дублирование матрицы `embeddings.weight` (537 МБ) в памяти
`GrassmannianDecoder` использует `embeddings.weight` (`131072 x 1024`) как привязанную выходную голову (`tie_word_embeddings = True`).
**Решение:** При совместной сборке всегда передавать энкодер в декодер: `dec = get_decoder(encoder=enc, device=device)`. При автономном вызове `get_decoder()` без аргумента `encoder` матрица `embeddings.weight` подхватывается автоматически из `../encoder/weights.pt`.

### Ловушка 4: Взвешивание `PAD` и кратность `chunk_size = 4` (ADR 128)
Входная последовательность `input_ids` должна иметь длину, кратную `4` (дополняется `pad_token_id = 10`). Для неполных крайних окон при передаче пользовательских весов `weights` для `PAD`-позиций задаётся вес `<= 0.001` относительно реальных токенов.

### Ловушка 5: Возмущения клеток внутри шара $\mathbb{B}^{128}$ (ADR 129)
Для возврата зашумлённых или алгебраически преобразованных клеток в шар $\mathbb{B}^{128}$ использовать только `clamp_to_unit_ball(V)` (не `safe_squash`, который сжимает нормы внутри шара в ~11.3 раза).

---

## 4. Эталонный код сборки (Reference Implementation)

```python
import torch
from storage.components.tokenizer import get_tokenizer
from storage.components.encoder import get_encoder
from storage.components.decoder import get_decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = get_tokenizer()
encoder = get_encoder(device=device)
decoder = get_decoder(encoder=encoder, device=device)

sep_concept = torch.load("storage/components/sep_concept.pt", map_location=device, weights_only=False)
void_concept = torch.load("storage/components/void_concept.pt", map_location=device, weights_only=False)

text = "Спектральное латентное пространство Грассманиана."
ids = tokenizer.encode(text).ids
pad_id = 10
while len(ids) % 4 != 0:
    ids.append(pad_id)

input_ids = torch.tensor([ids], dtype=torch.long, device=device)
raw_w = torch.where(input_ids == pad_id, 0.001, 1.0).view(1, -1, 4)
weights = (raw_w / raw_w.sum(dim=-1, keepdim=True)).view(1, -1)

with torch.no_grad():
    V_cell = encoder(input_ids, weights=weights)  # (1, N_cells, 8, 128)
    logits = decoder(V_cell)                      # (1, 4 * N_cells, 131072)
    pred_ids = logits.argmax(dim=-1)[0].tolist()
    reconstructed = tokenizer.decode(pred_ids)
```
