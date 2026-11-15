# Компонента: Декодер Грассманиана Plan C (`GrassmannianDecoder`)

Автономный $\mathcal{O}(8)$-инвариантный декомпрессор **Phase 1 Plan C** (нормализованные EMA-веса чекпоинта `step_19500`), разворачивающий латентный холст концепт-клеток `(B, N, 8, 128)` обратно в логиты словаря Tekken `(B, 4N, 131072)`.

## Состав архитектуры
- `cell_in_norm` + `dec_context_layer` (1 слой `ModernBERTLayer` по холсту длины `N`, `SphereLinear` + `QK-norm`)
- `unpacker` (`SetCrossAttentionUnpacker` 1 -> 4: 4 обучаемых слота, Cross-Attention к 8 осям и инъекция $G = V V^T$ в `float32`)
- `dec_smooth_layers` (2 сглаживающих слоя `ModernBERTLayer` по полной длине `4N`) + `dec_final_norm`
- Tied Output Head: проецирует в `131072` через `embeddings.weight` строго в `float32` с отключённым `autocast` (ADR 125).

## Использование
```python
import torch
from storage.components.tokenizer import get_tokenizer
from storage.components.encoder import get_encoder
from storage.components.decoder import get_decoder

tok = get_tokenizer()
enc = get_encoder(device="cuda" if torch.cuda.is_available() else "cpu")
# Передаем encoder, чтобы не дублировать матрицу embeddings (131072 x 1024) в памяти
dec = get_decoder(encoder=enc, device=enc.embeddings.weight.device)

with torch.no_grad():
    V_cell = enc(input_ids)        # (B, N, 8, 128)
    logits = dec(V_cell)           # (B, 4N, 131072)
    pred_ids = logits.argmax(-1)
```
Если декодер используется отдельно от энкодера, `get_decoder()` автоматически подхватит `embeddings.weight` из `../encoder/weights.pt`.
