# Компонента: Декодер-Генератор Грассманиана Plan C (`GrassmannianDecoder`)

Автономный $\mathcal{O}(8)$-инвариантный генеративный декодер и радар шума **Phase 2 Plan C** (нормализованные EMA-веса чекпоинта `phase2_decoder_step_25000.pt`, ADR 137), разворачивающий латентный холст концепт-клеток `(B, N, 8, 128)` обратно в логиты словаря Tekken `(B, 4N, 131072)` и оценивающий поклеточный/потокенный уровень шума.

## Состав архитектуры
- Входная детерминированная $\mathcal{O}(8)$-канонизация базиса `canonicalize_cell_basis(V)` в `float32` с отключённым `autocast` (ADR 122, ADR 126, ADR 137)
- `cell_in_norm` + `dec_context_layer` (1 слой `ModernBERTLayer` по холсту длины `N`, `SphereLinear` + `QK-norm`)
- `unpacker` (`SetCrossAttentionUnpacker` 1 -> 4: 4 обучаемых слота, Cross-Attention к 8 осям и инъекция $G = V V^T$ в `float32`)
- **Выход А (`NoiseRadarHead`)**: быстрая оценка поклеточного шума `sigma_cell_hat (B, N)` и потокенной неуверенности `sigma_tok_hat (B, 4N)` через метод `estimate_noise(V_cell)` или флаг `return_radar=True`
- **Выход Б (Генератор)**: инъекция `sigma_tok_hat` через `noise_cond_proj` + 4 сглаживающих/генерирующих слоя `dec_smooth_layers` (`ModernBERTLayer` по полной длине `4N`) + `dec_final_norm`
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
    V_cell = enc(input_ids)                                 # (B, N, 8, 128)
    sigma_cell, sigma_tok, _ = dec.estimate_noise(V_cell)   # Выход А: Радар Шума
    logits, radar = dec(V_cell, return_radar=True)          # Выход Б: (B, 4N, 131072) + радар
    texts = dec.decode_text(V_cell, tokenizer=tok)          # Строки с удалением SLOT_PAD (id=10)
```
Если декодер используется отдельно от энкодера, `get_decoder()` автоматически подхватит `embeddings.weight` из `../encoder/weights.pt`. Также поддерживается передача `weights_path` напрямую на тренировочный чекпоинт Phase 2 (`decoder_state_dict` / `ema_state_dict`).
