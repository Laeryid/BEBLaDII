# Компонента: Энкодер Грассманиана Plan C (`GrassmannianEncoder`)

Автономный энкодер и спектральный пулер **Phase 1 Plan C** (нормализованные EMA-веса чекпоинта `step_19500`), отображающий последовательности токенов Tekken `(B, 4N)` в латентные концепт-клетки $\operatorname{Gr}(8, 128)$ формы `(B, N, 8, 128)`.

## Состав архитектуры
- `embeddings` (`131072 x 1024`) + `embed_norm` (`RMSNorm`)
- `lexical_mlp`: token-wise SwiGLU с `SphereLinear` -> `v_sem [0:96]` и `v_pos [112:128]`
- `context_encoder`: 2 слоя `ModernBERTLayer` (`SphereLinear` + per-head `QK-norm`) -> `v_syn [96:112]`
- `vae_head`: блочно-диагональная проекция `(96 | 16 | 16)`
- `normalizer`: `safe_squash` в шар $\mathbb{B}^{128}$, расчёт $\text{rk}_{\text{eff}} \ge 1.5$
- `pooler`: Partial RoPE на `112:128`, спектральный пулинг $4 \to 1$ через `LorentzianEigh` и Sign Fixing

## Использование
```python
import torch
from storage.components.tokenizer import get_tokenizer
from storage.components.encoder import get_encoder

tok = get_tokenizer()
enc = get_encoder(device="cuda" if torch.cuda.is_available() else "cpu")

ids = tok.encode("Спектральное латентное пространство Грассманиана").ids
# Дополняем до кратности 4 (pad_token_id=10)
while len(ids) % 4 != 0:
    ids.append(10)

input_ids = torch.tensor([ids], device=enc.embeddings.weight.device)
with torch.no_grad():
    V_cell = enc(input_ids)  # (1, N_cells, 8, 128)
```
