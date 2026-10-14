# KI: Сборка модели для инференса (Model Assembly Guide)

Этот документ описывает точную архитектуру, источники весов и критические нюансы реализации для корректной сборки полного пайплайна модели BEBLaDII (от токенов до диффузии и обратно в текст). 

---

## 1. Компоненты архитектуры и их происхождение

Пайплайн инференса состоит из независимых модулей, которые собираются из разных источников (HuggingFace + локальные чекпоинты разных фаз):

1. **Эмбеддинги (Qwen Embeddings)**: Переводят токены Qwen в начальные векторы.
   - *Архитектура*: `nn.Embedding` (размерность 1536).
   - *Источник*: HuggingFace `Qwen/Qwen2.5-1.5B`.
   - *Нюанс*: В Qwen2.5 `tie_word_embeddings = True`, поэтому матрица эмбеддингов строго равна весам `lm_head.weight`.
2. **Латентный энкодер (VAE Encoder - Phase 1)**: Сжимает эмбеддинги Qwen (1536) в латентное пространство (1024) на единичной сфере.
   - *Архитектура*: `LatentEncoder` (Свертки, SwiGLU, L2-нормализация).
   - *Источник*: Чекпоинт Фазы 1 (`planB_phase1_checkpoints_phase1_vae_step_20000.pth`).
3. **Главная диффузионная сеть (DUS Backbone - Phase 4)**: Обрабатывает контекст и предсказывает $x_0$.
   - *Архитектура*: `DUSModel` (40 слоев, собранных из 28 слоев `answerdotai/ModernBERT-large`).
   - *Источник*: Инициализация из HuggingFace `ModernBERT-large` + веса из чекпоинта Фазы 4 (`phase4_step_85995.pth`).
4. **Обвязка диффузии (AdaLN, t_proj - Phase 4)**: Инжектирует уровень шума $t$ в трансформер.
   - *Архитектура*: Кастомные линейные слои и MLP, живущие *снаружи* DUS (`adaLN_attn`, `adaLN_mlp`, `t_proj_global`, `t_proj_token`, `t_joint_proj`, `sep_embed`).
   - *Источник*: Чекпоинт Фазы 4.
5. **Слои управления промптом (CAPromptLayer - Phase 6)**: Инжектируют контекст вопроса в диффузионный холст.
   - *Архитектура*: `CAPromptLayer` (Cross-Attention + Self-Attention + AdaLN времени) на слоях 12, 24, 36 (ModernBERT блоки 11, 23, 35) через `Phase6BlockWrapper`.
   - *Размороженные блоки*: 4 слоя DUS после каждого CA (12–15, 24–27, 36–39) дообучаются вместе с CA.
   - *Источник*: Чекпоинт Фазы 6 (`planB_phase6_checkpoints_phase6_ca_layers_step_*.pth`).
6. **Декодер текстов (ModernLatentDecoder - Phase 2)**: Переводит чистые латентные векторы (1024) обратно в логиты Qwen (1536 -> Vocab).
   - *Архитектура*: 3 последних слоя от `DUSModel` + Линейная проекция (1024 -> 1536) + `lm_head` от Qwen.
   - *Источник*: Скелет инициализируется через DUS (`phase4_step_85995.pth`), веса декодера из чекпоинта Фазы 2 (`planB_phase2_checkpoints_decoder_step_9000.pth`), проекция в токены через `qwen_embeddings.weight`.
7. **Голова Уверенности (Confidence Head - Phase 5)**: Оценивает готовность токенов (опционально для оркестрации).
   - *Архитектура*: `LocalSelfAttention` (window=32) + MLP.
   - *Источник*: Чекпоинт Фазы 5.

---

## 2. Локальные файлы и пути к весам

```python
embed_model_id   = "Qwen/Qwen2.5-1.5B"
modernbert_id    = "answerdotai/ModernBERT-large"

# Локальные чекпоинты:
vae_ckpt         = "C:/Experiments/BEBLaDII/experiments/phase 1/planB_phase1_checkpoints_phase1_vae_step_20000.pth"
dec_ckpt         = "C:/Experiments/BEBLaDII/experiments/phase 2/planB_phase2_checkpoints_decoder_step_9000.pth"
phase4_ckpt      = "C:/Experiments/BEBLaDII/experiments/phase 4/local_checkpoints/phase4_step_85995.pth"
phase6_ckpt      = "C:/Experiments/BEBLaDII/experiments/phase 6/checkpoints/planB_phase6_checkpoints_phase6_ca_layers_step_6000.pth"
latent_dict_ckpt = "C:/Experiments/BEBLaDII/experiments/phase 5/local/latent_dict.pt"
sep_token_ckpt   = "C:/Experiments/BEBLaDII/storage/components/sep_token.pt"
```

---

## 3. Критические нюансы реализации (Ловушки)

### Ловушка 1: Неконтекстуализированный промпт в CA (Нарушение ADR 099)
Если подавать сырой выход VAE-энкодера $Z_{prompt}$ напрямую в `ca_layers`, Cross-Attention получает векторы из чужого распределения (косинус между сырым латентом и выходом слоя 11 ModernBERT составляет всего **~0.088**).
**Решение:** Строго следовать ADR 099: промпт $Q$ **обязательно** прогоняется через первые 12 слоев DUS (0–11) при $t=0$ с выключенным CA. Извлеченный кэш $Z_{prompt\_12} = \text{hidden\_states}[12][:, 1:, :]$ кэшируется один раз и устанавливается в `ca._current_Z_prompt`.

### Ловушка 2: Hubness Problem при декодировании через `latent_dict`
Попытка декодировать текст через поиск ближайшего соседа по словарю $\arg\max (z \cdot D^T)$ приводит к "словесной каше" из редких иероглифов и служебных токенов (хабов).
**Решение:** Для связного текста декодирование должно производиться через **Phase 2 `ModernLatentDecoder`** со снятием логитов через матрицу `qwen_embeddings.weight`:
```python
projected = decoder(z_canvas.float()) # [B, T, 1536]
logits = F.linear(projected, qwen_embeddings.weight.float())
text = tokenizer.decode(logits.argmax(dim=-1)[0], skip_special_tokens=True)
```
Словарь `latent_dict.pt` используется **исключительно** как сенсор для расчета $t_{reported} = 1.0 - RawDProx$.

### Ловушка 3: Случайная инициализация Декодера
Класс `ModernLatentDecoder` имеет аргумент `dus_weights_path`. Если передать туда `None`, он тихо инициализирует `nn.TransformerEncoder` **случайными весами**. Загрузка словаря `strict=False` проигнорирует несовпадение ключей, и декодер будет выдавать мусор.
**Решение:** Обязательно передавать путь к весам DUS (например, `phase4_step_85995.pth`):
```python
decoder = ModernLatentDecoder(latent_dim=1024, qwen_dim=1536, num_layers=3, dus_weights_path=phase4_ckpt)
dec_st = torch.load(dec_ckpt, map_location="cpu")
decoder.load_state_dict(dec_st["decoder"], strict=True) # 100% strict совпадение ключей!
```

### Ловушка 4: Потеря слоев обвязки диффузии (AdaLN)
Словарь чекпоинта Фазы 4 содержит ключи `dus_ema` (веса трансформера) и ключи `adaLN_attn_ema`, `t_proj_global_ema` и т.д. Если загрузить только `dus_ema`, диффузионная модель останется с рандомно инициализированными слоями времени.
**Решение:** Обязательно загружать компоненты `["adaLN_attn", "adaLN_mlp", "t_proj_global", "t_proj_token", "t_joint_proj", "sep_embed"]`.

### Ловушка 5: Префиксы ключей (torch.compile и wrapper'ы)
При загрузке весов всегда нужно чистить ключи от паразитных префиксов:
- Убирать `_orig_module.` (артефакт `torch.compile`).
- Убирать `student.model.` или `model.` для весов DUS.
- В чекпоинте Фазы 6 (`trainable_state`) ключи CA-слоев имеют вид `dus.layers.11.ca_layer.*` (или `ca_layers.12.*`), а размороженные слои — `dus.layers.12.*`.

---

## 4. Эталонный код сборки (Reference Implementation)

```python
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from src.beb_la_dii.model.vae import LatentEncoder
from src.beb_la_dii.model.dus import DUSModel
from src.beb_la_dii.model.modern_decoder import ModernLatentDecoder
from src.beb_la_dii.utils.loss import safe_normalize

def load_phase6_pipeline(device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Загрузка Phase 6 модели (включает QwenEmbeddings, VAE, DUS, AdaLN, CA-слои)
    # См. реализацию класса BEBLaDIIPhase6Eval в evaluate_phase6_metrics.py
    model = BEBLaDIIPhase6Eval(config).to(device).eval()

    # 2. Загрузка Phase 2 Декодера с ModernBERT бэкбоном
    decoder = ModernLatentDecoder(
        latent_dim=1024, qwen_dim=1536, num_layers=3, 
        dus_weights_path=config['phase4_path']
    ).to(device).eval()
    
    ckpt_dec = torch.load(config['phase2_decoder_path'], map_location=device, weights_only=False)
    decoder.load_state_dict(ckpt_dec["decoder"], strict=True)

    # 3. Сенсорный словарь для RawDProx
    latent_dict = torch.load(config['latent_dict_ckpt'], map_location=device).float()

    return model, decoder, latent_dict, tokenizer

def generate_answer(model, decoder, latent_dict, tokenizer, prompt_text, device="cpu", steps=25, max_len=64):
    q_enc = tokenizer(prompt_text, return_tensors='pt').to(device)

    # ADR 099: Предварительный расчет семантического кэша промпта Z_prompt_12
    Z_prompt_12 = model.encode_prompt(q_enc.input_ids, q_enc.attention_mask)

    # Slerp обратная диффузия от чистого шума t=1.0 до t=0.0
    B = 1
    z_canvas = safe_normalize(torch.randn(B, max_len, 1024, device=device), dim=-1)
    dt = 1.0 / steps

    for i in range(steps):
        t_val = 1.0 - i * dt
        t_actual = torch.full((B, max_len), t_val, device=device)

        # Вычисление t_reported через RawDProx
        sims = torch.matmul(z_canvas, latent_dict.T)
        raw_dprox, _ = sims.max(dim=-1)
        t_reported = (1.0 - raw_dprox).clamp(0.0, 1.0)

        # Шаг денойзинга DUS + CA
        h39, z_pred = model.forward_step(z_canvas, Z_prompt_12, q_enc.attention_mask, t_actual, t_reported)

        # Slerp шаг
        t_next = torch.full((B, max_len), max(0.0, t_val - dt), device=device)
        theta_now = t_actual.unsqueeze(-1) * (math.pi / 2)
        theta_next = t_next.unsqueeze(-1) * (math.pi / 2)
        sin_now = torch.sin(theta_now).clamp(min=1e-5)
        
        w_pred = torch.sin(theta_now - theta_next) / sin_now
        w_cur = torch.sin(theta_next) / sin_now
        z_next = w_pred * z_pred + w_cur * z_canvas
        z_canvas = torch.where(theta_now > 1e-5, safe_normalize(z_next, dim=-1), safe_normalize(z_pred, dim=-1))

    # Декодирование текста через ModernLatentDecoder
    with torch.no_grad():
        projected = decoder(z_canvas.float())
        logits = F.linear(projected, model.qwen_embeddings.weight.float())
        pred_ids = logits.argmax(dim=-1)
        return tokenizer.decode(pred_ids[0], skip_special_tokens=True)
```
