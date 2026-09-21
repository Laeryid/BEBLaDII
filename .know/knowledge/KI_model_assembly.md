# KI: Сборка модели для инференса (Model Assembly Guide)

Этот документ описывает точную архитектуру, источники весов и критические нюансы реализации для корректной сборки полного пайплайна модели BEBLaDII (от токенов до диффузии и обратно в текст). 

---

## 1. Компоненты архитектуры и их происхождение

Пайплайн инференса состоит из 6 независимых модулей, которые собираются из разных источников (HuggingFace + локальные чекпоинты разных фаз):

1. **Эмбеддинги (Qwen Embeddings)**: Переводят токены Qwen в начальные векторы.
   - *Архитектура*: `nn.Embedding` (размерность 1536).
   - *Источник*: HuggingFace `Qwen/Qwen2.5-1.5B`.
2. **Латентный энкодер (VAE Encoder - Phase 1)**: Сжимает эмбеддинги Qwen (1536) в латентное пространство (1024) на единичной сфере.
   - *Архитектура*: `LatentEncoder` (Свертки, SwiGLU, L2-нормализация).
   - *Источник*: Чекпоинт Фазы 1.
3. **Главная диффузионная сеть (DUS Backbone - Phase 4)**: Обрабатывает контекст и предсказывает $x_0$.
   - *Архитектура*: `DUSModel` (40 слоев, собранных из 28 слоев `answerdotai/ModernBERT-large`).
   - *Источник*: Инициализация из HuggingFace `ModernBERT-large` + веса из чекпоинта Фазы 4.
4. **Обвязка диффузии (AdaLN, t_proj, self_cond - Phase 4)**: Инжектирует уровень шума $t$ в трансформер.
   - *Архитектура*: Кастомные линейные слои и MLP, живущие *снаружи* DUS внутри класса `BEBLaDIIPhase4aEval`.
   - *Источник*: Тот же чекпоинт Фазы 4. **Критически важно не забыть их загрузить!**
5. **Декодер текстов (ModernLatentDecoder - Phase 2)**: Переводит чистые латентные векторы (1024) обратно в логиты Qwen (1536 -> Vocab).
   - *Архитектура*: 3 последних слоя от `DUSModel` + Линейная проекция + `lm_head` от Qwen.
   - *Источник*: Базовые веса слоев из `AWAKENED_WEIGHTS_FINAL.pt`, тренированные проекции из чекпоинта Фазы 2, голова из HuggingFace `Qwen/Qwen2.5-1.5B`.
6. **Голова Уверенности (Confidence Head - Phase 5)**: Оценивает готовность токенов (опционально для инференса).
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
dus_base_weights = "C:/Experiments/BEBLaDII/kaggle_upload_1_2/AWAKENED_WEIGHTS_FINAL.pt"
phase4_ckpt      = "C:/Experiments/BEBLaDII/experiments/phase 4/local_checkpoints/phase4_step_85995.pth"
conf_ckpt        = "C:/Experiments/BEBLaDII/experiments/phase 5/local/confidence_head_v2.pt"
latent_dict_ckpt = "C:/Experiments/BEBLaDII/experiments/phase 5/local/latent_dict.pt"
```

---

## 3. Критические нюансы реализации (Ловушки)

### Ловушка 1: Случайная инициализация Декодера
Класс `ModernLatentDecoder` имеет аргумент `dus_weights_path`. Если передать туда `None`, он тихо инициализирует `nn.TransformerEncoder` **случайными весами**. Загрузка словаря `strict=False` проигнорирует несовпадение ключей, и декодер будет выдавать полный бред (например, слово `DataExchange` на любой вход).
**Решение:** Обязательно передавать путь к предобученным весам `dus_weights_path=dus_base_weights`.

### Ловушка 2: Потеря слоев обвязки диффузии (AdaLN)
Словарь чекпоинта Фазы 4 содержит ключи `dus_ema` (веса трансформера) и ключи `adaLN_attn_ema`, `t_proj_global_ema` и т.д. Если загрузить только `dus_ema`, диффузионная модель останется с **рандомно инициализированными слоями времени**. Это приводит к полному уничтожению векторов (рост `TrueN`).
**Решение:** Обязательно парсить и загружать компоненты `["adaLN_attn", "adaLN_mlp", "t_proj_global", "t_proj_token", "t_joint_proj", "sep_embed", "self_cond_proj"]` вручную.

### Ловушка 3: Префиксы ключей (torch.compile и wrapper'ы)
При загрузке весов всегда нужно чистить ключи от паразитных префиксов:
- Убирать `_orig_module.` (артефакт `torch.compile`).
- Убирать `student.model.` или `model.` для весов DUS.
- Для VAE (в `dus_model.encoder`) чекпоинт Фазы 1 сохранен с ключами `proj.0.weight`. Префикса `encoder.` там нет.
- Для Декодера Фазы 2 ключи могут содержать префикс `decoder.` (артефакт сохранения).

### Ловушка 4: Qwen LM Head
Голова генерации `lm_head` слишком тяжелая, чтобы держать весь Qwen в памяти.
**Решение:** Создаем `AutoModelForCausalLM`, копируем `.lm_head.weight.data.clone()`, и сразу делаем `del causal_model`, чтобы освободить VRAM.

---

## 4. Эталонный код сборки (Reference Implementation)

```python
def load_models_for_inference(device):
    # 1. Загрузка базовой архитектуры Фазы 4 (она уже включает Qwen embeddings и пустой LatentEncoder)
    dus_model = BEBLaDIIPhase4aEval(
        embedding_model_path="Qwen/Qwen2.5-1.5B", 
        modernbert_path="answerdotai/ModernBERT-large"
    )

    # 2. Загрузка VAE (Фаза 1)
    vae_st = torch.load(vae_ckpt, map_location="cpu", weights_only=False)
    if 'encoder' in vae_st:
        # Убираем префикс encoder. если он есть, иначе грузим как есть
        clean_vae = {k.replace("encoder.", ""): v for k, v in vae_st["encoder"].items()}
        dus_model.encoder.load_state_dict(clean_vae, strict=False)

    # 3. Загрузка DUS и Диффузионной обвязки (Фаза 4)
    p4_st = torch.load(phase4_ckpt, map_location="cpu", weights_only=False)
    
    # 3.1 Сам трансформер (DUS)
    dus_ema = p4_st.get("dus_ema", p4_st.get("dus", {}))
    clean_dus = {}
    for k, v in dus_ema.items():
        clean_k = k.replace("_orig_module.", "").replace("student.model.", "").replace("model.", "")
        clean_dus[clean_k] = v
    dus_model.dus.load_state_dict(clean_dus, strict=False)

    # 3.2 Обвязка диффузии (КРИТИЧЕСКИ ВАЖНО)
    for name in ["adaLN_attn", "adaLN_mlp", "t_proj_global", "t_proj_token", "t_joint_proj", "sep_embed", "self_cond_proj"]:
        comp_state = p4_st.get(f"{name}_ema", p4_st.get(name, {}))
        if comp_state:
            if name == "sep_embed":
                dus_model.sep_embed.copy_(comp_state)
            else:
                clean_comp = {k.replace("_orig_module.", ""): v for k, v in comp_state.items()}
                getattr(dus_model, name).load_state_dict(clean_comp, strict=True)
                
    dus_model.to(device).eval()

    # 4. Загрузка Декодера (Фаза 2)
    # КРИТИЧЕСКИ ВАЖНО: передать dus_weights_path, иначе backbone будет рандомным
    decoder = ModernLatentDecoder(
        latent_dim=1024, qwen_dim=1536, num_layers=3, 
        dus_weights_path=dus_base_weights
    )
    dec_st = torch.load(dec_ckpt, map_location="cpu", weights_only=False)
    clean_dec = {k.replace("decoder.", ""): v for k, v in dec_st.get("decoder", dec_st).items()}
    decoder.load_state_dict(clean_dec, strict=False)
    decoder.to(device).eval()

    # 5. Загрузка LM Head (Qwen)
    from transformers import AutoModelForCausalLM
    causal = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", torch_dtype=torch.bfloat16)
    lm_head_weight = causal.lm_head.weight.data.clone().to(device).float()
    del causal

    # 6. Загрузка Головы Уверенности (Фаза 5)
    conf_head = ConfidenceHead(h39_dim=1024, attn_dim=256, num_heads=4, window_size=32, conf_dim=5, mlp_hidden=512)
    conf_head.load_state_dict(torch.load(conf_ckpt, map_location="cpu"))
    conf_head.to(device).eval()

    return dus_model, conf_head, decoder, lm_head_weight
```

---

## 4. Сборка Оркестратора и Геометрического Ансамбля (Phase 5/6)

Для правильного инференса (денойзинга) Оркестратор не должен использовать метрику `TrueN`. Решения о шагах принимаются исключительно на основе **Геометрического Ансамбля**, вычисляемого из латентного словаря.

### 4.1. Геометрические метрики Головы Уверенности
Голова уверенности (Confidence Head) или модуль оценки DUS возвращает 3 критические метрики:
1. `RawDProx` (Top-1 Similarity): Косинусное расстояние от текущего предсказания $z_{pred}$ до ближайшего слова из словаря $\max_{v \in Vocab} \cos(z_{pred}, v)$.
2. `Delta`: Разница между косинусным сходством Top-1 и Top-2 слов.
3. `ConflictSim`: Косинусное сходство между самими векторами Top-1 и Top-2 кандидата в латентном пространстве.

### 4.2. Маршрутизация в Оркестраторе
В модуле `phase5_orchestrator.py` логика принятия решений работает без словаря (алгебраически):
- Если токен зависает ($t_{local} = 0$, но `RawDProx` низкий) и `Delta` < 0.05:
  - Если `ConflictSim` > 0.70 -> **SYNTAX_VARIATION** (игнорировать, Декодер сам разрешит регистр/пробел).
  - Если `ConflictSim` < 0.30 -> **SEMANTIC_CONFLICT** (сформировать запрос в CLM/RAG для разрешения семантической многозначности).

### 4.3. Адаптивное расписание диффузии (Option 2)
Оркестратор использует жесткий математический таймер, не позволяющий токену застрять в "чистилище высоких шумов" из-за синонимов:
`t_next = max(t_local - delta_t, 0.0)`
Где `delta_t = max(t_local - (1.0 - RawDProx), dt_min)`.
Это заставляет модель перейти из режима **Broad Exploration** ($t > 0.6$) в режим **Local Exploitation** ($t < 0.4$), гарантируя кристаллизацию любого токена.
