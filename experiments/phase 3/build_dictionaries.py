"""
build_dictionaries.py — Phase 3: Предвычисление словарей D_X0 и D_L40.

Также сохраняет token_map.json — маппинг token_id → текст токена.
Это позволяет читать alpha-разложения в человекочитаемом виде:
  L40_ctx("кот сидел") → top-5: "кот"(0.31), "sat"(0.22), "feline"(0.14)...

За один forward pass через InputProjector + DUS получаем:
  - D_X0[i]  = mu-вектор на позиции 1 (выход InputProjector), 1024-dim
  - D_L40[i] = вектор на позиции 1 (выход DUS слоя 40), 1024-dim

для каждого токена словаря Qwen (~152 064 токенов) в шаблоне:
  <|thought|> {token}

Сохраняет:
  storage/dictionaries/D_X0.pt       — [N, 1024], FP16
  storage/dictionaries/D_L40.pt      — [N, 1024], FP16
  storage/dictionaries/D_L40_norm.pt — [N, 1024], FP16 (нормализованные, для cosine-поиска)

Использование:
  .venv\\Scripts\\python.exe experiments/phase 3/build_dictionaries.py \\
      --checkpoint storage/components/phase2_final_36k.pt \\
      --batch-size 128 \\
      --device cpu

После завершения — загрузка на GCS:
  gsutil cp storage/dictionaries/D_X0.pt  gs://bebladii-weigths/phase3/dictionaries/
  gsutil cp storage/dictionaries/D_L40.pt gs://bebladii-weigths/phase3/dictionaries/
  gsutil cp storage/dictionaries/D_L40_norm.pt gs://bebladii-weigths/phase3/dictionaries/
"""

import os
import sys
import time
import argparse
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import tqdm

# Добавляем корень проекта в sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.beb_la_dii.model.dus import DUSModel
from src.beb_la_dii.model.projectors import InputProjector


# ---------------------------------------------------------------------------
# Аргументы CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Build D_X0 and D_L40 dictionaries for Phase 3.")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Путь к чекпоинту Phase 2 (.pt файл с state_dict)."
    )
    parser.add_argument(
        "--tokenizer-id", type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="HF ID или локальный путь к токенайзеру Qwen2.5."
    )
    parser.add_argument(
        "--student-base-id", type=str,
        default="answerdotai/ModernBERT-large",
        help="HF ID или локальный путь к базе ModernBERT для DUS-скелета."
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Размер батча при инференсе (рекомендуется 128 для CPU с 16 GB RAM)."
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Устройство: 'cpu' или 'cuda'."
    )
    parser.add_argument(
        "--output-dir", type=str, default="storage/dictionaries",
        help="Директория для сохранения словарей."
    )
    parser.add_argument(
        "--thought-token", type=str, default="<|thought|>",
        help="Специальный токен-якорь для шаблона."
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Ограничить число токенов (для отладки). По умолчанию — весь словарь."
    )
    parser.add_argument(
        "--only-x0", action="store_true",
        help="Пересчитать и сохранить только D_X0 (без L40)."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Загрузка весов из чекпоинта Phase 2
# ---------------------------------------------------------------------------

def load_phase2_weights(checkpoint_path: str, student: DUSModel, input_projector: InputProjector):
    """
    Загружает веса DUS и InputProjector из чекпоинта Phase 2.
    Поддерживает как плоский FSDP-формат (ключи с _orig_module.),
    так и стандартный (из ComponentRegistry).
    """
    print(f"  Загрузка чекпоинта: {checkpoint_path}")
    sd = torch.load(checkpoint_path, map_location="cpu")

    # Извлекаем вложенный словарь, если чекпоинт сохранён целиком (состояние оптимизатора и т.д.)
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]

    # Определяем формат чекпоинта
    first_key = next(iter(sd.keys()), "")

    # Чистим FSDP-префиксы если есть
    cleaned = {}
    for k, v in sd.items():
        k_clean = k.replace("_orig_module.", "").replace("module.", "")
        cleaned[k_clean] = v

    # --- DUS (student) ---
    student_sd = {}
    for k, v in cleaned.items():
        if k.startswith("student.model."):
            student_sd[k[len("student.model."):]] = v  # → embeddings.*, layers.*
        elif k.startswith("model.layers.") or k.startswith("model.embeddings") or k.startswith("model.final_norm"):
            student_sd[k[len("model."):]] = v

    matched_student, total_student = 0, len(student.model.state_dict())
    if student_sd:
        result = student.model.load_state_dict(student_sd, strict=False)
        matched_student = total_student - len(result.missing_keys)
    print(f"  DUS: загружено {matched_student}/{total_student} параметров")

    # --- InputProjector ---
    ip_sd = {}
    for k, v in cleaned.items():
        if k.startswith("input_projector."):
            ip_sd[k[len("input_projector."):]] = v  # → proj.0.weight, mu_head.weight, ...
        elif k.startswith("proj.") or k.startswith("mu_head.") or k.startswith("logvar_head."):
            ip_sd[k] = v

    matched_ip, total_ip = 0, len(input_projector.state_dict())
    if ip_sd:
        result = input_projector.load_state_dict(ip_sd, strict=False)
        matched_ip = total_ip - len(result.missing_keys)
    print(f"  InputProjector: загружено {matched_ip}/{total_ip} параметров")

    if matched_student == 0 and matched_ip == 0:
        print("  WARN: Ни один параметр не загружен! Проверьте формат чекпоинта.")
        print(f"  Первые 5 ключей в чекпоинте: {list(sd.keys())[:5]}")


# ---------------------------------------------------------------------------
# Построение словарей
# ---------------------------------------------------------------------------

def build_dictionaries(
    student: DUSModel,
    input_projector: InputProjector,
    tokenizer,
    thought_token_id: int,
    device: torch.device,
    batch_size: int,
    max_tokens: int = None,
):
    """
    Прогоняет все токены через InputProjector + DUS и собирает D_X0 и D_L40.

    Шаблон: [thought_token_id, token_id]
    Позиция 1 (token_id) используется как вектор токена.

    D_X0[i]  = mu из InputProjector для токена i (до DUS)
    D_L40[i] = выход DUS слоя 40 для токена i

    Возвращает: (D_X0, D_L40) — оба тензора [N, 1024], float32
    """
    vocab_size = tokenizer.vocab_size
    if max_tokens is not None:
        vocab_size = min(vocab_size, max_tokens)

    print(f"  Словарь: {vocab_size} токенов, batch_size={batch_size}, device={device}")

    # Формируем все input_ids заранее: [[thought_id, tok_id], ...]
    # Делаем это батчами, чтобы не хранить всё в памяти
    all_x0 = []
    all_l40 = []

    # Hook для перехвата mu из InputProjector (вызывается до DUS)
    captured_mu = {}

    def _hook_mu(module, input, output):
        # output = (z, mu, logvar); нас интересует mu на позиции -1 (целевой токен)
        _, mu, _ = output
        captured_mu["mu"] = mu[:, -1, :].detach().float()  # [B, 1024]

    hook_handle = input_projector.register_forward_hook(_hook_mu)

    student.eval()
    input_projector.eval()

    # Получаем embeddings из DUS (ModernBERT принимает input_ids + attention_mask,
    # но InputProjector принимает уже эмбеддинги Qwen.
    # Нам нужен токенайзер Qwen чтобы получить token_id → embedding из Qwen.
    # НО: мы не загружаем Qwen-модель целиком. Вместо этого берём эмбеддинги
    # напрямую из embedding таблицы Qwen (только embedding layer).
    # Загружаем только embedding weights из чекпоинта или отдельно.)
    #
    # Более простой подход (не требует Qwen): InputProjector принимает
    # Qwen-эмбеддинги размерности 3584. Нам нужна embedding table Qwen.
    # Загрузим только её.
    raise NotImplementedError(
        "Этот блок требует embedding table Qwen — см. секцию ниже."
    )


def build_dictionaries_with_qwen_emb(
    student: DUSModel,
    input_projector: InputProjector,
    qwen_embed_weight: torch.Tensor,  # [vocab_size, 3584]
    thought_ids: list[int],           # IDс префикса (<|thought|> = [27, 91, 60565, 91, 29])
    device: torch.device,
    batch_size: int,
    max_tokens: int = None,
    only_x0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Основная функция построения словарей.

    qwen_embed_weight: embedding таблица Qwen2.5, [vocab_size, 3584], float32/float16
    thought_ids:       список ID токенов префикса. Шаблон: [*thought_ids, token_id].
    Вектор целевого токена = позиция -1 (last_hidden_state[:, -1, :]).
    """
    vocab_size = qwen_embed_weight.shape[0]
    if max_tokens is not None:
        vocab_size = min(vocab_size, max_tokens)

    prefix_len = len(thought_ids)
    seq_len = prefix_len + 1  # префикс + целевой токен
    print(f"  Словарь: {vocab_size} токенов | batch_size={batch_size} | device={device}")
    print(f"  Шаблон: {thought_ids} + [token_id], seq_len={seq_len}, вектор на поз. -1")
    if only_x0:
        print("  Режим --only-x0 включен. Пропускается DUS.")

    # Эмбеддинги префикса (одинаковы для всех батчей)
    thought_token_ids_t = torch.tensor(thought_ids, dtype=torch.long)
    prefix_embs = qwen_embed_weight[thought_token_ids_t].to(device)  # [prefix_len, 3584]

    all_x0 = []
    all_l40 = []

    # Hook: перехватываем X0_sphere из LayerNorm (конец proj)
    captured_mu = {}

    def _hook_h(module, inp, output):
        # output is the result of self.proj(x)
        captured_mu["x0_sphere"] = output.detach().float()  # [B, seq_len, 1024]

    hook_handle = input_projector.proj.register_forward_hook(_hook_h)

    student.eval()
    input_projector.eval()

    n_batches = (vocab_size + batch_size - 1) // batch_size
    t_start = time.time()

    with torch.no_grad():
        for batch_idx in tqdm(range(n_batches), desc="Building dictionaries", unit="batch"):
            start = batch_idx * batch_size
            end = min(start + batch_size, vocab_size)
            B = end - start

            # Формируем батч эмбеддингов: [B, seq_len, 3584]
            # [prefix_embs | tok_emb]
            tok_embs = qwen_embed_weight[start:end].to(device)          # [B, 3584]
            prefix_batch = prefix_embs.unsqueeze(0).expand(B, -1, -1)   # [B, prefix_len, 3584]
            tok_batch = tok_embs.unsqueeze(1)                            # [B, 1, 3584]
            seq_emb = torch.cat([prefix_batch, tok_batch], dim=1)       # [B, seq_len, 3584]

            # Attention mask: все токены видимы
            attention_mask = torch.ones(B, seq_len, dtype=torch.long, device=device)

            # --- Forward: InputProjector ---
            # InputProjector.forward(x) ожидает x: [B, seq_len, 3584]
            z, mu, logvar = input_projector(seq_emb)
            # captured_mu["x0_sphere"] заполнен hook'ом: [B, seq_len, 1024]
            if device.type == "xla":
                import torch_xla.core.xla_model as xm
                xm.mark_step()
            x0_batch = captured_mu["x0_sphere"][:, -1, :].cpu().float()   # [B, 1024] — поз. -1 = цель
            all_x0.append(x0_batch)

            if not only_x0:
                # --- Forward: DUS (latentBERT) ---
                # z: [B, seq_len, 1024] — выход InputProjector (в eval: z == mu)
                if device.type == "xla":
                    import torch_xla.core.xla_model as xm
                    xm.mark_step()
                dus_output = student.model(
                    inputs_embeds=z,
                    attention_mask=attention_mask,
                )
                # last_hidden_state: [B, seq_len, 1024]
                l40_batch = dus_output.last_hidden_state[:, -1, :].cpu().float()  # [B, 1024]
                all_l40.append(l40_batch)

            # Прогресс каждые 50 батчей
            if (batch_idx + 1) % 50 == 0:
                elapsed = time.time() - t_start
                done = (batch_idx + 1) * batch_size
                eta = elapsed / done * (vocab_size - done)
                tqdm.write(
                    f"  [{batch_idx+1}/{n_batches}] "
                    f"Обработано: {done}/{vocab_size} токенов | "
                    f"Прошло: {elapsed/60:.1f} мин | ETA: {eta/60:.1f} мин"
                )

    hook_handle.remove()

    D_X0 = torch.cat(all_x0, dim=0)   # [N, 1024]
    D_L40 = torch.cat(all_l40, dim=0) if not only_x0 else None

    print(f"  Готово за {(time.time()-t_start)/60:.1f} мин")
    print(f"  D_X0:  shape={D_X0.shape}, norm_cv={_norm_cv(D_X0):.4f}")
    if not only_x0:
        print(f"  D_L40: shape={D_L40.shape}, norm_cv={_norm_cv(D_L40):.4f}")

    return D_X0, D_L40


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _norm_cv(x: torch.Tensor) -> float:
    """Coefficient of variation нормы векторов (диагностика сферичности)."""
    norms = x.norm(dim=-1)
    return (norms.std() / norms.mean()).item()


def build_token_map(tokenizer) -> list[str]:
    """
    Строит список строк: token_map[token_id] = текст токена.
    Сохраняется в token_map.json рядом со словарями.

    Используется для human-readable аналитики alpha-разложений:
        top-k токены для контекстного вектора → читаемые слова/сабворды.
    """
    vocab_size = tokenizer.vocab_size
    token_map = ["" ] * vocab_size
    for token_id in range(vocab_size):
        try:
            # decode даёт читаемый текст (со спецсимволами и пробелами)
            text = tokenizer.decode([token_id], skip_special_tokens=False)
        except Exception:
            text = f"<unk_{token_id}>"
        token_map[token_id] = text
    return token_map


def sanity_check_x0(
    D_X0: torch.Tensor,
    token_map: list[str],
    probe_token_ids: list[int],
    k: int = 5,
):
    """
    Быстрый sanity check для словаря D_X0.
    Проверяет, что вектор целевого токена в словаре
    находит самого себя как top-1 соседа (cos_sim ≈ 1.0).
    """
    print("\n--- Sanity Check: поиск в D_X0 ---")
    D_X0_norm = F.normalize(D_X0.float(), dim=-1)
    
    with torch.no_grad():
        for tok_id in probe_token_ids:
            if tok_id >= D_X0_norm.shape[0]:
                continue
            query = D_X0_norm[tok_id].unsqueeze(0)           # [1, 1024]
            scores = (query @ D_X0_norm.T)                   # [1, N]
            topk = torch.topk(scores[0], k)

            tok_text = repr(token_map[tok_id])
            print(f"\n  Запрос (D_X0): {tok_text} (id={tok_id})")
            for rank, (idx, sim) in enumerate(zip(topk.indices.tolist(), topk.values.tolist())):
                marker = "✓" if idx == tok_id else " "
                print(f"    {marker} [{rank+1}] {repr(token_map[idx]):20s} cos={sim:.3f}  (id={idx})")
    print()


def sanity_check_decomposition(
    D_L40_norm: torch.Tensor,
    D_X0: torch.Tensor,
    token_map: list[str],
    thought_token_id: int,
    probe_token_ids: list[int],
    k: int = 8,
    tau: float = 0.1,
):
    """
    Быстрый sanity check: для нескольких тестовых токенов показывает
    их soft-разложение в словаре D_L40.

    Ожидаем: top-1 alpha ≈ сам токен (alpha_top1_self),
    остальные — семантически близкие.
    """
    print("\n--- Sanity Check: alpha-разложения ---")
    print(f"  tau={tau}, top-k={k}")

    with torch.no_grad():
        for tok_id in probe_token_ids:
            if tok_id >= D_L40_norm.shape[0]:
                continue
            # Берём нормализованный вектор токена из D_L40_norm
            query = D_L40_norm[tok_id].unsqueeze(0)           # [1, 1024]
            scores = (query @ D_L40_norm.T) / tau              # [1, N]
            topk = torch.topk(scores[0], k)
            alpha = torch.softmax(topk.values, dim=-1)

            tok_text = repr(token_map[tok_id])
            print(f"\n  Запрос: {tok_text} (id={tok_id})")
            for rank, (idx, a) in enumerate(zip(topk.indices.tolist(), alpha.tolist())):
                marker = "✓" if idx == tok_id else " "
                print(f"    {marker} [{rank+1}] {repr(token_map[idx]):20s} α={a:.3f}  (id={idx})")
    print()


def load_qwen_embeddings(checkpoint_path: str, tokenizer_id: str) -> torch.Tensor:
    """
    Извлекает embedding-таблицу Qwen из чекпоинта Phase 2.
    Если в чекпоинте нет embedding-весов учителя — загружает только
    embedding layer из HF (без полной модели).

    Возвращает: [vocab_size, 3584], float32
    """
    print("  Поиск embedding-таблицы Qwen в чекпоинте...")
    sd = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]

    # Пробуем найти embed_tokens в чекпоинте (teacher embeddings)
    embed_key = None
    for k in sd.keys():
        if "embed_tokens.weight" in k and ("teacher" in k or "model.embed" in k):
            embed_key = k
            break

    if embed_key:
        print(f"  Найдены embeddings в чекпоинте: {embed_key}")
        return sd[embed_key].float()

    # Fallback: загружаем только embedding layer из HF
    print(f"  Embeddings не найдены в чекпоинте. Загружаем из HF: {tokenizer_id}")
    print("  (Загружается только embedding layer, ~1.2 GB)")

    # Загружаем конфиг и только embedding weights через low_cpu_mem_usage
    from transformers import AutoModelForCausalLM, AutoConfig
    config = AutoConfig.from_pretrained(tokenizer_id, trust_remote_code=True)

    # Загружаем полную модель, но сразу извлекаем и удаляем
    # Используем float16 чтобы уместить в RAM при необходимости
    model = AutoModelForCausalLM.from_pretrained(
        tokenizer_id,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    embed_weight = model.model.embed_tokens.weight.detach().float()
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"  Embedding table загружена: {embed_weight.shape}")
    return embed_weight


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def download_checkpoint_if_needed(checkpoint_path: str) -> str:
    if checkpoint_path.startswith("gs://"):
        local_path = os.path.basename(checkpoint_path)
        if not os.path.exists(local_path):
            print(f"  Скачивание чекпоинта из GCS: {checkpoint_path} -> {local_path} ...")
            import subprocess
            try:
                subprocess.run(["gcloud", "storage", "cp", checkpoint_path, local_path], check=True)
            except (FileNotFoundError, subprocess.CalledProcessError):
                subprocess.run(["gsutil", "cp", checkpoint_path, local_path], check=True)
        else:
            print(f"  Чекпоинт {local_path} уже существует локально.")
        return local_path
    return checkpoint_path

def main():
    args = parse_args()
    if args.device == "xla":
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print("  Используется TPU (XLA)")
    else:
        device = torch.device(args.device)

    # Загрузка чекпоинта из Cloud Storage, если указан gs://
    local_checkpoint_path = download_checkpoint_if_needed(args.checkpoint)

    print("=" * 60)
    print("Phase 3: Build Dictionaries")
    print("=" * 60)
    print(f"  Чекпоинт:  {args.checkpoint}")
    print(f"  Device:    {device}")
    print(f"  Batch:     {args.batch_size}")
    print(f"  Output:    {args.output_dir}")
    print()

    # --- 1. Токенайзер ---
    print("[1/5] Загрузка токенайзера...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size
    print(f"  Vocab size: {vocab_size}")

    # Строим token_map сразу — понадобится и для sanity check, и для сохранения
    print("  Строим token_map (id → текст)...")
    token_map = build_token_map(tokenizer)

    # Ищем thought_token_ids (может быть несколько токенов)
    thought_ids = tokenizer.encode(args.thought_token, add_special_tokens=False)
    if not thought_ids:
        raise ValueError(f"--thought-token '{args.thought_token}' не дал токенов.")
    print(f"  thought_token '{args.thought_token}' → ids={thought_ids} ({len(thought_ids)} ток.)")
    # seq_len = len(thought_ids) + 1 (префикс + целевой токен)
    # вектор целевого токена = позиция -1 (last)
    thought_token_id = thought_ids[0]  # для sanity check anchor

    # --- 2. Модели ---
    print("\n[2/5] Создание DUS (latentBERT) скелета...")
    student = DUSModel.from_scratch(
        component_id="latentBERT",
        version="v1.0",
        config={"base_model_id": args.student_base_id, "target_layers": 40},
    )
    student = student.to(device)

    print("\n[3/5] Создание InputProjector...")
    input_projector = InputProjector.from_scratch(
        component_id="qwen_to_bert_input",
        version="v1.0",
    )
    input_projector = input_projector.to(device)

    # --- 3. Загрузка весов Phase 2 ---
    print("\n[4/5] Загрузка весов Phase 2...")
    print(f"  Загрузка чекпоинта: {local_checkpoint_path}")
    load_phase2_weights(local_checkpoint_path, student, input_projector)

    # 5. Эмбеддинги Qwen
    qwen_embed_weight = load_qwen_embeddings(local_checkpoint_path, args.tokenizer_id)
    # Приводим к float32 для стабильности CPU-инференса
    qwen_embed_weight = qwen_embed_weight.float()
    print(f"  Embedding table: {qwen_embed_weight.shape}")  # [152064, 3584]

    # --- 5. Построение словарей ---
    print("\n[5/5] Построение словарей D_X0 и D_L40...")
    D_X0, D_L40 = build_dictionaries_with_qwen_emb(
        student=student,
        input_projector=input_projector,
        qwen_embed_weight=qwen_embed_weight,
        thought_ids=thought_ids,
        device=device,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        only_x0=args.only_x0,
    )

    # --- 6. Sanity check: проверка D_X0 ---
    probe_texts = [" the", " is", " cat", " math", " step", "думать", "<|thought|>"]
    probe_ids = []
    for t in probe_texts:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if len(ids) == 1:
            probe_ids.append(ids[0])
    if thought_token_id not in probe_ids:
        probe_ids.insert(0, thought_token_id)

    sanity_check_x0(
        D_X0=D_X0,
        token_map=token_map,
        probe_token_ids=probe_ids,
        k=5,
    )

    if not args.only_x0:
        # --- 7. Нормализация D_L40 для cosine-поиска ---
        print("\nНормализация D_L40...")
        D_L40_norm = F.normalize(D_L40.float(), dim=-1)

        # --- 8. Sanity check: читаемые разложения для нескольких токенов ---
        probe_texts = [" the", " is", " cat", " math", " step", "думать", "<|thought|>"]
        probe_ids = []
        for t in probe_texts:
            ids = tokenizer.encode(t, add_special_tokens=False)
            if len(ids) == 1:
                probe_ids.append(ids[0])
        if thought_token_id not in probe_ids:
            probe_ids.insert(0, thought_token_id)

        sanity_check_decomposition(
            D_L40_norm=D_L40_norm,
            D_X0=D_X0,
            token_map=token_map,
            thought_token_id=thought_token_id,
            probe_token_ids=probe_ids,
            k=8,
            tau=0.1,  # начальное значение; потом подберём в Этапе 1
        )

    # --- 9. Конвертация в FP16 и сохранение ---
    print("Сохранение словарей...")
    os.makedirs(args.output_dir, exist_ok=True)

    paths = {
        "D_X0":       os.path.join(args.output_dir, "D_X0.pt"),
    }
    token_map_path = os.path.join(args.output_dir, "token_map.json")
    torch.save(D_X0.half(),       paths["D_X0"])

    if not args.only_x0:
        paths["D_L40"] = os.path.join(args.output_dir, "D_L40.pt")
        paths["D_L40_norm"] = os.path.join(args.output_dir, "D_L40_norm.pt")
        torch.save(D_L40.half(),      paths["D_L40"])
        torch.save(D_L40_norm.half(), paths["D_L40_norm"])

    print(f"  Сохранение token_map ({len(token_map)} токенов)...")
    with open(token_map_path, "w", encoding="utf-8") as f:
        json.dump(token_map, f, ensure_ascii=False)

    for name, path in paths.items():
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  {name}: {path} ({size_mb:.0f} MB)")
    token_map_size = os.path.getsize(token_map_path) / 1024 / 1024
    print(f"  token_map: {token_map_path} ({token_map_size:.1f} MB)")

    # --- 10. Итоговая диагностика ---
    print("\n" + "=" * 60)
    print("Итоговые метрики:")
    print(f"  D_X0  norm_cv: {_norm_cv(D_X0):.4f}")
    if not args.only_x0:
        print(f"  D_L40 norm_cv: {_norm_cv(D_L40):.4f}  (ожидается ~0.089 как у l40)")
    print()
    print("Загрузка на GCS:")
    all_files = list(paths.values()) + [token_map_path]
    for path in all_files:
        fname = os.path.basename(path)
        print(f"  gsutil cp {path} gs://bebladii-weigths/phase3/dictionaries/{fname}")
    print("=" * 60)


if __name__ == "__main__":
    main()
