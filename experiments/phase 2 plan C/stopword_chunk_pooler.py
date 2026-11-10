import os
import re
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = "C:/Experiments/BEBLaDII"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.beb_la_dii.utils.loss import safe_normalize

# =========================================================================
# 1. Список стоп-слов и связок (Русский + Базовый английский)
# =========================================================================
BASE_CONNECTORS = {
    # Предлоги русские
    "в", "во", "на", "с", "со", "к", "ко", "по", "из", "изо", "у", "о", "об", "обо",
    "от", "ото", "до", "за", "над", "под", "подо", "пред", "предо", "при", "про",
    "без", "безо", "для", "через", "сквозь", "между", "меж", "среди", "около", "вокруг",
    "после", "кроме", "вместо", "ради", "насчет", "вслед", "навстречу",
    # Союзы русские
    "и", "да", "а", "но", "или", "либо", "что", "чтобы", "чтоб", "как", "когда", "если",
    "то", "хотя", "пока", "будто", "словно", "точно", "потому", "поэтому", "также", "тоже",
    # Частицы русские
    "же", "ж", "бы", "б", "ли", "ль", "не", "ни", "вот", "ведь", "даже", "лишь",
    "только", "уже", "еще", "ещё", "так", "разве", "неужели",
    # Местоимения-связки и вспомогательные глаголы
    "он", "она", "оно", "они", "его", "ее", "её", "их", "ему", "ей", "им", "нем", "ней",
    "мы", "вы", "я", "ты", "меня", "тебя", "нас", "вас", "мне", "тебе", "нам", "вам",
    "свой", "своя", "свое", "своё", "свои", "себе", "себя", "собой", "собою",
    "этот", "эта", "это", "эти", "этого", "этой", "этих", "этому", "этим",
    "тот", "та", "то", "те", "того", "той", "тех", "тому", "тем",
    "весь", "вся", "все", "всё", "всего", "всей", "всех", "всему", "всеми",
    "который", "которая", "которое", "которые", "которого", "которой", "которых", "которому",
    "быть", "был", "была", "было", "были", "есть", "будет", "будут", "буду", "будем",
    # Базовые английские stopwords
    "the", "a", "an", "in", "on", "at", "to", "for", "with", "from", "of", "by", "as",
    "and", "but", "or", "so", "if", "than", "that", "this", "these", "those",
    "it", "its", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had"
}


# =========================================================================
# 2. Построитель булевой маски связок с защитой BPE-границ
# =========================================================================
def get_or_build_connector_mask(tokenizer, cache_path=None, vocab_size=None) -> torch.Tensor:
    """
    Создает или загружает булев тензор [vocab_size], где True соответствует
    только целым токенам-связкам (предлоги, союзы, знаки препинания, спецтокены).
    Защищает от ложного зашумления BPE-слогов и суффиксов внутри содержательных слов.
    """
    total_vocab = max(len(tokenizer), getattr(tokenizer, "vocab_size", 0))
    if vocab_size is None or vocab_size < total_vocab:
        vocab_size = total_vocab

    if cache_path and os.path.exists(cache_path):
        try:
            mask = torch.load(cache_path, map_location="cpu", weights_only=False)
            if torch.is_tensor(mask) and mask.shape[0] >= vocab_size:
                return mask[:vocab_size].bool()
        except Exception:
            pass

    mask = torch.zeros(vocab_size, dtype=torch.bool)
    punct_pattern = re.compile(r"^[\s\W_]+$")

    for token_id in range(vocab_size):
        try:
            text = tokenizer.decode([token_id], skip_special_tokens=False)
        except Exception:
            continue

        # 1. Специальные токены (<|endoftext|>, <|im_start|>, <|pad|> и т.д.)
        if text.startswith("<|") and text.endswith("|>"):
            mask[token_id] = True
            continue

        # 2. Проверка на пунктуацию / пробелы
        if punct_pattern.match(text):
            mask[token_id] = True
            continue

        clean_text = text.strip().lower()

        # 3. Проверка на словарные связки С УЧЕТОМ BPE-границы:
        # Токен считается изолированной связкой только если он начинается с пробела
        # (в Qwen токенизаторе это означает начало отдельного слова).
        # Если пробела в начале нет, это внутренний слог или окончание (например, "то", "со", "бы").
        if clean_text in BASE_CONNECTORS:
            is_word_start = text.startswith(" ") or text.startswith("Ġ") or text.startswith("\n")
            if is_word_start:
                mask[token_id] = True
                continue

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(mask, cache_path)

    return mask


# =========================================================================
# 3. StopwordAwareChunkPooler
# =========================================================================
class StopwordAwareChunkPooler(nn.Module):
    """
    Сжимает последовательность латентов в K раз (по умолчанию K=4),
    подавляя и зашумляя токены-связки, чтобы смысловой патч формировался
    исключительно из содержательных корней.
    Корректно обрабатывает padding_mask, исключая PAD-токены из пулинга.
    """
    def __init__(self, tokenizer, cache_dir=None, chunk_size: int = 4):
        super().__init__()
        self.chunk_size = chunk_size
        cache_path = os.path.join(cache_dir, "connector_mask.pt") if cache_dir else None
        mask_cpu = get_or_build_connector_mask(tokenizer, cache_path=cache_path)
        self.register_buffer("connector_mask", mask_cpu)

    def forward(
        self,
        z_orig: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        stopword_weight: float = 0.05,
        noise_level: float = 0.80,
        training: bool = True
    ):
        """
        z_orig: [B, T, D=1024]
        input_ids: [B, T]
        attention_mask: [B, T] (1 = активный токен, 0 = PAD)
        Возвращает:
            z_pooled [B, N = T // chunk_size, D=1024]
            (если attention_mask передан, возвращает кортеж: z_pooled, attention_mask_N)
        """
        B, T, D = z_orig.shape
        K = self.chunk_size
        assert T % K == 0, f"Sequence length T={T} must be divisible by chunk_size K={K}"
        N = T // K

        # Определяем маску связок для текущего батча
        safe_ids = input_ids.clamp(0, self.connector_mask.shape[0] - 1)
        is_conn = self.connector_mask[safe_ids]  # [B, T]

        # Базовые веса для усреднения
        weights = torch.ones((B, T), device=z_orig.device, dtype=z_orig.dtype)

        if training:
            # Случайное подавление веса связок в диапазоне [0.0, stopword_weight * 2]
            rand_w = torch.rand((B, T), device=z_orig.device, dtype=z_orig.dtype) * (stopword_weight * 2.0)
            weights = torch.where(is_conn, rand_w, weights)

            # Добавляем сферический шум к векторам связок
            eps = torch.randn_like(z_orig)
            proj = (eps * z_orig).sum(dim=-1, keepdim=True) * z_orig
            eps_ortho = safe_normalize(eps - proj, dim=-1)

            t_val = noise_level
            mu = math.cos(t_val * (math.pi / 2))
            sigma = math.sin(t_val * (math.pi / 2))
            z_noisy_conn = safe_normalize(mu * z_orig + sigma * eps_ortho, dim=-1)

            # Подменяем только токены-связки зашумленными
            is_conn_3d = is_conn.unsqueeze(-1).expand_as(z_orig)
            z_effective = torch.where(is_conn_3d, z_noisy_conn, z_orig)
        else:
            # Eval: фиксированный минимальный вес для связок, без добавления шума
            weights = torch.where(is_conn, torch.tensor(stopword_weight, device=z_orig.device, dtype=z_orig.dtype), weights)
            z_effective = z_orig

        # Обработка PAD-токенов: если передан attention_mask, вес PAD строго 0.0
        if attention_mask is not None:
            mask_bool = attention_mask.bool()
            weights = torch.where(mask_bool, weights, torch.zeros_like(weights))
            # Для чистых PAD-токенов латенты не должны искажать среднее
            z_effective = torch.where(mask_bool.unsqueeze(-1), z_effective, torch.zeros_like(z_effective))

        # Решейпим в чанки по K токенов
        z_chunks = z_effective.view(B, N, K, D)
        w_chunks = weights.view(B, N, K, 1)

        # Сумма весов внутри чанка с защитой от деления на 0
        w_sum = w_chunks.sum(dim=2, keepdim=True)
        has_weight = (w_sum > 1e-6).float()
        w_norm = torch.where(has_weight.bool(), w_chunks / w_sum.clamp(min=1e-4), torch.full_like(w_chunks, 1.0 / K))

        # Взвешенное суммирование внутри чанка + сферическая нормализация
        z_pooled = (z_chunks * w_norm).sum(dim=2)  # [B, N, D]
        z_pooled = safe_normalize(z_pooled, dim=-1)

        if attention_mask is not None:
            # Чанк считается активным, если хотя бы один токен внутри чанка активен
            mask_N = attention_mask.view(B, N, K).any(dim=2).long()
            return z_pooled, mask_N

        return z_pooled
