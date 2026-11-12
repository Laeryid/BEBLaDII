"""
Скрипт глубокого анализа многообразия Грассманиана и квантовой логики над концепт-клетками:
1. Замеры покрытия латентного пространства (Manifold Coverage Scan):
   - Эффективная размерность (Participation Ratio / Isotropy) по ковариации клеток
   - Геодезический морфинг Грассманиана (Slerp / Principal Angles Geodesic)
   - Радиус устойчивости к изотропному шуму (Perturbation Resilience Margin)
   - Безусловное сэмплирование случайных клеток (Prior Sampling Validity)
2. Булева и спектральная алгебра над концептами:
   - Семантическая композиция Эккарта–Янга и близость Грассманиана (Sim_Gr):
     ("дом", "платеж") vs "ипотека" vs контроль ("банан")
   - Булева логика цветов (Красный, Синий, Белый):
     * Дистрибутивность: (Кр ИЛИ С) И (Кр ИЛИ Б) == Кр
     * Противоречие: Кр И НЕ(Кр) == VOID
     * Исключение: (Кр ИЛИ Б) И НЕ(Б) == Кр
3. Топология и равномерность языкового словаря (Vocabulary Uniformity & Isotropy Scan):
   - Participation Ratio по словарю Tekken
   - Распределение rk_eff по токенам
   - Кластеризация и межъязыковая близость (RU vs EN)
"""

import os
import sys
import math
import argparse
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from train_phase1 import (
    GrassmannianPhase1Model,
    EnvironmentConfig,
    load_tekken_tokenizer,
)


def load_model_and_tokenizer(ckpt_path: str, device: torch.device) -> Tuple[GrassmannianPhase1Model, Any, Dict[str, Any]]:
    env = EnvironmentConfig.detect()
    tokenizer = load_tekken_tokenizer(env)

    print(f"\n[1/4] Загрузка чекпоинта: {ckpt_path} ...", flush=True)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Файл чекпоинта не найден: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    step = checkpoint.get("step", 0)
    metrics = checkpoint.get("metrics", {})
    val_loss = metrics.get("val/loss", checkpoint.get("val_loss", None))
    print(f"      -> Чекпоинт шага: {step} | val_loss: {val_loss}", flush=True)

    model = GrassmannianPhase1Model(
        vocab_size=131072,
        hidden_size=1024,
        intermediate_size=2048,
        num_axes=8,
        axis_dim=128,
        chunk_size=4,
        pad_token_id=10,
    )

    # Приоритет теневым весам EMA
    if "ema_state_dict" in checkpoint and checkpoint["ema_state_dict"] is not None:
        print("      -> Применяем сглаженные EMA shadow веса.", flush=True)
        raw_sd = checkpoint["ema_state_dict"]
        if isinstance(raw_sd, dict) and "shadow" in raw_sd:
            raw_sd = raw_sd["shadow"]
    else:
        print("      -> Применяем стандартные веса модели.", flush=True)
        raw_sd = checkpoint.get("model_state_dict", checkpoint)

    cleaned_sd = {k.replace("module.", ""): v for k, v in raw_sd.items()}
    model.load_state_dict(cleaned_sd, strict=True)
    model.to(device)
    model.eval()
    print("      -> Модель переведена в режим eval().", flush=True)

    meta = {"step": step, "val_loss": val_loss}
    return model, tokenizer, meta


def encode_phrase_to_cell(phrase: str, model: GrassmannianPhase1Model, tokenizer: Any, device: torch.device) -> torch.Tensor:
    """Кодирует текстовую фразу ровно в 1 концепт-клетку V_cell in R^{1 x 1 x 8 x 128} с честным взвешиванием PAD."""
    enc = tokenizer.encode(phrase)
    token_ids = enc.ids[:4]
    pad_id = 10
    n_real = len(token_ids)
    if len(token_ids) < 4:
        token_ids = token_ids + [pad_id] * (4 - len(token_ids))

    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
    # Честные веса: 1.0 для реальных слов, 0.001 для PAD (как в DualRegimeDataset)
    raw_weights = [1.0 if i < n_real else 0.001 for i in range(4)]
    w_sum = sum(raw_weights)
    weights = torch.tensor([[w / w_sum for w in raw_weights]], dtype=torch.float32, device=device)

    with torch.no_grad():
        V_cell, _, _, _ = model.encode_and_pool(input_tensor, weights)
    return V_cell  # (1, 1, 8, 128)


def encode_words_to_cell(words: List[str], model: GrassmannianPhase1Model, tokenizer: Any, device: torch.device) -> torch.Tensor:
    """Кодирует список отдельных слов в 1 концепт-клетку окна 4 токена, распределяя веса поровну между словами."""
    token_ids = []
    for w in words[:4]:
        enc = tokenizer.encode(w)
        if len(enc.ids) > 0:
            token_ids.append(enc.ids[0])
    pad_id = 10
    n_real = len(token_ids)
    if len(token_ids) < 4:
        token_ids = token_ids + [pad_id] * (4 - len(token_ids))

    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
    raw_weights = [1.0 if i < n_real else 0.001 for i in range(4)]
    w_sum = sum(raw_weights)
    weights = torch.tensor([[w / w_sum for w in raw_weights]], dtype=torch.float32, device=device)

    with torch.no_grad():
        V_cell, _, _, _ = model.encode_and_pool(input_tensor, weights)
    return V_cell  # (1, 1, 8, 128)


def decode_cell_to_tokens(V_cell: torch.Tensor, model: GrassmannianPhase1Model, tokenizer: Any) -> Tuple[str, List[List[Tuple[str, float]]]]:
    """
    Декодирует концепт-клетку V_cell in R^{1 x 1 x 8 x 128} в текст длины 4 токена.
    Возвращает декодированный текст и Top-3 наиболее вероятных токена для каждой позиции окна.
    """
    with torch.no_grad():
        logits = model.decode(V_cell)  # (1, 4, 131072)
        probs = F.softmax(logits[0].float(), dim=-1)
        pred_ids = logits.argmax(dim=-1)[0].cpu().tolist()
        rec_text = tokenizer.decode(pred_ids)

        topk_info = []
        for pos in range(4):
            top_probs, top_indices = torch.topk(probs[pos], k=3)
            pos_candidates = []
            for p, idx in zip(top_probs.tolist(), top_indices.tolist()):
                tok_str = tokenizer.decode([idx])
                pos_candidates.append((tok_str, p))
            topk_info.append(pos_candidates)

    return rec_text, topk_info


def safe_squash_np(V: torch.Tensor, alpha: float = 1.0 / math.sqrt(128)) -> torch.Tensor:
    """Проекция строк матрицы в единичный шар B^128."""
    norms = torch.norm(V, p=2, dim=-1, keepdim=True).clamp(min=1e-7)
    direction = V / norms
    scale = torch.tanh(alpha * norms)
    return (direction * scale).contiguous()


# ==============================================================================
# МОДУЛЬ 1: КВАНТОВО-ГЕОМЕТРИЧЕСКАЯ АЛГЕБРА НАД КОНЦЕПТАМИ
# ==============================================================================

def get_orthonormal_basis(V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    V: (8, 128)
    Возвращает:
      Q: (128, 8) ортонормированный базис подпространства
      S: (8,) сингулярные числа осей
    """
    V_f32 = V.float()
    U, S, Vh = torch.linalg.svd(V_f32, full_matrices=False)
    Q = Vh.T  # (128, 8)
    return Q, S


def grassmann_similarity(V_A: torch.Tensor, V_B: torch.Tensor) -> Tuple[float, List[float]]:
    """
    Метрика близости Грассманиана между концепт-клетками A и B:
    1. Перекрытие проекторов матрицы Грама:
       Sim_Gr = Tr(rho_A @ rho_B) / (||rho_A||_F * ||rho_B||_F) in [0, 1]
       где rho = V^T @ V in R^{128 x 128}.
       Без вычисления матриц 128x128: ||V_A @ V_B^T||_F^2 / (||V_A @ V_A^T||_F * ||V_B @ V_B^T||_F).
    2. Главные углы: cos theta_1..8
    """
    v_a = V_A[0, 0].float()  # (8, 128)
    v_b = V_B[0, 0].float()  # (8, 128)

    # Матрицы Грама
    G_a = torch.matmul(v_a, v_a.T)  # (8, 8)
    G_b = torch.matmul(v_b, v_b.T)  # (8, 8)
    frob_a = torch.norm(G_a, p="fro")
    frob_b = torch.norm(G_b, p="fro")

    # Перекрытие rho_A и rho_B: Tr(V_A^T V_A V_B^T V_B) = ||V_A V_B^T||_F^2
    M_ab = torch.matmul(v_a, v_b.T)  # (8, 8)
    overlap = (M_ab ** 2).sum()
    sim_gr = (overlap / (frob_a * frob_b + 1e-7)).clamp(0.0, 1.0).item()

    # Главные углы между подпространствами
    Q_A, _ = get_orthonormal_basis(v_a)
    Q_B, _ = get_orthonormal_basis(v_b)
    M_ortho = torch.matmul(Q_A.T, Q_B)
    _, S_ortho, _ = torch.linalg.svd(M_ortho)
    cos_thetas = S_ortho.clamp(0.0, 1.0).tolist()

    return sim_gr, cos_thetas


def concept_compose(V_A: torch.Tensor, V_B: torch.Tensor, w_a: float = 0.5, w_b: float = 0.5) -> torch.Tensor:
    """
    Аддитивная спектральная композиция понятий A и B (по пулингу Эккарта–Янга):
    W_tilde = [sqrt(w_a) * V_A; sqrt(w_b) * V_B] in R^{16 x 128}
    Находит топ-8 осей взвешенного союза.
    """
    v_a = V_A[0, 0].float() * math.sqrt(w_a)
    v_b = V_B[0, 0].float() * math.sqrt(w_b)
    combined = torch.cat([v_a, v_b], dim=0)  # (16, 128)
    U, S, Vh = torch.linalg.svd(combined, full_matrices=False)
    top_axes = Vh[:8] * S[:8, None]
    res_cell = safe_squash_np(top_axes)
    return res_cell.unsqueeze(0).unsqueeze(0).to(V_A.dtype)


def concept_or(V_A: torch.Tensor, V_B: torch.Tensor) -> torch.Tensor:
    """
    Операция ИЛИ (A OR B) — Прямая сумма подпространств A + B.
    Объединяет оси двух клеток (16 x 128) и находит топ-8 главных компонент через SVD.
    """
    v_a = V_A[0, 0].float()
    v_b = V_B[0, 0].float()
    combined = torch.cat([v_a, v_b], dim=0)  # (16, 128)

    # SVD по теореме Эккарта-Янга
    U, S, Vh = torch.linalg.svd(combined, full_matrices=False)
    top_axes = Vh[:8] * S[:8, None]
    res_cell = safe_squash_np(top_axes)
    return res_cell.unsqueeze(0).unsqueeze(0).to(V_A.dtype)


def concept_and(V_A: torch.Tensor, V_B: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
    """
    Операция И (A AND B) — Пересечение подпространств через главные углы (Principal Angles).
    Находит общие семантические измерения, в которых A и B согласованы.
    """
    v_a = V_A[0, 0].float()
    v_b = V_B[0, 0].float()
    Q_A, S_A = get_orthonormal_basis(v_a)  # (128, 8)
    Q_B, S_B = get_orthonormal_basis(v_b)  # (128, 8)

    # Матрица взаимных скалярных произведений
    M = torch.matmul(Q_A.T, Q_B)  # (8, 8)
    U_m, S_m, Vh_m = torch.linalg.svd(M)
    cos_thetas = S_m.clamp(0.0, 1.0).tolist()  # косинусы главных углов между подпространствами

    # Канонические векторы пересечения
    X = torch.matmul(Q_A, U_m)     # (128, 8)
    Y = torch.matmul(Q_B, Vh_m.T)   # (128, 8)
    # Симметризованный консенсус, взвешенный силой пересечения
    Consensus = 0.5 * (X + Y) * S_m[None, :]  # (128, 8)
    top_axes = Consensus.T  # (8, 128)

    res_cell = safe_squash_np(top_axes)
    return res_cell.unsqueeze(0).unsqueeze(0).to(V_A.dtype), cos_thetas


def concept_not_relative(V_Universe: torch.Tensor, V_Exclude: torch.Tensor) -> torch.Tensor:
    """
    Операция НЕ в контексте Universe: (Universe AND NOT Exclude).
    Проецирует подпространство Universe на ортогональное дополнение к Exclude.
    """
    v_u = V_Universe[0, 0].float()
    v_e = V_Exclude[0, 0].float()
    Q_E, _ = get_orthonormal_basis(v_e)  # (128, 8)
    
    # Проектор на ортогональное дополнение к Exclude: P_perp = I - Q_E @ Q_E^T
    I_128 = torch.eye(128, device=v_u.device, dtype=torch.float32)
    P_perp = I_128 - torch.matmul(Q_E, Q_E.T)  # (128, 128)

    # Проецируем оси Вселенной
    projected = torch.matmul(v_u, P_perp)  # (8, 128)
    U, S, Vh = torch.linalg.svd(projected, full_matrices=False)
    top_axes = Vh[:8] * S[:8, None]
    res_cell = safe_squash_np(top_axes)
    return res_cell.unsqueeze(0).unsqueeze(0).to(V_Universe.dtype)


def run_logic_experiments(model: GrassmannianPhase1Model, tokenizer: Any, device: torch.device):
    print("\n" + "=" * 80)
    print("=== ЭКСПЕРИМЕНТ 1: СЕМАНТИЧЕСКАЯ БЛИЗОСТЬ И КОМПОЗИЦИЯ ПОНЯТИЙ ===")
    print("=" * 80)

    triplets = [
        ("ипотека", ["дом", "платеж"], "банан", "Ипотека vs (Дом + Платеж) vs Контроль (Банан)"),
        ("программа", ["компьютер", "код"], "помидор", "Программа vs (Компьютер + Код) vs Контроль (Помидор)"),
        ("лекарство", ["врач", "болезнь"], "трактор", "Лекарство vs (Врач + Болезнь) vs Контроль (Трактор)"),
        ("пар", ["огонь", "вода"], "карандаш", "Пар vs (Огонь + Вода) vs Контроль (Карандаш)"),
    ]

    for target_word, source_words, random_ctrl, title in triplets:
        print(f"\n--- Тест: {title} ---")
        c_target = encode_phrase_to_cell(target_word, model, tokenizer, device)
        c_composed = encode_words_to_cell(source_words, model, tokenizer, device)
        c_random = encode_phrase_to_cell(random_ctrl, model, tokenizer, device)

        rec_target, _ = decode_cell_to_tokens(c_target, model, tokenizer)
        rec_composed, topk_comp = decode_cell_to_tokens(c_composed, model, tokenizer)
        rec_random, _ = decode_cell_to_tokens(c_random, model, tokenizer)

        print(f"  * Таргет (\"{target_word}\") декодируется как: \"{rec_target.strip()}\"")
        print(f"  * Композиция {source_words} декодируется как: \"{rec_composed.strip()}\"")
        print(f"  * Контроль (\"{random_ctrl}\") декодируется как: \"{rec_random.strip()}\"")

        # Замер Грассманианова сходства:
        sim_target_comp, angles_comp = grassmann_similarity(c_target, c_composed)
        sim_target_rand, angles_rand = grassmann_similarity(c_target, c_random)

        cos_comp = [round(c, 3) for c in angles_comp[:4]]
        cos_rand = [round(c, 3) for c in angles_rand[:4]]

        print(f"  * Сходство Грассманиана Sim_Gr(Таргет, Композиция): {sim_target_comp:.4f} | Топ-4 угла: {cos_comp}")
        print(f"  * Сходство Грассманиана Sim_Gr(Таргет, Контроль):    {sim_target_rand:.4f} | Топ-4 угла: {cos_rand}")
        
        ratio = (sim_target_comp / (sim_target_rand + 1e-7))
        if ratio > 1.2:
            print(f"  -> ВЫВОД: Семантическое сближение подтверждено! Сходство выше контрольного в {ratio:.2f} раз.")
        else:
            print(f"  -> ВЫВОД: Сходство сопоставимо с базовым шумом (ratio={ratio:.2f}). Требуется диффузионное ядро.")

    print("\n" + "=" * 80)
    print("=== ЭКСПЕРИМЕНТ 2: БУЛЕВА АЛГЕБРА ЦВЕТОВ (С ЧЕСТНЫМИ ВЕСАМИ БЕЗ PAD-ИСКАЖЕНИЙ) ===")
    print("=" * 80)

    c_red = encode_phrase_to_cell("красный", model, tokenizer, device)
    c_blue = encode_phrase_to_cell("синий", model, tokenizer, device)
    c_white = encode_phrase_to_cell("белый", model, tokenizer, device)

    rec_red, _ = decode_cell_to_tokens(c_red, model, tokenizer)
    rec_blue, _ = decode_cell_to_tokens(c_blue, model, tokenizer)
    rec_white, _ = decode_cell_to_tokens(c_white, model, tokenizer)

    print(f"Базовые концепты:")
    print(f"  A [Красный]: \"{rec_red.strip()}\"")
    print(f"  B [Синий]:   \"{rec_blue.strip()}\"")
    print(f"  C [Белый]:   \"{rec_white.strip()}\"")

    # 1. (Кр И С) И (Кр И Б)
    print("\n1. Формула: (Красный AND Синий) AND (Красный AND Белый)")
    c_rb, _ = concept_and(c_red, c_blue)
    c_rw, _ = concept_and(c_red, c_white)
    c_res1, angles1 = concept_and(c_rb, c_rw)
    rec1, _ = decode_cell_to_tokens(c_res1, model, tokenizer)
    print(f"   -> Результат декодирования: \"{rec1.strip()}\" | Главные углы: {[round(c, 3) for c in angles1[:4]]}")

    # 2. Дистрибутивность: (Красный OR Синий) AND (Красный OR Белый) -> должно сходиться к Красному
    print("\n2. Формула дистрибутивности: (Красный OR Синий) AND (Красный OR Белый) [теория = Красный]")
    c_r_or_b = concept_or(c_red, c_blue)
    c_r_or_w = concept_or(c_red, c_white)
    c_res2, angles2 = concept_and(c_r_or_b, c_r_or_w)
    rec2, _ = decode_cell_to_tokens(c_res2, model, tokenizer)
    sim_distrib, _ = grassmann_similarity(c_res2, c_red)
    print(f"   -> Результат декодирования: \"{rec2.strip()}\" | Sim_Gr с Красным: {sim_distrib:.4f}")

    # 3. Закон противоречия: Красный AND NOT(Красный) -> пустота (VOID)
    print("\n3. Закон противоречия: Красный AND NOT(Красный) [теория = Пустое множество / VOID]")
    c_annihilate = concept_not_relative(c_red, c_red)
    rec3, _ = decode_cell_to_tokens(c_annihilate, model, tokenizer)
    energy_annihilate = torch.norm(c_annihilate).item()
    print(f"   -> Результат декодирования: \"{rec3.strip()}\" | Остаточная энергия: {energy_annihilate:.6f}")

    # 4. Закон исключения: (Красный OR Белый) AND NOT(Белый) -> Красный
    print("\n4. Закон исключения: (Красный OR Белый) AND NOT(Белый) [теория = Красный]")
    c_rw_union = concept_or(c_red, c_white)
    c_recovered_red = concept_not_relative(c_rw_union, c_white)
    rec4, _ = decode_cell_to_tokens(c_recovered_red, model, tokenizer)
    sim_excl, _ = grassmann_similarity(c_recovered_red, c_red)
    print(f"   -> Результат декодирования: \"{rec4.strip()}\" | Sim_Gr с Красным: {sim_excl:.4f}")


# ==============================================================================
# МОДУЛЬ 2: ЗАМЕРЫ ПОКРЫТИЯ ПРОСТРАНСТВА ДЕКОДЕРОМ (MANIFOLD COVERAGE)
# ==============================================================================

def run_manifold_coverage_scan(model: GrassmannianPhase1Model, tokenizer: Any, device: torch.device, num_batches: int = 30):
    print("\n" + "=" * 80)
    print("=== ЭКСПЕРИМЕНТ 3: ЭФФЕКТИВНАЯ РАЗМЕРНОСТЬ И ИЗОТРОПИЯ МНОГООБРАЗИЯ ===")
    print("=" * 80)

    env = EnvironmentConfig.detect()
    val_path = env.val_data_path

    if not os.path.exists(val_path):
        print(f"[Warning] val.parquet не найден по пути {val_path}. Пропускаем ковариационный тест.", flush=True)
        return

    import pandas as pd
    df = pd.read_parquet(val_path)
    col = "text" if "text" in df.columns else df.columns[0]
    samples = df[col].dropna().tolist()[:num_batches]

    collected_cells = []
    pad_id = 10

    for text in samples:
        enc = tokenizer.encode(text)
        tokens = enc.ids[:128]
        rem = len(tokens) % 4
        if rem != 0:
            tokens += [pad_id] * (4 - rem)
        if len(tokens) == 0:
            continue
        inp = torch.tensor([tokens], dtype=torch.long, device=device)
        w = torch.ones(1, len(tokens), dtype=torch.float32, device=device)
        with torch.no_grad():
            V_c, _, _, _ = model.encode_and_pool(inp, w)
            collected_cells.append(V_c.view(-1, 128).float().cpu())

    if not collected_cells:
        print("[Warning] Не удалось собрать латентные клетки.")
        return

    # Матрица всех осей: (M, 128)
    all_axes = torch.cat(collected_cells, dim=0)  # (M, 128)
    M, D = all_axes.shape
    print(f"Собрано {M} латентных осей концепт-клеток из {len(samples)} валидационных текстов.")

    # Ковариация 128 измерений
    centered = all_axes - all_axes.mean(dim=0, keepdim=True)
    cov = torch.matmul(centered.T, centered) / (M - 1)  # (128, 128)
    eigs = torch.linalg.eigvalsh(cov).clamp(min=1e-8)
    eigs = eigs.flip(dims=[-1])  # по убыванию

    # Participation ratio (эффективная размерность)
    sum_eigs = eigs.sum().item()
    sum_sq_eigs = (eigs ** 2).sum().item()
    dim_eff = (sum_eigs ** 2) / (sum_sq_eigs + 1e-9)

    # Профиль концентрации энергии
    cum_energy = (torch.cumsum(eigs, dim=0) / sum_eigs * 100.0).tolist()

    print(f"\n[Результаты спектрального анализа Gr(8, 128)]:")
    print(f"  * Эффективная размерность (Participation Ratio): {dim_eff:.2f} из 128 каналов")
    print(f"  * Концентрация энергии спектра:")
    print(f"      - Топ-8  измерений:  {cum_energy[7]:5.1f}%")
    print(f"      - Топ-16 измерений:  {cum_energy[15]:5.1f}%")
    print(f"      - Топ-32 измерений:  {cum_energy[31]:5.1f}%")
    print(f"      - Топ-64 измерений:  {cum_energy[63]:5.1f}%")
    print(f"      - Все 128 измерений: 100.0%")

    if dim_eff > 60:
        print("  -> ВЫВОД: Пространство высоко-изотропно! Декодер задействует широкий объем шара B^128.")
    elif dim_eff > 25:
        print("  -> ВЫВОД: Умеренная изотропия, есть запас по емкости.")
    else:
        print("  -> ВЫВОД: Низкая эффективная размерность (анизотропный сплющенный эллипсоид).")


def run_perturbation_and_prior_tests(model: GrassmannianPhase1Model, tokenizer: Any, device: torch.device):
    print("\n" + "=" * 80)
    print("=== ЭКСПЕРИМЕНТ 4: РАДИУС УСТОЙЧИВОСТИ К ШУМУ И PRIOR SAMPLING ===")
    print("=" * 80)

    base_phrase = "искусственный интеллект"
    c_base = encode_phrase_to_cell(base_phrase, model, tokenizer, device)
    rec_orig, _ = decode_cell_to_tokens(c_base, model, tokenizer)
    print(f"Базовая клетка (\"{base_phrase}\") декодируется как: \"{rec_orig.strip()}\"")

    sigmas = [0.05, 0.10, 0.20, 0.35, 0.50]
    print("\n1. Инъекция изотропного гауссова шума V_cell + sigma * N(0, I):")
    for s in sigmas:
        noise = torch.randn_like(c_base) * s
        c_noisy = safe_squash_np(c_base + noise)
        rec_noisy, _ = decode_cell_to_tokens(c_noisy, model, tokenizer)
        print(f"   * Шум sigma = {s:4.2f} -> Декодирование: \"{rec_noisy.strip()}\"")

    print("\n2. Безусловное сэмплирование (Prior Sampling Validity):")
    print("   Генерируем 3 случайные клетки из шара B^128 (без энкодера) и подаем в декодер:")
    for i in range(3):
        rand_v = torch.randn(1, 1, 8, 128, device=device)
        rand_cell = safe_squash_np(rand_v)
        rec_rand, _ = decode_cell_to_tokens(rand_cell, model, tokenizer)
        print(f"   * Сэмпл #{i+1}: \"{rec_rand.strip()}\"")


def run_vocabulary_distribution_scan(model: GrassmannianPhase1Model, tokenizer: Any, device: torch.device, num_tokens: int = 4000):
    print("\n" + "=" * 80)
    print(f"=== ЭКСПЕРИМЕНТ 5: ТОПОЛОГИЯ И РАВНОМЕРНОСТЬ СЛОВАРЯ ({num_tokens} ТОКЕНОВ) ===")
    print("=" * 80)

    vocab = tokenizer.get_vocab()
    ru_ids = []
    en_ids = []
    other_ids = []

    # Быстро классифицируем первые 20 000 токенов через decode
    max_scan = min(len(vocab), 25000)
    for tid in range(max_scan):
        text = tokenizer.decode([tid]).strip()
        if not text:
            other_ids.append(tid)
            continue
        has_cyrillic = any('\u0400' <= ch <= '\u04FF' for ch in text)
        has_latin = any(('a' <= ch <= 'z') or ('A' <= ch <= 'Z') for ch in text)
        if has_cyrillic:
            ru_ids.append(tid)
        elif has_latin:
            en_ids.append(tid)
        else:
            other_ids.append(tid)

    print(f"Разбиение первых {max_scan} токенов Tekken:")
    print(f"  * Русские токены (кириллица): {len(ru_ids)}")
    print(f"  * Английские токены (латиница): {len(en_ids)}")
    print(f"  * Служебные/символы/числа: {len(other_ids)}")

    n_per_lang = min(len(ru_ids), len(en_ids), num_tokens // 2)
    sample_ru = torch.tensor(ru_ids[:n_per_lang], dtype=torch.long, device=device)
    sample_en = torch.tensor(en_ids[:n_per_lang], dtype=torch.long, device=device)
    all_sample_ids = torch.cat([sample_ru, sample_en], dim=0)
    all_sample_ids_2d = all_sample_ids.unsqueeze(0)  # (1, S)

    # Прогоняем токены через статический энкодер (3D тензор: 1, S, 1024)
    with torch.no_grad():
        x_emb = model.embed_norm(model.embeddings(all_sample_ids_2d))  # (1, S, 1024)
        h_sem, h_pos = model.lexical_mlp(x_emb)  # (1, S, 8, 96), (1, S, 8, 16)
        h_syn_zero = torch.zeros(1, all_sample_ids.shape[0], model.num_axes, 16, device=device, dtype=h_sem.dtype)
        mu, _ = model.vae_head(h_sem, h_syn_zero, h_pos)  # (1, S, 8, 128)
        V_tok, rk_eff, _ = model.normalizer(mu)  # (1, S, 8, 128)
        V_tok = V_tok.squeeze(0)  # (S, 8, 128)
        rk_eff = rk_eff.squeeze(0)  # (S,)

    S, K, D = V_tok.shape
    axes = V_tok.view(S * K, D).float()

    # 1. Спектральный анализ ковариации словаря (Participation Ratio)
    centered = axes - axes.mean(dim=0, keepdim=True)
    cov = torch.matmul(centered.T, centered) / (axes.shape[0] - 1)
    eigs = torch.linalg.eigvalsh(cov).clamp(min=1e-8).flip(dims=[-1])
    sum_eigs = eigs.sum().item()
    sum_sq_eigs = (eigs ** 2).sum().item()
    dim_eff = (sum_eigs ** 2) / (sum_sq_eigs + 1e-9)
    cum_energy = (torch.cumsum(eigs, dim=0) / sum_eigs * 100.0).tolist()

    print(f"\n[Спектральный анализ словарного пространства (Gr(8, 128))]:")
    print(f"  * Эффективная размерность словаря (Participation Ratio): {dim_eff:.2f} из 128 осей")
    print(f"  * Доли спектральной энергии:")
    print(f"      - Топ-8 осей:   {cum_energy[7]:5.1f}%")
    print(f"      - Топ-16 осей:  {cum_energy[15]:5.1f}%")
    print(f"      - Топ-32 осей:  {cum_energy[31]:5.1f}%")
    print(f"      - Топ-64 осей:  {cum_energy[63]:5.1f}%")

    # 2. Распределение эффективных рангов токенов rk_eff
    rk_flat = rk_eff.view(-1).cpu()
    rk_mean = rk_flat.mean().item()
    rk_min = rk_flat.min().item()
    rk_max = rk_flat.max().item()
    pct_degenerate = (rk_flat < 1.5).float().mean().item() * 100.0
    pct_rich = (rk_flat >= 2.0).float().mean().item() * 100.0

    print(f"\n[Распределение эффективных рангов токенов (rk_eff)]:")
    print(f"  * Средний rk_eff: {rk_mean:.3f} (min: {rk_min:.3f}, max: {rk_max:.3f})")
    print(f"  * Токенов с рангом < 1.5 (вырожденные): {pct_degenerate:.2f}%")
    print(f"  * Токенов с рангом >= 2.0 (богатый спектр): {pct_rich:.2f}%")

    # 3. Межъязыковая геометрия и Uniformity
    V_ru = V_tok[:n_per_lang].float()
    V_en = V_tok[n_per_lang:].float()

    n_pairs = min(300, n_per_lang)
    idx_a = torch.randint(0, n_per_lang, (n_pairs,))
    idx_b = torch.randint(0, n_per_lang, (n_pairs,))

    # RU - RU перекрытие:
    M_ruru = torch.matmul(V_ru[idx_a], V_ru[idx_b].transpose(-2, -1))
    sim_ruru = (M_ruru ** 2).sum(dim=[-2, -1]) / (torch.norm(torch.matmul(V_ru[idx_a], V_ru[idx_a].transpose(-2, -1)), dim=[-2, -1]) * torch.norm(torch.matmul(V_ru[idx_b], V_ru[idx_b].transpose(-2, -1)), dim=[-2, -1]) + 1e-7)

    # EN - EN перекрытие:
    M_enen = torch.matmul(V_en[idx_a], V_en[idx_b].transpose(-2, -1))
    sim_enen = (M_enen ** 2).sum(dim=[-2, -1]) / (torch.norm(torch.matmul(V_en[idx_a], V_en[idx_a].transpose(-2, -1)), dim=[-2, -1]) * torch.norm(torch.matmul(V_en[idx_b], V_en[idx_b].transpose(-2, -1)), dim=[-2, -1]) + 1e-7)

    # RU - EN перекрытие (межъязыковое):
    M_ruen = torch.matmul(V_ru[idx_a], V_en[idx_b].transpose(-2, -1))
    sim_ruen = (M_ruen ** 2).sum(dim=[-2, -1]) / (torch.norm(torch.matmul(V_ru[idx_a], V_ru[idx_a].transpose(-2, -1)), dim=[-2, -1]) * torch.norm(torch.matmul(V_en[idx_b], V_en[idx_b].transpose(-2, -1)), dim=[-2, -1]) + 1e-7)

    unif_val = torch.log(torch.exp(2.0 * sim_ruen).mean()).item()

    print(f"\n[Межъязыковая геометрия и Uniformity по Wang & Isola]:")
    print(f"  * Среднее перекрытие RU - RU: {sim_ruru.mean().item():.4f}")
    print(f"  * Среднее перекрытие EN - EN: {sim_enen.mean().item():.4f}")
    print(f"  * Межъязыковое перекрытие RU - EN: {sim_ruen.mean().item():.4f}")
    print(f"  * Метрика Grassmannian Uniformity (t=2.0): {unif_val:.4f}")
    if sim_ruen.mean().item() < 0.05:
        print("  -> ВЫВОД: Языки занимают изолированные кластеры на Грассманиане.")
    else:
        print("  -> ВЫВОД: Языки разделяют общее сбалансированное многообразие.")


def main():
    parser = argparse.ArgumentParser(description="Диагностика покрытия пространства и квантовой логики над концептами Phase 1")
    parser.add_argument("--ckpt", type=str, default="experiments/phase 1/checkpoints/phase1_best_model.pt", help="Путь к чекпоинту")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--skip-coverage", action="store_true", help="Пропустить сканирование валидационного датасета")
    parser.add_argument("--skip-vocab", action="store_true", help="Пропустить сканирование словаря Tekken")
    args = parser.parse_args()

    dev = torch.device(args.device)
    model, tokenizer, meta = load_model_and_tokenizer(args.ckpt, dev)

    # 1. Логические операции и семантическая композиция над концепт-клетками
    run_logic_experiments(model, tokenizer, dev)

    # 2. Стресс-тест на шум и Prior Sampling
    run_perturbation_and_prior_tests(model, tokenizer, dev)

    # 3. Полное сканирование изотропии и размерности покрытия по валидационному датасету
    if not args.skip_coverage:
        run_manifold_coverage_scan(model, tokenizer, dev)

    # 4. Сканирование топологии и равномерности словаря Tekken
    if not args.skip_vocab:
        run_vocabulary_distribution_scan(model, tokenizer, dev)

    print("\n" + "=" * 80)
    print("=== ДИАГНОСТИЧЕСКОЕ ИССЛЕДОВАНИЕ УСПЕШНО ЗАВЕРШЕНО ===")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
