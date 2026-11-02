"""Комплексная оценка чекпоинта Phase 6 в 4 актуальных режимах (ADR 116, ADR 117).

Актуальные режимы (void_aware_dprox=True включен по умолчанию):
1. baseline: Свободный холст без обрезки (SmartCrop=none, VoidDProx=ON).
2. adaptive_buf16: Флагман ADR 116 — Adaptive SmartCrop (стабилизация границы при t <= 0.70 + 16 токенов войд-буфера).
3. soft: Мягкий SmartCrop (tau=0.90) с фиксацией войд-хвоста без изменения физической длины внимания.
4. spec_adaptive: Продвинутый сэмплер Specificity-Aware LR-Gated Guidance (arXiv:2610.00395) + Adaptive SmartCrop.

Модель загружается 1 раз в память, после чего выполняется запуск выбранных режимов.
Метрики вычисляются на базе GPT-2 (PPL), N-граммного разнообразия (distinct-1/2/3),
доли копирования слов промпта (для ModernLatentDecoder и LatentDict NN) и финального DProx.
"""
import argparse
import datetime
import html
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PHASE6_DIR = PROJECT_ROOT / "experiments" / "phase 6"
if str(PHASE6_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE6_DIR))

# Импорт необходимых классов и функций из evaluate_phase6_metrics
from evaluate_phase6_metrics import (
    BEBLaDIIPhase6Eval,
    ModernLatentDecoder,
    generate_html,
    normalize_key,
    run_generation,
    validate_environment,
)

DEFAULT_PROMPT = "How does your generation mechanism contrast with traditional next-token prediction models?"

# Базовый набор 4 режимов генерации (ADR 116, 117)
MODES = [
    {
        "name": "baseline",
        "desc": "baseline (SmartCrop=none, VoidDProx=ON)",
        "suffix": "baseline",
        "kwargs": {"smartcrop": "none", "void_aware_dprox": True, "sampler": "hybrid"},
    },
    {
        "name": "adaptive_buf16",
        "desc": "adaptive SmartCrop + 16 void buffer (ADR 116 default)",
        "suffix": "smartcrop_adaptive_buffered16",
        "kwargs": {"smartcrop": "adaptive", "smartcrop_buffer": 16, "void_aware_dprox": True, "sampler": "hybrid"},
    },
    {
        "name": "soft",
        "desc": "soft SmartCrop (tau=0.90)",
        "suffix": "smartcrop_soft",
        "kwargs": {"smartcrop": "soft", "smartcrop_tau": 0.90, "void_aware_dprox": True, "sampler": "hybrid"},
    },
    {
        "name": "spec_adaptive",
        "desc": "spec-adaptive (Specificity LR-Gated Guidance + Adaptive Crop)",
        "suffix": "spec_adaptive",
        "kwargs": {
            "smartcrop": "adaptive",
            "smartcrop_buffer": 16,
            "sampler": "specificity_hybrid",
            "spec_tau": 0.75,
            "spec_temp": 0.35,
            "void_aware_dprox": True,
        },
    },
]


def find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    """Ищет самый свежий чекпоинт Phase 6 по номеру шага в имени файла."""
    if not ckpt_dir.exists():
        return None
    candidates = list(ckpt_dir.glob("planB_phase6_checkpoints_phase6_ca_layers_step_*.pth"))
    if not candidates:
        candidates = list(ckpt_dir.glob("*.pth"))
    if not candidates:
        return None

    def _extract_step(p: Path) -> int:
        m = re.search(r"step_(\d+)", p.name)
        return int(m.group(1)) if m else -1

    candidates.sort(key=_extract_step, reverse=True)
    return candidates[0]


def extract_both_from_html(path: Path) -> tuple[str, str, float | None]:
    """Резервное извлечение текстов и DProx из сохраненного HTML-файла."""
    raw = html.unescape(path.read_text(encoding="utf-8"))
    m_dec = re.search(r"=== FINAL ANSWER \(ModernLatentDecoder\) ===\s*(.*?)\s*=== FINAL ANSWER \(LatentDict", raw, re.S)
    m_nn = re.search(r"=== FINAL ANSWER \(LatentDict NN\) ===\s*(.*?)(?:<div|<pre|===|$)", raw, re.S)
    m_dprox = re.findall(r"DProx=([0-9.]+)", raw)
    final_dprox = float(m_dprox[-1]) if m_dprox else None

    txt_dec = re.sub(r"<[^>]+>", "", m_dec.group(1)).strip() if m_dec else ""
    txt_nn = re.sub(r"<[^>]+>", "", m_nn.group(1)).strip() if m_nn else ""
    return txt_dec, txt_nn, final_dprox


def strip_void(text: str) -> tuple[str, int]:
    """Удаляет маркеры <|void|> и <|void xN|>, возвращая очищенный текст и счетчик войд-токенов."""
    n_void = sum(int(x) for x in re.findall(r"<\|void x(\d+)\|>", text))
    n_void += len(re.findall(r"<\|void\|>", text))
    clean = re.sub(r"<\|void(?: x\d+)?\|>", " ", text)
    return clean.strip(), n_void


def distinct_n(words: list[str], n: int) -> float:
    """Вычисляет коэффициент лексического разнообразия distinct-n."""
    if not words or len(words) < n:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / max(1, len(grams))


def main():
    parser = argparse.ArgumentParser(description="Evaluation Suite for Phase 6 checkpoints in 4 modern modes (ADR 116, 117)")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint (auto-detects latest if omitted)")
    parser.add_argument("--tag", type=str, default=None, help="Tag for output filenames (e.g. 111k)")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Prompt text for evaluation")
    parser.add_argument("--modes", type=str, default="all",
                        help="Comma-separated mode names to run (baseline,adaptive_buf16,soft,spec_adaptive) or 'all'")
    parser.add_argument("--max_len", type=int, default=500, help="Canvas length (default: 500)")
    parser.add_argument("--steps", type=int, default=25, help="Diffusion steps (default: 25)")
    parser.add_argument("--cfg_max", type=float, default=4.0, help="Max CFG scale at t=1.0 (default: 4.0)")
    parser.add_argument("--cfg_min", type=float, default=1.0, help="Min CFG scale at t=0.04 (default: 1.0)")
    parser.add_argument("--cfg_power", type=float, default=1.0, help="CFG decay schedule power exponent (default: 1.0)")
    parser.add_argument("--re_noise_t", type=float, default=0.88, help="Target noise level for anchor jump (default: 0.88)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for generation (default: 42)")
    args = parser.parse_args()

    ckpt_dir = PROJECT_ROOT / "experiments" / "phase 6" / "checkpoints"
    if args.ckpt:
        target_ckpt = Path(args.ckpt).resolve()
    else:
        latest = find_latest_checkpoint(ckpt_dir)
        if latest is None:
            print(f"[ERROR] No Phase 6 checkpoints found in {ckpt_dir}!")
            sys.exit(1)
        target_ckpt = latest

    if not target_ckpt.exists():
        print(f"[ERROR] Checkpoint not found: {target_ckpt}")
        sys.exit(1)

    if args.tag:
        tag = args.tag
    else:
        m = re.search(r"step_(\d+)", target_ckpt.name)
        if m:
            step_num = int(m.group(1))
            tag = f"{step_num // 1000}k" if step_num >= 1000 and step_num % 1000 == 0 else f"{step_num}"
        else:
            tag = target_ckpt.stem

    # Фильтрация выбранных режимов
    modes_by_name = {m["name"]: m for m in MODES}
    if args.modes.strip().lower() in ["all", "default"]:
        selected_modes = MODES
    else:
        chosen_names = [x.strip() for x in args.modes.split(",") if x.strip()]
        selected_modes = []
        for cn in chosen_names:
            if cn in modes_by_name:
                selected_modes.append(modes_by_name[cn])
            else:
                print(f"[WARNING] Unknown mode '{cn}', available: {list(modes_by_name.keys())}")
        if not selected_modes:
            print("[ERROR] No valid modes selected! Aborting.")
            sys.exit(1)

    config = {
        'qwen_path': "Qwen/Qwen2.5-1.5B",
        'modernbert_path': "answerdotai/ModernBERT-large",
        'encoder_path': os.path.join(PROJECT_ROOT, "experiments", "phase 1", "planB_phase1_checkpoints_phase1_vae_step_20000.pth"),
        'phase4_path': os.path.join(PROJECT_ROOT, "experiments", "phase 4", "local_checkpoints", "phase4_step_85995.pth"),
        'phase2_decoder_path': os.path.join(PROJECT_ROOT, "experiments", "phase 2", "planB_phase2_checkpoints_decoder_step_9000.pth"),
        'sep_token_path': os.path.join(PROJECT_ROOT, "storage", "components", "sep_token.pt"),
        'void_token_path': os.path.join(PROJECT_ROOT, "storage", "components", "void_token.pt"),
        'phase6_ckpt': str(target_ckpt),
    }

    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Starting {tag} Evaluation Suite ({len(selected_modes)} Modes)...")
    print(f"Target Checkpoint: {target_ckpt}")
    print(f"Evaluation Prompt: '{args.prompt}'")
    print(f"Active Modes     : {[m['name'] for m in selected_modes]}")

    dict_primary = os.path.join(PROJECT_ROOT, "experiments", "phase 5", "local", "latent_dict.pt")
    dict_fallback = os.path.join(PROJECT_ROOT, "BEBLaDII-planB-Phase5-Data", "latent_dict.pt")
    dict_paths = [dict_primary, dict_fallback]
    ds_path = os.path.join(PROJECT_ROOT, "BEBLaDII-planB-Phase6-Data", "phase 6", "data", "train_phase6.parquet")

    all_ok, resolved_dict = validate_environment(config, dict_paths, ds_path, verbose=True)
    if not all_ok:
        print("[ERROR] Environment validation failed!")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    # 1. Загрузка токенизатора и словаря латентов
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Loading Tokenizer & LatentDict...")
    tokenizer = AutoTokenizer.from_pretrained(config['qwen_path'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    latent_dict = torch.load(resolved_dict, map_location=device).float()

    # 2. Загрузка модели BEBLaDIIPhase6Eval (ADR 116, 117 с GeGLU и prompt cache)
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Loading Phase 6 Model ({tag})...")
    t0_load = time.time()
    model = BEBLaDIIPhase6Eval(config).to(device)
    print(f"Phase 6 Model loaded in {time.time() - t0_load:.1f}s.")

    # 3. Загрузка ModernLatentDecoder
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Loading ModernLatentDecoder...")
    decoder = ModernLatentDecoder(
        latent_dim=1024,
        qwen_dim=1536,
        num_layers=3,
        dus_weights_path=config['phase4_path']
    ).to(device)
    ckpt_dec = torch.load(config['phase2_decoder_path'], map_location=device, weights_only=False)
    st_dec = ckpt_dec.get("decoder", ckpt_dec)
    clean_st_dec = {normalize_key(k).replace("decoder.", ""): v for k, v in st_dec.items()}
    decoder.load_state_dict(clean_st_dec, strict=False)
    decoder.eval()
    print("Decoder ready.")

    out_dir = PROJECT_ROOT / "experiments" / "phase 6"
    run_timings = {}
    generation_results = {}

    # 4. Прогон всех выбранных режимов генерации
    print("\n" + "=" * 60)
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Executing {len(selected_modes)} Generation Modes:")
    print("=" * 60)

    for mode in selected_modes:
        name = mode["name"]
        desc = mode["desc"]
        html_file = out_dir / f"demo_{tag}_{mode['suffix']}.html"
        kwargs = mode["kwargs"]

        print(f"\n>>> Running Mode: {desc} ...")
        t_start = time.time()
        res = run_generation(
            model=model,
            decoder=decoder,
            latent_dict=latent_dict,
            tokenizer=tokenizer,
            query_text=args.prompt,
            device=device,
            max_len=args.max_len,
            steps=args.steps,
            cfg_scale=args.cfg_min,
            cfg_max=args.cfg_max,
            cfg_min=args.cfg_min,
            cfg_power=args.cfg_power,
            seed=args.seed,
            re_noise_t=args.re_noise_t,
            eps_scale=1.0,
            **kwargs,
        )
        elapsed = time.time() - t_start
        run_timings[name] = elapsed
        generation_results[name] = res
        print(f">>> Mode '{name}' finished in {elapsed / 60:.2f} min ({elapsed:.1f} s). Saving {html_file.name}...")
        generate_html([res], str(html_file))

    # 5. Оценка качества через GPT-2 PPL, Diversity, DProx и долю промпта
    print("\n" + "=" * 60)
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Scoring Generations with GPT-2 & Diversity Metrics...")
    print("=" * 60)

    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    gpt2_lm = AutoModelForCausalLM.from_pretrained("gpt2").eval().to(device)

    pw = set(re.findall(r"\w+", args.prompt.lower()))
    scores = {}
    clean_texts = {}

    for mode in selected_modes:
        name = mode["name"]
        res = generation_results[name]

        # Извлечение текста декодера и NN
        full_dec = res.get("final", "")
        full_nn = res.get("final_nn", "")

        # Запасное извлечение из HTML, если пусто
        if not full_dec:
            html_file = out_dir / f"demo_{tag}_{mode['suffix']}.html"
            full_dec, full_nn, html_dprox = extract_both_from_html(html_file)

        clean, n_void = strip_void(full_dec)
        clean_nn, _ = strip_void(full_nn)

        words = re.findall(r"\w+", clean.lower())
        words_nn = re.findall(r"\w+", clean_nn.lower())

        # Финальный DProx и void_pct из истории генерации
        last_step_entry = res["history"][-1] if res.get("history") else {}
        final_dprox = last_step_entry.get("raw_dprox", None)
        final_void_pct = last_step_entry.get("void_pct", 0.0)

        with torch.no_grad():
            if len(clean) > 0:
                ids = gpt2_tok(clean, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(device)
                loss = gpt2_lm(ids, labels=ids).loss.item()
                ppl = round(float(torch.exp(torch.tensor(loss))), 1)
            else:
                ppl = float("nan")

        prompt_word_share = round(sum(w in pw for w in words) / max(1, len(words)), 3)
        nn_prompt_share = round(sum(w in pw for w in words_nn) / max(1, len(words_nn)), 3) if words_nn else 0.0

        scores[name] = {
            "mode_desc": mode["desc"],
            "time_min": round(run_timings[name] / 60, 2),
            "content_words": len(words),
            "void_tokens": n_void,
            "void_pct": round(final_void_pct, 1),
            "dprox": round(float(final_dprox), 3) if final_dprox is not None else None,
            "ppl_gpt2": ppl,
            "distinct1": round(distinct_n(words, 1), 3),
            "distinct2": round(distinct_n(words, 2), 3),
            "distinct3": round(distinct_n(words, 3), 3),
            "prompt_word_share": prompt_word_share,
            "nn_prompt_share": nn_prompt_share,
        }
        clean_texts[name] = clean

    # Сохранение JSON со структурированными метриками
    out_json = out_dir / f"score_generations_{tag}.json"
    out_json.write_text(json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved metric scores to {out_json}")

    # Подготовка файлов слепой оценки
    names = list(clean_texts.keys())
    m_num = re.search(r"\d+", tag)
    seed_val = int(m_num.group()) if m_num else 42
    random.Random(seed_val).shuffle(names)
    blind_key = {f"Ответ {chr(65 + i)}": n for i, n in enumerate(names)}

    blind_md_lines = [
        f"# Слепая оценка генераций Phase 6 ({tag})\n",
        f"**Вопрос**: {args.prompt}\n",
        "Оцените качество и связность каждого ответа по шкале от 1 (бессвязный повтор / зацикливание) до 5 (отличный связный осмысленный ответ).\n",
    ]
    for i, n in enumerate(names):
        blind_md_lines.append(f"## Ответ {chr(65 + i)}\n\n{clean_texts[n]}\n")

    blind_md_path = out_dir / f"blind_judge_{tag}.md"
    blind_key_path = out_dir / f"blind_judge_key_{tag}.json"
    blind_md_path.write_text("\n".join(blind_md_lines), encoding="utf-8")
    blind_key_path.write_text(json.dumps(blind_key, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved blind judge documents to {blind_md_path.name} & {blind_key_path.name}")

    # Печать итоговой консольной таблицы
    print("\n" + "=" * 105)
    print(f"{'Режим':20} {'Время':>8} {'Слов':>6} {'Void':>5} {'DProx':>6} {'PPL':>8} {'d1':>6} {'d2':>6} {'d3':>6} {'prompt%':>8} {'nn_p%':>6}")
    print("-" * 105)
    for k, v in scores.items():
        dp_str = f"{v['dprox']:.3f}" if v.get('dprox') is not None else " N/A "
        nn_p_str = f"{v['nn_prompt_share']:.3f}" if v.get('nn_prompt_share') is not None else " N/A "
        print(
            f"{k:20} {v['time_min']:>7.1f}м {v['content_words']:6d} {v['void_tokens']:5d} {dp_str:>6} "
            f"{v['ppl_gpt2']:8.1f} {v['distinct1']:6.3f} {v['distinct2']:6.3f} {v['distinct3']:6.3f} "
            f"{v['prompt_word_share']:8.3f} {nn_p_str:>6}"
        )
    print("=" * 105)


if __name__ == "__main__":
    main()
