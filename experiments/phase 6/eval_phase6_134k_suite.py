"""Комплексная оценка чекпоинта Phase 6 Step 134000 в 5 режимах из ADR 115.

Режимы:
1. baseline: smartcrop="none", void_aware_dprox=False
2. void_dprox: smartcrop="none", void_aware_dprox=True
3. soft: smartcrop="soft", smartcrop_tau=0.90
4. hard_v2: smartcrop="hard", smartcrop_tau=0.90, smartcrop_buffer=0
5. hard_buf16: smartcrop="hard", smartcrop_tau=0.90, smartcrop_buffer=16

Модель загружается 1 раз в память, после чего выполняется запуск всех 5 режимов.
Метрики вычисляются на базе GPT-2 (PPL) и N-граммного разнообразия (distinct-1/2/3).
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
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Импорт необходимых классов и функций из evaluate_phase6_metrics
from evaluate_phase6_metrics import (
    BEBLaDIIPhase6Eval,
    ModernLatentDecoder,
    generate_html,
    normalize_key,
    run_generation,
    validate_environment,
)

PROMPT = "How does your generation mechanism contrast with traditional next-token prediction models?"

CKPT_PATH = os.path.join(
    PROJECT_ROOT,
    "experiments",
    "phase 6",
    "checkpoints",
    "planB_phase6_checkpoints_phase6_ca_layers_step_134000.pth",
)

CONFIG = {
    'qwen_path': "Qwen/Qwen2.5-1.5B",
    'modernbert_path': "answerdotai/ModernBERT-large",
    'encoder_path': os.path.join(PROJECT_ROOT, "experiments", "phase 1", "planB_phase1_checkpoints_phase1_vae_step_20000.pth"),
    'phase4_path': os.path.join(PROJECT_ROOT, "experiments", "phase 4", "local_checkpoints", "phase4_step_85995.pth"),
    'phase2_decoder_path': os.path.join(PROJECT_ROOT, "experiments", "phase 2", "planB_phase2_checkpoints_decoder_step_9000.pth"),
    'sep_token_path': os.path.join(PROJECT_ROOT, "storage", "components", "sep_token.pt"),
    'void_token_path': os.path.join(PROJECT_ROOT, "storage", "components", "void_token.pt"),
    'phase6_ckpt': CKPT_PATH,
}

MODES = [
    {
        "name": "baseline",
        "desc": "baseline (SmartCrop=none, VoidDProx=OFF)",
        "suffix": "baseline",
        "kwargs": {"smartcrop": "none", "void_aware_dprox": False},
    },
    {
        "name": "void_dprox",
        "desc": "void_dprox (SmartCrop=none, VoidDProx=ON)",
        "suffix": "void_dprox",
        "kwargs": {"smartcrop": "none", "void_aware_dprox": True},
    },
    {
        "name": "soft",
        "desc": "soft SmartCrop (tau=0.90)",
        "suffix": "smartcrop_soft",
        "kwargs": {"smartcrop": "soft", "smartcrop_tau": 0.90, "void_aware_dprox": False},
    },
    {
        "name": "hard_v2",
        "desc": "hard SmartCrop v2 (strict crop, buffer=0)",
        "suffix": "smartcrop_hard_v2",
        "kwargs": {"smartcrop": "hard", "smartcrop_tau": 0.90, "smartcrop_buffer": 0, "void_aware_dprox": False},
    },
    {
        "name": "hard_buf16",
        "desc": "hard SmartCrop + 16 void buffer (1A)",
        "suffix": "smartcrop_hard_buffered16",
        "kwargs": {"smartcrop": "hard", "smartcrop_tau": 0.90, "smartcrop_buffer": 16, "void_aware_dprox": False},
    },
]


def extract_final(path: Path) -> str:
    raw = html.unescape(path.read_text(encoding="utf-8"))
    m = re.search(r"=== FINAL ANSWER \(ModernLatentDecoder\) ===\s*(.*?)\s*=== FINAL ANSWER \(LatentDict", raw, re.S)
    if not m:
        raise ValueError(f"final answer not found in {path}")
    txt = re.sub(r"<[^>]+>", "", m.group(1)) if "<div" in m.group(1) else m.group(1)
    return txt.strip()


def strip_void(text: str):
    n_void = sum(int(x) for x in re.findall(r"<\|void x(\d+)\|>", text))
    n_void += len(re.findall(r"<\|void\|>", text))
    clean = re.sub(r"<\|void(?: x\d+)?\|>", " ", text)
    return clean.strip(), n_void


def distinct_n(words, n):
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / max(1, len(grams))


def main():
    parser = argparse.ArgumentParser(description="Evaluation Suite for Phase 6 checkpoints in 5 modes (ADR 115)")
    parser.add_argument("--ckpt", type=str, default=CKPT_PATH, help="Path to checkpoint")
    parser.add_argument("--tag", type=str, default=None, help="Tag for output filenames (e.g. 122k)")
    args = parser.parse_args()

    target_ckpt = os.path.abspath(args.ckpt)
    if not os.path.exists(target_ckpt):
        print(f"[ERROR] Checkpoint not found: {target_ckpt}")
        sys.exit(1)

    if args.tag:
        tag = args.tag
    else:
        m = re.search(r"step_(\d+)", os.path.basename(target_ckpt))
        if m:
            step_num = int(m.group(1))
            tag = f"{step_num // 1000}k" if step_num >= 1000 and step_num % 1000 == 0 else f"{step_num}"
        else:
            tag = Path(target_ckpt).stem

    CONFIG['phase6_ckpt'] = target_ckpt

    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Starting {tag} Evaluation Suite (5 Modes)...")
    print(f"Target Checkpoint: {target_ckpt}")

    dict_primary = os.path.join(PROJECT_ROOT, "experiments", "phase 5", "local", "latent_dict.pt")
    dict_fallback = os.path.join(PROJECT_ROOT, "BEBLaDII-planB-Phase5-Data", "latent_dict.pt")
    dict_paths = [dict_primary, dict_fallback]
    ds_path = os.path.join(PROJECT_ROOT, "BEBLaDII-planB-Phase6-Data", "phase 6", "data", "train_phase6.parquet")

    all_ok, resolved_dict = validate_environment(CONFIG, dict_paths, ds_path, verbose=True)
    if not all_ok:
        print("[ERROR] Environment validation failed!")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    # 1. Загрузка токенизатора и словарей
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Loading Tokenizer & LatentDict...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['qwen_path'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    latent_dict = torch.load(resolved_dict, map_location=device).float()

    # 2. Загрузка BEBLaDIIPhase6Eval
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Loading Phase 6 Model ({tag})...")
    t0_load = time.time()
    model = BEBLaDIIPhase6Eval(CONFIG).to(device)
    print(f"Phase 6 Model loaded in {time.time() - t0_load:.1f}s.")

    # 3. Загрузка ModernLatentDecoder
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Loading ModernLatentDecoder...")
    decoder = ModernLatentDecoder(
        latent_dim=1024,
        qwen_dim=1536,
        num_layers=3,
        dus_weights_path=CONFIG['phase4_path']
    ).to(device)
    ckpt_dec = torch.load(CONFIG['phase2_decoder_path'], map_location=device, weights_only=False)
    st_dec = ckpt_dec.get("decoder", ckpt_dec)
    clean_st_dec = {normalize_key(k).replace("decoder.", ""): v for k, v in st_dec.items()}
    decoder.load_state_dict(clean_st_dec, strict=False)
    decoder.eval()
    print("Decoder ready.")

    out_dir = PROJECT_ROOT / "experiments" / "phase 6"
    run_timings = {}

    # 4. Прогон всех 5 режимов генерации
    print("\n" + "=" * 60)
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Executing 5 Generation Modes for Prompt:")
    print(f"'{PROMPT}'")
    print("=" * 60)

    for mode in MODES:
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
            query_text=PROMPT,
            device=device,
            max_len=500,
            steps=25,
            cfg_scale=1.0,
            cfg_max=4.0,
            cfg_min=1.0,
            cfg_power=1.0,
            sampler="hybrid",
            seed=42,
            re_noise_t=0.88,
            eps_scale=1.0,
            **kwargs,
        )
        elapsed = time.time() - t_start
        run_timings[name] = elapsed
        print(f">>> Mode '{name}' finished in {elapsed / 60:.2f} min ({elapsed:.1f} s). Saving {html_file.name}...")
        generate_html([res], str(html_file))

    # 5. Оценка качества через GPT-2 PPL и N-граммы
    print("\n" + "=" * 60)
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Scoring Generations with GPT-2 & Diversity Metrics...")
    print("=" * 60)

    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    gpt2_lm = AutoModelForCausalLM.from_pretrained("gpt2").eval().to(device)

    pw = set(re.findall(r"\w+", PROMPT.lower()))
    scores = {}
    clean_texts = {}

    for mode in MODES:
        name = mode["name"]
        html_file = out_dir / f"demo_{tag}_{mode['suffix']}.html"
        full_text = extract_final(html_file)
        clean, n_void = strip_void(full_text)
        words = re.findall(r"\w+", clean.lower())

        with torch.no_grad():
            if len(clean) > 0:
                ids = gpt2_tok(clean, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(device)
                loss = gpt2_lm(ids, labels=ids).loss.item()
                ppl = round(float(torch.exp(torch.tensor(loss))), 1)
            else:
                ppl = float("nan")

        scores[name] = {
            "mode_desc": mode["desc"],
            "time_min": round(run_timings[name] / 60, 2),
            "content_words": len(words),
            "void_tokens": n_void,
            "ppl_gpt2": ppl,
            "distinct1": round(distinct_n(words, 1), 3),
            "distinct2": round(distinct_n(words, 2), 3),
            "distinct3": round(distinct_n(words, 3), 3),
            "prompt_word_share": round(sum(w in pw for w in words) / max(1, len(words)), 3),
        }
        clean_texts[name] = clean

    # Сохранение JSON с метриками
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
        f"**Вопрос**: {PROMPT}\n",
        "Оцените качество и связность каждого ответа по шкале от 1 (бессвязный повтор / зацикливание) до 5 (отличный связный осмысленный ответ).\n"
    ]
    for i, n in enumerate(names):
        blind_md_lines.append(f"## Ответ {chr(65 + i)}\n\n{clean_texts[n]}\n")

    blind_md_path = out_dir / f"blind_judge_{tag}.md"
    blind_key_path = out_dir / f"blind_judge_key_{tag}.json"
    blind_md_path.write_text("\n".join(blind_md_lines), encoding="utf-8")
    blind_key_path.write_text(json.dumps(blind_key, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved blind judge documents to {blind_md_path.name} & {blind_key_path.name}")
    blind_key_path.write_text(json.dumps(blind_key, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved blind judge documents to {blind_md_path.name} & {blind_key_path.name}")

    # Печать итоговой таблицы
    print("\n" + "=" * 90)
    print(f"{'Режим':22} {'Время':>8} {'Слов':>6} {'Void':>5} {'PPL':>8} {'d1':>6} {'d2':>6} {'d3':>6} {'prompt%':>8}")
    print("-" * 90)
    for k, v in scores.items():
        print(
            f"{k:22} {v['time_min']:>7.1f}м {v['content_words']:6d} {v['void_tokens']:5d} "
            f"{v['ppl_gpt2']:8.1f} {v['distinct1']:6.3f} {v['distinct2']:6.3f} {v['distinct3']:6.3f} "
            f"{v['prompt_word_share']:8.3f}"
        )
    print("=" * 90)


if __name__ == "__main__":
    main()
