"""Оценка качества сгенерированных ответов Phase 6 из HTML-отчетов.

Метрики (только по не-void части): GPT-2 perplexity, distinct-1/2/3,
доля токенов, скопированных из промпта, длина содержательной части.
Также пишет blind_judge.md с перемешанными ответами для слепой оценки.
"""
import argparse
import html
import json
import random
import re
from pathlib import Path

import torch

PROMPT = "How does your generation mechanism contrast with traditional next-token prediction models?"
RUNS = {
    "baseline": "demo_baseline_len500_s25.html",
    "void_dprox": "demo_void_dprox.html",
    "soft": "demo_smartcrop_soft.html",
    "hard_v2": "demo_smartcrop_hard_v2.html",
    "hard_buf16": "demo_smartcrop_hard_buffered16.html",
}


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="experiments/phase 6")
    ap.add_argument("--lm", default="gpt2")
    ap.add_argument("--out", default="experiments/phase 6/score_generations.json")
    args = ap.parse_args()
    d = Path(args.dir)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.lm)
    lm = AutoModelForCausalLM.from_pretrained(args.lm).eval()

    pw = set(re.findall(r"\w+", PROMPT.lower()))
    res, texts = {}, {}
    for name, fn in RUNS.items():
        full = extract_final(d / fn)
        clean, n_void = strip_void(full)
        words = re.findall(r"\w+", clean.lower())
        ids = tok(clean, return_tensors="pt", truncation=True, max_length=1024).input_ids
        with torch.no_grad():
            loss = lm(ids, labels=ids).loss.item()
        res[name] = {
            "content_words": len(words),
            "void_tokens": n_void,
            "ppl_gpt2": round(float(torch.exp(torch.tensor(loss))), 1),
            "distinct1": round(distinct_n(words, 1), 3),
            "distinct2": round(distinct_n(words, 2), 3),
            "distinct3": round(distinct_n(words, 3), 3),
            "prompt_word_share": round(sum(w in pw for w in words) / max(1, len(words)), 3),
        }
        texts[name] = clean

    Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{'run':12} {'words':>6} {'void':>5} {'ppl':>8} {'d1':>6} {'d2':>6} {'d3':>6} {'prompt%':>8}")
    for k, v in res.items():
        print(f"{k:12} {v['content_words']:6} {v['void_tokens']:5} {v['ppl_gpt2']:8} {v['distinct1']:6} "
              f"{v['distinct2']:6} {v['distinct3']:6} {v['prompt_word_share']:8}")

    names = list(texts)
    random.Random(0).shuffle(names)
    key = {f"Ответ {chr(65 + i)}": n for i, n in enumerate(names)}
    lines = [f"# Слепая оценка\n\nВопрос: {PROMPT}\n"]
    for i, n in enumerate(names):
        lines.append(f"## Ответ {chr(65 + i)}\n\n{texts[n]}\n")
    Path(d / "blind_judge.md").write_text("\n".join(lines), encoding="utf-8")
    Path(d / "blind_judge_key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
