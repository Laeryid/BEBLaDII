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


def extract_both(path: Path):
    raw = html.unescape(path.read_text(encoding="utf-8"))
    m_dec = re.search(r"=== FINAL ANSWER \(ModernLatentDecoder\) ===\s*(.*?)\s*=== FINAL ANSWER \(LatentDict", raw, re.S)
    m_nn = re.search(r"=== FINAL ANSWER \(LatentDict NN\) ===\s*(.*?)(?:<div|<pre|===|$)", raw, re.S)
    m_dprox = re.findall(r"DProx=([0-9.]+)", raw)
    final_dprox = float(m_dprox[-1]) if m_dprox else None

    txt_dec = re.sub(r"<[^>]+>", "", m_dec.group(1)).strip() if m_dec else ""
    txt_nn = re.sub(r"<[^>]+>", "", m_nn.group(1)).strip() if m_nn else ""
    return txt_dec, txt_nn, final_dprox


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
        fp = d / fn
        if not fp.exists():
            continue
        full_dec, full_nn, dprox_val = extract_both(fp)
        clean, n_void = strip_void(full_dec)
        clean_nn, _ = strip_void(full_nn)
        words = re.findall(r"\w+", clean.lower())
        words_nn = re.findall(r"\w+", clean_nn.lower())
        ids = tok(clean, return_tensors="pt", truncation=True, max_length=1024).input_ids
        with torch.no_grad():
            loss = lm(ids, labels=ids).loss.item()
        res[name] = {
            "content_words": len(words),
            "void_tokens": n_void,
            "dprox": round(dprox_val, 3) if dprox_val is not None else None,
            "ppl_gpt2": round(float(torch.exp(torch.tensor(loss))), 1),
            "distinct1": round(distinct_n(words, 1), 3),
            "distinct2": round(distinct_n(words, 2), 3),
            "distinct3": round(distinct_n(words, 3), 3),
            "prompt_word_share": round(sum(w in pw for w in words) / max(1, len(words)), 3),
            "nn_prompt_share": round(sum(w in pw for w in words_nn) / max(1, len(words_nn)), 3) if words_nn else 0.0,
        }
        texts[name] = clean

    Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{'run':12} {'words':>6} {'void':>5} {'dprox':>6} {'ppl':>8} {'d1':>6} {'d2':>6} {'d3':>6} {'prompt%':>8} {'nn_p%':>6}")
    for k, v in res.items():
        dp_str = f"{v['dprox']:.3f}" if v['dprox'] is not None else " N/A "
        print(f"{k:12} {v['content_words']:6} {v['void_tokens']:5} {dp_str:>6} {v['ppl_gpt2']:8} {v['distinct1']:6} "
              f"{v['distinct2']:6} {v['distinct3']:6} {v['prompt_word_share']:8} {v['nn_prompt_share']:6}")

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
