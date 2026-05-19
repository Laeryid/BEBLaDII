import json, sys
sys.stdout.reconfigure(encoding='utf-8')

FILES = {
    "OLD train": r"storage\experiments\20260516 Phase + Reasoning 5500 steps\checkpoints_history_5500.jsonl",
    "OLD val":   r"storage\experiments\20260516 Phase + Reasoning 5500 steps\checkpoints_history_val_5500.jsonl",
    "NEW train": r"tmp\history.jsonl",
    "NEW val":   r"tmp\history_val.jsonl",
}

def audit(name, path):
    print(f"\n{'='*60}")
    print(f"  {name}: {path}")
    print(f"{'='*60}")
    try:
        with open(path, encoding='utf-8') as f:
            raw_lines = f.readlines()
    except Exception as e:
        print(f"  ERROR opening file: {e}")
        return

    print(f"  Total lines: {len(raw_lines)}")

    records = []
    parse_errors = 0
    for i, line in enumerate(raw_lines):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            parse_errors += 1
            print(f"  JSON parse error on line {i+1}: {line[:80]}")

    print(f"  Valid JSON records: {len(records)}")
    print(f"  Parse errors: {parse_errors}")
    if not records:
        return

    # Step coverage
    steps = [r.get("global_step") for r in records if "global_step" in r]
    if steps:
        print(f"  Step range: {min(steps)} -> {max(steps)}")
        diffs = [steps[i+1]-steps[i] for i in range(len(steps)-1)]
        if diffs:
            print(f"  Step interval: min={min(diffs)}, max={max(diffs)}, typical={sorted(diffs)[len(diffs)//2]}")
        gaps = [(steps[i], steps[i+1]) for i in range(len(steps)-1) if steps[i+1]-steps[i] > 200]
        if gaps:
            print(f"  Gaps (>200 steps): {gaps}")
        else:
            print(f"  Gaps: none")

    # Key metrics presence
    all_keys = set()
    for r in records:
        all_keys.update(r.keys())

    IMPORTANT_KEYS = [
        "global_step", "train/loss", "val/loss",
        "train/full_l40_cos", "val/l40_cos",
        "val/l40_isotropy", "val/l40_neighbor_recall",
        "val/l40_prior", "val/l40_mu_drift", "val/l40_var_drift",
        "train/beta", "train/lr",
        "val/l20_mse", "val/l30_mse",
    ]
    print(f"\n  Key metric availability:")
    for k in IMPORTANT_KEYS:
        count = sum(1 for r in records if k in r)
        if count > 0:
            sample_vals = [r[k] for r in records if k in r]
            vmin, vmax = min(sample_vals), max(sample_vals)
            print(f"    {k:<35} present in {count:>4}/{len(records)} records  [{vmin:.4g} .. {vmax:.4g}]")
        else:
            print(f"    {k:<35} MISSING")

    # Check for NaN/Inf
    nan_inf_count = 0
    for r in records:
        for k, v in r.items():
            if isinstance(v, float) and (v != v or abs(v) == float('inf')):
                nan_inf_count += 1
    print(f"\n  NaN/Inf values found: {nan_inf_count}")

    # Unknown keys
    known = set(IMPORTANT_KEYS)
    extra = all_keys - known
    print(f"  Extra keys (not in expected list): {sorted(extra)}")

for name, path in FILES.items():
    audit(name, path)
