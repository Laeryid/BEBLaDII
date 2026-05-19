import json, sys
sys.stdout.reconfigure(encoding='utf-8')

OLD_VAL  = r"storage\experiments\20260516 Phase + Reasoning 5500 steps\checkpoints_history_val_5500.jsonl"
OLD_TRAIN = r"storage\experiments\20260516 Phase + Reasoning 5500 steps\checkpoints_history_5500.jsonl"
NEW_VAL  = r"tmp\history_val.jsonl"
NEW_TRAIN = r"tmp\history.jsonl"

def load(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]

old_val   = load(OLD_VAL)
old_train = load(OLD_TRAIN)
new_val   = load(NEW_VAL)
new_train = load(NEW_TRAIN)

print(f"OLD run: {len(old_val)} val checkpoints, {len(old_train)} train logs")
print(f"NEW run: {len(new_val)} val checkpoints, {len(new_train)} train logs")
print()

# --- Key val metrics comparison at matching steps ---
KEY_METRICS = [
    ("val/loss",                 "Loss"),
    ("val/l40_cos",              "L40 Cosine"),
    ("val/l40_prior",            "L40 Prior"),
    ("val/l40_isotropy",         "L40 Isotropy"),
    ("val/l40_neighbor_recall",  "L40 Neighbor Recall"),
    ("val/l40_mu_drift",         "L40 Mu Drift"),
    ("val/l40_var_drift",        "L40 Var Drift"),
]

def find_step(data, step):
    for d in data:
        if abs(d.get("global_step", -1) - step) <= 30:
            return d
    return None

COMPARE_STEPS = [40, 100, 200, 500, 1000, 1500, 2000, 2490]

print("=" * 90)
print(f"{'Metric':<25} | {'Step':>5} | {'OLD (pre-fix)':>15} | {'NEW (post-fix)':>15} | {'Delta':>12}")
print("=" * 90)

for metric, label in KEY_METRICS:
    first = True
    for step in COMPARE_STEPS:
        old_d = find_step(old_val, step)
        new_d = find_step(new_val, step)
        if old_d is None and new_d is None:
            continue
        old_v = old_d.get(metric, None) if old_d else None
        new_v = new_d.get(metric, None) if new_d else None

        old_s = f"{old_v:.5f}" if old_v is not None else "N/A"
        new_s = f"{new_v:.5f}" if new_v is not None else "N/A"
        if old_v is not None and new_v is not None:
            delta = new_v - old_v
            delta_s = f"{delta:+.5f}"
        else:
            delta_s = "N/A"

        lbl = label if first else ""
        print(f"{lbl:<25} | {step:>5} | {old_s:>15} | {new_s:>15} | {delta_s:>12}")
        first = False
    print("-" * 90)

print()
print("=== ISOTROPY TRAJECTORY ===")
print(f"{'':>3} {'step':>6} | {'OLD iso':>12} | {'NEW iso':>12} | {'ratio NEW/OLD':>14}")
print("-" * 60)
for step in COMPARE_STEPS:
    old_d = find_step(old_val, step)
    new_d = find_step(new_val, step)
    old_iso = old_d.get("val/l40_isotropy", None) if old_d else None
    new_iso = new_d.get("val/l40_isotropy", None) if new_d else None
    old_s = f"{old_iso:.7f}" if old_iso else "N/A    "
    new_s = f"{new_iso:.7f}" if new_iso else "N/A    "
    ratio = f"{new_iso/old_iso:.2f}x" if (old_iso and new_iso) else "N/A"
    print(f"    {step:>6} | {old_s:>12} | {new_s:>12} | {ratio:>14}")

print()
print("=== PRIOR LOSS TRAJECTORY (key indicator of centering) ===")
print(f"{'step':>6} | {'OLD prior':>12} | {'NEW prior':>12}")
print("-" * 40)
for step in COMPARE_STEPS:
    old_d = find_step(old_val, step)
    new_d = find_step(new_val, step)
    old_p = old_d.get("val/l40_prior", None) if old_d else None
    new_p = new_d.get("val/l40_prior", None) if new_d else None
    old_s = f"{old_p:.2f}" if old_p is not None else "N/A"
    new_s = f"{new_p:.2f}" if new_p is not None else "N/A"
    print(f"{step:>6} | {old_s:>12} | {new_s:>12}")
