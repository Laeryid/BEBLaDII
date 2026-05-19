import json, sys
sys.stdout.reconfigure(encoding='utf-8')

lines = open('tmp/history_val.jsonl', encoding='utf-8').readlines()
data = [json.loads(l) for l in lines]

print(f"{'step':>6} | {'loss':>8} | {'l40_cos':>8} | {'l40_isotropy':>14} | {'l40_nr':>8} | {'prior':>8} | {'mu_drift':>10}")
print("-" * 85)
for d in data:
    step  = d.get("global_step", 0)
    loss  = d.get("val/loss", 0)
    cos   = d.get("val/l40_cos", 0)
    iso   = d.get("val/l40_isotropy", 0)
    nr    = d.get("val/l40_neighbor_recall", 0)
    prior = d.get("val/l40_prior", 0)
    mu    = d.get("val/l40_mu_drift", 0)
    print(f"{step:>6} | {loss:>8.3f} | {cos:>8.4f} | {iso:>14.8f} | {nr:>8.5f} | {prior:>8.3f} | {mu:>10.5f}")

print()
# Trend analysis
if len(data) >= 3:
    first = data[0]
    last  = data[-1]
    mid_idx = len(data)//2
    mid   = data[mid_idx]

    iso_first = first.get("val/l40_isotropy", 0)
    iso_mid   = mid.get("val/l40_isotropy", 0)
    iso_last  = last.get("val/l40_isotropy", 0)
    
    nr_first = first.get("val/l40_neighbor_recall", 0)
    nr_last  = last.get("val/l40_neighbor_recall", 0)
    
    cos_first = first.get("val/l40_cos", 0)
    cos_last  = last.get("val/l40_cos", 0)
    
    mu_first = first.get("val/l40_mu_drift", 0)
    mu_last  = last.get("val/l40_mu_drift", 0)

    print("=== TREND ANALYSIS ===")
    print(f"Isotropy:  {iso_first:.6f} -> {iso_mid:.6f} -> {iso_last:.6f}  (x{iso_last/max(iso_first,1e-9):.1f})")
    print(f"Neighbor recall: {nr_first:.5f} -> {nr_last:.5f}  (delta: {nr_last-nr_first:+.5f})")
    print(f"L40 cosine:      {cos_first:.4f}  -> {cos_last:.4f}  (delta: {cos_last-cos_first:+.4f})")
    print(f"Mu drift:        {mu_first:.5f}  -> {mu_last:.5f}  (delta: {mu_last-mu_first:+.5f})")
    print(f"Steps range:     {first.get('global_step')} -> {last.get('global_step')}")
