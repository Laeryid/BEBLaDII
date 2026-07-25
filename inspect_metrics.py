import csv

out_csv = "C:\\Experiments\\BEBLaDII\\metrics_history.csv"
print(f"Reading {out_csv}...")

with open(out_csv, 'r') as f:
    reader = csv.DictReader(f)
    metrics = list(reader)

print("\n--- Spikes across the whole training ---")

# Let's find all steps where loss > 0.01
for m in metrics:
    step = int(m['step'])
    loss = float(m.get('loss', 0))
    if loss > 0.01 and step > 1000:
        print(f"Step {step:05d} | loss={loss:.4f} | grad_norm={float(m['grad_norm']):.4f} | norm_Llast={float(m['norm_Llast']):.4f} | norm_L0={float(m['norm_L0']):.4f} | t_mean={float(m['t_mean']):.4f} | cov_loss={float(m['cov_loss']):.4f}")

# Let's print metrics at 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000 to see the cycle effect
print("\n--- Metrics at Restart boundaries ---")
for m in metrics:
    step = int(m['step'])
    if step % 2000 == 0:
        print(f"Step {step:05d} | loss={float(m['loss']):.4f} | lr_dus={float(m['lr_dus']):.2e} | lr_new={float(m['lr_new']):.2e} | norm_Llast={float(m['norm_Llast']):.2f}")

