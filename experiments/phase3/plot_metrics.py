import os
import sys
import torch

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

ckpt_path = r"C:\Experiments\BEBLaDII\experiments\phase3\planB_phase3_checkpoints_phase3_step_17995.pth"

if not os.path.exists(ckpt_path):
    print(f"File not found: {ckpt_path}")
    sys.exit(1)

print(f"Loading checkpoint metadata from: {ckpt_path}")
state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

metrics_history = state.get("metrics_history", [])
print(f"Total metrics entries recorded: {len(metrics_history)}")

if not metrics_history:
    print("No metrics_history found in checkpoint!")
    sys.exit(0)

sample = metrics_history[0]
print(f"Keys in metrics_history: {list(sample.keys())}")

print("\n" + "=" * 110)
print(f"{'Step':<7} | {'Loss':<8} | {'DenoiseL':<8} | {'CosAll':<8} | {'CosLow(t<.3)':<12} | {'CosHi(t>.7)':<12} | {'PriorL':<8} | {'GradNorm':<8} | {'LR_DUS':<9}")
print("=" * 110)

interval = max(1, len(metrics_history) // 25)

for i in range(0, len(metrics_history), interval):
    entry = metrics_history[i]
    step = entry.get("step", i * 10)
    loss = entry.get("loss", 0.0)
    denoise_loss = entry.get("denoising_loss", 0.0)
    cos_all = entry.get("cos_sim_all", 0.0)
    cos_low = entry.get("cos_sim_t_low", 0.0)
    cos_hi = entry.get("cos_sim_t_high", 0.0)
    prior_loss = entry.get("prior_loss", 0.0)
    grad_norm = entry.get("grad_norm", 0.0)
    lr_dus = entry.get("lr_dus", 0.0)
    
    print(f"{step:<7} | {loss:<8.4f} | {denoise_loss:<8.4f} | {cos_all:<8.4f} | {cos_low:<12.4f} | {cos_hi:<12.4f} | {prior_loss:<8.4f} | {grad_norm:<8.4f} | {lr_dus:<9.2e}")

last = metrics_history[-1]
step = last.get("step", len(metrics_history) * 10)
loss = last.get("loss", 0.0)
denoise_loss = last.get("denoising_loss", 0.0)
cos_all = last.get("cos_sim_all", 0.0)
cos_low = last.get("cos_sim_t_low", 0.0)
cos_hi = last.get("cos_sim_t_high", 0.0)
prior_loss = last.get("prior_loss", 0.0)
grad_norm = last.get("grad_norm", 0.0)
lr_dus = last.get("lr_dus", 0.0)
print(f"{step:<7} | {loss:<8.4f} | {denoise_loss:<8.4f} | {cos_all:<8.4f} | {cos_low:<12.4f} | {cos_hi:<12.4f} | {prior_loss:<8.4f} | {grad_norm:<8.4f} | {lr_dus:<9.2e}")
print("=" * 110)

restarts = []
for i in range(1, len(metrics_history)):
    prev_step = metrics_history[i-1].get("step", 0)
    curr_step = metrics_history[i].get("step", 0)
    if curr_step < prev_step or (curr_step - prev_step > 100):
        restarts.append((prev_step, curr_step))

if restarts:
    print(f"\nDetected potential restart points in step progression: {restarts}")

cos_hi_vals = [e.get("cos_sim_t_high", 0.0) for e in metrics_history if "cos_sim_t_high" in e]
cos_lo_vals = [e.get("cos_sim_t_low", 0.0) for e in metrics_history if "cos_sim_t_low" in e]

if cos_hi_vals:
    print(f"\ncos_sim_t_high: min={min(cos_hi_vals):.4f}, max={max(cos_hi_vals):.4f}, start={cos_hi_vals[0]:.4f}, end={cos_hi_vals[-1]:.4f}")
if cos_lo_vals:
    print(f"cos_sim_t_low:  min={min(cos_lo_vals):.4f}, max={max(cos_lo_vals):.4f}, start={cos_lo_vals[0]:.4f}, end={cos_lo_vals[-1]:.4f}")
