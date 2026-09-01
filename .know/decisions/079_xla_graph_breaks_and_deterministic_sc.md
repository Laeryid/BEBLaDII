<!-- created: 2026-09-02 -->
# ADR 079: XLA Graph Breaks and Deterministic Self-Conditioning

## Context and Problem
During Phase 4a (TPU Kaggle) training, `ExecuteReplicatedTime` spikes reached 2–4 seconds per step (with some steps taking >140s). Analysis of XLA `metrics.txt` revealed two major issues:
1. **Catastrophic Graph Breaks (`aten::_local_scalar_dense`)**: Calling `.item()` synchronously every step (e.g., `grad_norm.item()`) caused TPU to stall, waiting for CPU sync. During logging (every 10 steps), the loop calling `v.mean().item()` for 28 metrics caused 40-second blockages.
2. **Graph Recompilation (UncachedCompile) & Thrashing**: 
   - `torch.rand(1).item() < 0.5` inside the training loop (for Self-Conditioning) forced PyTorch XLA to compile and alternate between two completely different massive computational graphs (`TensorsGraphSize ~ 33804`). 
   - The cyclic LR scheduler updated Python float `lr` values periodically (e.g. `current_optim_step % 50 == 1`). Since XLA embeds Python floats as constants into the graph, every LR change triggered a 140s recompilation.
   - The `compute_adaln_diagnostics` function executed Python loops over layers every 10 steps, repeatedly injecting new operations into the graph, causing massive one-time graph compilations (+9 UncachedCompile).

## Decisions Made and Lessons Learned
- **What was tried and didn't work:** Logging metrics inline via `.item()` directly off the TPU, dynamic probabilistic Python `if` statements for computational logic (SC), and updating the LR every 50 steps via Python conditionals.
- **Successful Solution (Best Practice):**
  - **Deterministic Self-Conditioning (In-Graph):** Removed `if torch.rand(1).item() < 0.5`. Instead, the `no_grad()` pass is executed for the entire batch, and `sc_mask` explicitly zero-masks the first half (`batch[:B//2]`). This mathematically guarantees exactly one unified, stable XLA graph.
  - **Asynchronous / Batched Logging:** All logging scalars are concatenated into a single stacked tensor and retrieved via one `.cpu().tolist()` call, slashing graph breaks from ~30 per log step down to exactly 1.
  - **Static Hyperparameters:** Removed the `XLA FIX` for LR changing. The learning rate is now static (using `PACE` with `warmup_steps=0`), eliminating graph invalidation.
  - **Removed `compute_adaln_diagnostics`:** Disabled diagnostic layer loops to preserve graph footprint stability.

## Impact
- **Positive:** Training steps stabilized completely at ~1.1 to ~1.6 seconds per step, without massive recompilations (UncachedCompile) or sudden 40+ second CPU wait states.
- **Negative:** Self-Conditioning is now explicitly applied to exactly 50% of every batch rather than stochastically per full batch, but this is a negligible mathematical trade-off for enormous TPU stability gains.
