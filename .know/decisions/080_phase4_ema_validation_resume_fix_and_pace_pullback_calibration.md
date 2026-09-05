<!-- created: 2026-09-03 -->
# ADR 080: Phase 4 EMA Validation, TPU Shadow Resume Fix and PACE Pullback Calibration

## Context and Problem
During a 9-hour training run on Kaggle TPU v5e-8 (Phase 4a Hierarchical Noise, steps 8,995 to ~22,500), training telemetry showed:
1. **XLA Warmup**: Initial graph compilation completed within the first 10–20 steps (throughput ramped from ~0 to ~38 samples/sec by step 9,020).
2. **Early Gradient Spike**: At step ~9,120, a gradient norm spike (`grad_norm ~ 2.6`) occurred following the transition to large-batch TPU training.
3. **Severe Metric Plateau**: Training and validation loss stalled completely at `~0.15` for over 13,000 steps without improvement, despite processing hundreds of thousands of samples.
4. **Offline Evaluation Discrepancy**: Local evaluation of checkpoint `phase4_step_21995.pth` via `evaluate_phase4_checkpoints.py` initially reported catastrophic collapse (`cos_h39 ~ 0.05` across test sequences). Subsequent investigation revealed three major interconnected flaws in the training infrastructure:

### Root Causes
- **Flaw 1: Validation evaluated LIVE weights instead of EMA**:
  In `train_phase4_tpu_notebook.py`, validation was executed directly on `model.eval()` without invoking `ema.apply(actual_model)`. All WandB validation curves (`val_loss`, `val_cos_h39_*`, `val_adaln_*`) reflected raw LIVE optimizer weights rather than iterate-averaged EMA weights, violating `KI_pace_optimizer.md` and masking EMA degradation.
- **Flaw 2: CPU Tensor Overwrite of Sharded TPU EMA Shadows on Resume**:
  In `load_checkpoint_split`, the resume logic loaded the checkpoint dictionary to host CPU and performed `ema.shadow.update(ema_update)`. This assigned raw CPU tensors directly into `ema.shadow`, destroying device placement (`xla:0`) and SPMD FSDP sharding specs. Furthermore, `self_cond_proj` shadow re-initialization contained an erroneous `.cpu()` call.
- **Flaw 3: Excessive PACE Pullback Alpha ("The Anchor Plateau")**:
  With `pullback_alpha = 0.016` and constant DUS learning rate (`5e-5`), the pullback force `0.016 * (param - EMA)` was ~50–300× stronger than typical AdamW step updates. This acted as a rigid attractor, locking live model parameters to a stagnant or contaminated EMA shadow and preventing gradient descent into lower loss regions.

## Decisions Made
1. **Enforce EMA Validation with Protected Restoration**:
   - In `train_phase4_tpu_notebook.py`, the validation loop is now explicitly wrapped in:
     ```python
     model.eval()
     ema.apply(actual_model)
     try:
         # Validation loop on EMA weights...
     finally:
         ema.restore(actual_model)
         model.train()
     ```
   - All validation metrics (`[VAL - EMA]`, `val_loss`, `val_cos_h39_*`, `val_layer_divergence`, `val_adaln_*`) now faithfully measure the production EMA model.
2. **In-Place Sharded Copy for EMA Shadows on Resume**:
   - Replaced dictionary reassignment with in-place `.copy_()` targeting existing sharded TPU tensors:
     ```python
     if target_key in ema.shadow:
         ema.shadow[target_key].copy_(v.to(device=ema.shadow[target_key].device, dtype=torch.float32))
     ```
   - Removed all `.cpu()` casts in `self_cond_proj` initialization to maintain pure TPU `float32` residency without graph breaks.
3. **Calibrate PACE Pullback Alpha (ADR 077 Compliance)**:
   - Reduced `pullback_alpha` from `0.016` to `0.001` (aligned with the EMA update rate `1 - decay = 0.01`). This preserves stabilization from iterate averaging while removing the artificial constraint that caused the loss plateau.
4. **Dual-Mode Offline Checkpoint Evaluation**:
   - Updated `evaluate_phase4_checkpoints.py` to independently evaluate both `LIVE` and `EMA` parameter states from any `.pth` file, displaying explicit side-by-side diagnostic sections.

## Consequences and Next Steps
- **Checkpoint Invalidation**: Checkpoint `phase4_step_21995.pth` is contaminated and must be discarded.
- **Restart Target**: Training will resume from the healthy checkpoint `phase4_step_8995.pth`.
- **Expected Outcome**: With calibrated `pullback_alpha = 0.001`, in-place sharded EMA resumption, and true EMA validation, the network will be free to escape the 0.15 plateau and continue converge on high-noise tokens (`t > 0.7`).
