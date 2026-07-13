<!-- created: 2026-07-13 -->
# ADR 057: Phase 3 bfloat16 Gradient Collapse and Loss Function Robustness

## Context and Problem
During Phase 3 (DUS Core training), we encountered a complete failure of the model to learn denoising or even retain the identity transformation. The `eval_phase3_logic_local.py` script showed that the DUS layers output complete garbage ("китайские иероглифы").
Analysis via layer norms (`layer_analysis.py`) and direct weight comparison (`compare_weights.py`) revealed that the weights in checkpoint `step_8000.pth` matched the initialization from `AWAKENED_WEIGHTS_FINAL.pt` down to exactly 4 decimal places, with an absolute zero difference. 

The root causes of this "frozen weights" catastrophe were two-fold:
1. **bfloat16 Precision Limits**: `bfloat16` has only 7 bits of mantissa. The gradients from the optimizer were so small that they could not exceed the epsilon of `bfloat16` rounding. Consequently, all gradient updates were completely erased by the hardware rounding, acting as if `lr=0`.
2. **Loss Function Suppression (`gamma=20`)**: The `identity_penalty` was explicitly multiplied by `w_c = torch.pow(c_true.float(), gamma)`, where `gamma=20`. This exponential suppression effectively flattened the loss gradients for the majority of tokens, ensuring that the gradient magnitudes remained far below the `bfloat16` representable threshold.

Additionally, the evaluation script failed to properly initialize the `ModernLatentDecoder` due to an incorrect argument parsing logic (attempting to load decoder weights from the `--encoder` path), which obfuscated the debugging process by causing the `AE Only` test to fail.

## Decisions Made and Lessons Learned
- **What was tried and didn't work:** 
  - Applying `c_true ** 20` directly as a multiplier for the `identity_penalty` loss component. This creates excessively small gradients that cannot survive `bfloat16` truncation.
  - Relying on `clip_grad_norm_` without verifying that the raw gradients exceed the hardware precision floor.
- **Successful Solution (Best Practice):** 
  - **Decouple `gamma=20` from the Loss Multiplier**: The high exponent (`gamma=20`) should be strictly reserved for generating the `c_embed` conditioning signal to push the embedding to extreme values (pure noise vs pure clean). It must **not** be used as a scalar multiplier for the loss itself. The loss weight `w_c` should scale linearly or smoothly (e.g., `c_true`).
  - **Compute Loss and Penalties in `float32`**: All loss computations, including cosine similarities and penalty accumulations, must be performed in `torch.float32`. 
  - **Aggressive Initial LR**: Phase 3 requires an initial LR boost (similar to ADR 056 for Phase 2) to break the initial symmetry of the copied DUS layers and generate gradients large enough to bypass `bfloat16` precision limits.

## Impact
- **Positive:** We will eliminate the "silent freeze" bug caused by precision loss. The DUS layers will actually begin learning the topological alignment and denoising tasks. The decoder is now correctly verified to reconstruct the text flawlessly (the AE is healthy).
- **Negative:** We must be more rigorous with gradient monitoring in future TPU runs. High LR might cause temporary instability during the first few steps, but it is necessary to shatter the rounding floor.
