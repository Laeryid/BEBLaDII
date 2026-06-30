# Phase 3: Chronicle of Failed Attempts and Approach Evolution (June 2026)

This report details the step-by-step evolution of Phase 3 (DUS Logic & Denoising Training). Following the logic of previous phases, every day of major architectural decisions (ADRs) represents a distinct "attempt" — a full cycle of running, profiling, discovering bottlenecks, and applying structural corrections.

---

## Attempt 1: Removing Anchor Loss and ZCA Whitening
**Date:** 23.06.2026 (ADR 040)

**Goal:** Distill the logic of the DeepSeek teacher into DUS without collapsing into Shortcut Learning.

**What went wrong:** Attempting to distill logic via a `LogicAdapter` (Anchor Loss) led to Shortcut Learning, where the adapter merely fit the VAE outputs and lost the teacher's semantic topology. The extreme anisotropy of the teacher caused cosine similarities to collapse near 1.0, making it impossible to isolate meaningful signal.

**Correction:** Removed the `LogicAdapter` entirely. Introduced ZCA Whitening for the teacher's vectors to restore true semantic angles, and implemented Pearson RKD on 5x5 non-overlapping token windows. Restored `cov_loss` to prevent dimensional collapse.

## Attempt 2: Spherical Topology Enforcement
**Date:** 25.06.2026 (ADR 041)

**Goal:** Fix the orthogonal convergence and break the plateau in denoising metrics.

**What went wrong:** Unnormalized DUS outputs found a loophole: they cheated the Huber (L2) Loss by manipulating vector norms rather than angles. Furthermore, Huber Loss fundamentally conflicted with the rotation-invariant Pearson RKD, causing the latent space to rotate orthogonally.

**Correction:** Enforced strict spherical topology (`safe_normalize`) on the DUS output. Replaced Huber Loss with Angular Loss (`1.0 - cos_sim`). Added a variance floor (`var_floor`) to prevent coordinate collapse on the sphere.

## Attempt 3: Depth Decoupling and Precision Protection
**Date:** 26.06.2026 (ADRs 042, 043)

**Goal:** Resolve gradient conflicts on the final layer and stop bfloat16 precision loss.

**What went wrong:** `identity_penalty` and `denoise_loss` flatlined at ~0.5. The final layer (40) was mathematically torn between predicting absolute coordinates for denoising and mimicking the teacher's relative topology for logic. Concurrently, orthogonal gradients from spherical losses caused unbounded random walks in activation lengths, reaching ~8000 (ULP 64.0) and destroying fine-grained topological differences in `bfloat16`.

**Correction:** Separated tasks by depth: layer 40 now exclusively handles denoising via a direct Macro-Residual connection, while layer 33 focuses on the teacher's relative topology. Added an Activation L2 Penalty to gently constrain raw vector lengths within the optimal 0-1 precision range.

## Attempt 4: Conditional Diffusion and The Shortcut Collapse
**Date:** 27.06.2026 (ADRs 044, 045, 046, 047)

**Goal:** Provide noise-level conditioning to DUS and stabilize the stochastic training landscape.

**What went wrong:** The network initially struggled to denoise because it lacked explicit knowledge of the noise level (`c_true`). Adding `c_embed` along with an external residual connection created new conflicts. An EMA was introduced for stability but failed to update due to XLA Dead Code Elimination. When we tried moving `c_embed` into the targets to "help" the network, it triggered a catastrophic Shortcut Collapse: the network blew up the `c_embed` norm to 14.5, overpowering the real signal, predicting its own bias for a near-perfect loss, but outputting garbage at inference.

**Correction:** Upgraded DUS to a Conditional Diffusion model by injecting `c_embed` via an MLP, removing the redundant external residual. Fixed EMA with `xm.mark_step()`. Reverted to "Pure Targets" (`z_clean` / `z_noisy`) and strictly bounded the `c_embed` norm to 0.1 to eliminate the shortcut vulnerability forever.

## Attempt 5: Breaking the Identity Plateau
**Date:** 28.06.2026 (ADRs 048, 049, 050)

**Goal:** Clear the `c_embed` bias from the final output and accelerate convergence.

**What went wrong:** The network couldn't easily subtract the injected `c_embed` bias from its residual stream, causing `identity_penalty` to plateau. A deeper architectural conflict became obvious: distilling DeepSeek's logic on layer 33 while doing Qwen-space denoising on layer 40 required an impossible zigzag trajectory for a 40-layer network. Gradient analysis showed logic gradients were mathematically dead.

**Correction:** Analytically subtracted `c_embed` from the pre-norm hidden states (Negative Skip Connection). Radically removed the DeepSeek teacher and `logic_loss` entirely, shifting focus 100% to denoising (allowing logic to form implicitly). Attempted to boost gradients using Layer-wise LR Decay (LLRD) and L1 Sparsity for `c_embed`.

## Attempt 6: Reversing LLRD and Fixing Validation Shift
**Date:** 29.06.2026 (ADRs 051, 052)

**Goal:** Recover from the ADR 050 catastrophe and close the severe gap between train and val metrics.

**What went wrong:** The LLRD combined disastrously with AdamW, freezing the bottom layers and severely overfitting the top layers. Furthermore, the frozen target generator (`LatentEncoder`) was inadvertently left in `.train()` mode. This meant Dropout shifted the `z_clean` targets during training, but turning off Dropout during validation teleported the target manifold, collapsing the metrics.

**Correction:** Completely rolled back ADR 050 (removed LLRD and sparsity loss, reset `c_embed` norm to 0.1). Explicitly forced `.eval()` mode for all frozen generators within the orchestrator to ensure stable target distributions.

---

## Summary of Phase 3 (In Progress)
Phase 3 (DUS Logic & Denoising Training) is currently **in progress**. While the early stages required intense restructuring, we are not at a dead end. 

Over 6 dense iteration cycles, the architecture has been systematically purified:
- `c_embed` conditioning is now handled correctly without causing shortcut learning or polluting the attention space.
- The pipeline is fully protected against `bfloat16` activation explosion and Dropout-induced target shifts.
- The training objective has been streamlined to focus purely on denoising, avoiding contradictory topological mimicry.

The foundation is now mathematically sound and stable, ready to proceed with the remaining training iterations.
