# Phase 2: Chronicle of Failed Attempts and Approach Evolution (May 2026)

This report details the step-by-step evolution of Phase 2 (Reasoning Distillation on TPU). Every day of major architectural decisions (ADRs) since late April represents a distinct "attempt" to solve emerging bottlenecks.

May 2026 was largely spent destroying naive assumptions about hardware, metrics, and diffusion-to-LLM alignment.

---

## Attempt 1: The First TPU Onslaught (FSDP, SPMD, and Checkpointing)
**Date:** 04.05.2026 (ADRs 003, 004, 005)
**Goal:** Transition to XLA FSDP to run the large context model without OOM.
**What went wrong:** 
- The classic Data Parallel approach immediately caused OOM. 
- Moving to legacy FSDP caused instability, and moving to SPMD FSDPv2 revealed that XLA replicated batches across cores without explicit sharding, requiring 93GB HBM. 
- The teacher's attention bottleneck limited context to 4096 tokens. 
- Enabling `gradient_checkpointing` for ModernBERT triggered XLA to fall back to eager attention (due to Sliding Window Attention) and rematerialize all layers in parallel, requesting 21GB of temp memory instantly and crashing the TPU.
**Correction:** We explicitly called `xs.mark_sharding` for inputs, permanently capped context at 4096 tokens, and completely disabled `gradient_checkpointing` for ModernBERT on XLA.

## Attempt 2: Loading and Launch Stabilization
**Date:** 05.05.2026 (ADRs 006, 007)
**Goal:** Stabilize the checkpoint loading process and multi-process launch (`torchrun`).
**What went wrong:** 
- Loading weights *after* the FSDP wrapper caused XLA to overwrite sharded parameters with CPU tensors, causing massive OOM. Kaggle checkpoints also had mismatched `_orig_module.` keys.
- Multi-process PJRT initialization led to fatal `ABORTED` lockfile conflicts and `sflag` memory exhaustion due to the complex ModernBERT HLO graph.
**Correction:** Keys were sanitized and weights strictly loaded *before* the FSDP wrapper. We forced `torchrun --nproc_per_node=4` with batch accumulation to reduce `sflag` load per process, alongside strict PJRT environment variables.

## Attempt 3: Single-Process Victory and Aggressive LR
**Date:** 06.05.2026 (ADRs 008, 009)
**Goal:** Escape the remaining lockfile conflicts and break out of high-MSE plateaus.
**What went wrong:** `torchrun` continued to cause fatal lockfile errors and massive I/O delays during weight loading. Additionally, the model was stuck in bad projector initializations when the learning rate dropped.
**Correction:** We abandoned `torchrun` entirely for a single python process with an SPMD mesh, which eliminated lockfile crashes. To force the model out of its plateau, we raised the LR ceiling and floor (`eta_min`) and set rapid cycles (`T_mult=1`).

## Attempt 4: Clean Slate Restart
**Date:** 07.05.2026 (ADR 010)
**Goal:** Restart Reasoning Phase 1 from scratch.
**What went wrong:** We realized the weights were heavily "polluted" from the previous 6 failed sharding/OOM experiments. The W&B history was fragmented.
**Correction:** Restarted the training entirely with stable SPMD and no gradient checkpointing to establish a clean metrics baseline.

## Attempt 5: Breaking the LayerNorm Ceiling
**Date:** 09.05.2026 (ADR 011)
**Goal:** Fix the "scale barrier" keeping intermediate layer MSE artificially high.
**What went wrong:** The student's LayerNorm forced vector norms to ~59, while the teacher's target vectors had norms of ~24. The student could not shrink its vectors to match the teacher without destroying the directional information.
**Correction:** Added a `learnable output_scale` to `FeatureProjector` so the network could independently scale its output.

## Attempt 6: Latent Space Isotropization (Preparing for Diffusion)
**Date:** 10.05.2026 (ADR 012)
**Goal:** Stop "pinching" the latent space with absolute MSE coordinates and make it diffusion-ready.
**What went wrong:** Strict MSE was forcing the 1024D ModernBERT to exactly mimic DeepSeek's absolute numerical coordinates, ruining its own topological capacity.
**Correction:** Dropped MSE on the final layer (l40) in favor of Cosine Similarity and a `Prior Loss` to enforce a Gaussian N(0,1) distribution. Disabled bias in the projector.

## Attempt 7: Trajectory-Aware Distillation
**Date:** 12.05.2026 (ADR 013)
**Goal:** Teach the student "reasoning trajectories" instead of static end-states.
**What went wrong:** Training only on full sequences left the diffusion model without a semantic landscape to traverse (from noise to clarity).
**Correction:** Introduced Growing Masks (25/50/75/100% of text) within a single static XLA shape `[4B, 4096]` and introduced `L_delta` to penalize incorrect semantic jumps between steps.

## Attempt 8: Fixing Padding Metrics and "Student Laziness"
**Date:** 15.05.2026 (ADR 014)
**Goal:** Fix the `L_delta` trajectory distillation.
**What went wrong:** The student found a cheat code: it shrank the delta vector length to near zero, formally minimizing `L_delta` without actually making reasoning steps. Also, our isotropy metrics were an illusion because they included padding tokens.
**Correction:** Fixed masking in validation metrics and added a Magnitude Penalty to `L_delta` to force the student to take full-sized reasoning steps.

## Attempt 9: Centered Cosine to stop Rank-1 Collapse
**Date:** 17.05.2026 (ADR 015)
**Goal:** Fix dimensional collapse.
**What went wrong:** Using absolute cosine similarity pulled all tokens into a single narrow cone (Rank-1 collapse), destroying the isotropy we fought for.
**Correction:** Swapped absolute cosine for Pearson Correlation (Centered Cosine Similarity).

## Attempt 10: Decoupling Absolute Space entirely
**Date:** 19.05.2026 (ADR 016)
**Goal:** Apply the centered cosine revelation globally.
**Correction:** Removed MSE entirely from all intermediate layers, replacing it with Centered Cosine, abandoning absolute space geometry completely.

## Attempt 11: The Illusion of Token Independence (RKD)
**Date:** 20.05.2026 (ADR 017)
**Goal:** Ensure the student learns the actual meaning of the text, not just token-by-token alignment.
**What went wrong:** Aligning individual tokens failed to transfer the internal *topology* (angles and distances between different tokens).
**Correction:** Introduced Relational Knowledge Distillation (RKD) to align pairwise similarity matrices, and Norm Correlation to align the amplitude rhythm.

## Attempt 12: The Geometric Conflict (Removing Absolute Cosine)
**Date:** 21.05.2026 (ADR 018)
**Goal:** Stop the gradient conflict between Prior Loss and Cosine Loss.
**What went wrong:** Prior Loss forces a fully isotropic N(0,1) space, but the Teacher's latent space is low-rank. A high-dimensional isotropic vector is mathematically nearly orthogonal to any fixed low-rank subspace. The optimizer was tearing itself apart trying to satisfy both.
**Correction:** Set `cos_weight = 0.0`. Rely entirely on functional invariants (RKD, delta, norm_corr) and let Prior Loss anchor the distribution.

## Attempt 13: The Error of Architectural Mimicry
**Date:** 23.05.2026 (ADR 019)
**Goal:** Finalize the loss function landscape.
**What went wrong:** We realized a fundamental flaw in layer-by-layer alignment. A Diffusion network (Student) performs *one denoising step* per forward pass. An LLM (Teacher) performs *gradual syntax-to-semantics abstraction* per forward pass. The intermediate layers of these two architectures have entirely different jobs; forcing them to align was destroying the final layer's performance.
**Correction:** Completely decoupled intermediate layers from the teacher. Deleted projectors for layers 20 and 30, focusing 100% of the gradient signal on the final reasoning output (layer 40).

---

## Summary of Phase 2 (So Far)
We have not yet produced a final Reasoning-distilled model. However, 13 distinct attempts have forged a robust engineering foundation:
- A stable TPU v6e XLA pipeline with single-process SPMD.
- "Cheat-proof" metrics that ignore padding and penalize delta-shrinking.
- A conceptually sound loss landscape focusing on topological invariants (RKD, centered norms) rather than absolute coordinates.
- An architecturally correct setup that acknowledges the differences between Diffusion and Autoregressive models by severing intermediate layer alignment.

If Attempt 13 does not show convergence, we can confidently declare that the remaining bottleneck is not in the loss function or architecture, but likely in the hyperparameters or dataset quality.
