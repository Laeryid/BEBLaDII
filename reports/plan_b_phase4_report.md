# Phase 4 Report: Hierarchical Noise Diffusion & Semantic Skepticism

This report is based on the scripts used in this phase, training quality evaluation, and architectural decisions (ADR 75-86), and reflects the results of Phase 4 training.

## 1. Core Training Concepts (Phase 4a)

In Phase 4, we transitioned from canonical diffusion with a uniform noise level for all tokens (Phase 3) to hierarchical noise. The model learns contextual restoration (in-painting) and evaluates the confidence level of each individual token considering the global context.

### Hierarchical Noise and False Confident Tokens ([ADR 076](../.know/decisions/076_phase4_hierarchical_noise_per_token_diffusion.md), [ADR 083](../.know/decisions/083_phase4_semantic_skepticism_scl_and_anchors.md))
To teach the model semantic skepticism (distrust of doubtful tokens), the following mechanisms were introduced:
*   **Hierarchical Noise**: A global parameter $t_{global} \sim U(0, 1)$ was introduced, defining the batch "temperature", alongside individual token noise levels ($t_{actual}$ and $t_{reported}$).
*   **False Confident Tokens (Liars)**: Tokens with high actual noise $t_{actual}$, but reporting low noise $t_{reported} \in [0.02, 0.15]$. The proportion of such tokens $p_{false}$ scales with $t_{global}$, and they are only added at a medium noise level $0.3 < t_{global} < 0.7$. This is necessary for the model to understand that in the middle of the diffusion process, new information might force it to change tokens it was previously confident in.
*   **True Anchors**: To prevent optimizer cheating (Data Leak), 10% of tokens always remain "honest anchors" with a true low $t_{actual}$ regardless of $t_{global}$. This forces the Attention mechanism to analyze the semantic context, rather than simply relying on the AdaLN numerical pattern, assuming at certain stages that low-noise tokens just shouldn't be trusted.

### Loss Function Modifications ([ADR 076](../.know/decisions/076_phase4_hierarchical_noise_per_token_diffusion.md), [ADR 083](../.know/decisions/083_phase4_semantic_skepticism_scl_and_anchors.md))
*   **Min-SNR Removal**: Min-SNR weighting was completely removed, as tokens with high $t_{actual}$ are critical for contextual restoration. (Min-SNR used to change the error weight for highly noised tokens so as not to hinder the model from learning on more important signals for it).
*   **Target-Aware Angular SCL**: A contrastive regularization with a "Half-Angle Margin" was introduced to overcome model "laziness" (copying noise). Repulsion from the noisy input only works in the first half of the diffusion path, zeroing out near the clean target $z_{clean}$, which eliminates gradient conflicts. This serves as an additional signal for the model — if it concludes that a token is falsely clean, it should try to change it.

## 2. TPU Training Optimization and Stabilization ([ADR 077](../.know/decisions/077_pace_optimizer_pullback_alpha_math_fix.md), [ADR 080](../.know/decisions/080_phase4_ema_validation_resume_fix_and_pace_pullback_calibration.md), [ADR 081](../.know/decisions/081_phase4_checkpoints_evaluation_and_pace_schedule.md), [ADR 082](../.know/decisions/082_tpu_precision_floor_fix_and_lr_restoration.md))

During scaling to TPU v5e-8 (Kaggle), several issues with metric stagnation and gradient collapse were resolved:

*   **PACE Optimizer Fine-Tuning**: The `pullback_alpha` value was radically reduced (to 0.001) to balance it with the exponential moving average (EMA) update rate. This eliminated the model "freezing".
*   **Tensor-Based PACE Schedule**: A cyclic schedule (Cosine Cyclic Schedule) for `pullback_alpha` on TPU tensors was implemented, avoiding XLA graph recompilations.
*   **EMA Validation Fix**: Fixed a bug where `LIVE` weights were measured during validation instead of `EMA` weights. In-place restoration of EMA shadow tensors from checkpoints on TPU was introduced to preserve FSDP sharding.
*   **Dropping bfloat16 in Optimizer (Precision Floor Fix)**: It was found that the `XLA_USE_BF16=1` flag caused zeroing of small AdamW optimizer gradients on TPU (gradient collapse). The model, AdaLN, and AdamW were transitioned to strict `float32`. Afterwards, the Learning Rate was restored to optimal GPU values (`dus_learning_rate = 2e-5`).

## 3. Loss Function Graphs

![Denoising Loss](../experiments/phase%204/phase4_components_loss.png)
![Cosine Similarities](../experiments/phase%204/phase4_cosine_similarities.png)

<!-- comment
GPU
2026-08-30 11:02:41    0-4000
2026-08-30 19:21:40    4000-9000
2026-09-05 20:12:20    9000-15000
2026-09-09 06:55:43    15000-21000
2026-09-09 19:29:24    21000-23000
2026-09-09 23:30:46    23000-30000
2026-09-12 09:50:38    30000-36000
2026-09-12 22:29:31    36000-42000
2026-09-13 14:47:57    42000-44000
2026-09-13 19:04:38    44000-50000
TPU
16-09-2026 21:19:32    50000-62000
17-09-2026 12:19:32    62000-73000
18-09-2026 01:12:21    73000-86000
-->

The graphs show noticeable changes in dynamics (jumps and trend breaks) at the following key moments:

*   **Step ~21,000 ([ADR 083](../.know/decisions/083_phase4_semantic_skepticism_scl_and_anchors.md) — Semantic Skepticism & True Anchors):**
    True Anchors (10% of tokens are forcibly made clean) were introduced to eliminate Data Leak. The model stopped guessing lying tokens by AdaLN numerical values and started using the attention mechanism to check the context. Target-Aware Angular SCL (half-margin repulsion) was also introduced.
*   **Step ~42,000 ([ADR 085](../.know/decisions/085_noise_aware_context_trust_and_isolation.md) — Noise-Aware Context Trust):**
    Curriculum Batching (artificial islands of clean tokens in a sea of noise) and an isolation penalty ($L_{isolation}$) were introduced. The model learned to physically disconnect attention weights (K-vectors) from heavily noised context to avoid hallucinations.
*   **Step ~50,000 (Transition to TPU XLA):**
    We migrated training from GPU to Kaggle TPU. The sharp change in the graphs here is related to an 8-fold increase in batch size (due to Data Parallelism / FSDP). In addition, the cyclic Learning Rate schedule was abandoned in favor of the PACE optimizer to maintain stability and avoid constant XLA graph recompilations.
*   **Step ~73,000 ([ADR 086](../.know/decisions/086_type_c_full_range_and_tglobal_cap.md) — Type C Full Range & t_global Cap):**
    Restrictions on noise generation were lifted: $t_{actual}$ started being sampled across the full `[0, 1]` range randomly. This forced the model to learn to restore meaning from clean prompt anchors at all generation stages. In previous runs, the model didn't see examples of what to do with highly noised tokens at the final diffusion stages. Additionally, a limit of $t_{global} \le 0.99$ was set to prevent meaningless training on purely 100% noise.

## 4. Latest Checkpoint Analysis (step 85995)
- [Evaluation code](<../experiments\phase 4\evaluate_phase4_checkpoints.py>)
- [Evaluation results](<../experiments\phase 4\local_checkpoints\evaluation_phase4_step_85995.txt>)


Metrics were extracted from the `phase4_step_85995.pth` checkpoint, and the evaluation report was analyzed:

*   **Topology and Identity**: In the EMA model at a high noise level ($t=0.9$), `Rank1` on the 39th layer decreased, indicating no collapse (Iso=0.5050, R1=0.2613). The model's output has high isotropy, which meets the properties of the latent space from Phase 1.
*   **AdaLN Sensitivity**: High sensitivity of AdaLN modules to the time parameter $t$ is observed, $\Delta scale$ reaches $0.09-0.20$, confirming the active influence of diffusion conditioning.
*   **Hierarchical Token Denoising**: The model successfully differentiates tokens. At $t_{global} = 0.5$, the anchor confidence for English text is 0.9995, while False Confident tokens are recognized and denoised (0.9996). 
*   **Multi-Step Diffusion (Slerp 25 steps)**: When sampling from 100% noise (t=1.0), the cosine similarity of the generated texts with the original (Baseline) is $\sim 0.84 - 0.91$, confirming the model's ability to restore text semantics from scratch.

## Conclusion
In Phase 4, the diffusion core successfully learned to denoise the canvas under its non-uniform noising. Now the Confidence Head and the Orchestrator will be able to control which tokens to consider trustworthy, and which ones need further work.

The diffusion core has serious problems clearing tokens whose initial noise level is close to 1. It does not fill them with a semantically and logically justified value, which could be either a desired or undesired behavior in the end. For now, it has been decided to accept this behavior and consider Phase 4 successfully completed.
