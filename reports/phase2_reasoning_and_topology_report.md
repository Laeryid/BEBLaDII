# Phase 2: Reasoning and Topology Report (May-June 2026)

## 1. Introduction and Overview
Phase 2 (Reasoning Distillation) was a critical stage in transitioning from classical autoregressive legacy to a latent diffusion architecture. Initially, we planned to naively copy the knowledge (weights and activations) from a strong logical teacher into our 40-layer `Diffusion Backbone` using standard MSE. However, this approach led to a catastrophe: the diffusion manifold was incompatible with the rigid, absolute coordinate grid of the teacher.

The main outcome of this phase was a complete overhaul of the distillation paradigm. We moved from "copying absolute coordinates" to **Topological Alignment**. We successfully formed a spherical, isotropic latent space that preserves the teacher's logic but is fully prepared for the denoising process in the next phases. Phase 2 is officially considered successful and fully completed.

## 2. Datasets Used
As sources of "pure logic" for the Reasoning subset, we formed a strictly curated data mixture:
- **`Magpie-Reasoning-V2`**: Sampled down to exactly **80,000** examples.
- **`OpenThoughts-114k`**: Used in full (subject to token limits).
- **`CulturaX` (RU/CS)**: We incorporated **30,000** examples (15k Russian, 15k Czech) to maintain multilinguality and structural diversity.

**Length Filtering and Constraints:**
Due to strict TPU HBM limits and the architectural constraints of the context window, the data underwent rigorous length filtering:
- CoT logic chains (`magpie` and `open_thoughts`) were filtered to be strictly **under 4096 tokens**.
- `CulturaX` samples were filtered specifically to fall within the **3000-4000 token** range to act as high-density long-context anchors.

**Crucial Architectural Detail: The `<|thought|>` Tag Injection**
To cleanly separate the reasoning process from the conversational wrapper, we explicitly modified the dataset format during preparation. Immediately after the `<|im_start|>assistant\n` tag, we injected the special `<|thought|>\n` token into all CoT examples. This tag serves as a mandatory, hardcoded geometric anchor that forces the model to transition into the "Latent Thought Process" (System 2) before generating any response. All datasets were then pre-tokenized using Qwen2.5 and saved in Parquet format for high-speed TPU loading.

## 3. Loss Function and Metrics
Blindly applying `MSE` across all layers caused a gradient conflict with the diffusion prior. The final Loss Landscape was engineered as follows:

1. **Decoupling Intermediate Layers**: Intermediate layers (20, 30) were completely decoupled from the teacher (projectors were removed). The nature of diffusion (one denoising step per pass) and autoregression (deepening semantics per pass) are fundamentally different. We focused the gradient signal exclusively on the raw output of the 40th layer.
2. **Centered RKD (Relational Knowledge Distillation)**: Instead of matching the absolute coordinates of vectors, we began matching the *angles and distances* between tokens. Pre-centering the vectors saved us from collapsing into a narrow cone (Rank-1 Collapse).
3. **Huber Loss for Covariance**: Using MSE for covariance caused an O(S³) gradient explosion. Replacing it with an XLA-friendly Huber Loss (manually implemented via `torch.where`) stabilized computations on the TPU.
4. **Trajectory Delta ($L_{\Delta}$)**: This was a highly unconventional and critical move. Because we are distilling an Autoregressive teacher into a Diffusion student, static matching is insufficient—diffusion requires a *trajectory* (from noise to clarity). To simulate this without breaking the rigid static shapes of XLA, we introduced **Growing Masks**. Within a single static batch shape of `[4B, 4096]`, we applied progressive masking (e.g., revealing 25%, 50%, 75%, and 100% of the reasoning text). The student was forced to predict the semantic *step* (the delta vector) from a heavily masked state to a clearer state, effectively learning the "direction of thought." However, the student quickly found a shortcut: outputting a near-zero delta vector to mathematically minimize the error. To counter this, we added a strict **Magnitude Penalty** that penalized any artificial shrinking of the step size, forcing the model to take full, meaningful reasoning steps.

## Mathematical Formulation of the Loss Landscape
The final topological loss function for the 40th layer is defined as:

$$ \mathcal{L}_{total} = \lambda_{RKD}\mathcal{L}_{RKD} + \lambda_{\Delta}\mathcal{L}_{\Delta} + \lambda_{prior}\mathcal{L}_{prior} $$

Where the components are:
- **$\mathcal{L}_{RKD}$ (Centered Relational Knowledge Distillation)**: Ensures the student manifold mimics the internal topology (distances and angles between tokens) of the teacher. Vectors are explicitly centered before comparison to prevent conflict with the prior and avoid Rank-1 Collapse.
- **$\mathcal{L}_{\Delta}$ (Trajectory Delta Loss)**: Controls the semantic trajectory. It consists of a Centered Cosine penalty (ensuring the student's semantic step points in the same direction as the teacher's) plus the Magnitude Penalty (preventing the delta length from collapsing to zero).
- **$\mathcal{L}_{prior}$ (Prior Loss)**: Enforces the spherical, isotropic nature of the base diffusion space. It is a hybrid regularization containing a Variance Loss (using Huber Loss and a Variance Floor to prevent variance collapse) and a Scale-Invariant Covariance Loss (penalizing correlation between dimensions to ensure isotropy).

## 4. Target Properties and Achievements
**Our Goals:**
- Isotropy and the absence of preferred directions.
- A Spherical Manifold compatible with N(0, 1) Gaussian noise.
- Robustness against Shortcut Learning.

**Achievements:**
- Rank-1 Collapse was completely suppressed. Vectors distributed evenly across the surface of the hypersphere (monitored via Norm CV).
- Shortcut Learning was halted by introducing a Variance Floor and Scale-Invariant Covariance.
- A stable TPU v6e XLA SPMD (single-process) infrastructure was established.

## 5. Two-Stage Training
The training process proved nonlinear and split into two major stages:
- **Run 1 (Steps 0 - 18,000)**: We hit a plateau. The `rank1_ratio` metric was dropping extremely slowly. Shortcut Learning emerged: the student learned to "hide" from penalties by shrinking the vector variance to near zero (variance collapse).
- **Run 2 (Restart from 18,000)**: We introduced a Variance Floor and Scale-Invariant Covariance. To accelerate the isotropization of the space (independent of the anisotropic teacher), the `cov_loss` weight within the prior was increased from 0.1 to 0.3. We also migrated the data from EU GCS buckets to the US region (`us-central2`), which radically accelerated loading and eliminated intercontinental TPU-GCS latency.

## 6. Required Graphs (Wandb)
For visual confirmation of the topological alignment and completeness of the report, please download and attach the following Wandb plots as images:
1. `val/l40_rank1_ratio` — The primary indicator of victory over manifold collapse. Should show a confident decline below 0.40.
2. `train/loss` — The overall convergence trend (the dynamics after the 18k restart are particularly telling).
3. `train/delta_cos` — An indicator of the correct "direction" of the prediction step.
4. `train/full_l40_prior` — An indicator of the "roundness" of the space (penalty for exceeding Gaussian bounds).
5. `val/l40_norm_cv` — The coefficient of variation for norms, confirming that vectors lie on the surface of a hypersphere.

## 7. Linear Probing (NLI) Test: Evaluating Latent Space Quality
Because we entirely decoupled `MSE` and abandoned direct language output in Phase 2, we need a metric to confirm that the student actually "understands" the logic, rather than just memorizing the geometry of noise.

To achieve this, we will employ a **Linear Probing test on Natural Language Inference (NLI) tasks**:
- **Methodology**: We freeze the weights of the 40-layer backbone (Backbone frozen, no gradients pass through).
- **Adapter**: A simple, lightweight fully connected linear network (Linear Probe) is placed on top of the raw 40th layer.
- **Data**: We feed classic logical benchmarks (MNLI, SNLI) into the network.
- **Evaluation**: If the linear classifier can solve logical relationship classification tasks (Entailment / Contradiction / Neutral) with high accuracy, it will prove that the raw latent space contains easily extractable, linearly separable logic and semantics.

This test will serve as the final bridge between the abstract topology of Phase 2 and the onset of diffusion processes in Phase 3.
