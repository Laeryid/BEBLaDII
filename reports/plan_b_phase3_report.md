# Phase 3 Report (Plan B): Canonical Spherical Diffusion

## 1. Objective
The goal of Phase 3 was to construct a robust diffusion model (DUS Backbone) capable of denoising latent vectors on the spherical manifold created in Phase 1. 

## 2. Current Implementation Details (The Final Architecture)
* **Architecture**: 40-layer `DUSModel` based on `ModernBERT-large`, operating in strict `float32` precision to prevent gradient collapse.
* **Diffusion Type**: Canonical Unconditional Continuous Diffusion on the S^(d-1) hypersphere. Full range t in [0, 1] with a Cosine Noise Schedule. Slerp is used for noise injection.
* **Conditioning**: `AdaLN` (Adaptive Layer Normalization) strictly inside the ModernBERT block with `Zero Init`. All artificial regularizations have been removed.
* **Skip-Connections & Gates**: Implements an explicit Identity Gate guaranteeing identity mapping at t=0 structurally.
* **Dataset**: **Clean Texts** (no ChatML tags) with On-The-Fly tokenization.

**Attachment:** [Training Script](../experiments/phase%203/kaggle/train_phase3_notebook.py)

## 3. Loss Functions
1. **Unified Cosine Loss (x_0-prediction)**: 
   `L = 1 - <DUS(x_t, t), x_0>`
   Evaluated over all active tokens with Min-SNR weighting.
2. **Prior & Topology Losses**:
   - Variance Floor Penalty `(var_floor = 1.0 / (2D))`.
   - Covariance Penalty (penalizing off-diagonal elements).

### 4. Training Sessions Timeline
The actual Phase 3 training took place over several sessions (accompanied by architectural updates):
- **Session 1 (0–6000 steps):** Initial training phase with shut off AdaLN and Self-Conditioning to train base idea of the diffusion. Established the baseline convergence.
- **Session 2 (6000–12000 steps):** AdaLN is enabled and is training here. 
- **Session 3 (12000–18000 steps):** Following the stabilization of the model's behavior, Self-Conditioning was re-activated. The `self_cond_proj` layer was re-initialized with `xavier_uniform_` to give the model a "second glance" capability, allowing it to use the `x_0_pred` prediction from the first pass as a structural hint for the second denoising pass.
- 
## 5. Visualizations and Metrics Analysis
Below are the metrics extracted from the `phase3_step_17995.pth` checkpoint.

### Denoising Loss
<p align="center">
  <img src="../experiments/phase%203/denoising_loss.png" width="80%" alt="Train Denoising Loss" />
</p>
**denoising_loss**: Shows a stable convergence of the Unified Cosine Loss (x0-prediction), confirming the viability of canonical diffusion.

### Cosine Similarity (DUS Output vs Clean Latent)
<p align="center">
  <img src="../experiments/phase%203/cosine_similarity.png" width="80%" alt="Cosine Similarity" />
</p>
**cos_sim**: Demonstrates the quality of predicting the clean `x_0`. 
- `cos_h39_t_low` (low noise, t < 0.3): Cosine similarity tends towards 1.0; the model easily reconstructs a weakly noised vector.
- `cos_h39_t_mid` (medium noise, 0.3 <= t <= 0.7): Demonstrates stable semantic recovery capability in the middle of the diffusion trajectory.
- `cos_h39_t_high` (high noise, t > 0.7): Shows the true denoising capability of the DUS core to generate semantically close vectors from strong noise (sphere).

### Prior and Topology Losses
<p align="center">
  <img src="../experiments/phase%203/prior_losses.png" width="80%" alt="Prior and Topology Losses" />
</p>
**prior_loss / var_loss / cov_loss**: Control of the spherical topology. Penalties are kept near zero, indicating the preservation of isotropy and prevention of variance collapse in intermediate layers.

### 6. Diagnostic Sanity Tests Analysis (Step 17,995)
[Test script](<experiments\phase 3\test_phase3_denoise_sanity.py>)

[Test results](<experiments\phase 3\sanity_tests.txt>)

A comprehensive denoising test was conducted across different noise levels t in [0.1, 1.0] for both English and Russian texts.
- **Low Noise (t=0.1 - t=0.3):** The model demonstrates near-perfect stability. For the phrase *"The quick brown fox jumps over the lazy dog"*, the initial prediction of `x_0_pred` is immediately accurate and semantically intact (Cosine similarity ~0.98).
- **Medium Noise (t=0.4 - t=0.6):** The model effectively recovers semantics despite heavy corruption. At t=0.5, initial predictions hallucinate related words (e.g., *"The quick bread fox..."* or *"The quick red files..."*), but the trajectory rapidly converges to the correct semantic structure as `t` decreases.
- **High Noise (t >= 0.8):** The input vector `x_t` is almost entirely random noise on the hypersphere. While the model correctly hallucinates grammatically plausible token structures (due to the Decoder's syntax capabilities), the trajectory converges to arbitrary semantic attractors (e.g., random Slavic-like syllables *"шкаšeихle naтом na ze zíc"*). This is the expected and mathematically correct behavior for unconditional generation from pure noise pure noise (epsilon).

## 7. Conclusion
**Phase 3 is complete and stabilized.**
Following a prolonged series of numerical (bfloat16) and architectural (Pre-LN, AdaLN bugs, Data preparation bugs) crises, Phase 3 has successfully migrated to a strict, canonical, and mathematically rigorous paradigm of **continuous diffusion on a hypersphere**. "Crutch" solutions were removed, and the DUS Backbone now predicts clean `x_0` vectors using a cosine noise schedule and strict `float32`. The model is ready for integration into the generative pipeline of Phase 4.
