<!-- created: 2026-07-25 -->
# ADR 063: Phase 3 Advanced Diffusion Techniques Integration (LD4LG & DiffuSeq-v2)

## Context and Problem
During the development of Phase 3 (canonical spherical diffusion), we needed to further improve the stability and convergence speed of the model. Analysis of relevant papers on text continuous diffusion (LD4LG, DiffuSeq-v2) revealed several proven techniques:
1. Self-Conditioning: using the model's own x0_pred prediction from the previous step as additional conditioning significantly accelerates the crystallization of latents at low noise levels.
2. Shifted Noise Schedule: shifting the sampling distribution of t during training towards higher noise levels helps the model better learn denoising in complex regions (combats the "deafness" problem).
3. Mid-noise Diagnostics: lack of metrics to track denoising quality in the middle noise range (0.3 <= t <= 0.7).

## Decisions Made and Lessons Learned
- Architectural Preparation for Self-Conditioning: Added a self_cond_proj layer (with zero initialization) to BEBLaDIIPhase3, which allows mixing x0_pred into z_noisy. Thanks to the zero initialization, the current behavior of the model remains unchanged, ensuring backward compatibility and the possibility of gradually enabling this mechanism after achieving baseline convergence.
- Implementation of Shifted Noise Schedule: Added the t_sample_alpha=2.0 parameter to the forward function. When sampling t, the formula u = 1.0 - rand()^alpha is used, which shifts the distribution towards t_max, focusing the model's attention on the early (highly noisy) generation stages.
- Mid-range Diagnostics: Added the cos_sim_t_mid metric to monitor cosine similarity in the 0.3 <= t <= 0.7 range, providing a more complete picture of the training process.

## Impact
- Positive: The architecture is ready for Self-Conditioning without interrupting current training or losing checkpoint compatibility. The shifted t schedule should improve the model's denoising ability in the early diffusion steps. More detailed diagnostics will simplify debugging.
- Negative: Slight complication of the forward code and the potential need to fine-tune the t_sample_alpha hyperparameter.