<!-- created: 2026-09-05 -->
# ADR 081: Phase 4 Checkpoint Evaluation (8995 vs 10995) and Tensor-Based PACE Dynamic Schedule

## Context and Problem
Following the TPU v5e-8 pipeline optimizations (batch size 64, SPMD, ~45–50 samples/sec, completing the 150k dataset in ~2,344 steps), Phase 4a Hierarchical Noise training was resumed from checkpoint `phase4_step_8995.pth` up to step ~11,700.

During training, telemetry on WandB showed:
1. **Steady Train Loss Decrease**: `loss`, `denoising_loss`, and `prior_loss` continued to decline stably with stable `grad_norm ~ 0.1`.
2. **Apparent Validation Drift**: `val_loss` and `val_denoising_loss` drifted upwards (~0.132 to ~0.137), while `val_layer_divergence` grew monotonically (0.0495 to 0.0510).
3. **Question of Overfitting vs Optimization Artifact**: With only ~1 TPU epoch completed (2,700 steps × 64 = 172k samples), an urgent comparison was needed between checkpoint `phase4_step_8995.pth` and `phase4_step_10995.pth` to determine whether the model was degrading or genuinely learning.

## Investigation and Comparative Evaluation
Both checkpoints were evaluated locally using `evaluate_phase4_checkpoints.py` across multiple noise levels ($t \in [0.1, 0.5, 0.9]$), test languages, and network layers:

| Diagnostic Metric | Step 8995 | Step 10995 | Implication |
|---|---|---|---|
| **Layer Divergence (LIVE)** | `0.0516` | `0.0553` (+7.1%) | Healthy functional specialization of duplicated DUS layers |
| **AdaLN L0 $\Delta\text{scale}$ (attn)** | `0.0485` | `0.1177` (**+142%**) | Sharper conditioning response to timestep $t$ |
| **AdaLN L0 $\Delta\text{scale}$ (mlp)** | `0.0273` | `0.0773` (**+182%**) | Major improvement in timestep sensitivity |
| **Cos Sim ($h_{39}$ vs Clean) $t=0.9$ (LIVE)** | Eng: 0.3335 / Rus: 0.5237 | Eng: **0.4272** / Rus: **0.5690** | Substantial gain on difficult high-noise denoising |
| **EMA Topology at $t=0.9$ (L39)** | $\text{Iso}=0.4832$, $\mathbf{R_1=0.5215}$ (`rank1 high`) | $\mathbf{\text{Iso}=0.6819}$, $\mathbf{R_1=0.1737}$ | **Rank-1 collapse completely eliminated in EMA** (3× drop in $R_1$) |
| **Hierarchical Denoising $t_g=0.5, t>0.7$ (EMA)** | Eng: 0.5264 / Rus: 0.6578 | Eng: **0.7254** / Rus: **0.8634** | Massive +0.20 improvement on masked token denoising |

### Diagnostic Conclusion
The apparent increase in `val_loss` was not caused by overfitting. Instead, the heightened sensitivity of AdaLN modulations (scale deltas growing 2.5–2.8×) increases quadratic deviation penalties on single coordinate outliers, while the **semantic cosine similarity and topological health (isotropy, rank-1 ratio) improved substantially**. Most critically, EMA weights at step 10,995 completely resolved the near-collapse observed at step 8,995.

## Decisions Made

1. **Retain Checkpoint 10995 as Production Foundation**:
   - Checkpoint `phase4_step_10995.pth` is confirmed healthy and superior to step 8,995. Training will resume from step 10,995 onwards.

2. **Tensor-Based PACE Schedule (Zero XLA Recompilation)**:
   - In PyTorch/XLA, updating scalar floats in optimizer ops or using `alpha=float` in `tensor.sub_()` causes constant graph re-tracing (`UncachedCompile`).
   - `pullback_alpha` is now strictly instantiated and maintained as an in-place `torch.Tensor` directly on the TPU device (`requires_grad=False`, NOT an `nn.Parameter`):
     ```python
     self.pullback_alpha_tensor = torch.tensor(float(pullback_alpha), dtype=torch.float32, device=dev)
     self.pullback_alpha_tensor.requires_grad_(False)
     ```
   - In `EMA.step()`, pullback is applied via pure tensor multiplication:
     ```python
     ema_casted = self.shadow[name].to(param.dtype)
     param.data.sub_((param.data - ema_casted) * self.pullback_alpha_tensor.to(param.dtype))
     ```

3. **Dynamic Warmup & Cosine Cyclic Schedule**:
   - **Warmup ($0 \le s < 1000$)**: Linearly decays $\alpha$ from `0.03` to `0.001`. A high initial $\alpha$ strongly tethers live parameters to the stable EMA checkpoint, effectively substituting for LR warmup without touching optimizer parameter groups.
   - **Cyclic Modulation ($s \ge 1000$)**: Cosine cycle between `0.001` (exploration phase) and `0.01` (consolidation phase) with period $T = 2000$ steps (matching 1 dataset epoch):
     $$\alpha = \alpha_{\text{min}} + 0.5 \cdot (\alpha_{\text{max}} - \alpha_{\text{min}}) \cdot \left(1 - \cos\left(\frac{2\pi \cdot (s - W)}{T}\right)\right)$$
   - Enabled `pullback_warmup_on_resume = True` to provide stabilization on every training restart.

4. **Multi-File Checkpoint Evaluation**:
   - Modified `evaluate_phase4_checkpoints.py` to output separate evaluation reports per checkpoint (`evaluation_{ckpt_name}.txt`).

## Consequences and Validation
- Graph compilation on Kaggle TPU remains completely static and cached; no recompilation spikes occur during alpha modulation.
- Real-time visualization of `pullback_alpha` in WandB and console progress bar (`p_alpha`).
- Training is safe to continue on TPU from step 10,995 towards the 40,000 max steps target.
