# Phase 3: Chronicle of Output Projector Development and Diffusion Return Failure (June 2026)

This report details the evolution of Phase 3 (`OutputProjector` training). Each key architectural decision (ADR 031–036) represents an attempt to stabilize the mapping from the DUS space (L40) back to the original token space (X0).

Despite deep theoretical foundation, **we acknowledge the failure of the current approach**: the projector cannot reliably return the diffusion core to the diffusion space without critical semantic loss during recursive cycles.

---

## 1. Acknowledgement of Approach Failure

The fundamental problem lies in the fact that the `OutputProjector` is unable to form a definitive injective mapping from L40 to X0. During recursive inference (cycle: `X0 -> InputProjector -> DUS(L40) -> OutputProjector -> Z_pred -> ...`), starting from the second cycle, the predicted vector drifts towards irrelevant, "garbage" tokens from the Qwen dictionary.

The model does not reconstruct the original geometry of the diffusion space but smears probability across random neighbors. The DUS topology proved too complex or entangled for a simple MLP projector to learn the inverse transformation relying solely on local cosine distances.

---

## 2. Sanity Checks as Visual Confirmation

The problem is visually apparent in the `Sanity Check: alpha-decomposition` results during dictionary building. The Soft Dictionary Matching algorithm (τ=0.1, top-k=8) was used to calculate the weights $\alpha$.

It was expected that the query would confidently match either itself or close synonyms. However, in practice:

**Successful example (rare):**
```text
  Query: ' math' (id=6888)
    ✓ [1] ' math'              α=0.831
      [2] ' incremented'       α=0.028
      ...
```

**Failed examples (vast majority):**
```text
  Query: ' cat' (id=8251)
    ✓ [1] ' cat'               α=0.125
      [2] ' authorize'         α=0.125
      [3] ' Swipe'             α=0.125
      ...
```
*Commentary:* The vector for the word `cat` in the L40 space turned out to be equally close (by cosine) to completely random words (`authorize`, `Swipe`), which results in a uniform smearing of $\alpha$ weights (0.125 each across top-8). The semantic signal is destroyed.

---

## 3. Code Logic: How Phase 3 Was Conducted

The entire Phase 3 pipeline is concentrated in the `experiments/phase 3` folder. The logic is divided into dictionary precomputation and TPU training.

### 3.1. Dictionary Precomputation (`build_dictionaries.py`)
To avoid running the entire DUS on every step, we build static dictionaries (mappings) once on CPU/GPU:
1. Take the entire Qwen embedding table (152,064 tokens).
2. Pass each token through the `InputProjector` $\rightarrow$ obtain the **`D_X0`** dictionary.
3. Pass it further through 40 `DUS` layers $\rightarrow$ obtain the **`D_L40`** dictionary.
Both dictionaries have a shape of `[152064, 1024]`. The goal of the OutputProjector is to learn the function $f(D_{L40}) \approx D_{X0}$.

### 3.2. Soft Dictionary Matching and Whitening Logic
The algorithm for calculating the target point `Z_hat_target` on the sphere is implemented in the `soft_dictionary_matching` function.
- The `D_L40` space is highly anisotropic (rank1_ratio ≈ 0.7). If we compute the cosine directly, all tokens appear identical.
- **Whitening:** We calculate the global mean (`whitening_mu`) and standard deviation (`whitening_sigma`) for `D_L40`. Then we standardize the vectors $L40_{sphere} = (L40 - \mu) / \sigma$ and normalize them onto the sphere.
- Neighbor search occurs in this "whitened" spherical space. We compute `scores = (L40_norm @ D_L40_sphere.T) / tau`, take `top-k`, and apply `softmax`, obtaining the weights $\alpha$.
- **Reconstructing $Z_{target}$:** These weights $\alpha$ are applied to the **original** vectors from the `D_X0` dictionary. $Z_{hat} = \sum \alpha_i \cdot D\_X0_i$. The target vector is normalized and multiplied by the weighted expected norm.

### 3.3. Loss Function and Anchors (`train_phase3.py`)
Besides training on contextualized tokens from the batch (where the target is formed via soft-matching), the script uses **Anchor Loss**:
- A hybrid anchor sample is selected (in-distribution tokens + random ones).
- RKD-loss (Relational Knowledge Distillation) is applied to them: it minimizes the difference between the pairwise cosine matrix for OP outputs and the matrix for target `D_X0`. This protects against the "Hubness problem" and vector collapsing.
- Angular error component: we center the vectors ($Z - mean\_X0$) and penalize for cosine deviation (`1 - cos_sim`), plus add a small Huber penalty on the norm.

---

## 4. Attempt Chronicle (Based on ADR 031 - 036)

### Attempt 1: Baseline Soft Dictionary Matching
**Date:** 11.06.2026 (ADR 031)
**Goal:** Use Soft Dictionary Matching to create continuous targets.
**What went wrong:** Initial runs showed the projector converges, but predictions rapidly degenerate during recursive inference.

### Attempt 2: Whitening the L40 Space
**Date:** 13.06.2026 (ADR 032)
**Goal:** Solve the problem where Softmax distributed uniform weights `k_eff ≈ 27.6 out of 28`.
**What went wrong:** The L40 space was tightly pulled toward a single "pole".
**Correction:** Introduced diagonal Whitening of L40 before searching for neighbors by cosine, which allowed stretching semantic axes and lowering `k_eff`.

### Attempt 3: Spherical Latent Space
**Date:** (ADR 033)
**Correction:** Introduced strict spherical topology and Slerp (Spherical Linear Interpolation) to control the diffusion space. Added LayerNorm at the OP output.

### Attempt 4: Transition to Angular Loss Function
**Date:** 15.06.2026 (ADR 034)
**Goal:** Eliminate false convergence plateau.
**What went wrong:** Huber-loss minimized Cartesian distance, giving a low loss (`0.009`) but terrible angle (cos_sim ≈ 0.81, 11° deviation).
**Correction:** Replaced Huber with `1 - cos_sim` (angular loss) with a weak norm penalty (`cosine+norm`).

### Attempt 5: Centered Cosine and Contrastive Loss (InfoNCE)
**Date:** 16.06.2026 (ADR 035)
**Goal:** Defeat Rank-1 collapse and Garbage Collapse (drift into garbage tokens).
**What went wrong:** `D_X0` anisotropy forced the gradient to fit a massive mean vector. Anchor loss on the entire dictionary overloaded the network.
**Correction:** Vectors were centered (`Z - mean_X0`) before computing cosine (Pearson Correlation). Dynamic anchor selection and InfoNCE loss were introduced to repel from "garbage" clusters.

### Attempt 6: Abandoning InfoNCE in favor of Angle-wise RKD
**Date:** 17.06.2026 (ADR 036)
**Goal:** Fix loss stalling and geometry destruction.
**What went wrong:** InfoNCE (contrast_loss) conflicted with centering and suffered from "False Negative Repulsion", aggressively pushing away semantically close synonyms that didn't make the top-k.
**Correction:** Completely abandoned InfoNCE. Replaced with Angle-wise RKD on anchor tokens — the `[B, B]` pairwise cosine matrix of predictions is aligned with the target matrix. This instantly pushed hubs apart without destroying local topology.

---

## 5. Required Graphs for the Report

For full visualization of the failure and confirmation of architectural conclusions, the following graphs must be attached to the final documentation (or W&B dashboard):

1. **Alignment Quality Metrics (Geometry):**
   - `train/cos_sim` vs `train/anchor_cos_sim` — will show that despite cosine growth on train, anchors (isolated tokens) align significantly worse.
   - `train/loss_ctx` and `train/loss_anchor` — convergence of contextual and isolated loss.
   - `train/rkd_loss` — decay curve of the pairwise cosine distance error.

2. **Soft Matching Statistics (Cause of Failure):**
   - `train/k_eff` (Softmax Effective Entropy) — will demonstrate how whitening helped reduce `k_eff`, but semantics still remained smeared.
   - `train/top1_self` — percentage of weight $\alpha$ given to the true token. It is critical to show its low value.

3. **Topological Invariants of the Space:**
   - `train/op_norm_cv` — norm coefficient of variation (will show space contraction).
   - *Offline metric:* `Cycle Cosine Degradation` (graph of cosine similarity between $Z_{pred}$ and the ideal target from cycle to cycle $1 \rightarrow 5$). The main proof of recursive degradation.

## Summary
The Phase 3 pipeline is flawlessly implemented from an engineering perspective (XLA optimizations, RKD, Whitening, Centered Cosine); however, the very hypothesis that semantics can be recovered from the compressed L40 space via a simple projection layer was not confirmed. A revision of the diffusion core paradigm or the introduction of skip-connections from early DUS layers is required.
