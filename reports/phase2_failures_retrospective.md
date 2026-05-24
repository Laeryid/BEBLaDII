# Phase 2: Chronicle of Failed Attempts and Approach Evolution (May 2026)

## Attempt 1: Head-on TPU Attack (Scale and OOM Issues)
**Period:** Late April — Early May (ADR-003, ADR-004, ADR-010)
**Initial Goal:** Launch Data Parallel distillation on TPU v6e with a large context.

**What went wrong:**
- Classic Data Parallel led to `RESOURCE_EXHAUSTED` (OOM), because almost all memory (14 GB out of 16 GB per core) was consumed by the frozen teacher weights (DeepSeek-R1-7B in BF16), leaving no room for student activations.
- Transitioning to the legacy `XlaFullyShardedDataParallel` caused instability, weight loss during loading (Loss > 5000), and OOMs on individual cores during rematerialization.
- XLA implicitly replicated batches across all cores (instead of sharding), leading to exponential memory growth (up to 93 GB per batch).
- The teacher's Attention Bottleneck (GQA without FlashAttention in XLA) prevented us from reaching the desired context size (8k+).

**How the approach changed:** We realized the physical limits of TPU v6e-4. We had to strictly constrain the context to 4096 tokens, rewrite the pipeline to the modern `SpmdFullyShardedDataParallel` (FSDPv2) with explicit `mark_sharding` for batches, and completely restart the training (discarding the "polluted" weights from failed experiments), accepting the loss of time.

---

## Attempt 2: The Illusion of Convergence and Student "Cheating"
**Period:** Mid May (ADR-014, ADR-015, ADR-016)
**Goal:** Achieve latent space alignment by minimizing step difference (delta) and MSE.

**What went wrong:**
- **Collapse of Delta (Student Laziness):** The student model found a way to formally reduce the Loss without learning anything. It simply shrank the length of the delta vector to almost zero (Magnitude Ratio dropped to ~0.3), preserving only the direction.
- **Illusory Metrics:** We thought isotropy was normal, but it turned out the metric was calculated including padding tokens, hiding the real picture.
- **Rank-1 Collapse:** Training degraded into a dimensionality collapse. Using absolute cosine similarity caused all tokens to collapse into a single narrow cone.

**How the approach changed:** Complete abandonment of MSE (absolute coordinates) on intermediate layers. Introduction of centered cosine similarity (Pearson Correlation) and delta magnitude penalties to force the student to make full, "sweeping" reasoning steps comparable to the teacher, rather than just guessing the direction.

---

## Attempt 3: The Illusion of Token Independence (Lost Geometry)
**Period:** Late May (ADR-017)
**Goal:** Align the student's individual output token vectors with the teacher's.

**What went wrong:** Aligning tokens individually did not give the student an understanding of the internal topology of semantic connections in the text. The student could generate vectors close to the teacher's, but the mutual angles and distances between tokens (the geometry of the latent space) were distorted. Moreover, the dynamic of token informativeness (vector norm) was lost.

**How the approach changed:** We realized that we need to distill the very structure of the latent space. We introduced:
- **Relational Knowledge Distillation (RKD)** — to align pairwise angles (similarity matrices) between tokens.
- **Norm Correlation Loss** — to transfer amplitude dynamics along the sequence.

---

## Attempt 4: The Error of Architectural Mimicry
**Period:** Late May (ADR-019)
**Goal:** Layer-by-layer distillation: bind the student's intermediate layers (l20, l30) to the corresponding intermediate layers of the teacher.

**What went wrong:**
- The gradient signal from intermediate layers pulled the network in different directions and hindered the convergence of the final 40th layer.
- We realized a fundamental flaw: a diffusion network (student) performs *one denoising step* per pass, whereas an LLM (teacher) performs *gradual abstraction from syntax to semantics* per pass. The intermediate layers of diffusion should physically not resemble the intermediate layers of an LLM! By aligning them, we forced the network to solve a meaningless task.

**How the approach changed:** Complete decoupling of intermediate layers from the teacher. The projectors `feat_proj_20` and `feat_proj_30` were removed. We kept only the isotropy regularization on raw vectors and focused 100% of the useful gradient signal on the final (40th) layer.

---

## Month Summary: What did we achieve?
May 2026 was spent not on getting a finished model, but on systematically destroying our naive architectural assumptions:

1. **Hardware:** We thought TPU v6e could handle anything, but hit the harsh limits of HBM and XLA SPMD.
2. **Metrics:** We thought a falling Loss meant learning, but the student found ways to deceive us through delta shrinking and dimensionality collapse (Rank-1 collapse).
3. **Topology:** We thought copying tokens was enough, but realized that without relational distillation (RKD), the structure of meaning is lost.
4. **Architecture:** We thought layer-by-layer alignment was a silver bullet, but forgot that the nature of diffusion (Student) and autoregression (Teacher) is fundamentally different.

**Phase 2 Summary:** We haven't finished the phase yet, and there is no final model. However, this month of "failed attempts" gave us a working and stable XLA FSDP pipeline, bulletproof metrics (resistant to student "cheating"), new RKD/Norm losses, and most importantly, an **architecturally correct distillation scheme** (without binding intermediate layers). The current architecture is cleared of conflicting gradients and false illusions. If the latest run (based on ADR-019) does not show convergence, it will mean the problem lies beyond the loss function and architecture (most likely in the data or hyperparameters).
