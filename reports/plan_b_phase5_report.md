# Phase 5 Report: Confidence Head and Geometric Denoising Ensemble

## 1. Architecture Evolution and Rejected Ideas (ADR 87-90)

During the development of Phase 5 (Confidence Head), the main objective was to provide the Orchestrator with a reliable numerical signal to control the diffusion process (stopping, re-noising, calling RAG). In the process, we went through several architectural iterations and concepts.

### Rejected Approaches:
1. **Layer 39 Hacking (ADR 87)**: Initially, it was planned to extract the confidence signal directly from the internal states of the 39th layer. However, this was abandoned in favor of a full-fledged **Confidence Head** with custom Local Self-Attention and a preliminary fusion mechanism, which allowed us to eliminate the "Context Infection" effect (where a single noisy token would zero out the confidence of clean neighbors).
2. **Geometry-Based Diffusion Rollback (Tug-of-War)**: The problem of L2-normalization (ADR 033) was identified — in cases of semantic ambivalence (when the model hesitates between two words), the hybrid vector is pushed out onto the sphere's surface into a void (RawDProx stuck at ~0.55). Attempting to force the $t_{local}$ scheduler to roll back the noise in such situations resulted in an endless loop.

### Final Mechanism: Geometric Ensemble
Instead of rolling back the noise, we transitioned to an **Adaptive ODE solver (Option 2)**. The timer forcibly and monotonically drives $t_{local}$ to zero. Resolving the void problem was delegated to a geometric ensemble consisting of three metrics:
- **RawDProx (Top-1 Sim)**: Cosine similarity of the vector to the closest word in the dictionary.
- **Delta**: The difference in similarity between the Top-1 and Top-2 candidates.
- **ConflictSim (Inter-Token Similarity)**: Cosine similarity between the Top-1 and Top-2 vectors themselves.

## 2. Achieved Results (Based on Tests)
- [Evaluation code](<../experiments\phase 5\evaluate_phase5_metrics.py>)
- [Evaluation results](<../experiments\phase 5\phase5_evaluation_report.txt>)
- [Evaluation demonstration](<https://laeryid.github.io/BEBLaDII/experiments/phase%20/phase5_denoising_demo.html>)
 
Analysis of the logs (`phase5_evaluation_report.txt`) and visualizations (`phase5_denoising_demo.html`) shows that the introduced Orchestrator Decision Matrix successfully handles the classification of final token states. 

We achieved an algorithmic separation of semantic dead ends and harmless syntactic variations:

1. **Clean Token Crystallization (`DECODER_READY (CRYSTALLIZED)`)**:
   Tokens that did not have critical noise (Low Noise) converge successfully and quickly. The RawDProx metric reaches high values (>0.80), and the Orchestrator registers the token as ready.
   *Example: `brown` (RawDProx 0.975), `jumps`, `thousand`.*

2. **Safe Syntactic Variation (`DECODER_READY (SYNTAX_VARIATION)`)**:
   When a token gets stuck between variants with high mutual correlation (`ConflictSim > 0.70`), the system recognizes this as a variation in case or morphology (e.g., `over` vs `OVER`, `miles` vs `Miles`). The Orchestrator ignores the conflict, as the LM Head will safely round such a vector during final decoding.
   *Log example: Token `over` (RawDProx ~0.34, ConflictSim 0.826) — sent to the decoder without invoking the LLM.*

3. **Semantic Conflict / Dead End (`CLM_HELP (SEMANTIC_CONFLICT)`)**:
   If a vector gets stuck in a void (`Delta` is close to zero) and the Top-1 and Top-2 candidates have entirely different meanings (`ConflictSim < 0.30`), the Orchestrator recognizes a diffusion failure.
   *Log example: Token `hockey` vs `הם` (ConflictSim 0.234) or meaningless characters like `г`. The Orchestrator stops denoising attempts at the diffusion level and signals the need for language model intervention (CLM_HELP) for logical context resolution (RAG or inference).*

**Conclusion**: Phase 5 successfully implemented an autonomous stopping and routing mechanism. The model can now safely "give up" on irrecoverable noisy segments by delegating them to the Orchestrator, while confidently crystallizing clean or syntactically variant tokens.
