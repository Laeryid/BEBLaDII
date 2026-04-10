<p align="center">
  <img src="Logo.png" width="400" />
</p>

<h1 align="center">BEBLaDII</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Logic-System%202-blue" alt="System 2" />
  <img src="https://img.shields.io/badge/Model-Latent%20Diffusion-orange" alt="Latent Diffusion" />
  <img src="https://img.shields.io/badge/Architecture-ModernBERT-green" alt="ModernBERT" />
  <img src="https://img.shields.io/badge/Memory-RAG-red" alt="RAG" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-ee4c2c" alt="PyTorch" />
</p>

## Architecture

**BEBLaDII** (Bidirectional Encoder Based Lathent Diffusion with Information Injection) is a modular framework designed to implement the principle of "Intelligent Diffusion." The architecture effectively decouples deep logical analysis (**System 2**) from surface-level linguistic generation (**System 1**).

### Core Mechanics:

1.  **Latent Projection**: Discrete text is mapped into the continuous vector space of ModernBERT, establishing a rich semantic foundation.
2.  **Diffusion Reasoning**: Instead of traditional token-by-token generation, the model iteratively refines latent representations of the entire sequence, simulating a deliberative "reflection" process.
3.  **Retrieval-Augmented Reasoning (RAG)**: The system dynamically interacts with an external latent memory index. It injects relevant context through cross-attention layers directly within the diffusion loop to stabilize and ground the output.
4.  **Decoding**: The refined latent concepts are projected back into the decoder space for final transformation into human-readable text.

---
*Detailed specifications and data flow diagrams can be found in [Architecture.md](ideas/Architecture.md).*

## Roadmap

The development of BEBLaDII follows a structured five-phase cycle as detailed in [Training phases.md](ideas/Training%20phases.md):

- **Phase 1: Alignment & DUS** — Initial "intelligence" setup and capacity expansion via Depth Up-Scaling (DUS) and distillation from a teacher model.
- **Phase 2: Decoder Bridge** — Training the **Adapter** to bridge the diffusion latent space with the frozen decoder's embedding space.
- **Phase 3: Semantic Indexing** — Developing the **RAG-pooling** mechanism for efficient latent-to-latent retrieval.
- **Phase 4: Iterative Crystallization** — Training the **Denoiser** for multi-step latent refinement (simulating System 2 deliberation).
- **Phase 5: RAG Integration** — Final integration of Cross-Attention layers to ground the diffusion process in external knowledge.

## Infrastructure

### Component System
The project uses a modular component system based on the `BEComponent` class. All major architectural elements (projectors, models, distillers) are implemented as components, ensuring consistency and ease of integration.

### Versioning
Weight management and configuration history are handled by the `ComponentRegistry`. Metadata and model weights are stored in a structured format within `storage/components/`, allowing for precise version tracking of every system part.

### Experiments
Training and research are managed via the `ExperimentManager`. It tracks hyperparameters, provides state snapshots, and integrates with **Weights & Biases (WandB)** for real-time monitoring of experiments stored in `storage/experiments/`.

## Citation

> [!IMPORTANT]
> **Mandatory Attribution**
> If you utilize the code in this repository, or if you build upon the core architectural concepts of the **Reasoning Latent Diffusion** framework (System 2 reasoning via iterative latent refinement), **citation is strictly required**. Intellectual credit must be given for both the implementation and the underlying theoretical framework.

Please use the provided [CITATION.cff](CITATION.cff) file or the following metadata:
- **Title**: BEBLaDII: Bidirectional Encoder Based Lathent Diffusion with Information Injection
- **Author**: Bogdan Buliakov
- **URL**: [https://github.com/Laeryid/BEBLaDII](https://github.com/Laeryid/BEBLaDII)
