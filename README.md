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

# BEBLaDII: Reasoning Latent Diffusion Model

**BEBLaDII** stands for **Bidirectional Encoder Based Latent Diffusion with Information Injection**.

## Purpose
BEBLaDII is an advanced AI model designed to use **Complementary Latent Memory (CLM) directly as its own external memory**. Unlike standard auto-regressive models that predict the next discrete token, BEBLaDII separates logical reasoning from linguistic generation. It continuously "thinks", doubts its own representations, and iteratively crystallizes meaning inside a continuous latent space. When the model detects high uncertainty in its thoughts, it directly queries external CLM memory to stabilize its latent representations before finally translating them into human-readable text.

## Architecture
BEBLaDII is a discrete diffusion model with soft latent anchoring. The architecture is modular and processes data through four key stages. 

```mermaid
graph TD
    classDef verbal fill:#2d3436,stroke:#74b9ff,stroke-width:2px,color:#fff
    classDef latent fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff
    classDef clm fill:#27ae60,stroke:#2ecc71,stroke-width:2px,color:#fff

    subgraph "Discrete Verbal Space (System 1)"
        A["Input Text"] --> B["Tokenizer (e.g., Qwen)"]
        B --> C["Decoder Embeddings (4096-dim)"]
        
        K["4096-dim Decoded Vectors"] --> L["Frozen LM-Head"]
        L --> M["Final Output Text"]
    end

    subgraph "Continuous Latent Space (System 2)"
        C --> D["Input Projector & μ-VAE"]
        D -->|"768-dim Normalized"| E{"latentBERT Diffusion Core"}
        
        E --> F["Uncertainty Head (Denoiser)"]
        F -->|"Confidence Maps"| G{"Orchestrator"}
        
        G -->|"High Confidence (Proceed) or no ideas"| E
        G -->|"Low Confidence (Trigger Memory)"| H["CLM-Pooling & Cross-Attention"]
        
        H -->|"Inject Knowledge"| E
        
        G -->|"Final Denoised State"| J["Adapter MLP"]
        J --> K
    end
    
    subgraph "External Memory (CLM Space)"
        H -.->|"Queries"| I[("FAISS Vector Index (Latent Chunks)")]
        I -.->|"Retrieves Context"| H
    end

    class A,B,C,K,L,M verbal
    class D,E,F,G,J latent
    class H,I clm
```

1. **Latent Ingestion (Entry)**
   - **Tokenizer**: Converts input text into embeddings using a frozen decoder vocabulary (e.g., Qwen) to ensure strong baseline linguistic features.
   - **Projector & µ-VAE**: Maps the discrete decoder space into a continuous, Gaussian-friendly **Latent Diffusion Space**. The µ-VAE normalizes the space to prevent diffusion collapse.

2. **Diffusion Core (Reasoning Engine)**
   - **latentBERT**: The core engine is a Depth Up-Scaled ModernBERT (expanded to 40 layers via overlap). Instead of generating tokens sequentially, it acts as a global critic, refining the entire sequence simultaneously over multiple denoising steps.
   - **Uncertainty Head (Denoiser)**: A specialized lightweight neural classifier that evaluates the confidence ($\alpha$) of each latent vector.
   - **Orchestrator**: The control logic module that reads the confidence maps and makes decisive actions (e.g., triggering calls to RAG external memory if uncertainty is high, or continuing the diffusion process).

3. **CLM (Complementary Latent Memory)**
   - **Why not RAG?**: While conceptually similar to Retrieval-Augmented Generation, the execution differs fundamentally. Instead of retrieving discrete text and prepending it to an input prompt, CLM retrieves continuous mathematical vectors ("latent chunks") and injects them directly into the diffusion core's reasoning space via cross-attention. This is why we use a distinct term.
   - **CLM-Pooling**: Compresses a sequence of uncertain diffusion tokens into a highly concentrated vector query.
   - **Index & Cross-Attention**: The system queries a vector database (FAISS) directly in latent coordinates. The retrieved "latent chunks" of knowledge are injected back into the ModernBERT core dynamically via Cross-Attention layers.

4. **Decoder (Output Generation)**
   - **Adapter**: A lightweight MLP that performs the inverse projection, mapping the refined continuous vectors back into the original vocabulary space.
   - **LM-Head**: A frozen language model head translates these final stable vectors into the discrete final text.

## Components
The information flows through the system in a strict sequence. Each component acts as a specialized transformer of data:

1. **Tokenizer (Embedder)**
   - **Base Technology:** Frozen LLM Vocabulary & Embedder (e.g., Qwen 2.5).
   - **Our Changes:** Completely removed the native ModernBERT tokenizer in favor of a much richer multi-lingual LLM tokenizer.
   - **Role:** Converts raw text strings into high-dimensional numerical vectors (e.g., 4096-dim). It acts as the initial discrete linguistic anchor.
   - **I/O:** Receives raw text from the user -> Transmits continuous embeddings to the Input Projector.

2. **Input Projector**
   - **Base Technology:** Multi-Layer Perceptron (MLP).
   - **Our Changes:** Replaces the native embedding layer of ModernBERT to serve as an architectural bridge.
   - **Role:** Geometrically maps vectors from the large decoder space (4096-dim) down into the more compact latent workspace of ModernBERT (768-dim).
   - **I/O:** Receives decoder embeddings -> Transmits 768-dim vectors to the µ-VAE Normalization Head.

3. **µ-VAE Normalization Head**
   - **Base Technology:** Variational Autoencoder (VAE) continuous projection.
   - **Our Changes:** Custom dual linear layers (`mu_head` and `logvar_head`) permanently baked onto the Projector.
   - **Role:** Forces the continuous space to be Gaussian-friendly and suitable for diffusion mechanics. It prevents vectors from instantly collapsing into noise during the forward diffusion process.
   - **I/O:** Receives unnormalized latent vectors -> Transmits stable, normalized diffusion vectors to the ModernBERT Core.

4. **latentBERT (Reasoning Engine)**
   - **Base Technology:** ModernBERT-large.
   - **Our Changes:** Extended from 28 to 40 layers using Depth Up-Scaling (DUS) with overlapping blocks (1-20 and 9-28), effectively increasing reasoning depth without starting from scratch.
   - **Role:** The core iterative diffusion processor. It analyzes the entire sequence of "clouds of meaning" in parallel, applying multi-step denoising refinement based on surrounding semantic context.
   - **I/O:** Receives normalized latent vectors and CLM context -> Transmits conceptually refined latent sequences to the Denoiser.

5. **Uncertainty Head (Denoiser)**
   - **Base Technology:** Lightweight Sigmoid Classifier Head.
   - **Our Changes:** Replaces standard mathematical variance metrics (which are unstable in textual domains) with a trained neural head.
   - **Role:** Calculates the confidence score ($\alpha$) for each latent token, effectively acting as the neural evaluator of certainty.
   - **I/O:** Receives refined vectors from latentBERT -> Transmits confidence maps to the Orchestrator.

6. **Orchestrator**
   - **Base Technology:** Algorithmic control logic (non-neural).
   - **Our Changes:** Custom logic layer acting as the system's executive decision-maker.
   - **Role:** Routes the progression of the diffusion loop, detects uncertainty, requests external knowledge, and interprets latent vectors.
   - **I/O:** Receives confidence maps -> Transmits commands to CLM, shifts $T$, or inserts tokens.

7. **CLM-Pooling & Router Head (Knowledge Injectors)**
   - **Base Technology:** Pooling layer + FAISS Vector Index + Cross-Attention blocks + Router MLP.
   - **Our Changes:** Custom retrieval logic working purely in latent vector spaces, governed by a trainable Router Head to evaluate context relevance.
   - **Role:** Pools an uncertain sequence, evaluates knowledge sufficiency, and dynamically injects "latent chunks" or explicit "no data" signals.
   - **I/O:** Receives doubt-heavy queries -> Transmits weighted external context back into latentBERT.

8. **Latent Task Context (Capabilities Index)**
   - **Base Technology:** The exact same encoder architecture as the main CLM + specific index for tools/tasks.
   - **Our Changes:** A distinct operational memory reserved exclusively for strict contextual data.
   - **Role:** Supplies exact formats and descriptions for available tools, MCP servers, parsed file structures, and other contextual elements necessary for formulating precise tool commands. Because it shares the main CLM encoder, capability vectors can be perfectly compared against semantic query vectors.
   - **I/O:** Receives specific capability queries from the Orchestrator -> Transmits precise context back into latentBERT via CA_Capabilities.

9. **Adapter MLP**
   - **Base Technology:** Multi-Layer Perceptron.
   - **Our Changes:** Custom bridging layer trained via Direct Cross-Entropy.
   - **Role:** The inverse of the Input Projector. Translates the final structured 768-dim latent thoughts back into the specific 4096-dim geometric footprint recognized by the external LLM vocabulary.
   - **I/O:** Receives fully crystallized latent vectors from the final diffusion step -> Transmits 4096-dim vectors to the LM-Head.

10. **LM-Head (Decoder)**
   - **Base Technology:** Frozen LLM Head (e.g., Qwen).
   - **Our Changes:** Entirely frozen and detached from its native transformer blocks.
   - **Role:** Acts as the final discrete linguistic "renderer," decoding the abstract numerical concepts back into human-readable words and sentences.
   - **I/O:** Receives 4096-dim continuous vectors from the Adapter -> Transmits final discrete text output to the user.

## Detailed Component Breakdown

### 1. latentBERT
The core reasoning engine is built on a Depth Up-Scaled ModernBERT, expanding it to 40 layers. The most critical modification is the integration of three distinct types of **Cross-Attention (CA)** modules. Instead of replacing existing layers, these CA modules are inserted parallel to the residual connections, typically starting from the second third of the network's depth (e.g., every second or third layer). The three parallel CA formats are:
- **CA_Prompt:** Continuously embeds the original user prompt or task instruction to maintain strict alignment.
- **CA_Knowledge (Memory):** Injects external factual data retrieved from the main CLM database.
- **CA_Capabilities (Task Context):** Acts as a "Latent Task Context" for tool-use, incorporating active task context, API outputs, or MCP server payloads.

## The Orchestrator (Executive Logic)
The Orchestrator is the algorithmic control center, acting as the system's executive decision-maker (System 2 Controller). Its core functions include:

*   **Doubt Pattern Recognition:** It does not blindly trigger CLM calls. It analyzes the confidence maps provided by the Uncertainty Head for specific patterns of uncertainty—since an entirely noisy vector might just need more internal denoising, whereas a vector oscillating between two conflicting facts is a prime candidate for an external query.
*   **Vector Interpretation:** It interprets the continuous latent vectors to understand if the model lacks the space to form a complex concept or if it has reached a state of clarity.
*   **Dynamic Space Expansion (Dilation):** If the Orchestrator detects an entropy "bottleneck" where the model struggles to fit a multi-word concept into a single token slot, it performs *Dynamic Upsampling*. It inserts new interpolated noise tokens, effectively giving the model more space to crystallize thoughts. Positional relationships are preserved thanks to the relative nature of **RoPE** (Rotary Positional Embeddings).
*   **Time-Travel & Diffusion Control:** It manages the temporal parameter $T$ (the noise level). 
    *   **Early Stopping:** Decides to end the diffusion early if the vectors converge quickly, saving compute.
    *   **Time-Travel (Epiphany):** If CLM retrieves a critically new fact late in the process, the Orchestrator can "roll back" time by adding noise (increasing $T$), melting the incorrect rigid vectors and allowing them to re-crystallize correctly around the new facts.

## Complementary Latent Memory (CLM Engine)
The CLM system stores and processes information directly in the continuous diffusion space.

### Latent Memory Structure
*   **Text Chunks:** Used for debugging, interpretability, and rebuilding the index if the model is updated.
*   **Latent Token Chunks:** Sequences of "ideal" vectors in the diffusion space corresponding to the text.
*   **Aggregated Tokens:** Highly concentrated single vectors representing the entire semantic chunk, optimized for rapid similarity search against pooled query vectors.

### Router Head & Null-Embedding
*   **Router Head:** Instead of relying solely on static cosine similarity, a lightweight neural `Router Head` evaluates the retrieved chunks. It determines if the lexical match actually contains the *sufficient* facts required to resolve the Orchestrator's doubt, balancing weights between the Main CLM and the Latent Task Context.
*   **The "No-Data" Signal:** If the Router Head determines that no relevant facts exist, the system does not simply stay silent (which would force the model to hallucinate from its weights). Instead, it injects a fully trainable `null_clm_embedding` with maximum attention weight. The model treats this "absence of knowledge" as a concrete fact, organically generating an honest "I don't know" response without breaking grammatical structure.

## Logic of the Operational Spaces
The model achieves "reasoning" by transitioning between continuous shapes of meaning and rigid words, bridging a critical gap through three isolated spaces:

* **Latent Diffusion Space:** A continuous mathematical space where true "thinking" occurs. Rather than operating on exact words, it operates on "clouds of meaning". A vector here might represent a hesitant guess or a soft hypothesis. The vectors are penalized over time for changing, simulating an energy barrier that forces gradual "crystallization" of ideas without locking them in prematurely.
* **CLM Search Space:** The space used for querying external knowledge. Sequence embeddings from the latent diffusion space are pooled together to form a highly specific question vector. It connects the model's internal doubts with pre-calculated, verified "latent chunks" residing in the database.
* **Verbal / Decoding Space:** The discrete linguistic space containing distinct tokens and words from a frozen tokenizer (e.g., Qwen's vocabulary). This space is purely functional—it provides the initial entry constraints and acts as the final "voice" of the model, strictly decoding the crystallized latent thoughts into readable human text at the very last step.

## Temporal Logic & Planning (t-dependent Reasoning)
Because the diffusion process is a progressive refinement over time, `latentBERT` can be trained to implement fundamentally different reasoning strategies at different stages of the denoising cycle, dictated by the time parameter $t$ (or $\tau$). 

*   **Latent Priors (Warm Start):** Instead of starting the generation process from pure Gaussian noise ($X_T \sim \mathcal{N}(0, I)$), the model can start from a directed *Prior*—a predefined "semantic cloud" representing a document template (e.g., a report structure or a JSON schema). This frees the model from the heavy macro-work of formatting and prevents "ghost text" hallucinations, as the prior consists of abstract meanings rather than concrete words.
*   **Abstract Planning:** Early stages (large $t$) are characterized by high uncertainty. The model deliberately constructs a high-level **Response Plan** (the conceptual scaffolding) without rushing to define specific concepts.
*   **Saliency-based Diffusion (Curriculum Learning):** The model is trained using a ranked noise approach. Instead of uniform masking, tokens are prioritized by their semantic importance. At large $t$, the model learns to recover the heavy "semantic skeleton" (top 10% crucial words). As $t$ decreases, it focuses on secondary facts, and finally refines grammar and prepositions at the lowest $t$. 
*   **Iterative Crystallization:** During later steps (small $t$), the model shifts from abstract planning to concrete conceptual generation, remaining entirely within the continuous Latent Space until the very final step when the Adapter translates it to discrete text. This time-dependent logic effectively layers abstract planning and concrete execution within the exact same iterative diffusion loop.

## Implementation Plan
The training lifecycle is divided into five strictly separated phases, moving from basic semantic alignment to complex external memory integration:

**Phase 1: Alignment & DUS (Depth Up-Scaling)**
Extending the ModernBERT architecture to 40 layers and aligning its semantic capacity. We distill abstract logical reasoning from a powerful teacher (e.g., DeepSeek-R1-7B) into the ModernBERT core, ignoring linguistic formatting and focusing purely on matching the latent thought representations. Here we try to give the model the ability to think abstractly and logically.

**Phase 2: Adapter Training (Decoder Bridge)**
Connecting the "intellect" with the "voice". All components are frozen except the Adapter, which is trained via Direct Cross-Entropy to correctly project diffusion vectors into the frozen Qwen LM-head, ensuring perfect textual reconstruction of concepts. Here we try to give the model the ability to speak.

**Phase 3: CLM-Pooling (Semantic Indexing)**
Training the mechanism that forms queries to the external database. We teach the model to reliably compress long latent chunk sequences into robust single query vectors that can successfully identify their related chunks in a FAISS index. Here we try to give the model the ability to remember.

**Phase 4: Denoiser & Iterative Crystallization**
Training the Uncertainty Head (Sigmoid classifier) to accurately predict confidence maps. We additionally train initial Cross-Attention layers to properly integrate the input prompt alongside a trainable `gamma` balancing parameter. During this phase, we apply **Saliency-based Diffusion** to teach the model a Curriculum: construct a high-level **Response Plan** (semantic skeleton) in the early, noisy steps of the diffusion process (large $t$), and then gradually crystallize these abstract thoughts into concrete grammar and syntax in the later steps (small $t$). Here we try to give the model the ability to plan and to crystallize its thoughts.

**Phase 5: CLM Integration**
The final assembly of the system. Cross-Attention layers are embedded throughout the ModernBERT architecture to dynamically pull in the retrieved latent knowledge chunks exactly when and where the Denoiser flags uncertainty, completing the complementary latent memory loop. Here we try to give the model the ability to use its memory.

**Phase 6: Capabilities Integration (Latent Task Context)**
Implementation of the task-specific operational index. The model learns to utilize contextual tool descriptions, MCP server connections, and directory/file structures via the CA_Capabilities layer, enabling precise, context-aware command generation. Here we try to give the model the ability to use tools.


## Reports

* Phase 1
  * [Phase 1 Awakening Report](reports/phase1_awakening_report.md)

## Citation

> [!IMPORTANT]
> **Mandatory Attribution**
> If you utilize the code in this repository, or if you build upon the core architectural concepts of the **Reasoning Latent Diffusion** framework (Latent Diffusion-based reasoning via iterative latent refinement), **citation is strictly required**. Intellectual credit must be given for both the implementation and the underlying theoretical framework.

Please use the provided [CITATION.cff](CITATION.cff) file or the following metadata:
- **Title**: BEBLaDII: Bidirectional Encoder Based Lathent Diffusion with Information Injection
- **Author**: Bogdan Buliakov
- **URL**: [https://github.com/Laeryid/BEBLaDII](https://github.com/Laeryid/BEBLaDII)
