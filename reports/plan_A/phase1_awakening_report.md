# Phase 1: Step Awakening Report

## Executive Summary
**Phase 1: Step Awakening** is the foundational stage of the BEBLaDII project. Its primary objective was to "awaken" the latent space of the newly constructed 40-layer Depth Up-Scaled (DUS) model and train the **Input and Feature Projectors**. This ensures that the latent embeddings from the frozen teacher model (DeepSeek-R1-Distill-Qwen-7B) are correctly mapped into the semantic space of the student model (ModernBERT-large based DUS), and the student's outputs are mapped back for loss calculation.

## Architectural Overview
The system employs a **System 1 (Voice) / System 2 (Reasoning)** split. During this phase:
- **Teacher**: DeepSeek-R1-7B (frozen, 4-bit NF4) provides the target semantic distributions.
- **Student (System 2 Engine)**: A 40-layer `DUSModel` initialized from `ModernBERT-large` using block duplication.
- **Projectors**: An **Input Projector** mapping Qwen's hidden dimension (3584) to ModernBERT's latent dimension (1024), and **Feature Projectors** mapping student states back to Qwen's space (1024 -> 3584). All projectors were unfrozen and trained.

## Dataset Preparation
Data was prepared using `prepare_phase1_data_v2.py`, focusing on short, high-quality samples for rapid alignment:
- **CulturaX (CS/RU)**: General language knowledge, sampled and truncated to 500 tokens.
- **Magpie Reasoning**: Synthetic reasoning chains, filtered and truncated to 500 tokens.
- **Logic**: The short context (Max Length 500) allowed the model to focus on embedding alignment rather than long-range dependencies.

During this phase, we have cut the elements length to 256 in the script for speedups.

## Training Configuration & Stability
Training was conducted on **Kaggle T4 GPUs** with the following key parameters:
- **Optimizer**: Standard **AdamW** (`torch.optim`).
- **Learning Rate**: 5e-5 with a single epoch pass.
- **Precision**: **FP32** was utilized for loss calculation to prevent gradient collapse (NaN losses) which occurred during early experiments with FP16.
- **Loss Function**: `DistillationLoss` combining MSE and Cosine Similarity with **Attention Masking** to ignore padding tokens.

## Results Analysis
The training demonstrated stable convergence as captured in the Weights & Biases logs:
- **Training Loss**: 
![Training Loss](../storage/experiments/20260413%20Phase%201%20Awakening%20weights/W&B%20Chart%2014.%204.%202026%2017_20_32.png)
- **Validation Loss**: 
![Validation Loss](../storage/experiments/20260413%20Phase%201%20Awakening%20weights/W&B%20Chart%2014.%204.%202026%2017_20_18.png)

The charts indicate a consistent downward trend without significant spikes, suggesting that the **Mirror-Load architecture** and masked loss effectively stabilized the initial training phase. The validation loss mirrors the training curve, indicating good generalization to unseen text within the same distribution.

## Key Takeaways
1. **Mirror-Load Success**: Using pre-built weights and local resource mapping avoided internet-dependency issues in Kaggle.
2. **NaN Loss Mitigation**: Switching to FP32 and enforcing attention masks in the loss function was critical for the T4 GPU architecture.
3. **Latent Alignment**: The successful alignment of the Input and Feature Projectors creates a stable bridge between the teacher and student spaces. Despite early architecture plans separating projector training stages, the notebook run trained all projectors simultaneously to ensure proper loss calculation across layers.

---
*Report generated on: 2026-04-25*
*Project: BEBLaDII (Bidirectional Encoder Based Latent Diffusion with Information Injection)*
