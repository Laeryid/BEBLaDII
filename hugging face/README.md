---
license: apache-2.0
language:
- ru
- en
base_model:
- answerdotai/ModernBERT-large
tags:
- experimental
- text_diffusion
- lathent_space
- custom-model
- pytorch
- diffusion
- vae
---

# BEBLaDII Foundation Weights (Phases 1-5)

This repository contains the foundation weights for the training phases of the experimental **BEBLaDII** architecture. (**Git Repository:** 
[github.com/Laeryid/BEBLaDII](https://github.com/Laeryid/BEBLaDII))
These weights are intended for further training (Phase 6 and beyond) and are not a standalone, chat-ready LLM.

## 📦 Checkpoints Included

1. **Phase 1 (VAE/Latent Encoder):** `phase1_vae_step_20000.pth`
   * Responsible for shaping the spherical latent space (Spherical Topology, `F.normalize`) and initial compression.
2. **Phase 2 (Latent Decoder):** `phase2_decoder_step_9000.pth`
   * Initialized from ModernBERT-large. Decodes latent representations back into raw text.
3. **Phase 3 (Canonical Diffusion / DUS):** `phase3_diffusion_step_17995.pth`
   * Continuous diffusion model on a sphere using AdaLN and UNet-style skip-connections. Provides Depth Up-Scaling (DUS) capabilities.
4. **Phase 4 (Hierarchical Per-Token Noise Diffusion core / DUS):** `phase4_diffusion_core_step_85995.pth`
   * As phase 3, but now tokens may have individual noise level.
5. **Separator Token:** `sep_token.pt`
   * A custom separator embedding tensor. Initially designed to pass `t` onto the canvas before switching to AdaLN. Preserved for potential future utility.
6. **Phase 5 (Confidence Head):** `phase5_confidence_head_v2.pt`
   * Computes confidence and geometric token status signals for the algorithmic Orchestrator to resolve lexical ambiguity without explicit teacher decoding.
7. **Latent Dictionary:** `latent_dict.pt`
   * Pre-computed normalized latent vectors for the entire Qwen2.5-1.5B vocabulary (151,936 tokens) through the Phase 1 VAE. Used by the Orchestrator's Sensor Ensemble to calculate geometric metrics instantly without re-running the encoder.

## 🚀 Usage

These weights are designed to be loaded into the respective modules of the BEBLaDII framework. Example of loading weights in PyTorch:

```python
import torch
from huggingface_hub import hf_hub_download

# Example: Loading the diffusion component
file_path = hf_hub_download(repo_id="YOUR_USERNAME/bebladii-foundation-weights", filename="phase3_diffusion_step_17995.pth")
state_dict = torch.load(file_path, map_location="cpu")
# model.load_state_dict(state_dict)
```

## ⚖️ License and Attribution
This project is licensed under **Apache 2.0**.
The architecture incorporates concepts and base weight initializations from the **ModernBERT** and **DeepSeek** model families.
