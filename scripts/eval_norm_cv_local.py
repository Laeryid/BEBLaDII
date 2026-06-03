import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.beb_la_dii.model.dus import create_latentbert
from transformers import AutoTokenizer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Init tokenizer and latentBERT
    print("Loading tokenizer and model structure...")
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
    model = create_latentbert(model_id="answerdotai/ModernBERT-large", target_layers=40)
    
    # 2. Load weights
    ckpt_path = r"C:\Experiments\BEBLaDII\storage\experiments\20260601 Phase + Reasoning adr 27\checkpoints_latest_checkpoint.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    state_dict = ckpt["model_state_dict"]
    
    # Strip common prefixes from FSDP/DDP/Wrapper
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("module."):
            new_k = new_k.replace("module.", "", 1)
        if new_k.startswith("student."):
            new_k = new_k.replace("student.", "", 1)
        if new_k.startswith("model."):
            new_k = new_k.replace("model.", "", 1)
        new_state_dict[new_k] = v
        
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded weights. Missing keys: {len(missing)} (mostly projectors/teacher if any)")
    
    model.to(device)
    model.eval()
    
    # 3. Create dummy texts
    texts = [
        "In the realm of latent space models, evaluating the variance of token norms is a crucial step towards understanding the topology of the learned manifold.",
        "When the coefficient of variation approaches zero, it indicates that the embeddings are distributed on a hypersphere, also known as the Zen Solution.",
        "Deep learning architectures such as ModernBERT use deeply up-scaled layers to align with teacher models."
    ]
    
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    print("Running forward pass...")
    with torch.no_grad():
        # In BEBLaDII, latent_bert is called either with inputs_embeds or input_ids.
        # Here we test it with input_ids directly.
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True
        )
    
    # 4. Calculate Norm CV for l40
    # outputs.hidden_states will contain embedding + 40 layers. 
    # l40 is the last layer (index 40)
    l40_states = outputs.hidden_states[40] # shape: (B, T, D)
    mask = inputs["attention_mask"].bool()
    
    active_tokens = l40_states[mask] # shape: (N, D)
    norms = active_tokens.norm(dim=-1)
    
    cv = norms.std() / norms.mean()
    print("\n" + "=" * 60)
    print(f"Calculated norm_cv_l40_raw : {cv.item():.5f}")
    print(f"Mean norm                  : {norms.mean().item():.3f}")
    print(f"Std deviation of norms     : {norms.std().item():.3f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
