import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

project_root = "C:\\Experiments\\BEBLaDII"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.beb_la_dii.model.vae import LatentEncoder
from src.beb_la_dii.utils.loss import safe_normalize

def test_thoughts_encoding():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    qwen_path = "Qwen/Qwen2.5-1.5B"
    phase1_ckpt = r"C:\Experiments\BEBLaDII\experiments\phase 1\planB_phase1_checkpoints_phase1_vae_step_20000.pth"
    
    print("Loading tokenizer and embeddings...")
    tokenizer = AutoTokenizer.from_pretrained(qwen_path)
    qwen = AutoModelForCausalLM.from_pretrained(qwen_path, torch_dtype=torch.bfloat16)
    qwen_embed_weight = qwen.get_input_embeddings().weight.detach().clone().to(device)
    del qwen
    
    print(f"Loading LatentEncoder from {phase1_ckpt}...")
    encoder = LatentEncoder().to(device).to(torch.bfloat16)
    state = torch.load(phase1_ckpt, map_location=device)
    if "encoder" in state:
        state = state["encoder"]
    encoder.load_state_dict(state, strict=False)
    encoder.eval()
    
    with torch.no_grad():
        text = "<|thoughts|>"
        # Tokenize (using add_special_tokens=False because we want the exact subwords if it's not a single token)
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        print(f"\nText: {text}")
        print(f"Token IDs: {inputs.input_ids[0].tolist()}")
        print(f"Tokens: {[tokenizer.decode([idx]) for idx in inputs.input_ids[0]]}")
        
        # Get Qwen Embeddings
        thoughts_qwen = F.embedding(inputs.input_ids, qwen_embed_weight)
        print(f"Qwen Embeddings Shape: {thoughts_qwen.shape}")
        
        # Encode via LatentEncoder
        z_thoughts, _, _ = encoder(thoughts_qwen)
        print(f"LatentEncoder Output Shape: {z_thoughts.shape}")
        
        # Pool and Normalize
        sep = safe_normalize(z_thoughts.mean(dim=1), dim=-1)
        print(f"Final Separator Token Shape: {sep.shape}")
        print(f"Final Norm: {sep.norm().item():.4f}")
        print(f"First 5 values of separator token: {sep[0, :5].tolist()}")
        
        save_path = r"C:\Experiments\BEBLaDII\storage\components\sep_token.pt"
        torch.save(sep.squeeze(0).cpu(), save_path)
        print(f"Saved sep_token to {save_path}")

if __name__ == "__main__":
    test_thoughts_encoding()
