import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

project_root = r"C:\Experiments\BEBLaDII"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.beb_la_dii.model.vae import LatentEncoder
from src.beb_la_dii.utils.loss import safe_normalize

def generate_void_token():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    qwen_path = "Qwen/Qwen2.5-1.5B"
    phase1_ckpt = r"C:\Experiments\BEBLaDII\experiments\phase 1\planB_phase1_checkpoints_phase1_vae_step_20000.pth"
    output_path = r"C:\Experiments\BEBLaDII\storage\components\void_token.pt"
    
    print(f"Using device: {device}")
    print("Loading tokenizer and embeddings...")
    tokenizer = AutoTokenizer.from_pretrained(qwen_path)
    qwen = AutoModelForCausalLM.from_pretrained(qwen_path, torch_dtype=torch.bfloat16)
    qwen_embed_weight = qwen.get_input_embeddings().weight.detach().clone().to(device)
    del qwen
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"Loading LatentEncoder from {phase1_ckpt}...")
    encoder = LatentEncoder().to(device).to(torch.bfloat16)
    state = torch.load(phase1_ckpt, map_location=device, weights_only=False)
    if "encoder" in state:
        state = state["encoder"]
    encoder.load_state_dict(state, strict=False)
    encoder.eval()
    
    with torch.no_grad():
        text = "<|void|>"
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        print(f"\nText: {text}")
        print(f"Token IDs: {inputs.input_ids[0].tolist()}")
        print(f"Tokens: {[tokenizer.decode([idx]) for idx in inputs.input_ids[0]]}")
        
        # Qwen Embeddings
        void_qwen = F.embedding(inputs.input_ids, qwen_embed_weight)
        print(f"Qwen Embeddings Shape: {void_qwen.shape}")
        
        # LatentEncoder
        z_void, _, _ = encoder(void_qwen)
        print(f"LatentEncoder Output Shape: {z_void.shape}")
        
        # Mean pooling and spherical normalization
        void_token = safe_normalize(z_void.mean(dim=1), dim=-1)
        void_token = void_token.squeeze(0).to(torch.bfloat16).cpu()
        
        print(f"Final <|void|> Token Shape: {void_token.shape}")
        print(f"Final Norm: {void_token.float().norm().item():.4f}")
        print(f"Dtype: {void_token.dtype}")
        print(f"First 5 values: {void_token[:5].tolist()}")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(void_token, output_path)
        print(f"Saved void_token to {output_path}")

if __name__ == "__main__":
    generate_void_token()
