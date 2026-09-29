import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

project_root = r"C:\Experiments\BEBLaDII"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.beb_la_dii.model.modern_decoder import ModernLatentDecoder
from src.beb_la_dii.model.dus import DUSModel

def test_decoding_special_tokens():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    qwen_path = "Qwen/Qwen2.5-1.5B"
    dec_ckpt_path = r"C:\Experiments\BEBLaDII\experiments\phase 2\planB_phase2_checkpoints_decoder_step_9000.pth"
    sep_path = r"C:\Experiments\BEBLaDII\storage\components\sep_token.pt"
    void_path = r"C:\Experiments\BEBLaDII\storage\components\void_token.pt"
    
    print("Loading tokenizer and Qwen lm_head...")
    tokenizer = AutoTokenizer.from_pretrained(qwen_path)
    qwen = AutoModelForCausalLM.from_pretrained(qwen_path, torch_dtype=torch.bfloat16)
    lm_head_weight = qwen.lm_head.weight.detach().clone().to(device).float()
    del qwen
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    print("Building ModernLatentDecoder backbone...")
    dus = DUSModel.from_scratch(weights_path=None)
    decoder = ModernLatentDecoder(latent_dim=1024, qwen_dim=1536, num_layers=3)
    decoder.backbone = dus.model
    decoder.backbone.layers = torch.nn.ModuleList(decoder.backbone.layers[-3:])
    decoder.use_modern_bert = True
    
    print(f"Loading decoder weights from {dec_ckpt_path}...")
    st = torch.load(dec_ckpt_path, map_location="cpu", weights_only=False)
    state = st["decoder"] if "decoder" in st else st
    clean_state = {k.replace("decoder.", ""): v for k, v in state.items()}
    decoder.load_state_dict(clean_state, strict=True)
    decoder.to(device).float()
    decoder.eval()
    
    # Load tokens
    sep_tok = torch.load(sep_path, map_location=device, weights_only=False).float()
    void_tok = torch.load(void_path, map_location=device, weights_only=False).float()
    
    print(f"sep_token shape: {sep_tok.shape}, norm: {sep_tok.norm().item():.4f}")
    print(f"void_token shape: {void_tok.shape}, norm: {void_tok.norm().item():.4f}")
    print(f"Cosine similarity between sep and void: {(sep_tok @ void_tok).item():.4f}\n")
    
    T = 10
    z_sep = sep_tok.unsqueeze(0).unsqueeze(0).expand(1, T, -1).clone()   # [1, 10, 1024]
    z_void = void_tok.unsqueeze(0).unsqueeze(0).expand(1, T, -1).clone() # [1, 10, 1024]
    
    def decode_sequence(z_input, name):
        with torch.no_grad():
            dec_out = decoder(z_input) # [1, 10, 1536]
            logits = F.linear(dec_out, lm_head_weight) # [1, 10, vocab_size]
            token_ids = logits.argmax(dim=-1).squeeze(0).tolist()
            
            raw_tokens = [tokenizer.decode([idx]) for idx in token_ids]
            full_text = tokenizer.decode(token_ids, skip_special_tokens=False)
            
            print(f"=== DECODING RESULTS FOR {name} (10 tokens) ===")
            print(f"Full decoded string: '{full_text}'")
            print(f"Raw token IDs: {token_ids}")
            print(f"Tokens by position:")
            probs = F.softmax(logits.squeeze(0), dim=-1)
            for pos, tid in enumerate(token_ids):
                top_k = torch.topk(probs[pos], k=3)
                top_info = ", ".join([f"'{tokenizer.decode([idx.item()])}' ({p.item():.2%})" for idx, p in zip(top_k.indices, top_k.values)])
                print(f"  Pos {pos:02d}: ID {tid:6d} -> '{raw_tokens[pos]}' | Top-3: {top_info}")
            print()
            return token_ids, raw_tokens
            
    decode_sequence(z_sep, "SEP_TOKEN (<|thoughts|>)")
    decode_sequence(z_void, "VOID_TOKEN (<|void|>)")

if __name__ == "__main__":
    test_decoding_special_tokens()
