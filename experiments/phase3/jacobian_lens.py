import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

project_root = r"C:\Experiments\BEBLaDII"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.phase3.inspect_and_test_phase3 import load_checkpoint_and_inspect, load_models
from src.beb_la_dii.utils.loss import safe_normalize

def analyze_gradients(ckpt_path, diff_model, tokenizer, device):
    print(f"\n{'='*80}\nJacobian Lens (Gradient Flow): {os.path.basename(ckpt_path)}\n{'='*80}")
    state = load_checkpoint_and_inspect(ckpt_path)
    
    dus_ema = state.get("dus_ema", state.get("dus", {}))
    clean_dus = {k.replace("_orig_module.", ""): v for k, v in dus_ema.items()}
    diff_model.dus.load_state_dict(clean_dus, strict=False)
    
    for name in ["adaLN_attn", "adaLN_mlp", "t_proj", "sep_embed"]:
        ema = state.get(f"{name}_ema", state.get(name, {}))
        if ema:
            if name == "sep_embed":
                diff_model.sep_embed.copy_(ema)
            else:
                getattr(diff_model, name).load_state_dict(ema, strict=True)
            
    diff_model.train() # Need train mode for gradients
    for p in diff_model.parameters():
        p.requires_grad = True

    text = "The quick brown fox jumps over the lazy dog."
    tok = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = tok.input_ids.to(device)
    attn_mask = tok.attention_mask.to(device)
    
    with torch.no_grad():
        qwen_embeds = diff_model.qwen_embeddings(input_ids)
        z_clean, _, _ = diff_model.encoder(qwen_embeds)
        z_clean = F.normalize(z_clean.float(), dim=-1).detach()
    z_clean.requires_grad_(False)
    
    B, T, D = z_clean.shape
    
    # We will test t = 0.9 (high noise) and t = 0.1 (low noise)
    for t_val in [0.9, 0.1]:
        print(f"\n--- Gradient Flow at t={t_val} ---")
        diff_model.zero_grad()
        
        # Add noise manually so we can track its gradients
        mu = torch.cos(torch.tensor(t_val) * (torch.pi / 2)).item()
        sigma = torch.sin(torch.tensor(t_val) * (torch.pi / 2)).item()
        epsilon = torch.randn_like(z_clean)
        z_t = safe_normalize(mu * z_clean + sigma * epsilon, dim=-1)
        z_t.requires_grad_(True)
        
        t_tensor = torch.tensor([t_val], device=device)
        
        # Hooks to capture hidden states and their gradients
        hidden_states = {}
        def get_hook(layer_idx):
            def hook(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                h.retain_grad()
                hidden_states[layer_idx] = h
            return hook
            
        hooks = []
        for i, layer in enumerate(diff_model.dus.layers):
            h = layer.register_forward_hook(get_hook(i))
            hooks.append(h)
            
        # Forward pass using z_noisy_override
        out = diff_model(input_ids, attn_mask, t_tensor, z_noisy_override=z_t)
        dus_final = out["dus_final"]
        
        # Calculate Cosine Loss (ADR-060)
        target = z_clean[attn_mask.bool()]
        pred = dus_final[attn_mask.bool()]
        cos_sim = (pred * target).sum(dim=-1)
        loss = (1.0 - cos_sim).mean()
        
        loss.backward()
        
        # Remove hooks
        for h in hooks:
            h.remove()
            
        # Analyze gradients
        print(f"{'Layer':<6} | {'Grad Norm':<12} | {'Activation Norm':<15}")
        print("-" * 40)
        
        # z_t gradient (Input)
        zt_grad_norm = z_t.grad.norm().item() if z_t.grad is not None else 0.0
        zt_norm = z_t.norm().item()
        print(f"{'Input':<6} | {zt_grad_norm:<12.6f} | {zt_norm:<15.6f}")
        
        for i in range(len(diff_model.dus.layers)):
            if i in hidden_states and hidden_states[i].grad is not None:
                grad_norm = hidden_states[i].grad.norm().item()
                act_norm = hidden_states[i].norm().item()
                
                # Highlight if gradient explodes or vanishes
                marker = ""
                if grad_norm < 1e-6:
                    marker = "<< DEAD >>"
                elif grad_norm > 10.0:
                    marker = "<< EXPLODING >>"
                    
                print(f"{i:<6} | {grad_norm:<12.6f} | {act_norm:<15.6f} {marker}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    print("Loading models...")
    dummy_state = {}
    tokenizer, diff_model, decoder, lm_head_weight = load_models(device, dtype, dummy_state)
    
    ckpt_path = r"C:\Experiments\BEBLaDII\experiments\phase3\local_checkpoints\phase3_step_18995.pth"
    if os.path.exists(ckpt_path):
        analyze_gradients(ckpt_path, diff_model, tokenizer, device)
    else:
        print(f"[!] CKPT not found: {ckpt_path}")

if __name__ == "__main__":
    main()
