import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

project_root = r"C:\Experiments\BEBLaDII"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.phase3.inspect_and_test_phase3 import load_checkpoint_and_inspect, load_models, encode_clean_text

def analyze_adaln_sensitivity(diff_model, device):
    print("\n" + "="*80)
    print("DIAGNOSTIC 1: AdaLN Sensitivity to t (Check if model is 'deaf' to t)")
    print("="*80)
    
    t_vals = torch.tensor([0.01, 0.5, 0.99], device=device)
    t_sin = diff_model.t_sin_embed(t_vals)
    t_emb = diff_model.t_proj(t_sin)
    
    print(f"t_proj output norm for t=[0.01, 0.5, 0.99]: {t_emb.norm(dim=-1).tolist()}")
    
    layer_0_attn_shift, layer_0_attn_scale = diff_model.adaLN_attn[0](t_emb)
    layer_39_attn_shift, layer_39_attn_scale = diff_model.adaLN_attn[-1](t_emb)
    
    print(f"Layer 0 AdaLN_attn scale mean: {layer_0_attn_scale.squeeze(1).mean(dim=-1).tolist()}")
    print(f"Layer 0 AdaLN_attn shift std:  {layer_0_attn_shift.squeeze(1).std(dim=-1).tolist()}")
    print(f"Layer 39 AdaLN_attn scale mean: {layer_39_attn_scale.squeeze(1).mean(dim=-1).tolist()}")
    print(f"Layer 39 AdaLN_attn shift std:  {layer_39_attn_shift.squeeze(1).std(dim=-1).tolist()}")
    
    diff_t01_t99 = (diff_model.adaLN_attn[0](t_emb[0:1])[0] - diff_model.adaLN_attn[0](t_emb[2:3])[0]).norm().item()
    print(f"Difference in Layer 0 AdaLN modulation between t=0.01 and t=0.99: {diff_t01_t99:.6f}")
    if diff_t01_t99 < 1e-4:
        print("  -> RESULT: Model is COMPLETELY DEAF to t! AdaLN produces static modulation.")
    else:
        print("  -> RESULT: AdaLN DOES respond to t! Signal is reaching the layers.")

def analyze_layerwise_collapse(diff_model, tokenizer, embeddings, encoder, decoder, device, dtype):
    print("\n" + "="*80)
    print("DIAGNOSTIC 2: Layer-wise Representation Collapse (Where does 'CALE' form?)")
    print("="*80)
    
    text1 = "The quick brown fox jumps over the lazy dog."
    text2 = "Мама мыла раму, а папа чинил телевизор."
    text3 = "Quantum mechanics governs subatomic particle interactions."
    
    z1, m1 = encode_clean_text(text1, tokenizer, embeddings, encoder, device, dtype)
    z2, m2 = encode_clean_text(text2, tokenizer, embeddings, encoder, device, dtype)
    z3, m3 = encode_clean_text(text3, tokenizer, embeddings, encoder, device, dtype)
    
    hidden_states_map = {}
    
    def get_hook(layer_idx):
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            hidden_states_map[layer_idx] = h.detach().float()
        return hook
        
    hooks = []
    for idx, layer in enumerate(diff_model.dus.layers):
        hooks.append(layer.register_forward_hook(get_hook(idx)))
        
    t_test = torch.tensor([0.5], device=device)
    
    with torch.no_grad():
        out1 = diff_model(z1, m1, t_test)
        states1 = {k: v.clone() for k, v in hidden_states_map.items()}
        
        out2 = diff_model(z2, m2, t_test)
        states2 = {k: v.clone() for k, v in hidden_states_map.items()}
        
    for h in hooks:
        h.remove()
        
    print(f"{'Layer':<6} | {'CosSim(Text1, Text2)':<22} | {'Text1 Hidden Variance':<22} | {'Status'}")
    print("-" * 75)
    
    for idx in range(len(diff_model.dus.layers)):
        h1 = states1[idx].squeeze(0) # [T1, D]
        h2 = states2[idx].squeeze(0) # [T2, D]
        
        # Mean pooling per sample
        m1_vec = h1.mean(dim=0)
        m2_vec = h2.mean(dim=0)
        
        cos_between_samples = F.cosine_similarity(m1_vec.unsqueeze(0), m2_vec.unsqueeze(0)).item()
        var_within_sample = h1.var(dim=0).mean().item()
        
        status = "OK"
        if cos_between_samples > 0.95:
            status = "COLLAPSED (Rank-1)"
        elif cos_between_samples > 0.8:
            status = "NARROWING"
            
        if idx % 4 == 0 or idx == len(diff_model.dus.layers) - 1 or status == "COLLAPSED (Rank-1)":
            print(f"{idx:<6} | {cos_between_samples:<22.4f} | {var_within_sample:<22.6f} | {status}")

def analyze_gradients_and_weights(diff_model, device):
    print("\n" + "="*80)
    print("DIAGNOSTIC 3: Weight Norms and Gradient Flow Analysis")
    print("="*80)
    
    t_proj_weight_norm = diff_model.t_proj[0].weight.norm().item()
    t_proj_grad_scale = diff_model.t_proj[0].weight.std().item()
    print(f"t_proj first layer weight norm: {t_proj_weight_norm:.4f}, std: {t_proj_grad_scale:.6f}")
    
    adaLN_attn_weights = [m.modulation[-1].weight.norm().item() for m in diff_model.adaLN_attn]
    adaLN_attn_biases = [m.modulation[-1].bias.norm().item() for m in diff_model.adaLN_attn]
    
    print(f"AdaLN_attn weight norms min/max: {min(adaLN_attn_weights):.4f} / {max(adaLN_attn_weights):.4f}")
    print(f"AdaLN_attn bias norms min/max:   {min(adaLN_attn_biases):.4f} / {max(adaLN_attn_biases):.4f}")
    
    # LayerNorm final weights
    final_norm_w = diff_model.dus.final_norm.weight.norm().item()
    print(f"DUS final_norm weight norm: {final_norm_w:.4f}")

def main():
    ckpt_path = r"C:\Experiments\BEBLaDII\experiments\phase3\phase3_step_6995.pth"
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return
        
    state = load_checkpoint_and_inspect(ckpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    tokenizer, embeddings, encoder, decoder, diff_model, lm_head_weight, sep_embed = load_models(device, dtype, state)
    
    analyze_adaln_sensitivity(diff_model, device)
    analyze_layerwise_collapse(diff_model, tokenizer, embeddings, encoder, decoder, device, dtype)
    analyze_gradients_and_weights(diff_model, device)

if __name__ == "__main__":
    main()
