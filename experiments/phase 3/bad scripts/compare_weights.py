import torch
import copy

def get_norms(ckpt_path, key_prefix=""):
    print(f"\nAnalyzing: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # Try to extract DUS weights
    if "dus_ema" in state:
        dus = state["dus_ema"]
    elif "dus" in state:
        dus = state["dus"]
    elif "latentBERT_state_dict" in state:
        dus = state["latentBERT_state_dict"]
    elif "model_state_dict" in state:
        dus = state["model_state_dict"]
    else:
        dus = state

    clean_dus = {}
    for k, v in dus.items():
        k_clean = k.replace("student.model.", "").replace("model.", "").replace("_orig_module.", "").replace("_fsdp_wrapped_module.", "")
        clean_dus[k_clean] = v

    layer_norms = {}
    for i in range(40):
        layer_norms[i] = {
            "Wqkv": 0,
            "Wo": 0,
            "attn_norm": 0,
            "Wi": 0,
            "mlp_Wo": 0,
            "mlp_norm": 0
        }

    for k, v in clean_dus.items():
        if k.startswith("layers."):
            parts = k.split(".")
            layer_idx = int(parts[1])
            sub_comp = parts[2]
            if sub_comp == "attn":
                if parts[3] == "Wqkv": layer_norms[layer_idx]["Wqkv"] += v.norm().item()
                elif parts[3] == "Wo": layer_norms[layer_idx]["Wo"] += v.norm().item()
            elif sub_comp == "attn_norm": layer_norms[layer_idx]["attn_norm"] += v.norm().item()
            elif sub_comp == "mlp":
                if parts[3] == "Wi": layer_norms[layer_idx]["Wi"] += v.norm().item()
                elif parts[3] == "Wo": layer_norms[layer_idx]["mlp_Wo"] += v.norm().item()
            elif sub_comp == "mlp_norm": layer_norms[layer_idx]["mlp_norm"] += v.norm().item()

    return layer_norms

path_phase3 = r"C:\Experiments\BEBLaDII\experiments\phase3\planB_phase3_checkpoints_phase3_step_8000.pth"
path_awakened = r"C:\Experiments\BEBLaDII\kaggle_upload_1_2\AWAKENED_WEIGHTS_FINAL.pt"

norms_p3 = get_norms(path_phase3)
norms_aw = get_norms(path_awakened)

print("\nComparison of Layer 19 (Original) and Layer 31 (Copied from 19):")
print("-" * 60)
print("AWAKENED_WEIGHTS_FINAL.pt:")
print(f"Layer 19 Wqkv: {norms_aw[19]['Wqkv']:.4f}")
print(f"Layer 31 Wqkv: {norms_aw[31]['Wqkv']:.4f}")
print(f"Diff: {abs(norms_aw[19]['Wqkv'] - norms_aw[31]['Wqkv']):.4f}")

print("\nplanB_phase3_checkpoints_phase3_step_8000.pth:")
print(f"Layer 19 Wqkv: {norms_p3[19]['Wqkv']:.4f}")
print(f"Layer 31 Wqkv: {norms_p3[31]['Wqkv']:.4f}")
print(f"Diff: {abs(norms_p3[19]['Wqkv'] - norms_p3[31]['Wqkv']):.4f}")

print("\nHas Layer 31 moved between Awakened and Phase3?")
print(f"Awakened L31 Wqkv: {norms_aw[31]['Wqkv']:.4f}")
print(f"Phase3 L31 Wqkv:   {norms_p3[31]['Wqkv']:.4f}")
print(f"Diff: {abs(norms_aw[31]['Wqkv'] - norms_p3[31]['Wqkv']):.4f}")
