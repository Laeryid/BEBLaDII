import torch

ckpt_path = r"C:\Experiments\BEBLaDII\experiments\phase3\planB_phase3_checkpoints_phase3_step_8000.pth"
state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

dus = state.get("dus_ema", state.get("dus", {}))
if "model_state_dict" in dus:
    dus = dus["model_state_dict"]

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

print(f"{'Layer':<5} | {'Wqkv':<10} | {'Wo':<10} | {'AttnNorm':<10} | {'Wi':<10} | {'MlpWo':<10} | {'MlpNorm':<10}")
print("-" * 75)
for i in range(40):
    n = layer_norms[i]
    print(f"{i:<5} | {n['Wqkv']:<10.2f} | {n['Wo']:<10.2f} | {n['attn_norm']:<10.2f} | {n['Wi']:<10.2f} | {n['mlp_Wo']:<10.2f} | {n['mlp_norm']:<10.2f}")

print("-" * 75)
if "embeddings.tok_embeddings.weight" in clean_dus:
    print(f"Tok Embeddings: {clean_dus['embeddings.tok_embeddings.weight'].norm().item():.2f}")
if "embeddings.norm.weight" in clean_dus:
    print(f"Tok Norm: {clean_dus['embeddings.norm.weight'].norm().item():.2f}")
if "final_norm.weight" in clean_dus:
    print(f"Final Norm: {clean_dus['final_norm.weight'].norm().item():.2f}")
