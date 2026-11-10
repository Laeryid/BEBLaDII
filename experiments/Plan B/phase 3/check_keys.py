import torch

ckpt_path = r"C:\Experiments\BEBLaDII\experiments\phase3\planB_phase3_checkpoints_phase3_step_8000.pth"
state = torch.load(ckpt_path, map_location="cpu")

dus = state.get("dus", state)
if "model_state_dict" in dus:
    dus = dus["model_state_dict"]
elif "latentBERT_state_dict" in dus:
    dus = dus["latentBERT_state_dict"]

clean_dus = {}
for k, v in dus.items():
    k_clean = k.replace("student.model.", "").replace("model.", "").replace("_orig_module.", "").replace("_fsdp_wrapped_module.", "")
    clean_dus[k_clean] = v

print("All keys in checkpoint:")
for k in sorted(clean_dus.keys()):
    if "final" in k or "norm" in k:
        print(f"  {k}")
