import sys
import torch

def check_checkpoint(ckpt_path):
    print(f"Inspecting checkpoint: {ckpt_path}\n")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    for module_name in ["encoder", "decoder"]:
        state = ckpt.get(module_name, {})
        keys = list(state.keys())
        print(f"--- {module_name.upper()} ---")
        print(f"Keys count: {len(keys)}")
        if len(keys) > 0:
            print(f"Sample keys: {keys[:5]}")
        else:
            print("EMPTY!")
        print()

    if len(ckpt.get("encoder", {})) == 0 or len(ckpt.get("decoder", {})) == 0:
        print("❌ ERROR: Checkpoint is empty! State dict filtering failed.")
    else:
        print("✅ SUCCESS: Checkpoint contains weights.")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/phase1/phase1_vae_step_50.pth"
    check_checkpoint(path)
