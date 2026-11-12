import sys
import torch
import torch.nn.functional as F

sys.path.append("experiments/phase 1")
from train_phase1 import GrassmannianPhase1Model

def test_full_pipeline():
    ckpt_path = "experiments/phase 1/checkpoints/phase1_best_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    cleaned_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model = GrassmannianPhase1Model(vocab_size=131072)
    model.load_state_dict(cleaned_sd, strict=True)
    model.train()

    device = torch.device("cpu")
    B, N, K, D = 2, 64, 8, 128
    V_cell = torch.randn(B, N, K, D, device=device) * 200.0  # экстремальная норма

    print("--- ТЕСТ 1: Текущий прогон в float16 ---")
    with torch.autocast(device_type="cpu", dtype=torch.float16):
        logits = model.decode(V_cell)
        print("Logits fp16 finite:", torch.isfinite(logits).all().item())

    print("\n--- ТЕСТ 2: Текущий прогон в bfloat16 ---")
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = model.decode(V_cell)
        print("Logits bf16 finite:", torch.isfinite(logits).all().item(), "max:", logits.abs().max().item())

if __name__ == "__main__":
    test_full_pipeline()
