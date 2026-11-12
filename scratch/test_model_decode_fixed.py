import sys
import torch
import torch.nn.functional as F

sys.path.append("experiments/phase 1")
from train_phase1 import GrassmannianPhase1Model

def test_fixed_model():
    ckpt_path = "experiments/phase 1/checkpoints/phase1_best_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    cleaned_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model = GrassmannianPhase1Model(vocab_size=131072)
    model.load_state_dict(cleaned_sd, strict=True)
    model.eval()

    B, N, K, D = 2, 32, 8, 128
    torch.manual_seed(42)
    V_cell = torch.randn(B, N, K, D)
    V_cell, _ = torch.linalg.qr(V_cell.view(B * N, D, K))
    V_cell = V_cell.transpose(-2, -1).view(B, N, K, D)

    # Тестируем прямой вызов decode()
    print("Testing model.decode(V_cell) in FP32...")
    with torch.no_grad():
        logits_fp32 = model.decode(V_cell)
        print("FP32 logits finite:", torch.isfinite(logits_fp32).all().item(), "min/max:", logits_fp32.min().item(), logits_fp32.max().item())

    print("\nTesting model.decode(V_cell) in FP16 (emulating Kaggle T4 GPU)...")
    # Переводим в FP16
    model.half()
    # Возвращаем gram_mlp в FP32, как это происходит при autocast на GPU (параметры FP32, а autocast кастует в half)
    # Или проверяем чистый half режим
    with torch.no_grad():
        logits_f16 = model.decode(V_cell.half())
        print("FP16 logits finite:", torch.isfinite(logits_f16).all().item(), "min/max:", logits_f16.float().min().item(), logits_f16.float().max().item())

if __name__ == "__main__":
    test_fixed_model()
