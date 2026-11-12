import sys
import torch
import torch.nn.functional as F

sys.path.append("experiments/phase 1")
from train_phase1 import GrassmannianPhase1Model

def test_decode_synthetic():
    torch.manual_seed(42)
    device = torch.device("cpu")
    model = GrassmannianPhase1Model(vocab_size=131072).to(device)
    model.train()

    B, N, K, D = 2, 64, 8, 128

    print("=== ТЕСТ 1: Синтетические V_cell с нормами от 1 до 500 ===")
    for scale in [1.0, 5.0, 10.0, 50.0, 100.0, 200.0, 500.0]:
        V_cell = torch.randn(B, N, K, D, device=device) * scale

        # Тест в float16 autocast
        with torch.autocast(device_type="cpu", dtype=torch.float16):
            logits_fp16 = model.decode(V_cell)
            is_fin_fp16 = torch.isfinite(logits_fp16).all().item()

        # Тест в bfloat16 autocast
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits_bf16 = model.decode(V_cell)
            is_fin_bf16 = torch.isfinite(logits_bf16).all().item()

        print(f"Scale {scale:5.1f} | FP16 finite: {is_fin_fp16} (max={logits_fp16.abs().max().item():.1e}) | BF16 finite: {is_fin_bf16} (max={logits_bf16.abs().max().item():.1e})")

if __name__ == "__main__":
    test_decode_synthetic()
