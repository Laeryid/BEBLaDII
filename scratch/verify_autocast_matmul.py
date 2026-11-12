import sys
import torch
import torch.nn.functional as F

sys.path.append("experiments/phase 1")
from train_phase1 import GrassmannianPhase1Model

def test_fix():
    torch.manual_seed(42)
    device = torch.device("cpu")
    model = GrassmannianPhase1Model(vocab_size=131072).to(device)
    model.train()

    B, N, K, D = 2, 64, 8, 128
    V_cell = torch.randn(B, N, K, D, device=device) * 50.0

    print("--- БЕЗ enabled=False (текущее состояние) ---")
    with torch.autocast(device_type="cpu", dtype=torch.float16):
        V_f32 = V_cell.float()
        G_bad = torch.matmul(V_f32, V_f32.transpose(-2, -1))
        print("G dtype:", G_bad.dtype, "finite:", torch.isfinite(G_bad).all().item(), "max:", G_bad.abs().max().item())

    print("\n--- С ЯВНЫМ enabled=False (как положено для FP32 изоляции) ---")
    with torch.autocast(device_type="cpu", dtype=torch.float16):
        with torch.autocast(device_type="cpu", enabled=False):
            V_f32 = V_cell.float()
            G_good = torch.matmul(V_f32, V_f32.transpose(-2, -1))
            print("G dtype:", G_good.dtype, "finite:", torch.isfinite(G_good).all().item(), "max:", G_good.abs().max().item())

if __name__ == "__main__":
    test_fix()
