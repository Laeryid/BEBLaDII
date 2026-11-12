import sys
import torch

sys.path.append("experiments/phase 1")
from train_phase1 import GrassmannianPhase1Model, EnvironmentConfig

def test_e2e():
    env = EnvironmentConfig.detect()
    print(f"Detected device: {env.device}, dtype: {env.dtype}")

    device = torch.device("cpu")
    model = GrassmannianPhase1Model(vocab_size=2048).to(device)
    model.train()

    B, N, K, D = 2, 64, 8, 128

    print("Тестирование прямого прохода decode с реальным классом из train_phase1.py...")
    for scale in [1.0, 50.0, 200.0, 500.0]:
        V_cell = torch.randn(B, N, K, D, device=device) * scale
        with torch.autocast(device_type="cpu", dtype=torch.float16):
            logits = model.decode(V_cell)
            assert torch.isfinite(logits).all(), f"Logits contains non-finite values at scale {scale}!"
            print(f"Scale {scale:5.1f} | Decode FP16 finite: True | max={logits.abs().max().item():.2f}")

    print("\n[ALL CHECKS PASSED] Исправление в train_phase1.py полностью работоспособно!")

if __name__ == "__main__":
    test_e2e()
