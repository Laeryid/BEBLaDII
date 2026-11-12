import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("experiments/phase 1")
from train_phase1 import GrassmannianPhase1Model

def test_decode_fp16():
    device = torch.device("cpu")
    model = GrassmannianPhase1Model().to(device)
    model.eval()

    B, N, K, D = 2, 32, 8, 128
    V_cell = torch.randn(B, N, K, D, device=device)
    V_cell, _ = torch.linalg.qr(V_cell.view(B * N, D, K))
    V_cell = V_cell.transpose(-2, -1).view(B, N, K, D)

    # Проверяем в float16
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = model.decode(V_cell)
        print("Decode bfloat16:", torch.isfinite(logits).all().item(), "min/max:", logits.min().item(), logits.max().item())

    # Проверяем что происходит если веса чуть-чуть подросли (симуляция шага 300)
    with torch.no_grad():
        for p in model.dec_context_layer.parameters():
            p.mul_(1.5)
        for p in model.unpacker.parameters():
            p.mul_(1.5)
        for p in model.dec_smooth_layers.parameters():
            p.mul_(1.5)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = model.decode(V_cell)
        print("Decode scaled bfloat16:", torch.isfinite(logits).all().item(), "min/max:", logits.min().item(), logits.max().item())

    # А теперь эмулируем половинную точность float16
    model_f16 = GrassmannianPhase1Model().to(device).half()
    V_cell_f16 = V_cell.half()
    with torch.no_grad():
        for p in model_f16.dec_context_layer.parameters():
            p.mul_(1.5)
        for p in model_f16.unpacker.parameters():
            p.mul_(1.5)
        for p in model_f16.dec_smooth_layers.parameters():
            p.mul_(1.5)
        try:
            logits_f16 = model_f16.decode(V_cell_f16)
            print("Decode float16:", torch.isfinite(logits_f16).all().item(), "min/max:", logits_f16.min().item(), logits_f16.max().item())
        except Exception as e:
            print("Error in float16 decode:", e)

if __name__ == "__main__":
    test_decode_fp16()
