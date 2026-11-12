import sys
import torch
import torch.nn.functional as F

sys.path.append("experiments/phase 1")
from train_phase1 import GrassmannianPhase1Model, compute_phase1_loss

def test_tied_head_fp32():
    device = torch.device("cpu")
    model = GrassmannianPhase1Model(vocab_size=131072).to(device)
    model.train()

    B, N, K, D = 2, 8, 8, 128
    chunk_size = 4
    seq_len = N * chunk_size
    input_ids = torch.randint(0, 1000, (B, seq_len), device=device)
    targets = torch.randint(0, 1000, (B, seq_len), device=device)
    weights = torch.ones(B, seq_len, device=device)

    # Симулируем большие скрытые состояния перед Tied Head
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits, V_cell, kl_loss, rank_loss, rk_eff = model(input_ids, weights)
        print("Logits dtype:", logits.dtype)
        print("Logits finite:", torch.isfinite(logits).all().item())
        print("Logits min/max:", logits.min().item(), logits.max().item())

        total_loss, metrics = compute_phase1_loss(
            logits=logits,
            targets=targets,
            V_cell=V_cell,
            kl_loss=kl_loss,
            rank_loss=rank_loss,
        )
        print("Total loss:", total_loss.item(), "finite:", torch.isfinite(total_loss).item())

    # Проверяем backward pass
    total_loss.backward()

    emb_grad = model.embeddings.weight.grad
    print("Embeddings grad is not None:", emb_grad is not None)
    print("Embeddings grad dtype:", emb_grad.dtype)
    print("Embeddings grad finite:", torch.isfinite(emb_grad).all().item())

    # Проверяем с экстремальными активациями (> 10000)
    fake_h = torch.randn(B, seq_len, 1024, device=device) * 5000.0  # гигантские значения
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        extreme_logits = model.get_tied_head_logits(fake_h)
        print("Extreme logits finite:", torch.isfinite(extreme_logits).all().item())
        print("Extreme logits min/max:", extreme_logits.min().item(), extreme_logits.max().item())

    print("\n[SUCCESS] Tied Head FP32 test passed perfectly!")

if __name__ == "__main__":
    test_tied_head_fp32()
