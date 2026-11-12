import sys
import torch

sys.path.append("experiments/phase 1")
from train_phase1 import GrassmannianPhase1Model, create_hybrid_optimizers, EnvironmentConfig, compute_phase1_loss

def test_final():
    env = EnvironmentConfig.detect()
    print(f"Detected device: {env.device}, dtype: {env.dtype}")

    device = torch.device("cpu")
    model = GrassmannianPhase1Model(vocab_size=2048).to(device)
    model.train()

    opt_muon, opt_adamw = create_hybrid_optimizers(model, muon_lr=0.02, adamw_lr=3e-4, weight_decay=0.01)
    print("Muon optimizer groups:", len(opt_muon.param_groups), "weight_decay:", opt_muon.param_groups[0].get("weight_decay"))
    print("AdamW optimizer groups:", len(opt_adamw.param_groups), "weight_decay:", opt_adamw.param_groups[0].get("weight_decay"))

    B, N, K, D = 2, 4, 8, 128
    seq_len = N * 4
    input_ids = torch.randint(0, 1000, (B, seq_len), device=device)
    targets = torch.randint(0, 1000, (B, seq_len), device=device)
    weights = torch.ones(B, seq_len, device=device)

    # Тестируем как в float16, так и в bfloat16
    for dt in [torch.float16, torch.bfloat16]:
        print(f"\nТестирование прогона и шага оптимизатора под autocast({dt})...")
        with torch.autocast(device_type="cpu", dtype=dt):
            logits, V_cell, kl_loss, rank_loss, rk_eff = model(input_ids, weights)
            total_loss, metrics = compute_phase1_loss(
                logits=logits,
                targets=targets,
                V_cell=V_cell,
                kl_loss=kl_loss,
                rank_loss=rank_loss,
            )

        print(f"Loss finite: {torch.isfinite(total_loss).item()}, val: {total_loss.item():.4f}")
        assert torch.isfinite(total_loss), "Loss is non-finite!"

        opt_muon.zero_grad()
        opt_adamw.zero_grad()
        total_loss.backward()

        opt_muon.step()
        opt_adamw.step()
        print(f"Шаг оптимизаторов в {dt} успешно выполнен!")

    print("\n[ALL CHECKS PASSED] Обновленный train_phase1.py полностью стабилен!")

if __name__ == "__main__":
    test_final()
