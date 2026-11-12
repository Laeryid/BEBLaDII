import sys
import torch
import torch.nn.functional as F

sys.path.append("experiments/phase 1")
from train_phase1 import GrassmannianPhase1Model

def test_full_decode_with_fix():
    torch.manual_seed(42)
    device = torch.device("cpu")
    model = GrassmannianPhase1Model(vocab_size=131072).to(device)
    model.train()

    B, N, K, D = 2, 64, 8, 128

    # Патчим SetCrossAttentionUnpacker.forward с явным autocast enabled=False
    orig_forward = model.unpacker.forward

    def fixed_unpacker_forward(V_cell: torch.Tensor) -> torch.Tensor:
        B, N, K, D = V_cell.shape
        # Q: (1, 1, 4, 128)
        Q = model.unpacker.q_norm(model.unpacker.query_slots).unsqueeze(0).unsqueeze(0).expand(B, N, model.unpacker.num_slots, D).to(V_cell.dtype)
        # K, V: (B, N, 8, 128)
        K_mat = model.unpacker.k_norm(V_cell)
        V_mat = V_cell

        # Внимание: (B, N, 4, 8)
        scores = torch.matmul(Q, K_mat.transpose(-2, -1)) / (D ** 0.5)
        attn = F.softmax(scores.float(), dim=-1).to(V_mat.dtype)
        h_attn = torch.matmul(attn, V_mat)  # (B, N, 4, 128)

        # Матрица Грама клетки: расчет G и gram_mlp строго в float32 с явным отключением autocast!
        device_type = V_cell.device.type
        target_dtype = V_cell.dtype
        with torch.autocast(device_type=device_type, enabled=False):
            V_f32 = V_cell.float()
            G_f32 = torch.matmul(V_f32, V_f32.transpose(-2, -1))  # (B, N, 8, 8)
            G_flat_f32 = G_f32.view(B, N, 64)

            mlp_w0 = model.unpacker.gram_mlp[0].weight.float()
            mlp_b0 = model.unpacker.gram_mlp[0].bias.float() if model.unpacker.gram_mlp[0].bias is not None else None
            h1 = F.silu(F.linear(G_flat_f32, mlp_w0, mlp_b0))
            mlp_w2 = model.unpacker.gram_mlp[2].weight.float()
            mlp_b2 = model.unpacker.gram_mlp[2].bias.float() if model.unpacker.gram_mlp[2].bias is not None else None
            h2 = F.linear(h1, mlp_w2, mlp_b2)
            norm_w = model.unpacker.gram_mlp[3].weight.float()
            variance = h2.pow(2).mean(-1, keepdim=True)
            h_normed = h2 * torch.rsqrt(variance + 1e-6) * norm_w
            h_gram = h_normed.to(target_dtype).unsqueeze(2).expand(B, N, model.unpacker.num_slots, 128)

        # Конкатенация внимания и квадратичных инвариантов
        h_comb = torch.cat([h_attn, h_gram], dim=-1)  # (B, N, 4, 256)
        h_out = model.unpacker.out_norm(model.unpacker.out_proj(h_comb))  # (B, N, 4, 1024)

        return h_out.view(B, N * model.unpacker.num_slots, model.unpacker.hidden_size)

    model.unpacker.forward = fixed_unpacker_forward

    print("=== ТЕСТ: Прогон масштабов от 1 до 500 под autocast(float16) С ФИКСОМ ===")
    for scale in [1.0, 5.0, 10.0, 50.0, 100.0, 200.0, 500.0]:
        V_cell = torch.randn(B, N, K, D, device=device) * scale
        with torch.autocast(device_type="cpu", dtype=torch.float16):
            logits = model.decode(V_cell)
            is_fin = torch.isfinite(logits).all().item()
            print(f"Scale {scale:5.1f} | FP16 finite: {is_fin} | Logits max={logits.abs().max().item():.2f}")

if __name__ == "__main__":
    test_full_decode_with_fix()
