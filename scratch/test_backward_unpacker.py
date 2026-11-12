import sys
import torch
import torch.nn.functional as F

sys.path.append("experiments/phase 1")
from train_phase1 import GrassmannianPhase1Model, compute_phase1_loss

def test_fast_backward():
    torch.manual_seed(42)
    device = torch.device("cpu")
    # Используем меньший словарь для быстрого теста на CPU
    model = GrassmannianPhase1Model(vocab_size=2048).to(device)
    model.train()

    B, N, K, D = 2, 4, 8, 128

    def fixed_unpacker_forward(V_cell: torch.Tensor) -> torch.Tensor:
        B, N, K, D = V_cell.shape
        Q = model.unpacker.q_norm(model.unpacker.query_slots).unsqueeze(0).unsqueeze(0).expand(B, N, model.unpacker.num_slots, D).to(V_cell.dtype)
        K_mat = model.unpacker.k_norm(V_cell)
        V_mat = V_cell

        scores = torch.matmul(Q, K_mat.transpose(-2, -1)) / (D ** 0.5)
        attn = F.softmax(scores.float(), dim=-1).to(V_mat.dtype)
        h_attn = torch.matmul(attn, V_mat)

        device_type = V_cell.device.type
        target_dtype = V_cell.dtype
        with torch.autocast(device_type=device_type, enabled=False):
            V_f32 = V_cell.float()
            G_f32 = torch.matmul(V_f32, V_f32.transpose(-2, -1))
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

        h_comb = torch.cat([h_attn, h_gram], dim=-1)
        h_out = model.unpacker.out_norm(model.unpacker.out_proj(h_comb))
        return h_out.view(B, N * model.unpacker.num_slots, model.unpacker.hidden_size)

    model.unpacker.forward = fixed_unpacker_forward

    seq_len = N * 4
    input_ids = torch.randint(0, 1000, (B, seq_len), device=device)
    targets = torch.randint(0, 1000, (B, seq_len), device=device)
    weights = torch.ones(B, seq_len, device=device)

    with torch.autocast(device_type="cpu", dtype=torch.float16):
        logits, V_cell, kl_loss, rank_loss, rk_eff = model(input_ids, weights)
        total_loss, metrics = compute_phase1_loss(
            logits=logits,
            targets=targets,
            V_cell=V_cell,
            kl_loss=kl_loss,
            rank_loss=rank_loss,
        )

    print("Total loss:", total_loss.item(), "finite:", torch.isfinite(total_loss).item())
    total_loss.backward()

    # Проверяем градиенты
    all_grads_finite = True
    nan_params = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None or not torch.isfinite(p.grad).all():
                all_grads_finite = False
                nan_params.append(name)

    print("All gradients finite:", all_grads_finite)
    if not all_grads_finite:
        print("Params with non-finite grad:", nan_params)
    else:
        print("[SUCCESS] Все градиенты параметров модели строго конечны!")

if __name__ == "__main__":
    test_fast_backward()
