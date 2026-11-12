import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("experiments/phase 1")
from train_phase1 import SetCrossAttentionUnpacker

def test_unpacker_backward():
    unpacker = SetCrossAttentionUnpacker(axis_dim=128, num_slots=4, hidden_size=1024)
    # Симулируем autocast GPU: модель в FP32, но вход в Half
    B, N, K, D = 2, 32, 8, 128
    V_cell = torch.randn(B, N, K, D, requires_grad=True)

    # Прогоняем в half с защищенным float32 расчетом G и gram_mlp:
    Q = unpacker.q_norm(unpacker.query_slots).unsqueeze(0).unsqueeze(0).expand(B, N, unpacker.num_slots, D)
    K_mat = unpacker.k_norm(V_cell)
    scores = torch.matmul(Q, K_mat.transpose(-2, -1)) / (D ** 0.5)
    attn = F.softmax(scores, dim=-1)
    h_attn = torch.matmul(attn, V_cell)

    # G и gram_mlp в float32:
    V_f32 = V_cell.float()
    G = torch.matmul(V_f32, V_f32.transpose(-2, -1))
    G_flat = G.view(B, N, 64)
    # gram_mlp выполняется в float32
    h_gram = unpacker.gram_mlp(G_flat)
    h_gram_exp = h_gram.unsqueeze(2).expand(B, N, unpacker.num_slots, 128)

    h_comb = torch.cat([h_attn, h_gram_exp], dim=-1)
    h_out = unpacker.out_norm(unpacker.out_proj(h_comb))
    loss = h_out.sum()

    loss.backward()
    print("V_cell grad finite:", torch.isfinite(V_cell.grad).all().item())
    print("V_cell grad norm:", V_cell.grad.norm().item())
    print("gram_mlp weight grad finite:", torch.isfinite(unpacker.gram_mlp[0].weight.grad).all().item())
    print("gram_mlp weight grad norm:", unpacker.gram_mlp[0].weight.grad.norm().item())

if __name__ == "__main__":
    test_unpacker_backward()
