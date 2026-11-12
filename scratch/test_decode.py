import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Импортируем классы из train_phase1
sys.path.append("experiments/phase 1")
from train_phase1 import GrassmannianPhase1Model

def test_decode_stability():
    device = torch.device("cpu")
    model = GrassmannianPhase1Model().to(device)
    model.eval()

    B, N, K, D = 2, 32, 8, 128
    # V_cell: ортонормированный репер со случайными знаками
    V_cell = torch.randn(B, N, K, D, device=device)
    V_cell, _ = torch.linalg.qr(V_cell.view(B * N, D, K))
    V_cell = V_cell.transpose(-2, -1).view(B, N, K, D)

    print("V_cell is finite:", torch.isfinite(V_cell).all().item())
    print("V_cell norm:", torch.norm(V_cell, dim=-1).mean().item())

    # Прогоняем decode пошагово
    cell_vec = model.cell_in_norm(V_cell.view(B, N, K * D))
    cos_N, sin_N = model.rope(cell_vec, N)
    print("cell_vec finite:", torch.isfinite(cell_vec).all().item(), "norm:", cell_vec.norm(dim=-1).mean().item())

    h_cell = model.dec_context_layer(cell_vec, cos_N, sin_N)
    print("h_cell finite:", torch.isfinite(h_cell).all().item(), "norm:", h_cell.norm(dim=-1).mean().item())

    V_cell_mod = V_cell + h_cell.view(B, N, K, D)
    print("V_cell_mod norm:", torch.norm(V_cell_mod, dim=-1).mean().item())

    # Внутри unpacker:
    Q = model.unpacker.q_norm(model.unpacker.query_slots).unsqueeze(0).unsqueeze(0).expand(B, N, model.unpacker.num_slots, D)
    K_mat = model.unpacker.k_norm(V_cell_mod)
    scores = torch.matmul(Q, K_mat.transpose(-2, -1)) / (D ** 0.5)
    attn = F.softmax(scores, dim=-1)
    h_attn = torch.matmul(attn, V_cell_mod)
    print("h_attn norm:", h_attn.norm(dim=-1).mean().item())

    G = torch.matmul(V_cell_mod, V_cell_mod.transpose(-2, -1))
    print("G min/max:", G.min().item(), G.max().item())
    G_flat = G.view(B, N, 64)
    h_gram = model.unpacker.gram_mlp(G_flat)
    print("h_gram finite:", torch.isfinite(h_gram).all().item(), "norm:", h_gram.norm(dim=-1).mean().item())

    h_comb = torch.cat([h_attn, h_gram.unsqueeze(2).expand(B, N, model.unpacker.num_slots, 128)], dim=-1)
    h_unpacked = model.unpacker.out_norm(model.unpacker.out_proj(h_comb)).view(B, N * 4, 1024)
    print("h_unpacked norm:", h_unpacked.norm(dim=-1).mean().item())

    # Smooth layers:
    cos_full, sin_full = model.rope(h_unpacked, N * 4)
    x = h_unpacked
    for i, layer in enumerate(model.dec_smooth_layers):
        x = layer(x, cos_full, sin_full)
        print(f"layer {i} output norm:", x.norm(dim=-1).mean().item())
    x = model.dec_final_norm(x)
    logits = model.get_tied_head_logits(x)
    print("logits finite:", torch.isfinite(logits).all().item(), "min/max:", logits.min().item(), logits.max().item())

if __name__ == "__main__":
    test_decode_stability()
