import sys
import torch
import torch.nn.functional as F

sys.path.append("experiments/phase 1")
from train_phase1 import GrassmannianPhase1Model

def pinpoint():
    torch.manual_seed(42)
    device = torch.device("cpu")
    model = GrassmannianPhase1Model(vocab_size=131072).to(device)
    model.train()

    B, N, K, D = 2, 64, 8, 128
    V_cell = torch.randn(B, N, K, D, device=device) * 50.0

    with torch.autocast(device_type="cpu", dtype=torch.float16):
        # Пошагово внутри decode
        cell_vec = model.cell_in_norm(V_cell.view(B, N, K * D))
        print("cell_vec finite:", torch.isfinite(cell_vec).all().item(), "max:", cell_vec.abs().max().item())

        cos_N, sin_N = model.rope(cell_vec, N)
        h_cell = model.dec_context_layer(cell_vec, cos_N, sin_N)
        print("h_cell finite:", torch.isfinite(h_cell).all().item(), "max:", h_cell.abs().max().item())

        V_cell_mod = V_cell + h_cell.view(B, N, K, D)
        print("V_cell_mod finite:", torch.isfinite(V_cell_mod).all().item(), "max:", V_cell_mod.abs().max().item())

        # Внутри unpacker:
        Q = model.unpacker.q_norm(model.unpacker.query_slots).unsqueeze(0).unsqueeze(0).expand(B, N, model.unpacker.num_slots, D).to(V_cell_mod.dtype)
        K_mat = model.unpacker.k_norm(V_cell_mod)
        V_mat = V_cell_mod
        print("K_mat finite:", torch.isfinite(K_mat).all().item(), "max:", K_mat.abs().max().item())

        scores = torch.matmul(Q, K_mat.transpose(-2, -1)) / (D ** 0.5)
        print("scores finite:", torch.isfinite(scores).all().item(), "max:", scores.abs().max().item())

        attn = F.softmax(scores.float(), dim=-1).to(V_mat.dtype)
        print("attn finite:", torch.isfinite(attn).all().item(), "max:", attn.abs().max().item())

        h_attn = torch.matmul(attn, V_mat)
        print("h_attn finite:", torch.isfinite(h_attn).all().item(), "max:", h_attn.abs().max().item())

        # G_f32 и gram_mlp:
        target_dtype = V_cell_mod.dtype
        V_f32 = V_cell_mod.float()
        G_f32 = torch.matmul(V_f32, V_f32.transpose(-2, -1))
        G_flat_f32 = G_f32.view(B, N, 64)
        print("G_flat_f32 finite:", torch.isfinite(G_flat_f32).all().item(), "max:", G_flat_f32.abs().max().item())

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
        print("h_gram finite:", torch.isfinite(h_gram).all().item(), "max:", h_gram.abs().max().item())

        h_comb = torch.cat([h_attn, h_gram], dim=-1)
        print("h_comb finite:", torch.isfinite(h_comb).all().item(), "max:", h_comb.abs().max().item())

        out_linear = model.unpacker.out_proj(h_comb)
        print("out_linear finite:", torch.isfinite(out_linear).all().item(), "max:", out_linear.abs().max().item())

        h_out = model.unpacker.out_norm(out_linear)
        print("h_out finite:", torch.isfinite(h_out).all().item(), "max:", h_out.abs().max().item())

        h_unpacked = h_out.view(B, N * model.unpacker.num_slots, model.unpacker.hidden_size)

        full_len = N * model.chunk_size
        cos_full, sin_full = model.rope(h_unpacked, full_len)
        x = h_unpacked
        for l_idx, layer in enumerate(model.dec_smooth_layers):
            x = layer(x, cos_full, sin_full)
            print(f"dec_smooth_layers[{l_idx}] finite:", torch.isfinite(x).all().item(), "max:", x.abs().max().item())

        x_final = model.dec_final_norm(x)
        print("x_final finite:", torch.isfinite(x_final).all().item(), "max:", x_final.abs().max().item())

        logits = model.get_tied_head_logits(x_final)
        print("logits finite:", torch.isfinite(logits).all().item(), "max:", logits.abs().max().item())

if __name__ == "__main__":
    pinpoint()
