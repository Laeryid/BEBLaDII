import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("experiments/phase 1")
from train_phase1 import GrassmannianPhase1Model

def test_autocast_fix():
    ckpt_path = "experiments/phase 1/checkpoints/phase1_best_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    cleaned_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model = GrassmannianPhase1Model(vocab_size=131072)
    model.load_state_dict(cleaned_sd, strict=True)
    model.eval()

    # V_cell (ортонормированный)
    B, N, K, D = 2, 32, 8, 128
    torch.manual_seed(42)
    V_cell = torch.randn(B, N, K, D)
    V_cell, _ = torch.linalg.qr(V_cell.view(B * N, D, K))
    V_cell = V_cell.transpose(-2, -1).view(B, N, K, D)

    # 1. Проверяем текущий decode под autocast float16 (эмулируем поведение GPU)
    print("--- 1. Текущий decode с имитацией autocast float16 ---")
    with torch.no_grad():
        cell_vec = model.cell_in_norm(V_cell.view(B, N, K * D))
        cos_N, sin_N = model.rope(cell_vec, N)
        h_cell = model.dec_context_layer(cell_vec, cos_N, sin_N)
        V_cell_mod = V_cell + h_cell.view(B, N, K, D)

        # Симуляция FP16 matmul для G:
        G_f16 = torch.matmul(V_cell_mod.half(), V_cell_mod.half().transpose(-2, -1))
        print("G_f16 finite:", torch.isfinite(G_f16).all().item(), "min/max:", G_f16.min().item(), G_f16.max().item())

    # 2. Проверяем безопасный расчет G и gram_mlp в float32:
    print("\n--- 2. Безопасный decode с расчетом G и gram_mlp в float32 ---")
    model.half()
    with torch.no_grad():
        V_cell_mod_f16 = V_cell_mod.half()
        
        # Q, K, V в half
        Q = model.unpacker.q_norm(model.unpacker.query_slots).unsqueeze(0).unsqueeze(0).expand(B, N, model.unpacker.num_slots, D)
        K_mat = model.unpacker.k_norm(V_cell_mod_f16)
        scores = torch.matmul(Q, K_mat.transpose(-2, -1)) / (D ** 0.5)
        attn = F.softmax(scores.float(), dim=-1).half()
        h_attn = torch.matmul(attn, V_cell_mod_f16)
        print("h_attn finite:", torch.isfinite(h_attn).all().item(), "norm:", h_attn.float().norm(dim=-1).mean().item())

        # G и gram_mlp строго в float32:
        G_f32 = torch.matmul(V_cell_mod_f16.float(), V_cell_mod_f16.float().transpose(-2, -1))
        G_flat_f32 = G_f32.view(B, N, 64)
        # gram_mlp в float32:
        gram_mlp_f32 = model.unpacker.gram_mlp.float()
        h_gram_f32 = gram_mlp_f32(G_flat_f32)
        print("h_gram_f32 finite:", torch.isfinite(h_gram_f32).all().item(), "norm:", h_gram_f32.norm(dim=-1).mean().item())

        # Конкатенируем и проецируем в half:
        model.unpacker.half()
        h_gram_exp = h_gram_f32.half().unsqueeze(2).expand(B, N, model.unpacker.num_slots, 128)
        h_comb = torch.cat([h_attn, h_gram_exp], dim=-1)
        h_out = model.unpacker.out_norm(model.unpacker.out_proj(h_comb))
        h_unpacked = h_out.view(B, N * 4, 1024)
        print("h_unpacked finite:", torch.isfinite(h_unpacked).all().item(), "norm:", h_unpacked.float().norm(dim=-1).mean().item())

        # Smooth layers:
        cos_full, sin_full = model.rope(h_unpacked.float(), N * 4)
        x = h_unpacked
        for i, layer in enumerate(model.dec_smooth_layers):
            x = layer(x, cos_full.half(), sin_full.half())
            print(f"layer {i} finite:", torch.isfinite(x).all().item(), "max:", x.float().abs().max().item())

        x = model.dec_final_norm(x)
        # Tied logits
        logits = model.get_tied_head_logits(x)
        print("logits finite:", torch.isfinite(logits).all().item(), "min/max:", logits.float().min().item(), logits.float().max().item())

if __name__ == "__main__":
    test_autocast_fix()
