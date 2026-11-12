import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("experiments/phase 1")
from train_phase1 import GrassmannianPhase1Model

def inspect_checkpoint():
    ckpt_path = "experiments/phase 1/checkpoints/phase1_best_model.pt"
    print(f"Loading checkpoint: {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    step = ckpt.get("step", "unknown")
    val_loss = ckpt.get("val_loss", "unknown")
    print(f"Checkpoint info: step={step}, val_loss={val_loss}")

    model = GrassmannianPhase1Model(vocab_size=131072)
    # Загружаем shadow веса EMA или веса модели
    if "ema_state_dict" in ckpt and "shadow" in ckpt["ema_state_dict"]:
        print("Using EMA shadow weights...")
        state_dict = ckpt["ema_state_dict"]["shadow"]
    else:
        print("Using model state dict...")
        state_dict = ckpt["model_state_dict"]

    # Очищаем префикс module. если модель училась в DataParallel
    cleaned_sd = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "")
        cleaned_sd[new_k] = v

    model.load_state_dict(cleaned_sd, strict=True)
    model.eval()
    print("Model loaded successfully!")

    # Генерируем тестовый ортонормированный V_cell (2, 32, 8, 128)
    B, N, K, D = 2, 32, 8, 128
    torch.manual_seed(42)
    V_cell = torch.randn(B, N, K, D)
    V_cell, _ = torch.linalg.qr(V_cell.view(B * N, D, K))
    V_cell = V_cell.transpose(-2, -1).view(B, N, K, D)

    for dtype, dtype_name in [(torch.float32, "FP32"), (torch.float16, "FP16")]:
        print(f"\n================ Running decode in {dtype_name} ================")
        V_test = V_cell.to(dtype)
        m = model.to(dtype)

        with torch.no_grad():
            cell_vec = m.cell_in_norm(V_test.view(B, N, K * D))
            cos_N, sin_N = m.rope(cell_vec, N)
            h_cell = m.dec_context_layer(cell_vec, cos_N, sin_N)
            V_cell_mod = V_test + h_cell.view(B, N, K, D)

            print(f"cell_vec: finite={torch.isfinite(cell_vec).all().item()}, norm={cell_vec.norm(dim=-1).mean().item():.3f}, min={cell_vec.min().item():.3f}, max={cell_vec.max().item():.3f}")
            print(f"h_cell: finite={torch.isfinite(h_cell).all().item()}, norm={h_cell.norm(dim=-1).mean().item():.3f}, min={h_cell.min().item():.3f}, max={h_cell.max().item():.3f}")
            print(f"V_cell_mod: finite={torch.isfinite(V_cell_mod).all().item()}, norm={V_cell_mod.norm(dim=-1).mean().item():.3f}, min={V_cell_mod.min().item():.3f}, max={V_cell_mod.max().item():.3f}")

            # Inside unpacker:
            Q = m.unpacker.q_norm(m.unpacker.query_slots).unsqueeze(0).unsqueeze(0).expand(B, N, m.unpacker.num_slots, D)
            K_mat = m.unpacker.k_norm(V_cell_mod)
            V_mat = V_cell_mod

            scores = torch.matmul(Q, K_mat.transpose(-2, -1)) / (D ** 0.5)
            attn = F.softmax(scores, dim=-1)
            h_attn = torch.matmul(attn, V_mat)

            G = torch.matmul(V_cell_mod, V_cell_mod.transpose(-2, -1))
            G_flat = G.view(B, N, 64)
            h_gram = m.unpacker.gram_mlp(G_flat)

            print(f"scores: finite={torch.isfinite(scores).all().item()}, min={scores.min().item():.3f}, max={scores.max().item():.3f}")
            print(f"attn: finite={torch.isfinite(attn).all().item()}")
            print(f"h_attn: finite={torch.isfinite(h_attn).all().item()}, norm={h_attn.norm(dim=-1).mean().item():.3f}")
            print(f"G: finite={torch.isfinite(G).all().item()}, min={G.min().item():.3f}, max={G.max().item():.3f}")
            print(f"h_gram: finite={torch.isfinite(h_gram).all().item()}, norm={h_gram.norm(dim=-1).mean().item():.3f}, min={h_gram.min().item():.3f}, max={h_gram.max().item():.3f}")

            h_comb = torch.cat([h_attn, h_gram.unsqueeze(2).expand(B, N, m.unpacker.num_slots, 128)], dim=-1)
            h_out = m.unpacker.out_norm(m.unpacker.out_proj(h_comb))
            h_unpacked = h_out.view(B, N * 4, 1024)
            print(f"h_unpacked: finite={torch.isfinite(h_unpacked).all().item()}, norm={h_unpacked.norm(dim=-1).mean().item():.3f}")

            # Smooth layers:
            cos_full, sin_full = m.rope(h_unpacked, N * 4)
            x = h_unpacked
            for i, layer in enumerate(m.dec_smooth_layers):
                x = layer(x, cos_full, sin_full)
                print(f"smooth layer {i}: finite={torch.isfinite(x).all().item()}, norm={x.norm(dim=-1).mean().item():.3f}, max={x.max().item():.3f}")

            x = m.dec_final_norm(x)
            print(f"dec_final_norm: finite={torch.isfinite(x).all().item()}, norm={x.norm(dim=-1).mean().item():.3f}")

            logits = m.get_tied_head_logits(x)
            print(f"logits: finite={torch.isfinite(logits).all().item()}, min={logits.min().item():.3f}, max={logits.max().item():.3f}")

if __name__ == "__main__":
    inspect_checkpoint()
