import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.append("experiments/phase 1")
from train_phase1 import GrassmannianPhase1Model, DualRegimeDataset, load_tekken_tokenizer, EnvironmentConfig

def trace_decode_tensors():
    env = EnvironmentConfig.detect()
    tokenizer = load_tekken_tokenizer(env)

    ckpt_path = "experiments/phase 1/checkpoints/phase1_best_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    cleaned_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model = GrassmannianPhase1Model(vocab_size=131072)
    model.load_state_dict(cleaned_sd, strict=True)
    model.eval()

    # Загружаем датасет
    dataset = DualRegimeDataset(
        parquet_path=env.val_data_path,
        tokenizer=tokenizer,
        seq_len=256,
        chunk_size=4,
    )
    dataset = Subset(dataset, range(min(50, len(dataset))))
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    print("Прогоняем батчи через decode с пошаговой проверкой тензоров...")
    nan_found = False

    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"]
        weights = batch["weights"]

        with torch.no_grad():
            V_cell, kl_loss, rank_loss, rk_eff = model.encode_and_pool(input_ids, weights)
            
            # Пошагово внутри decode:
            B, N, K, D = V_cell.shape
            cell_vec = model.cell_in_norm(V_cell.view(B, N, K * D))
            cos_N, sin_N = model.rope(cell_vec, N)
            h_cell = model.dec_context_layer(cell_vec, cos_N, sin_N)
            V_cell_mod = V_cell + h_cell.view(B, N, K, D)

            # Внутри unpacker:
            Q = model.unpacker.q_norm(model.unpacker.query_slots).unsqueeze(0).unsqueeze(0).expand(B, N, model.unpacker.num_slots, D).to(V_cell_mod.dtype)
            K_mat = model.unpacker.k_norm(V_cell_mod)
            V_mat = V_cell_mod

            scores = torch.matmul(Q, K_mat.transpose(-2, -1)) / (D ** 0.5)
            attn = F.softmax(scores.float(), dim=-1).to(V_mat.dtype)
            h_attn = torch.matmul(attn, V_mat)

            # Матрица Грама и gram_mlp:
            V_f32 = V_cell_mod.float()
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
            h_gram = h_normed.to(V_cell_mod.dtype).unsqueeze(2).expand(B, N, model.unpacker.num_slots, 128)

            h_comb = torch.cat([h_attn, h_gram], dim=-1)
            h_out = model.unpacker.out_norm(model.unpacker.out_proj(h_comb))
            h_unpacked = h_out.view(B, N * model.unpacker.num_slots, model.unpacker.hidden_size)

            full_len = N * model.chunk_size
            cos_full, sin_full = model.rope(h_unpacked, full_len)
            x = h_unpacked
            for l_idx, layer in enumerate(model.dec_smooth_layers):
                x = layer(x, cos_full, sin_full)
                if not torch.isfinite(x).all():
                    print(f"[NAN DETECTED] batch {batch_idx}: dec_smooth_layers[{l_idx}] дал нечисловое значение! max={x.abs().max().item()}")
                    nan_found = True
                    break

            x_normed = model.dec_final_norm(x)
            logits = model.get_tied_head_logits(x_normed)

            check_dict = {
                "cell_vec": cell_vec,
                "h_cell": h_cell,
                "V_cell_mod": V_cell_mod,
                "scores": scores,
                "attn": attn,
                "h_attn": h_attn,
                "G_f32": G_f32,
                "h_gram": h_gram,
                "h_unpacked": h_unpacked,
                "x_normed": x_normed,
                "logits": logits,
            }
            for name, val in check_dict.items():
                if not torch.isfinite(val).all():
                    print(f"[NAN DETECTED] batch {batch_idx}: {name} содержит NaN/Inf! min={val.min().item()}, max={val.max().item()}")
                    nan_found = True
                    break

            if nan_found:
                break

    if not nan_found:
        print("В FP32 на 25 батчах все тензоры идеально конечны!")

if __name__ == "__main__":
    trace_decode_tensors()
