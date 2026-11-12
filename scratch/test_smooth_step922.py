import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.append("experiments/phase 1")
from train_phase1 import GrassmannianPhase1Model, DualRegimeDataset, load_tekken_tokenizer, EnvironmentConfig

def test_smooth_on_train_data():
    env = EnvironmentConfig.detect()
    tokenizer = load_tekken_tokenizer(env)

    ckpt_path = "experiments/phase 1/checkpoints/phase1_best_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    cleaned_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model = GrassmannianPhase1Model(vocab_size=131072)
    model.load_state_dict(cleaned_sd, strict=True)
    model.train()

    dataset = DualRegimeDataset(
        parquet_path=env.data_path,
        tokenizer=tokenizer,
        seq_len=256,
        chunk_size=4,
    )
    # Проверим батчи с 400 по 500 (как раз около шагов 800-900)
    dataset = Subset(dataset, range(800, min(860, len(dataset))))
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    print("Прогоняем батчи из train.parquet через слои dec_smooth_layers под autocast(float16)...")
    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"]
        weights = batch["weights"]

        with torch.autocast(device_type="cpu", dtype=torch.float16):
            V_cell, kl_loss, rank_loss, rk_eff = model.encode_and_pool(input_ids, weights)

            B, N, K, D = V_cell.shape
            cell_vec = model.cell_in_norm(V_cell.view(B, N, K * D))
            cos_N, sin_N = model.rope(cell_vec, N)
            h_cell = model.dec_context_layer(cell_vec, cos_N, sin_N)
            V_cell_mod = V_cell + h_cell.view(B, N, K, D)

            h_unpacked = model.unpacker(V_cell_mod)

            full_len = N * model.chunk_size
            cos_full, sin_full = model.rope(h_unpacked, full_len)
            x = h_unpacked

            for l_idx, layer in enumerate(model.dec_smooth_layers):
                normed_x = layer.attn_norm(x)
                q = layer.q_proj(normed_x).view(B, full_len, layer.num_heads, layer.head_dim).transpose(1, 2)
                k = layer.k_proj(normed_x).view(B, full_len, layer.num_heads, layer.head_dim).transpose(1, 2)
                v = layer.v_proj(normed_x).view(B, full_len, layer.num_heads, layer.head_dim).transpose(1, 2)
                q, k = layer.q_proj.weight.new_empty(0), layer.k_proj.weight.new_empty(0)  # free
                # SDPA
                attn_out = F.scaled_dot_product_attention(layer.q_proj(normed_x).view(B, full_len, 8, 128).transpose(1, 2),
                                                         layer.k_proj(normed_x).view(B, full_len, 8, 128).transpose(1, 2),
                                                         layer.v_proj(normed_x).view(B, full_len, 8, 128).transpose(1, 2),
                                                         attn_mask=None, dropout_p=0.0, is_causal=False)
                x = x + layer.o_proj(attn_out.transpose(1, 2).contiguous().view(B, full_len, 1024))

                normed_mlp = layer.mlp_norm(x)
                gate = F.silu(layer.gate_proj(normed_mlp))
                up = layer.up_proj(normed_mlp)
                gate_up = gate * up
                mlp_out = layer.down_proj(gate_up)
                x = x + mlp_out

                if not torch.isfinite(x).all():
                    print(f"!!! НАЙДЕН NAN на батче {batch_idx} в dec_smooth_layers[{l_idx}] !!!")
                    print(f"  normed_mlp max: {normed_mlp.abs().max().item()}")
                    print(f"  gate max: {gate.abs().max().item()}")
                    print(f"  up max: {up.abs().max().item()}")
                    print(f"  gate_up max: {gate_up.abs().max().item()}, finite: {torch.isfinite(gate_up).all().item()}")
                    print(f"  mlp_out max: {mlp_out.abs().max().item()}, finite: {torch.isfinite(mlp_out).all().item()}")
                    return
                else:
                    if batch_idx % 5 == 0:
                        print(f"Batch {batch_idx:2d} | L{l_idx}: x max={x.abs().max().item():.1f}, gate*up max={gate_up.abs().max().item():.1f}")

    print("Тест завершен.")

if __name__ == "__main__":
    test_smooth_on_train_data()
