import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.append("experiments/phase 1")
from train_phase1 import GrassmannianPhase1Model, DualRegimeDataset, load_tekken_tokenizer, EnvironmentConfig

def test_autocast_bfloat16():
    env = EnvironmentConfig.detect()
    tokenizer = load_tekken_tokenizer(env)

    ckpt_path = "experiments/phase 1/checkpoints/phase1_best_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    cleaned_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model = GrassmannianPhase1Model(vocab_size=131072)
    model.load_state_dict(cleaned_sd, strict=True)
    model.train()  # ВАЖНО: model.train(), как в обучении!

    dataset = DualRegimeDataset(
        parquet_path=env.val_data_path,
        tokenizer=tokenizer,
        seq_len=256,
        chunk_size=4,
    )
    dataset = Subset(dataset, range(min(50, len(dataset))))
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    print("Прогоняем 20 батчей под autocast(bfloat16) в режиме model.train()...")
    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"]
        weights = batch["weights"]
        targets = batch["targets"]

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits, V_cell, kl_loss, rank_loss, rk_eff = model(input_ids, weights)
            if not torch.isfinite(logits).all():
                print(f"[NAN DETECTED] batch {batch_idx}: logits contains NaN/Inf! min={logits.min().item()}, max={logits.max().item()}")
                # Теперь по шагам найдем, где именно
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
                    x = layer(x, cos_full, sin_full)
                    print(f"dec_smooth_layers[{l_idx}] finite:", torch.isfinite(x).all().item(), "max:", x.abs().max().item())
                x = model.dec_final_norm(x)
                print("dec_final_norm finite:", torch.isfinite(x).all().item(), "max:", x.abs().max().item())
                break
            else:
                pass
    print("Тест завершен.")

if __name__ == "__main__":
    test_autocast_bfloat16()
