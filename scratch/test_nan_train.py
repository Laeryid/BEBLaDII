import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.append("experiments/phase 1")
from train_phase1 import GrassmannianPhase1Model, DualRegimeDataset, load_tekken_tokenizer, EnvironmentConfig

def test_autocast_train():
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
    dataset = Subset(dataset, range(min(100, len(dataset))))
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    print("Прогоняем 50 батчей из train.parquet в режиме model.train()...")
    nan_count = 0
    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"]
        weights = batch["weights"]
        targets = batch["targets"]

        V_cell, kl_loss, rank_loss, rk_eff = model.encode_and_pool(input_ids, weights)
        
        # Пошаговая диагностика внутри decode
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
            if not torch.isfinite(x).all():
                print(f"[STAGE NAN] batch {batch_idx}: dec_smooth_layers[{l_idx}] contains NaN! max={x.abs().max().item()}")
        x = model.dec_final_norm(x)
        if not torch.isfinite(x).all():
            print(f"[STAGE NAN] batch {batch_idx}: dec_final_norm contains NaN!")

        logits = model.get_tied_head_logits(x)
        if not torch.isfinite(logits).all():
            print(f"[STAGE NAN] batch {batch_idx}: logits contains NaN!")
            nan_count += 1
        
        # Проверяем также max значения активаций!
        if batch_idx < 5 or not torch.isfinite(logits).all():
            print(f"Batch {batch_idx}: V_cell_mod max={V_cell_mod.abs().max().item():.2f}, h_unpacked max={h_unpacked.abs().max().item():.2f}, x max={x.abs().max().item():.2f}, logits max={logits.abs().max().item():.2f}")

    print(f"Тест завершен. Всего батчей с NaN: {nan_count}")

if __name__ == "__main__":
    test_autocast_train()
