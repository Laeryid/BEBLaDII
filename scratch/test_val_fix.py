import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("experiments/phase 1")
from train_phase1 import GrassmannianPhase1Model, DualRegimeDataset, load_tekken_tokenizer, EnvironmentConfig

def test_fixes_on_val():
    env = EnvironmentConfig.detect()
    tokenizer = load_tekken_tokenizer(env)

    ckpt_path = "experiments/phase 1/checkpoints/phase1_best_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    cleaned_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Загружаем датасет
    dataset = DualRegimeDataset(
        parquet_path=env.val_data_path,
        tokenizer=tokenizer,
        seq_len=512,
        chunk_size=4,
    )
    from torch.utils.data import DataLoader, Subset
    dataset = Subset(dataset, range(min(4, len(dataset))))
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    input_ids = batch["input_ids"]
    targets = batch["targets"]
    weights = batch["weights"]

    print("=== Тестирование вариантов фикса на реальных весах шага 800 ===")

    # Вариант 1: Чистый FP32 без изменений (эталон)
    model = GrassmannianPhase1Model(vocab_size=131072)
    model.load_state_dict(cleaned_sd, strict=True)
    model.eval()

    with torch.no_grad():
        logits_fp32, V_cell, kl_loss, rank_loss, rk_eff = model(input_ids, weights)
        recon_loss_fp32 = F.cross_entropy(logits_fp32.view(-1, 131072), targets.view(-1), ignore_index=-100)
        print(f"1. Исходная модель в FP32 (Эталон): Recon Loss = {recon_loss_fp32.item():.4f}, logits finite = {torch.isfinite(logits_fp32).all().item()}")

    # Вариант 2: Исходная модель в FP16 (как на Kaggle T4 сейчас)
    model_fp16 = GrassmannianPhase1Model(vocab_size=131072).half()
    model_fp16.load_state_dict(cleaned_sd, strict=True)
    model_fp16.eval()

    with torch.no_grad():
        try:
            logits_fp16, _, _, _, _ = model_fp16(input_ids, weights)
            recon_loss_fp16 = F.cross_entropy(logits_fp16.float().view(-1, 131072), targets.view(-1), ignore_index=-100)
            print(f"2. Исходная модель в FP16: Recon Loss = {recon_loss_fp16.item():.4f}, logits finite = {torch.isfinite(logits_fp16).all().item()}")
        except Exception as e:
            print(f"2. Исходная модель в FP16: УПАЛА с ошибкой: {e}")

    # Вариант 3: Фикс матрицы Грама в unpacker (расчет G в float32 + clamp в float16)
    # Посмотрим, даст ли расчет G в float32 полное совпадение с эталоном в FP16!
    class FixedUnpacker(nn.Module):
        def __init__(self, orig_unpacker):
            super().__init__()
            self.u = orig_unpacker

        def forward(self, V_cell):
            B, N, K, D = V_cell.shape
            Q = self.u.q_norm(self.u.query_slots).unsqueeze(0).unsqueeze(0).expand(B, N, self.u.num_slots, D)
            K_mat = self.u.k_norm(V_cell)
            V_mat = V_cell

            scores = torch.matmul(Q, K_mat.transpose(-2, -1)) / (D ** 0.5)
            attn = F.softmax(scores, dim=-1)
            h_attn = torch.matmul(attn, V_mat)

            # Расчет матрицы Грама в float32 для защиты от переполнения 65504
            V_f32 = V_cell.float()
            G = torch.matmul(V_f32, V_f32.transpose(-2, -1))
            G_flat = G.view(B, N, 64)
            # Вход в gram_mlp в float32, а затем каст в рабочий dtype
            h_gram = self.u.gram_mlp.to(torch.float32)(G_flat).to(V_cell.dtype)
            h_gram_exp = h_gram.unsqueeze(2).expand(B, N, self.u.num_slots, 128)

            h_comb = torch.cat([h_attn, h_gram_exp], dim=-1)
            h_out = self.u.out_norm(self.u.out_proj(h_comb))
            return h_out.view(B, N * self.u.num_slots, self.u.hidden_size)

    model_fp16.unpacker = FixedUnpacker(model_fp16.unpacker)
    with torch.no_grad():
        logits_fixed, _, _, _, _ = model_fp16(input_ids, weights)
        recon_loss_fixed = F.cross_entropy(logits_fixed.float().view(-1, 131072), targets.view(-1), ignore_index=-100)
        print(f"3. Фикс G в float32 в FP16: Recon Loss = {recon_loss_fixed.item():.4f}, logits finite = {torch.isfinite(logits_fixed).all().item()}")

if __name__ == "__main__":
    test_fixes_on_val()
