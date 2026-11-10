import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from tqdm.auto import tqdm
import argparse
import glob
import re
import csv
import shutil

# Добавляем src в путь (более надежный способ)
current_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../'))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

# Для отладки импортов:
# print(f"Debug: src_path = {src_path}")

from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from beb_la_dii.model.modern_decoder import ModernLatentDecoder
from beb_la_dii.model.vae import LatentEncoder 

class Phase2DecoderWrapper(nn.Module):
    def __init__(self, encoder, decoder, qwen_embed_weight, qwen_lm_head_weight):
        super().__init__()
        self.encoder = encoder
        
        # Строгая заморозка энкодера
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.decoder = decoder
        self.qwen_embed_weight = nn.Parameter(qwen_embed_weight, requires_grad=False)
        self.qwen_lm_head_weight = nn.Parameter(qwen_lm_head_weight, requires_grad=False)
        
    def forward(self, input_ids, attention_mask=None):
        inputs_embeds = F.embedding(input_ids, self.qwen_embed_weight)
        with torch.no_grad():
            z, _, _ = self.encoder(inputs_embeds)
        projected_embeds = self.decoder(z, attention_mask)
        logits = F.linear(projected_embeds, self.qwen_lm_head_weight)
        return logits

def get_dataloader(data_path, batch_size=2, max_length=128, tokenizer=None):
    from indexed_parquet_dataset import IndexedParquetDataset
    from torch.utils.data import DataLoader

    print(f"Loading parquet dataset from {data_path} using IndexedParquetDataset...")
    ds = IndexedParquetDataset.from_folder(data_path, pattern="*.parquet", auto_fill=True)
    ds = ds.shuffle(seed=42, rg_buffer=8)

    def collate_fn(batch):
        if tokenizer is not None and "text" in batch[0] and batch[0]["text"] is not None:
            texts = [item["text"] for item in batch]
            encoded = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
            return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}

        input_ids, attention_mask = [], []
        for item in batch:
            seq = item.get("input_ids", [])
            if not isinstance(seq, list): seq = list(seq)
            seq = seq[:max_length]
            pad_len = max_length - len(seq)
            padded_seq = seq + [tokenizer.pad_token_id if tokenizer and tokenizer.pad_token_id is not None else 151643] * pad_len
            mask = [1] * len(seq) + [0] * pad_len
            input_ids.append(padded_seq)
            attention_mask.append(mask)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True)

def cleanup_old_checkpoints(ckpt_dir, keep_last=4):
    ckpts = glob.glob(os.path.join(ckpt_dir, "decoder_local_step_*.pth"))
    ckpt_steps = []
    for c in ckpts:
        match = re.search(r'step_(\d+)\.pth', c)
        if match:
            ckpt_steps.append((int(match.group(1)), c))
    
    # Сортируем по возрастанию номера шага
    ckpt_steps.sort(key=lambda x: x[0])
    
    # Оставляем только keep_last файлов
    if len(ckpt_steps) > keep_last:
        for step, c in ckpt_steps[:-keep_last]:
            os.remove(c)
            # Удаляем соответствующие логи и графики
            plot_file = os.path.join(ckpt_dir, f"loss_plot_step_{step}.png")
            if os.path.exists(plot_file):
                os.remove(plot_file)
            log_file = os.path.join(ckpt_dir, f"loss_log_step_{step}.csv")
            if os.path.exists(log_file):
                os.remove(log_file)

def train_local(args):
    # Принудительно отключаем XLA для локального запуска
    os.environ["PJRT_DEVICE"] = ""
    os.environ["XRT_TPU_CONFIG"] = ""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Phase 2 Decoder LOCAL on device: {device}")
    
    # 1. Загрузка Qwen и извлечение матриц
    print(f"Loading weights from {args.qwen_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    qwen = AutoModelForCausalLM.from_pretrained(args.qwen_path, torch_dtype=torch.bfloat16)
    embed_weight = qwen.model.embed_tokens.weight.detach().clone().to(torch.bfloat16)
    lm_head_weight = qwen.lm_head.weight.detach().clone().to(torch.bfloat16)
    del qwen
    gc.collect()
    print("Qwen deleted from memory. Kept only Embeddings and LM Head.")
    
    # 2. Загрузка замороженного Encoder из Фазы 1
    encoder = LatentEncoder().to(torch.bfloat16)
    if os.path.exists(args.phase1_ckpt):
        print(f"Loading Phase 1 Encoder from {args.phase1_ckpt}")
        ckpt = torch.load(args.phase1_ckpt, map_location="cpu")
        encoder.load_state_dict(ckpt["encoder"] if "encoder" in ckpt else ckpt)
    else:
        print(f"Warning: {args.phase1_ckpt} not found. Using random LatentEncoder.")
    
    # 3. Инициализация Декодера
    decoder = ModernLatentDecoder(
        latent_dim=1024,
        qwen_dim=1536,
        num_layers=3,
    ).to(torch.bfloat16)
    
    start_step = 0
    loss_history = []
    
    if os.path.exists(args.phase2_resume_ckpt):
        print(f"Resuming Phase 2 Decoder from {args.phase2_resume_ckpt}")
        ckpt = torch.load(args.phase2_resume_ckpt, map_location="cpu")
        decoder.load_state_dict(ckpt["decoder"] if "decoder" in ckpt else ckpt)
        if "step" in ckpt:
            start_step = ckpt["step"]
            print(f"Resumed from step {start_step}")
        if "loss_history" in ckpt:
            loss_history = ckpt["loss_history"]
    
    # 4. Сборка модели
    model = Phase2DecoderWrapper(encoder, decoder, embed_weight, lm_head_weight)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=args.max_steps)
    if start_step > 0:
        for _ in range(start_step):
            scheduler.step()

    loss_fct = nn.CrossEntropyLoss(reduction="none")
    
    model.train()
    print("Model initialized. Ready for training loop.")
    
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    # Создаем/Обновляем актуальный лог
    current_log_file = os.path.join(args.ckpt_dir, "loss_log_current.csv")
    mode = "w" if start_step == 0 else "w" # Всегда перезаписываем при возобновлении, так как история хранится в .pth
    with open(current_log_file, mode, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss"])
        for s, l in loss_history:
            writer.writerow([s, l])
    
    dataloader = get_dataloader(args.data_path, batch_size=args.batch_size, max_length=args.max_length, tokenizer=tokenizer)
    
    data_iter = iter(dataloader)
    if start_step > 0:
        print(f"Fast-forwarding dataloader by {start_step} steps...")
        for _ in tqdm(range(start_step), desc="Skipping batches"):
            try:
                next(data_iter)
            except StopIteration:
                break
                
    step = start_step
    pbar = tqdm(total=args.max_steps, desc="Phase 2 Local Training", initial=start_step)
    
    # Автоматическое приведение типов для CPU/GPU
    autocast_device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    optimizer.zero_grad()
    accum_loss = 0.0
    micro_step = 0
    
    for batch in data_iter:
        if step >= args.max_steps:
            break
            
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
    
        with torch.autocast(device_type=autocast_device_type, dtype=torch.bfloat16):
            logits = model(input_ids, attention_mask)
            ce_loss_raw = loss_fct(logits.view(-1, logits.size(-1)), input_ids.view(-1))
            ce_loss_raw = ce_loss_raw.view(input_ids.size())
            
            attention_mask_bf = attention_mask.to(torch.bfloat16)
            ce_loss = (ce_loss_raw * attention_mask_bf).sum() / (attention_mask_bf.sum() + 1e-6)
            
            # Масштабируем лосс для backward
            loss = ce_loss / args.accum_steps
    
        loss.backward()
        accum_loss += ce_loss.item() / args.accum_steps
        micro_step += 1
        
        if micro_step % args.accum_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            step += 1
            loss_val = accum_loss
            accum_loss = 0.0
            
            loss_history.append((step, loss_val))
            
            # Дописываем текущий шаг в актуальный лог
            with open(current_log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([step, loss_val])
                
            pbar.set_postfix({"Loss": f"{loss_val:.4f}"})
            pbar.update(1)
            
            if step % args.save_every == 0:
                # Сохраняем веса, шаг и историю
                save_path = os.path.join(args.ckpt_dir, f"decoder_local_step_{step}.pth")
                torch.save({
                    "decoder": model.decoder.state_dict(),
                    "step": step,
                    "loss_history": loss_history
                }, save_path)
            
                # Сохраняем "снапшот" лога
                shutil.copy2(current_log_file, os.path.join(args.ckpt_dir, f"loss_log_step_{step}.csv"))
                
                # Строим и сохраняем график
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 6))
                    steps, losses = zip(*loss_history)
                    
                    # Сглаживание лосса
                    smoothed_losses = []
                    alpha = 0.95 # Сильное сглаживание для красоты
                    for i, l in enumerate(losses):
                        if i == 0: smoothed_losses.append(l)
                        else: smoothed_losses.append(alpha * smoothed_losses[-1] + (1 - alpha) * l)
                    
                    plt.plot(steps, losses, alpha=0.3, color='blue', label="Raw Loss")
                    plt.plot(steps, smoothed_losses, color='blue', linewidth=2, label="Smoothed Loss")
                    plt.xlabel("Steps")
                    plt.ylabel("Cross Entropy Loss")
                    plt.title(f"Phase 2 Local Training Loss (Step {step})")
                    plt.legend()
                    plt.grid(True)
                    
                    plot_path = os.path.join(args.ckpt_dir, f"loss_plot_step_{step}.png")
                    plt.savefig(plot_path)
                    plt.close()
                except ImportError:
                    print("\n[Warning] matplotlib is not installed. Plot was not generated.")
                
                # Очищаем старые версии
                cleanup_old_checkpoints(args.ckpt_dir, keep_last=4)
            
    pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen_path", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--phase1_ckpt", type=str, default=r"..\..\..\experiments\phase 1\planB_phase1_checkpoints_phase1_vae_step_20000.pth")
    parser.add_argument("--dus_weights", type=str, default=r"..\..\..\storage\components\model\latentBERT\v1.0\weights.pt")
    parser.add_argument("--phase2_resume_ckpt", type=str, default="")
    parser.add_argument("--data_path", type=str, default=r"..\..\..\experiments\phase 3\train_data\data")
    default_ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    parser.add_argument("--ckpt_dir", type=str, default=default_ckpt_dir)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accum_steps", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--save_every", type=int, default=500)
    args = parser.parse_args()
    
    train_local(args)
