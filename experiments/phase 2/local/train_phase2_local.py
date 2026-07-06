import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from tqdm.auto import tqdm
import argparse

# Добавляем src в путь
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(os.path.join(project_root, 'src'))

from transformers import AutoModelForCausalLM, AutoTokenizer
from beb_la_dii.model.modern_decoder import ModernLatentDecoder
from beb_la_dii.model.vae import LatentEncoder 

class Phase2DecoderWrapper(nn.Module):
    def __init__(self, encoder, decoder, qwen_embed_weight, qwen_lm_head_weight):
        super().__init__()
        self.encoder = encoder
        
        # Строгая заморозка энкодера (он обучен в Фазе 1)
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.decoder = decoder
        # Матрицы Qwen передаем как тензоры
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
        dus_weights_path=args.dus_weights
    ).to(torch.bfloat16)
    
    if os.path.exists(args.phase2_resume_ckpt):
        print(f"Resuming Phase 2 Decoder from {args.phase2_resume_ckpt}")
        ckpt = torch.load(args.phase2_resume_ckpt, map_location="cpu")
        decoder.load_state_dict(ckpt["decoder"] if "decoder" in ckpt else ckpt)
    
    # 4. Сборка модели
    model = Phase2DecoderWrapper(encoder, decoder, embed_weight, lm_head_weight)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    
    model.train()
    print("Model initialized. Ready for training loop.")
    
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    dataloader = get_dataloader(args.data_path, batch_size=args.batch_size, max_length=args.max_length, tokenizer=tokenizer)
    
    step = 0
    pbar = tqdm(total=args.max_steps, desc="Phase 2 Local Training")
    
    # Автоматическое приведение типов для CPU/GPU
    autocast_device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
    
        optimizer.zero_grad()
        
        with torch.autocast(device_type=autocast_device_type, dtype=torch.bfloat16):
            logits = model(input_ids, attention_mask)
            ce_loss_raw = loss_fct(logits.view(-1, logits.size(-1)), input_ids.view(-1))
            ce_loss_raw = ce_loss_raw.view(input_ids.size())
            
            attention_mask_bf = attention_mask.to(torch.bfloat16)
            ce_loss = (ce_loss_raw * attention_mask_bf).sum() / (attention_mask_bf.sum() + 1e-6)
    
        ce_loss.backward()
        optimizer.step()
        
        step += 1
        pbar.set_postfix({"Loss": f"{ce_loss.item():.4f}"})
        pbar.update(1)
        
        if step % args.save_every == 0:
            save_path = os.path.join(args.ckpt_dir, f"decoder_local_step_{step}.pth")
            torch.save({"decoder": model.decoder.state_dict()}, save_path)
            
        if step >= args.max_steps:
            break
            
    pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen_path", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--phase1_ckpt", type=str, default=r"..\..\..\experiments\phase 1\planB_phase1_checkpoints_phase1_vae_step_20000.pth")
    parser.add_argument("--dus_weights", type=str, default=r"..\..\..\storage\components\model\latentBERT\v1.0\weights.pt")
    parser.add_argument("--phase2_resume_ckpt", type=str, default="")
    # Используем путь, где могут лежать файлы, или попросите пользователя уточнить
    parser.add_argument("--data_path", type=str, default=r"..\..\..\experiments\phase 3\train_data\data")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=10)
    args = parser.parse_args()
    
    train_local(args)
