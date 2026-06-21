import os

# Настройка переменных окружения для TPU v4
def setup_env():
    os.environ["PJRT_DEVICE"] = "TPU"
    os.environ["XLA_USE_BF16"] = "1"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

setup_env()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel
import numpy as np
import sys
import gc
import subprocess

# TPU single-process initialization
xr.use_spmd()

def setup_spmd_mesh():
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (num_devices, 1)
    device_ids = np.array(range(num_devices))
    mesh = xs.Mesh(device_ids, mesh_shape, ("fsdp", "model"))
    xs.set_global_mesh(mesh)
    return mesh

# Добавляем src в путь
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from transformers import AutoModelForCausalLM, AutoTokenizer
from beb_la_dii.model.modern_decoder import ModernLatentDecoder
from beb_la_dii.model.vae import LatentEncoder 
import wandb

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
        # 1. Извлекаем эмбеддинги вручную из сохраненной матрицы Qwen
        inputs_embeds = F.embedding(input_ids, self.qwen_embed_weight)
        
        # 2. VAE Encoder -> Latent Space Z (заморожен)
        with torch.no_grad():
            z, _, _ = self.encoder(inputs_embeds)
            
        # 3. Наш новый ModernLatentDecoder
        projected_embeds = self.decoder(z, attention_mask)
        
        # 4. Проекция обратно в токены через сохраненную матрицу Qwen LM Head
        logits = F.linear(projected_embeds, self.qwen_lm_head_weight)
        
        return logits

def get_dataloader(data_path, batch_size=8, max_length=1024, tokenizer=None):
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
            padded_seq = seq + [151643] * pad_len
            mask = [1] * len(seq) + [0] * pad_len
            input_ids.append(padded_seq)
            attention_mask.append(mask)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True, num_workers=4, prefetch_factor=2)

def train_phase2():
    mesh = setup_spmd_mesh()
    device = xm.xla_device()
    is_master = xm.is_master_ordinal()
    
    if is_master:
        print(f"Running Phase 2 Decoder on device: {device} with SPMD FSDP (Mesh: {mesh.shape()})")
        wandb.init(project="BEBLaDII", name="Phase2_Bidirectional_Decoder", config={
            "learning_rate": 1e-4,
            "architecture": "ModernBERT-3L",
            "loss": "CrossEntropy"
        })
    
    # 1. Загрузка Qwen и извлечение матриц
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    print(f"Loading weights from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Загружаем модель только для того, чтобы вытащить 2 матрицы
    qwen = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    embed_weight = qwen.model.embed_tokens.weight.detach().clone().to(torch.bfloat16)
    lm_head_weight = qwen.lm_head.weight.detach().clone().to(torch.bfloat16)
    
    # СУПЕР-ОПТИМИЗАЦИЯ: Удаляем тушу Qwen из памяти!
    del qwen
    gc.collect()
    print("Qwen deleted from memory. Kept only Embeddings and LM Head. Massive VRAM saving!")
    
    # 2. Загрузка замороженного Encoder из Фазы 1
    phase1_ckpt_path = "checkpoints/phase1/phase1_vae_step_20000.pth"
    encoder = LatentEncoder().to(torch.bfloat16)
    if os.path.exists(phase1_ckpt_path):
        if is_master: print(f"Loading Phase 1 Encoder from {phase1_ckpt_path}")
        ckpt = torch.load(phase1_ckpt_path, map_location="cpu")
        encoder.load_state_dict(ckpt["encoder"])
    else:
        if is_master: print(f"Warning: {phase1_ckpt_path} not found. Using random LatentEncoder.")
    
    # 3. Инициализация нашего нового Декодера (с подгрузкой DUS)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    dus_weights_dir = os.path.join(project_root, "storage", "components", "model", "latentBERT", "v1.0")
    dus_weights = os.path.join(dus_weights_dir, "weights.pt")
    
    # Скачиваем веса DUS из GCS, если их нет
    if is_master and not os.path.exists(dus_weights):
        print(f"Downloading DUS weights from GCS to {dus_weights}...")
        os.makedirs(dus_weights_dir, exist_ok=True)
        try:
            subprocess.run(["gcloud", "storage", "cp", "gs://bebladii-weigths-us/components/model/latentBERT/v1.0/weights.pt", dus_weights], check=True)
            print("DUS weights downloaded successfully.")
        except Exception as e:
            print(f"Error downloading DUS weights: {e}")
            
    xm.rendezvous("weights_downloaded") # Ждем пока мастер скачает

    decoder = ModernLatentDecoder(
        latent_dim=1024, 
        qwen_dim=1536, 
        num_layers=3, 
        dus_weights_path=dus_weights
    ).to(torch.bfloat16)
    
    # 4. Сборка финальной модели и FSDP обертка
    model = Phase2DecoderWrapper(encoder, decoder, embed_weight, lm_head_weight)
    model = model.to(device)
    
    # Оборачиваем модель в FSDP
    model = SpmdFullyShardedDataParallel(model, mesh=mesh)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    
    model.train()
    if is_master: print("Model initialized. Ready for training loop.")
    
    # Создаем директорию для чекпойнтов Фазы 2
    ckpt_dir = "checkpoints/phase2"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # --- ПОДГОТОВКА ДАННЫХ И ЦИКЛ ОБУЧЕНИЯ ---
    data_path = "./data/train"  # Или твой gs:// путь, если данные не скачаны
    if is_master and not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
        try:
            print("Downloading dataset from GCS...")
            subprocess.run(["gcloud", "storage", "rsync", "-r", "gs://bebladii-data-us/train/", data_path], check=True)
        except Exception as e:
            print(f"Dataset download skipped or failed: {e}")
            
    xm.rendezvous("data_downloaded")
    
    dataloader = get_dataloader(data_path, batch_size=16, max_length=1024, tokenizer=tokenizer)
    
    from tqdm.auto import tqdm
    
    if is_master: print("Starting training loop...")
    
    step = 0
    pbar = tqdm(total=10000, desc="Phase 2 Decoder") if is_master else None
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
    
        # SPMD Sharding батча (КРИТИЧНО ДЛЯ FSDP)
        xs.mark_sharding(input_ids, mesh, ('fsdp', None))
        xs.mark_sharding(attention_mask, mesh, ('fsdp', None))
    
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
    
        # Вычисляем Loss
        ce_loss_raw = loss_fct(logits.view(-1, logits.size(-1)), input_ids.view(-1))
        ce_loss_raw = ce_loss_raw.view(input_ids.size())
        
        # Переводим маску во float, чтобы XLA не мучился с приведением типов
        attention_mask_bf = attention_mask.to(torch.bfloat16)
        ce_loss = (ce_loss_raw * attention_mask_bf).sum() / (attention_mask_bf.sum() + 1e-6)
    
        ce_loss.backward()
        
        # Для SPMD используем обычный step() и ручной mark_step(), 
        # так как xm.optimizer_step() включает DDP-логику (all-reduce), ломающую SPMD-граф.
        optimizer.step()
        xm.mark_step()
        
        step += 1
    
        if is_master:
            wandb.log({"train/ce_loss": ce_loss.item(), "step": step})
            pbar.set_postfix({"Loss": f"{ce_loss.item():.4f}"})
            pbar.update(1)
            
            import torch_xla.debug.metrics as met
            if step <= 10:
                print(f"\n[Step {step}] XLA Compilations: {met.metric_data('CompileTime')[0] if 'CompileTime' in met.metrics_report() else '0'} (Look for increasing Compile counts)")
    
            if step % 500 == 0:
                torch.save({"decoder": model.module.decoder.state_dict() if hasattr(model, 'module') else model.decoder.state_dict()}, os.path.join(ckpt_dir, f"decoder_step_{step}.pth"))
                
        if step >= 10000: # Лимит для тестов
            break
            
    if is_master: 
        pbar.close()
        wandb.finish()

if __name__ == "__main__":
    # Если запуск через xla_spawn или torchrun:
    train_phase2()
    
    # Если запуск как обычный питоновский скрипт, раскомментируй:
    # xmp.spawn(train_phase2, args=())
