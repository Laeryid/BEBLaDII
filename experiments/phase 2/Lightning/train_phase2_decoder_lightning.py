import os
import subprocess
import sys
import gc
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

# Ensure the root of the project is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.beb_la_dii.model.modern_decoder import ModernLatentDecoder
    from src.beb_la_dii.model.vae import LatentEncoder
except ImportError as e:
    print(f"Error: Failed to import project modules. Make sure you run this from the project root. Error: {e}")
    sys.exit(1)


# ==========================================
# 1. Configuration
# ==========================================
class Config:
    # Model and Data Paths
    qwen_model_path = "Qwen/Qwen2.5-1.5B" # Uses HuggingFace Hub directly
    dataset_path = "./data/train_data/data"
    
    local_encoder_weights = "./weights/planB_phase1_checkpoints_phase1_vae_step_20000.pth"
    local_dus_weights = "./weights/AWAKENED_WEIGHTS_FINAL.pt"
    
    # Checkpointing and GCS
    resume_from_checkpoint = True
    gcs_checkpoint_dir = "gs://bebladii-weigths-us/planB/phase2/checkpoints/"
    output_dir = "./checkpoints/phase2"
    
    # Architecture and Training
    batch_size = 4
    grad_accum_steps = 4
    max_length = 1024
    learning_rate = 5e-5
    epochs = 4
    max_steps = 20000
    log_steps = 10
    save_steps = 1000
    
    wandb_project = "BEBLaDII-Phase2-Lightning"


args = Config()


# ==========================================
# 2. Model Definition
# ==========================================
class Phase2DecoderWrapper(nn.Module):
    def __init__(self, encoder, decoder, qwen_embed_weight, qwen_lm_head_weight):
        super().__init__()
        self.encoder = encoder
        
        # Freeze encoder strictly
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.decoder = decoder
        self.qwen_embed_weight = nn.Parameter(qwen_embed_weight, requires_grad=False)
        self.qwen_lm_head_weight = nn.Parameter(qwen_lm_head_weight, requires_grad=False)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "encoder"):
            self.encoder.eval()

    def forward(self, input_ids, attention_mask=None):
        inputs_embeds = F.embedding(input_ids, self.qwen_embed_weight)
        
        with torch.no_grad():
            z, _, _ = self.encoder(inputs_embeds)
            
        projected_embeds = self.decoder(z, attention_mask)
        logits = F.linear(projected_embeds, self.qwen_lm_head_weight)
        
        return logits


# ==========================================
# 3. Utilities
# ==========================================
def get_dataloader(data_path, batch_size=4, max_length=1024, tokenizer=None):
    from indexed_parquet_dataset import IndexedParquetDataset
    
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
            seq = list(item.get("input_ids", []))[:max_length]
            pad_len = max_length - len(seq)
            padded_seq = seq + [151643] * pad_len # Qwen pad token
            mask = [1] * len(seq) + [0] * pad_len
            input_ids.append(padded_seq)
            attention_mask.append(mask)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    return DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, 
        drop_last=True, num_workers=4, prefetch_factor=2
    )

def get_latest_gcs_checkpoint(gcs_dir):
    try:
        result = subprocess.run(["gsutil", "ls", gcs_dir], capture_output=True, text=True, check=True)
        files = result.stdout.splitlines()
        ckpt_files = [f for f in files if "decoder_step_" in f and f.endswith(".pth")]
        if not ckpt_files:
            return None
        
        def extract_step(filename):
            try:
                return int(filename.split("_step_")[-1].split(".pth")[0])
            except ValueError:
                return -1
                
        ckpt_files.sort(key=extract_step)
        return ckpt_files[-1]
    except Exception as e:
        print(f"Failed to list GCS checkpoints: {e}")
        return None


# ==========================================
# 4. Training Loop
# ==========================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] Using device: {device}")
    num_gpus = torch.cuda.device_count()
    print(f"[Init] Available GPUs: {num_gpus}")

    # Auth logic for GCP and WandB
    if os.path.exists("gcp_sa.json"):
        try:
            subprocess.run(["gcloud", "auth", "activate-service-account", "--key-file", "gcp_sa.json"], check=True)
            print("[Init] GCP Authentication successful via gcp_sa.json.")
        except Exception as e_gcp:
            print(f"[Init] WARN: Could not authenticate GCP: {e_gcp}")
    
    if args.wandb_project:
        wandb_api = os.environ.get("WANDB_API_KEY")
        if wandb_api:
            wandb.login(key=wandb_api)
        wandb.init(project=args.wandb_project, config=vars(args), resume="allow")

    # 1. Load Qwen Matrices
    print(f"Loading weights from {args.qwen_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 151643

    qwen = AutoModelForCausalLM.from_pretrained(args.qwen_model_path, torch_dtype=torch.bfloat16)
    embed_weight = qwen.model.embed_tokens.weight.detach().clone().to(torch.bfloat16)
    lm_head_weight = qwen.lm_head.weight.detach().clone().to(torch.bfloat16)

    del qwen
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Qwen deleted from memory. Kept only Embeddings and LM Head.")

    # 2. Load Phase 1 Encoder
    encoder = LatentEncoder().to(torch.bfloat16)
    if os.path.exists(args.local_encoder_weights):
        print(f"Loading Phase 1 Encoder from {args.local_encoder_weights}")
        ckpt = torch.load(args.local_encoder_weights, map_location="cpu")
        encoder.load_state_dict(ckpt.get("encoder", ckpt))
    else:
        print(f"Warning: {args.local_encoder_weights} not found. Using random LatentEncoder.")

    # 3. Init Decoder
    decoder = ModernLatentDecoder(
        latent_dim=1024, qwen_dim=1536, num_layers=3,
        dus_weights_path=args.local_dus_weights if os.path.exists(args.local_dus_weights) else None,
    ).to(torch.bfloat16)

    # 4. Assemble Wrapper
    model = Phase2DecoderWrapper(encoder, decoder, embed_weight, lm_head_weight).to(device)
    
    if num_gpus > 1:
        print(f"[Init] Wrapping model in DataParallel across {num_gpus} GPUs")
        model = nn.DataParallel(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 5. Dataloader Setup
    try:
        dataloader = get_dataloader(args.dataset_path, batch_size=args.batch_size, max_length=args.max_length, tokenizer=tokenizer)
    except Exception as e:
        print(f"Dataset warning: {e}. Dataloader is empty.")
        dataloader = []
        
    total_steps = (min(args.max_steps, ((len(dataloader) + args.grad_accum_steps - 1) // args.grad_accum_steps) * args.epochs) 
                   if len(dataloader) > 0 else args.max_steps)
    
    warmup_steps = int(total_steps * 0.05)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # 6. Checkpoint Auto-Resume
    start_step = 0
    metrics_history = []
    
    if getattr(args, "resume_from_checkpoint", False):
        latest_gs_ckpt = get_latest_gcs_checkpoint(args.gcs_checkpoint_dir)
        if latest_gs_ckpt:
            print(f"Found latest checkpoint on GCS: {latest_gs_ckpt}")
            local_ckpt_path = os.path.join(args.output_dir, "resume_checkpoint.pth")
            try:
                subprocess.run(["gsutil", "-q", "cp", latest_gs_ckpt, local_ckpt_path], check=True)
                if os.path.exists(local_ckpt_path):
                    print(f"Loading checkpoint {local_ckpt_path}...")
                    ckpt = torch.load(local_ckpt_path, map_location="cpu")
                    
                    actual_model = model.module if isinstance(model, nn.DataParallel) else model
                    actual_model.decoder.load_state_dict(ckpt.get("decoder", ckpt))
                    
                    if "optimizer" in ckpt:
                        optimizer.load_state_dict(ckpt["optimizer"])
                    if "scheduler" in ckpt:
                        scheduler.load_state_dict(ckpt["scheduler"])
                    if "step" in ckpt:
                        start_step = ckpt["step"]
                    if "metrics_history" in ckpt:
                        metrics_history = ckpt["metrics_history"]
                        print(f"Loaded {len(metrics_history)} historical metrics data points.")
                        
                    print(f"Successfully resumed from step {start_step}!")
            except Exception as e:
                print(f"Warning: Failed to download or load checkpoint from GS! Error: {e}")

    model.train()
    from tqdm.auto import tqdm
    pbar = tqdm(total=total_steps, initial=start_step, desc="Phase 2 Decoder")
    step = start_step

    optimizer.zero_grad()
    
    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(dataloader):
            if step >= total_steps:
                break
                
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            logits = model(input_ids, attention_mask)
            
            ce_loss_raw = loss_fct(logits.view(-1, logits.size(-1)), input_ids.view(-1))
            ce_loss_raw = ce_loss_raw.view(input_ids.size())
            
            attention_mask_bf = attention_mask.to(torch.bfloat16)
            ce_loss = (ce_loss_raw * attention_mask_bf).sum() / (attention_mask_bf.sum() + 1e-6)
            
            if ce_loss.dim() > 0:
                ce_loss = ce_loss.mean()
                
            ce_loss = ce_loss / args.grad_accum_steps
            ce_loss.backward()

            if (batch_idx + 1) % args.grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                
                # Logging
                if step % args.log_steps == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    current_loss = ce_loss.item() * args.grad_accum_steps
                    
                    metrics = {
                        "train/ce_loss": current_loss, 
                        "lr": current_lr,
                        "step": step
                    }
                    metrics_history.append(metrics)
                    
                    if args.wandb_project:
                        wandb.log(metrics, step=step)
                        
                    pbar.set_postfix({"Loss": f"{current_loss:.4f}", "LR": f"{current_lr:.2e}"})
                
                # Saving
                if step > start_step and step % args.save_steps == 0:
                    ckpt_path = os.path.join(args.output_dir, f"decoder_step_{step}.pth")
                    actual_model = model.module if isinstance(model, nn.DataParallel) else model
                    
                    torch.save({
                        "decoder": actual_model.decoder.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step": step,
                        "metrics_history": metrics_history
                    }, ckpt_path)
                    
                    print(f"\n[SAVE] Saved checkpoint to {ckpt_path}")
                    try:
                        subprocess.Popen(["gsutil", "-q", "cp", ckpt_path, args.gcs_checkpoint_dir])
                    except Exception:
                        pass
                
                pbar.update(1)
                
        if step >= total_steps:
            break

    pbar.close()
    
    # Final Save
    final_path = os.path.join(args.output_dir, "decoder_final.pth")
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    torch.save({
        "decoder": actual_model.decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "metrics_history": metrics_history
    }, final_path)
    
    print(f"[SAVE] Final weights saved → {final_path}")
    try:
        subprocess.run(["gsutil", "-q", "cp", final_path, args.gcs_checkpoint_dir])
    except Exception:
        pass
        
    if args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    train()
