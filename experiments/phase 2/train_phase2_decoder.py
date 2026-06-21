import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.spmd as xs
import sys
import gc

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
        
        # Замораживаем энкодер Фазы 1 (он идеален)
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.decoder = decoder
        
        # Регистрируем входные эмбеддинги как константу
        self.register_buffer("embed_weight", qwen_embed_weight)
        
        # LM Head делаем слоем для SPMD-шардирования, но замораживаем
        self.lm_head = nn.Linear(qwen_lm_head_weight.size(1), qwen_lm_head_weight.size(0), bias=False)
        self.lm_head.weight.data.copy_(qwen_lm_head_weight)
        self.lm_head.weight.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            # 1. Получаем входные эмбеддинги (Frozen)
            inputs_embeds = F.embedding(input_ids, self.embed_weight)
            # 2. Получаем латенты из замороженного VAE (Frozen)
            # Предполагается интерфейс: z, mu, logvar = encoder(embeds)
            z, _, _ = self.encoder(inputs_embeds)
            
        # 3. Умный Декодер (здесь идут градиенты)
        decoded = self.decoder(z, attention_mask=attention_mask)
        
        # 4. Проекция в логиты (Frozen)
        logits = self.lm_head(decoded)
        return logits

def train_phase2():
    device = xm.xla_device()
    is_master = xm.is_master_ordinal()
    
    if is_master:
        print(f"Running Phase 2 Decoder on device: {device}")
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
    dus_weights = r"C:\Experiments\BEBLaDII\storage\components\model\latentBERT\v1.0\weights.pt"
    decoder = ModernLatentDecoder(
        latent_dim=1024, 
        qwen_dim=1536, 
        num_layers=3, 
        dus_weights_path=dus_weights
    ).to(torch.bfloat16)
    
    # 4. Сборка финальной модели
    model = Phase2DecoderWrapper(encoder, decoder, embed_weight, lm_head_weight)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=1e-4)
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    
    model.train()
    if is_master: print("Model initialized. Ready for training loop.")
    
    # Создаем директорию для чекпойнтов Фазы 2
    ckpt_dir = "checkpoints/phase2"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # --- ДЕМОНСТРАЦИОННЫЙ ЦИКЛ ОБУЧЕНИЯ ---
    # inputs = tokenizer("Quantum computing is solving problems.", return_tensors="pt").to(device)
    # input_ids = inputs.input_ids
    # attention_mask = inputs.attention_mask
    # 
    # optimizer.zero_grad()
    # logits = model(input_ids, attention_mask)
    # 
    # ce_loss_raw = loss_fct(logits.view(-1, logits.size(-1)), input_ids.view(-1))
    # ce_loss_raw = ce_loss_raw.view(input_ids.size())
    # ce_loss = (ce_loss_raw * attention_mask).sum() / (attention_mask.sum() + 1e-6)
    # 
    # ce_loss.backward()
    # xm.optimizer_step(optimizer)
    # 
    # if is_master:
    #     wandb.log({"train/ce_loss": ce_loss.item()})
    #     print(f"Step Loss: {ce_loss.item()}")
    # 
    #     # Сохранение чекпойнта
    #     torch.save({"decoder": model.decoder.state_dict()}, os.path.join(ckpt_dir, "decoder_step_100.pth"))
    
    # if is_master: wandb.finish()

if __name__ == "__main__":
    # Для запуска на TPU с SPMD раскомментируй:
    # xmp.spawn(train_phase2, args=())
    pass
