import os
import sys

# Настройка переменных окружения для TPU v6e
def setup_env():
    os.environ["PJRT_DEVICE"] = "TPU"
    os.environ["XLA_USE_BF16"] = "1"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "2,2,1"
    os.environ["TPU_NUM_DEVICES"] = "4"
    os.environ["XLA_USE_SPMD"] = "1"

setup_env()

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def step(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.backup[name] = param.data.clone().detach()
                    param.data.copy_(self.shadow[name])

    def restore(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(self.backup[name])
        self.backup = {}

import torch_xla.core.xla_model as xm
import torch_xla.experimental.xla_sharding as xs
import torch_xla.runtime as xr
import wandb
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel

from src.beb_la_dii.model.vae import LatentEncoder
from src.beb_la_dii.model.dus import DUSModel
from src.beb_la_dii.utils.loss import safe_normalize

xr.use_spmd()

# ---------------------------------------------------------------------------
# SPMD Mesh
# ---------------------------------------------------------------------------

def setup_spmd_mesh():
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (num_devices, 1)
    device_ids = np.array(range(num_devices))
    mesh = xs.Mesh(device_ids, mesh_shape, ("fsdp", "model"))
    xs.set_global_mesh(mesh)
    return mesh


# LogicAdapter has been removed in favor of Pearson RKD on Whitened matrices


# ---------------------------------------------------------------------------
# Основная модель Phase 3
# ---------------------------------------------------------------------------

class BEBLaDIIPhase3(nn.Module):
    def __init__(self, teacher_model_path: str, embedding_model_path: str,
                 dus_weights: str | None, encoder_weights: str | None,
                 teacher_dim: int = 3584):
        super().__init__()

        # 1. Замороженный учитель (DeepSeek-R1-7B, 3584d) — только для дистилляции скрытых состояний
        self.teacher = AutoModel.from_pretrained(teacher_model_path, torch_dtype=torch.bfloat16)
        for p in self.teacher.parameters():
            p.requires_grad = False

        # 1b. Замороженный Qwen2.5-1.5B — только embedding слой для получения 1536d входа энкодера
        # LatentEncoder обучен на эмбеддингах Qwen2.5-1.5B (1536d), поэтому нельзя использовать
        # эмбеддинги DeepSeek (тоже 3584d → иное пространство).
        _qwen_base = AutoModel.from_pretrained(embedding_model_path, torch_dtype=torch.bfloat16)
        self.qwen_embeddings = _qwen_base.get_input_embeddings()  # nn.Embedding (vocab, 1536)
        del _qwen_base
        for p in self.qwen_embeddings.parameters():
            p.requires_grad = False

        # 2. Замороженный LatentEncoder (input_dim=1536, output_dim=1024)
        self.encoder = LatentEncoder()
        if encoder_weights and os.path.exists(encoder_weights):
            state = torch.load(encoder_weights, map_location="cpu")
            # Веса могут быть вложены под ключом "encoder"
            self.encoder.load_state_dict(state, strict=False)
            print(f"[Init] LatentEncoder weights loaded from {encoder_weights}", flush=True)
        else:
            print(f"[Init] WARN: encoder_weights not found ({encoder_weights}), using random init", flush=True)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.to(torch.bfloat16)

        # 3. DUS Backbone (обучаемый)
        dus_wrapper = DUSModel.from_scratch(weights_path=None)
        if dus_weights and os.path.exists(dus_weights):
            state = torch.load(dus_weights, map_location="cpu")
            if "latentBERT_state_dict" in state:
                state = state["latentBERT_state_dict"]
            elif "model_state_dict" in state:
                state = state["model_state_dict"]
            
            clean_state = {}
            for k, v in state.items():
                k_clean = k.replace("student.model.", "").replace("model.", "")
                clean_state[k_clean] = v
                
            model_sd = dus_wrapper.model.state_dict()
            matched = sum(1 for k in model_sd.keys() if k in clean_state)
            total_keys = len(model_sd.keys())
                
            dus_wrapper.model.load_state_dict(clean_state, strict=False)
            print(f"[Init] Awakened DUS weights loaded from {dus_weights} (Matched {matched}/{total_keys} parameters)", flush=True)
            
        self.dus = dus_wrapper.model
        self.dus.to(torch.bfloat16)

        # Confidence Embedding (Additive Projection)
        self.confidence_proj = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, 1024)
        )
        # Zero-init the final layer to start with no interference
        nn.init.zeros_(self.confidence_proj[-1].weight)
        nn.init.zeros_(self.confidence_proj[-1].bias)
        self.confidence_proj.to(torch.bfloat16)

        # LogicAdapter has been removed.

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                low_noise_amp: float = 0.5) -> dict:
        """
        Возвращает словарь тензоров для вычисления лосса.
        Все вычисления учителя и энкодера — без градиентов.
        """
        with torch.no_grad():
            # Эмбеддинги 1536d из Qwen2.5-1.5B (то же пространство, на котором обучен LatentEncoder)
            qwen_embeds = self.qwen_embeddings(input_ids)  # (B, T, 1536)

            # Скрытые состояния учителя (DeepSeek-R1-7B, 3584d) для дистилляции
            # Используем его собственный embedding слой: input_ids одинаковые (tokenizer Qwen-совместим)
            teacher_outputs = self.teacher(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True
            )
            teacher_hidden = teacher_outputs.last_hidden_state  # (B, T, 3584)
            teacher_layer23 = teacher_outputs.hidden_states[23]

            # Кодируем Qwen-эмбеддинги в латентное пространство
            z_clean, _, _ = self.encoder(qwen_embeds)  # (B, T, 1024)

            B, T, D = z_clean.shape

            # --- Windows generation ---
            noise_window_size = 5
            logic_window_size = 15
            num_windows = 5
            block_size = T // num_windows
            
            # Генерируем стартовые индексы для окон зашумления (5 токенов)
            starts = []
            for w in range(num_windows):
                # Окно зашумления может быть в любом месте блока
                min_start = 0
                max_start = max(min_start + 1, block_size - noise_window_size)
                
                start = torch.randint(w * block_size + min_start, w * block_size + max_start, (B, 1), device=z_clean.device)
                starts.append(start)
            starts = torch.cat(starts, dim=1) # (B, num_windows)
            
            # Окно зашумления (5 токенов)
            noise_offsets = torch.arange(noise_window_size, device=z_clean.device).view(1, 1, noise_window_size)
            noise_window_indices = starts.unsqueeze(-1) + noise_offsets # (B, num_windows, 5)
            
            # Расширенное окно логики (15 токенов)
            logic_starts = starts - ((logic_window_size - noise_window_size) // 2)
            logic_starts = torch.clamp(logic_starts, min=0, max=T - logic_window_size)
            logic_offsets = torch.arange(logic_window_size, device=z_clean.device).view(1, 1, logic_window_size)
            logic_window_indices = logic_starts.unsqueeze(-1) + logic_offsets # (B, num_windows, 15)
            
            # Внутри 5-токенового окна выбираем случайное подмножество токенов для зашумления (от 1 до 5)
            rand_noise = torch.rand((B, num_windows, noise_window_size), device=z_clean.device)
            num_to_noise = torch.randint(1, noise_window_size + 1, (B, num_windows, 1), device=z_clean.device)
            _, noise_ranks = torch.sort(rand_noise, dim=-1)
            noise_subset_mask = (torch.arange(noise_window_size, device=z_clean.device).view(1, 1, noise_window_size) < num_to_noise).float()
            noise_subset_mask = torch.gather(noise_subset_mask, 2, noise_ranks.argsort(dim=-1))
            
            # Проецируем маску зашумления на полную последовательность
            flat_noise_indices = noise_window_indices.view(B, -1)
            full_noise_mask = torch.zeros((B, T), device=z_clean.device, dtype=z_clean.dtype)
            full_noise_mask.scatter_(1, flat_noise_indices, noise_subset_mask.view(B, -1))
            
            noise_mask = full_noise_mask * attention_mask.to(z_clean.dtype)

            # Нормированный случайный шум (в float32 для точности на малых векторах)
            noise = torch.randn_like(z_clean)
            noise = safe_normalize(noise.float(), dim=-1).to(torch.bfloat16) * low_noise_amp

            z_noisy = z_clean + noise * noise_mask.unsqueeze(-1)
            
            # Сферическая геометрия (жестко на единичную сферу, так как z_clean от LatentEncoder уже там)
            z_noisy = torch.where(
                noise_mask.unsqueeze(-1) > 0,
                safe_normalize(z_noisy.float(), dim=-1).to(torch.bfloat16),
                z_noisy
            )

            # Уверенность
            z_clean_normed = safe_normalize(z_clean, dim=-1)
            z_noisy_normed = safe_normalize(z_noisy, dim=-1)
            c_true = torch.clamp((z_clean_normed * z_noisy_normed).sum(dim=-1), min=0.0)

        # Проекция уверенности
        c_embed = self.confidence_proj(c_true.unsqueeze(-1).to(torch.bfloat16))
        dus_input = z_noisy.to(torch.bfloat16) + c_embed

        # Прогон зашумлённой последовательности через DUS
        dus_outputs = self.dus(
            inputs_embeds=dus_input, attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Трансформер уже включает входной вектор (residual stream) в выход.
        # Поэтому last_hidden_state - это УЖЕ смещенный вектор (z_noisy + delta).
        # Чтобы не удваивать z_noisy, мы берем final как есть, а дельту считаем математически.
        dus_final_raw = dus_outputs.last_hidden_state
        dus_delta = dus_final_raw - z_noisy.to(torch.bfloat16)
        dus_final = safe_normalize(dus_final_raw.float(), dim=-1).to(torch.bfloat16)
        
        dus_layer33 = dus_outputs.hidden_states[33]

        return {
            "z_clean": z_clean,
            "z_noisy": z_noisy,
            "c_true": c_true,
            "noise_mask": noise_mask,
            "dus_delta": dus_delta,
            "dus_final": dus_final,
            "dus_layer33": dus_layer33,
            "teacher_layer23": teacher_layer23,
            "logic_window_indices": logic_window_indices,
            "attention_mask": attention_mask,
            "c_embed": c_embed,
            "dus_input": dus_input
        }


# ---------------------------------------------------------------------------
# Функция потерь
# ---------------------------------------------------------------------------

def compute_phase3_loss(
    outputs: dict,
    gamma: float = 20.0,
    w_denoise: float = 1.0,
    w_logic: float = 1.0,
    w_identity: float = 5.0,
    denoise_delta: float = 5.0,
    whitening_w: torch.Tensor = None,
    whitening_mean: torch.Tensor = None,
) -> tuple[torch.Tensor, dict]:
    z_clean       = outputs["z_clean"]
    c_true        = outputs["c_true"]
    noise_mask    = outputs["noise_mask"]
    dus_delta     = outputs["dus_delta"]
    dus_final     = outputs["dus_final"]
    dus_layer33   = outputs["dus_layer33"]
    teacher_layer23 = outputs["teacher_layer23"]
    logic_window_indices = outputs["logic_window_indices"]
    c_embed       = outputs["c_embed"]
    dus_input     = outputs["dus_input"]
    attn_f         = outputs["attention_mask"].float()

    metrics: dict = {}

    B, T, D = z_clean.size()
    num_windows = logic_window_indices.size(1)
    logic_window_size = logic_window_indices.size(2)
    active_tokens  = attn_f.sum().clamp(min=1.0)
    noised_tokens  = noise_mask.sum().clamp(min=1.0)

    # 4.1 Геометрическое: Prior Loss (mean, var, cov)
    z_flat   = dus_final.view(-1, D)
    mask_flat = attn_f.view(-1, 1)
    m_state  = (z_flat * mask_flat).sum(dim=0) / active_tokens
    z_centered = z_flat - m_state
    
    v_state = (z_centered.pow(2) * mask_flat).sum(dim=0) / active_tokens
    var_floor = 1.0 / (D * 2)  # Нижняя граница дисперсии для единичной сферы
    var_loss = F.relu(var_floor - v_state).mean()
    
    cov = (z_centered.T @ (z_centered * mask_flat)) / active_tokens
    cov_off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = cov_off_diag.pow(2).sum() / D
    
    prior_loss = m_state.pow(2).mean() + var_loss + 0.1 * cov_loss
    metrics["prior_loss"] = prior_loss.detach()
    metrics["cov_loss"] = cov_loss.detach()

    total_loss = 0.1 * prior_loss

    # 4.2 Диффузное (Denoising): Cosine Loss на зашумлённых позициях
    denoise_target = safe_normalize(z_clean.float() + c_embed.float(), dim=-1).to(torch.bfloat16)
    cos_denoise = (dus_final.float() * denoise_target.float()).sum(dim=-1)
    denoise_elementwise = 1.0 - cos_denoise
    denoise_loss = (denoise_elementwise * noise_mask).sum() / noised_tokens
    metrics["denoise_loss"] = denoise_loss.detach()
    metrics["norm_dus_delta"] = (dus_delta.float().norm(dim=-1) * noise_mask).sum() / noised_tokens
    total_loss = total_loss + w_denoise * denoise_loss

    # 4.3 Дистилляционное (Логика): Pearson RKD на матрицах (15x15) выбеленного учителя
    flat_logic_indices = logic_window_indices.view(B, -1)
    
    dus_w = torch.gather(dus_layer33, 1, flat_logic_indices.unsqueeze(-1).expand(-1, -1, dus_layer33.size(-1))).view(B, num_windows, logic_window_size, dus_layer33.size(-1))
    tea_w = torch.gather(teacher_layer23, 1, flat_logic_indices.unsqueeze(-1).expand(-1, -1, teacher_layer23.size(-1))).view(B, num_windows, logic_window_size, teacher_layer23.size(-1))
    
    attn_w = torch.gather(attn_f, 1, flat_logic_indices).view(B, num_windows, logic_window_size)
    window_valid = (attn_w.sum(dim=-1) == logic_window_size).float()
    
    metrics["norm_tea_raw"] = tea_w.float().norm(dim=-1).mean().detach()
    metrics["norm_dus_l33"] = dus_w.float().norm(dim=-1).mean().detach()

    if whitening_w is not None and whitening_mean is not None:
        tea_w = tea_w - whitening_mean
        tea_w = torch.matmul(tea_w, whitening_w)
        metrics["norm_tea_white"] = tea_w.float().norm(dim=-1).mean().detach()
    else:
        metrics["norm_tea_white"] = metrics["norm_tea_raw"]
        
    dus_w_norm = safe_normalize(dus_w, dim=-1)
    tea_w_norm = safe_normalize(tea_w, dim=-1)
    
    S_dus = torch.matmul(dus_w_norm, dus_w_norm.transpose(-1, -2))
    S_tea = torch.matmul(tea_w_norm, tea_w_norm.transpose(-1, -2))
    
    mask = torch.triu(torch.ones((logic_window_size, logic_window_size), dtype=torch.bool, device=S_dus.device), diagonal=1)
    dus_triu = S_dus[:, :, mask]
    tea_triu = S_tea[:, :, mask]
    
    def pearson_corr(x, y):
        x_c = x - x.mean(dim=-1, keepdim=True)
        y_c = y - y.mean(dim=-1, keepdim=True)
        num = (x_c * y_c).sum(dim=-1)
        den = torch.sqrt((x_c.pow(2).sum(dim=-1) + 1e-8) * (y_c.pow(2).sum(dim=-1) + 1e-8))
        return num / den

    corr = pearson_corr(dus_triu, tea_triu)
    logic_loss_per_window = 1.0 - corr
    
    valid_windows_sum = window_valid.sum().clamp(min=1.0)
    logic_loss = (logic_loss_per_window * window_valid).sum() / valid_windows_sum
    metrics["logic_loss"] = logic_loss.detach()
    metrics["logic_pearson_r"] = (corr * window_valid).sum().detach() / valid_windows_sum
    total_loss = total_loss + w_logic * logic_loss

    # 4.4 Identity Penalty (Cosine Loss)
    w_c = torch.pow(c_true.float(), gamma)
    identity_target = safe_normalize(dus_input.float(), dim=-1).to(torch.bfloat16)
    cos_identity = (dus_final.float() * identity_target.float()).sum(dim=-1)
    penalty_elementwise = 1.0 - cos_identity
    identity_penalty = (penalty_elementwise * w_c * attn_f).sum() / active_tokens
    metrics["identity_penalty"] = identity_penalty.detach()
    total_loss = total_loss + w_identity * identity_penalty

    # Метрики мониторинга
    c_true_mean = (c_true * attn_f).sum() / active_tokens
    metrics["c_true_mean"] = c_true_mean.detach()

    return total_loss, metrics


# DataLoader удален, используем встроенный в utils.data


# ---------------------------------------------------------------------------
# Обучение
# ---------------------------------------------------------------------------

def train(args):
    mesh = setup_spmd_mesh()
    device = xm.xla_device()

    if os.path.exists(args.local_whitening):
        whitening_data = torch.load(args.local_whitening, map_location="cpu")
        whitening_w = whitening_data["W"].to(device).to(torch.bfloat16)
        whitening_mean = whitening_data.get("mean")
        if whitening_mean is not None:
            whitening_mean = whitening_mean.to(device).to(torch.bfloat16)
            xs.mark_sharding(whitening_mean, mesh, (None, None))
        
        xs.mark_sharding(whitening_w, mesh, (None, None))
        print("[Init] Whitening matrix and mean loaded and moved to XLA.", flush=True)
    else:
        whitening_w = None
        whitening_mean = None
        print("[Init] WARN: Whitening matrix NOT found, using raw teacher vectors.", flush=True)

    print("[Init] Instantiating BEBLaDIIPhase3 model... (this includes loading 7B Teacher)", flush=True)

    model = BEBLaDIIPhase3(
        teacher_model_path=args.teacher_model_path,
        embedding_model_path=args.embedding_model_path,
        dus_weights=args.local_dus_weights,
        encoder_weights=args.local_encoder_weights,
        teacher_dim=args.teacher_dim,
    ).to(device)

    def shard_output(output, mesh):
        # FSDP требует этот callable для не-тензорных выходов от HF Models
        return None

    # Шардируем все тяжёлые компоненты через FSDP:
    print("[Init] Starting SPMD FSDP sharding across TPU... (THIS TAKES 2-4 MINUTES!)", flush=True)
    model.teacher          = SpmdFullyShardedDataParallel(model.teacher, mesh=mesh, shard_output=shard_output)
    model.qwen_embeddings  = SpmdFullyShardedDataParallel(model.qwen_embeddings, mesh=mesh, shard_output=shard_output)
    model.encoder          = SpmdFullyShardedDataParallel(model.encoder, mesh=mesh, shard_output=shard_output)
    model.dus              = SpmdFullyShardedDataParallel(model.dus, mesh=mesh, shard_output=shard_output)
    model.confidence_proj  = SpmdFullyShardedDataParallel(model.confidence_proj, mesh=mesh, shard_output=shard_output)
    print("[Init] FSDP sharding completed!", flush=True)

    trainable_params = list(model.dus.parameters()) + list(model.confidence_proj.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-2)

    ema = EMA(model, decay=0.999)

    if xm.is_master_ordinal() and args.wandb_project:
        wandb.init(project=args.wandb_project, config=vars(args))

    from src.beb_la_dii.utils.data import get_dataloader

    dataloader = get_dataloader(
        stage='reasoning', 
        batch_size=args.batch_size, 
        max_length=args.max_length, 
        split='train', 
        val_ratio=0.0
    )
    val_dataloader = get_dataloader(
        stage='reasoning', 
        batch_size=args.batch_size, 
        max_length=args.max_length, 
        split='val', 
        val_ratio=0.0
    )

    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2000, T_mult=1, eta_min=1e-6)

    model.train()
    step = 0

    total_steps = min(args.max_steps, len(dataloader) * args.epochs)

    from tqdm.auto import tqdm
    pbar = tqdm(total=total_steps, desc="Training Phase 3") if xm.is_master_ordinal() else None

    for epoch in range(args.epochs):
        for batch in dataloader:
            if step >= total_steps:
                break

            input_ids     = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Шардируем батч по FSDP-оси (КРИТИЧНО для TPU скорости)
            xs.mark_sharding(input_ids,      mesh, ("fsdp", None))
            xs.mark_sharding(attention_mask, mesh, ("fsdp", None))

            optimizer.zero_grad()

            fwd_outputs = model(input_ids, attention_mask=attention_mask,
                                low_noise_amp=args.low_noise_amp)
            loss, metrics = compute_phase3_loss(
                fwd_outputs, gamma=args.gamma,
                w_denoise=args.w_denoise, w_logic=args.w_logic,
                w_identity=args.w_identity, denoise_delta=args.denoise_delta,
                whitening_w=whitening_w, whitening_mean=whitening_mean
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            xm.optimizer_step(optimizer, barrier=True)
            ema.step(model)
            xm.mark_step() # ПРИНУДИТЕЛЬНО материализуем in-place апдейты EMA
            scheduler.step()

            # --- Цикличный прогрев LR (Initial & Restart Warmups) ---
            current_optim_step = step + 1
            warmup_steps = min(1000, int(args.max_steps * 0.1))
            
            if current_optim_step <= warmup_steps:
                # 1. Начальный прогрев
                lr_warmup_factor = max(0.01, current_optim_step / warmup_steps)
                for idx_p, param_group in enumerate(optimizer.param_groups):
                    param_group['lr'] = scheduler.base_lrs[idx_p] * lr_warmup_factor
            else:
                # 2. Прогрев при автоматических рестартах (каждые 2000 шагов)
                rel_step = current_optim_step % 2000
                restart_warmup_steps = 200
                if rel_step < restart_warmup_steps:
                    lr_warmup_factor = max(0.01, rel_step / restart_warmup_steps)
                    for idx_p, param_group in enumerate(optimizer.param_groups):
                        param_group['lr'] = param_group['lr'] * lr_warmup_factor
            # ---------------------------------------------------------

            # Логирование
            if step % args.log_steps == 0:
                # Диагностика EMA
                diff_sum = 0.0
                num_params = 0
                for n, p in model.named_parameters():
                    if p.requires_grad and n in ema.shadow:
                        diff_sum += (p.data - ema.shadow[n]).abs().mean().item()
                        num_params += 1
                ema_diff = diff_sum / max(1, num_params)

                metrics_dict = {k: v.item() for k, v in metrics.items()}
                metrics_dict["loss"] = loss.item()
                metrics_dict["lr"]   = optimizer.param_groups[0]['lr']
                metrics_dict["ema_diff"] = ema_diff
                if xm.is_master_ordinal():
                    if args.wandb_project:
                        wandb.log(metrics_dict, step=step)
                    if pbar:
                        pbar.set_postfix({
                            "loss":    f"{metrics_dict['loss']:.4f}",
                            "logic":   f"{metrics_dict.get('logic_loss', 0):.4f}",
                            "c_true":  f"{metrics_dict.get('c_true_mean', 0):.4f}",
                            "ema_df":  f"{ema_diff:.6f}",
                        })

            # Валидация
            if val_dataloader is not None and step > 0 and step % args.val_steps == 0:
                ema.apply(model)
                model.eval()
                v_loss_sum = torch.tensor(0.0, device=device)
                max_val_batches = 50
                num_v_batches = 0
                
                with torch.no_grad():
                    for v_step, v_batch in enumerate(val_dataloader):
                        if v_step >= max_val_batches: break
                        
                        v_input_ids = v_batch["input_ids"].to(device)
                        v_attention_mask = v_batch["attention_mask"].to(device)
                        xs.mark_sharding(v_input_ids, mesh, ('fsdp', None))
                        xs.mark_sharding(v_attention_mask, mesh, ('fsdp', None))
                        
                        v_fwd = model(v_input_ids, attention_mask=v_attention_mask, low_noise_amp=args.low_noise_amp)
                        v_loss, _ = compute_phase3_loss(
                            v_fwd, gamma=args.gamma,
                            w_denoise=args.w_denoise, w_logic=args.w_logic,
                            w_identity=args.w_identity, denoise_delta=args.denoise_delta,
                            whitening_w=whitening_w, whitening_mean=whitening_mean
                        )
                        v_loss_sum += v_loss
                        num_v_batches += 1
                        xm.mark_step()
                
                if num_v_batches > 0:
                    v_loss_avg = (v_loss_sum / num_v_batches).item()
                    if xm.is_master_ordinal():
                        if args.wandb_project: wandb.log({"val/loss": v_loss_avg}, step=step)
                        print(f"\n[VAL] Step {step} | Loss: {v_loss_avg:.4f} (EMA)")
                model.train()
                ema.restore(model)

            # Сохранение чекпоинта и GCS Sync
            if step > 0 and step % args.save_steps == 0:
                xm.mark_step()
                if xm.is_master_ordinal():
                    os.makedirs(args.output_dir, exist_ok=True)
                    ckpt_path = os.path.join(args.output_dir, f"phase3_step_{step}.pth")
                    dus_state     = {k: v.cpu() for k, v in model.dus.state_dict().items()}
                    
                    ema.apply(model)
                    dus_ema_state = {k: v.cpu() for k, v in model.dus.state_dict().items()}
                    proj_ema_state = {k: v.cpu() for k, v in model.confidence_proj.state_dict().items()}
                    ema.restore(model)
                    
                    torch.save({
                        "dus": dus_state, 
                        "dus_ema": dus_ema_state, 
                        "confidence_proj_ema": proj_ema_state
                    }, ckpt_path)
                    print(f"\n[SAVE] Checkpoint (Hot + EMA) saved → {ckpt_path}")
                    
                    if args.gcs_checkpoint_dir:
                        import subprocess
                        print(f"[GCS] Syncing checkpoints to {args.gcs_checkpoint_dir}...")
                        subprocess.Popen(["gcloud", "storage", "rsync", "-r", args.output_dir, args.gcs_checkpoint_dir])

            step += 1
            if pbar:
                pbar.update(1)

        if step >= total_steps:
            break

    if pbar:
        pbar.close()

    # Финальное сохранение
    xm.mark_step()
    if xm.is_master_ordinal():
        os.makedirs(args.output_dir, exist_ok=True)
        final_path = os.path.join(args.output_dir, "phase3_final.pth")
        
        ema.apply(model)
        dus_ema_state = {k: v.cpu() for k, v in model.dus.state_dict().items()}
        proj_ema_state = {k: v.cpu() for k, v in model.confidence_proj.state_dict().items()}
        ema.restore(model)
        
        torch.save({
            "dus": {k: v.cpu() for k, v in model.dus.state_dict().items()},
            "dus_ema": dus_ema_state,
            "confidence_proj_ema": proj_ema_state
        }, final_path)
        print(f"[SAVE] Final weights (Hot + EMA) saved → {final_path}")
        if args.wandb_project:
            wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BEBLaDII Phase 3 — DUS Logic Training")

    # Модели
    parser.add_argument("--teacher_model_path", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="HuggingFace path или локальная папка модели-учителя (DeepSeek-R1-7B)")
    parser.add_argument("--embedding_model_path", type=str,
                        default="Qwen/Qwen2.5-1.5B",
                        help="Модель, из которой берём embedding слой для LatentEncoder (1536d)")
    parser.add_argument("--teacher_dim", type=int, default=3584,
                        help="Размерность last_hidden_state учителя (3584 для DeepSeek-R1-7B)")

    # GCS пути к весам
    parser.add_argument("--encoder_weights_gs", type=str,
                        default="gs://bebladii-weigths-us/planB/phase1/checkpoints/phase1_vae_step_10000.pth")
    parser.add_argument("--dus_weights_gs", type=str,
                        default="gs://bebladii-weigths-us/kaggle_upload_1_2/AWAKENED_WEIGHTS_FINAL.pt")
    parser.add_argument("--data_path", type=str,
                        default="gs://bebladii-datasets-us/phase 3/train_data/data")
    parser.add_argument("--val_data_path", type=str, default="",
                        help="Путь к данным валидации. Пусто = выключено.")
    parser.add_argument("--whitening_gs", type=str,
                        default="gs://bebladii-weigths-us/planB/phase3/whitening_matrix_l23.pth")
    parser.add_argument("--gcs_checkpoint_dir", type=str, default="",
                        help="Бакет для бэкапа весов (например: gs://bebladii-weigths-us/planB/phase3/checkpoints/)")

    # Параметры обучения
    parser.add_argument("--output_dir",    type=str,   default="checkpoints/phase3")
    parser.add_argument("--batch_size",    type=int,   default=8)
    parser.add_argument("--max_length",    type=int,   default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs",        type=int,   default=1)
    parser.add_argument("--max_steps",     type=int,   default=40000)
    parser.add_argument("--log_steps",     type=int,   default=10)
    parser.add_argument("--val_steps",     type=int,   default=1000)
    parser.add_argument("--save_steps",    type=int,   default=1000)

    # Гиперпараметры Phase 3
    parser.add_argument("--low_noise_amp", type=float, default=0.5,
                        help="Амплитуда шума LowNoiseAmp (норма случайного вектора)")
    parser.add_argument("--gamma",    type=float, default=20.0,
                        help="Степень нелинейности штрафа стабильности (W_c = c_true^gamma)")
    parser.add_argument("--w_denoise", type=float, default=10.0, help="Вес Denoising loss")
    parser.add_argument("--w_logic",   type=float, default=1.0, help="Вес Logic distillation loss")
    parser.add_argument("--w_identity", type=float, default=5.0, help="Вес Identity penalty")
    parser.add_argument("--denoise_delta", type=float, default=5.0, help="Delta для Huber Loss")

    parser.add_argument("--wandb_project", type=str, default="BEBLaDII-Phase3")

    args = parser.parse_args()

    # Загрузка ресурсов с GCS (только master rank)
    if xm.is_master_ordinal():
        import subprocess
        os.makedirs("./weights_cache", exist_ok=True)

        args.local_encoder_weights = "./weights_cache/encoder.pth"
        if not os.path.exists(args.local_encoder_weights):
            print(f"[GCS] Downloading encoder: {args.encoder_weights_gs}")
            subprocess.run(
                ["gcloud", "storage", "cp", args.encoder_weights_gs, args.local_encoder_weights],
                check=True
            )

        args.local_dus_weights = "./weights_cache/dus.pth"
        if not os.path.exists(args.local_dus_weights):
            print(f"[GCS] Downloading DUS weights: {args.dus_weights_gs}")
            subprocess.run(
                ["gcloud", "storage", "cp", args.dus_weights_gs, args.local_dus_weights],
                check=True
            )

        args.local_whitening = "./weights_cache/whitening_matrix_l23.pth"
        if not os.path.exists(args.local_whitening):
            print(f"[GCS] Downloading whitening matrix: {args.whitening_gs}")
            try:
                subprocess.run(["gcloud", "storage", "cp", args.whitening_gs, args.local_whitening], check=True)
            except subprocess.CalledProcessError:
                print(f"[GCS] WARN: Whitening matrix not found at {args.whitening_gs}")

        # Синк всего датасета в ./data (как в Phase 1)
        os.makedirs("./data", exist_ok=True)
        try:
            # Синкаем директорию с данными в корень ./data
            subprocess.run(["gsutil", "-m", "rsync", "-r", "gs://bebladii-datasets-us/data/", "./data"], check=True)
            print("[GCS] Successfully synced dataset to ./data")
        except Exception as e:
            print(f"[GCS] ERROR syncing dataset: {e}")
    else:
        args.local_encoder_weights = "./weights_cache/encoder.pth"
        args.local_dus_weights     = "./weights_cache/dus.pth"
        args.local_whitening       = "./weights_cache/whitening_matrix_l23.pth"

    # Ждём пока master закончит скачивание
    xm.rendezvous("resources_prepared")

    train(args)
