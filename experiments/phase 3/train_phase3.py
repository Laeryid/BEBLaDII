"""
train_phase3.py — Обучение OutputProjector (Phase 3).
Включает XLA/SPMD для TPU v4-8 и --debug режим для локальной проверки (CPU/GPU).
"""

import os
import sys
import argparse
import subprocess
import time
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.beb_la_dii.model.dus import DUSModel
from src.beb_la_dii.model.projectors import InputProjector, OutputProjector

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.experimental.xla_sharding as xs
    import torch_xla.runtime as xr
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False


# =============================================================================
# Настройка окружения и аргументы
# =============================================================================

def setup_env():
    os.environ["PJRT_DEVICE"] = "TPU"
    os.environ["XLA_USE_BF16"] = "1"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "2,2,1"
    os.environ["TPU_NUM_DEVICES"] = "4"
    os.environ["XLA_USE_SPMD"] = "1"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="phase3_test")
    parser.add_argument("--tau", type=float, default=0.1, help="Температура для Soft Matching")
    parser.add_argument("--k", type=int, default=28, help="Топ-k для Soft Matching")
    parser.add_argument("--num-layers", type=int, default=2, help="Слои OP: 2 или 3")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--debug", action="store_true", help="10 шагов локально (без XLA)")
    return parser.parse_args()


# =============================================================================
# Загрузка весов и словарей
# =============================================================================

def load_phase2_weights(checkpoint_path: str, student: DUSModel, input_projector: InputProjector):
    sd = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    cleaned = {}
    for key, v in sd.items():
        k_clean = key.replace("_orig_module.", "").replace("module.", "")
        cleaned[k_clean] = v

    student_sd = {}
    for key, v in cleaned.items():
        if key.startswith("student.model."):
            student_sd[key[len("student.model."):]] = v
        elif key.startswith("model.layers.") or key.startswith("model.embeddings") or key.startswith("model.final_norm"):
            student_sd[key[len("model."):]] = v

    if student_sd:
        student.model.load_state_dict(student_sd, strict=False)

    ip_sd = {}
    for key, v in cleaned.items():
        if key.startswith("input_projector."):
            ip_sd[key[len("input_projector."):]] = v
        elif key.startswith("proj.") or key.startswith("mu_head.") or key.startswith("logvar_head."):
            ip_sd[key] = v

    if ip_sd:
        input_projector.load_state_dict(ip_sd, strict=False)


# =============================================================================
# Данные (Pre-tokenized Dataset)
# =============================================================================

class PretokenizedDataset(Dataset):
    """Читает уже токенизированные parquet файлы (колонка 'input_ids')"""
    def __init__(self, data_dir: str, max_length=512):
        self.max_length = max_length
        try:
            from indexed_parquet import IndexedParquetDataset
        except ImportError:
            from indexed_parquet_dataset import IndexedParquetDataset
            
        print(f"Loading pre-tokenized data from {data_dir}...")
        # from_folder склеит все parquet файлы в один логический датасет
        self.ds = IndexedParquetDataset.from_folder(data_dir, auto_fill=True)
        # rg_buffer: количество row-groups в буфере памяти.
        # Сэмпл берётся из буфера, группа заменяется только когда все её элементы выбраны.
        self.ds = self.ds.shuffle(rg_buffer=32)
        print(f"Total samples: {len(self.ds)}")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        input_ids = item.get("input_ids", [])
        
        # Обрезаем или паддим
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = [1] * self.max_length
        else:
            pad_len = self.max_length - len(input_ids)
            attention_mask = [1] * len(input_ids) + [0] * pad_len
            input_ids = input_ids + [151643] * pad_len  # 151643 = pad_token (Qwen)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }


# =============================================================================
# Функции Loss и Metrics
# =============================================================================

def soft_dictionary_matching(L40_ctx, D_X0, D_L40_norm_sphere, whitening_mu, whitening_sigma, tau, k):
    """
    Soft Dictionary Matching с отбеливанием пространства L40.
    
    Отбеливание (whitening) устраняет доминирующие компоненты пространства
    (rank1_ratio ~0.7), не меняя семантической структуры соседей.
    Веса alpha вычисляются по сферическому пространству, но применяются
    к оригинальным векторам D_X0 — алгоритм разложения при этом корректен.
    """
    B, seq_len, D = L40_ctx.shape
    L40_flat = L40_ctx.reshape(-1, D)
    
    # Отбеливание: сжимаем доминирующие оси до масштаба остальных
    L40_sphere = (L40_flat - whitening_mu) / whitening_sigma
    L40_norm = F.normalize(L40_sphere, dim=-1)
    
    # Поиск соседей по сферическому пространству
    scores = (L40_norm @ D_L40_norm_sphere.T) / tau
    topk_scores, topk_idx = torch.topk(scores, k=k, dim=-1)
    alpha = F.softmax(topk_scores, dim=-1)
    
    # Сборка целевого вектора из ОРИГИНАЛЬНЫХ векторов Qwen (не сферических!)
    top_dx0 = D_X0[topk_idx]
    Z_hat_raw = (alpha.unsqueeze(-1) * top_dx0).sum(dim=1)
    
    top_dx0_norms = top_dx0.norm(dim=-1)
    L_expected = (alpha * top_dx0_norms).sum(dim=1, keepdim=True)
    Z_hat_target = F.normalize(Z_hat_raw, dim=-1) * L_expected
    
    eps = 1e-9
    H = -(alpha * torch.log(alpha + eps)).sum(dim=-1)
    k_eff = torch.exp(H).mean().item()
    top1_self = alpha[:, 0].mean().item()
    
    return Z_hat_target.reshape(B, seq_len, D), k_eff, top1_self

def get_hub_loss(pred, target, delta=1.0):
    diff = pred - target
    abs_diff = diff.abs()
    loss = torch.where(abs_diff < delta, 0.5 * diff ** 2, delta * (abs_diff - 0.5 * delta))
    return loss.mean()

def _norm_cv(x: torch.Tensor) -> float:
    norms = x.norm(dim=-1)
    return (norms.std() / norms.mean()).item()


# =============================================================================
# Обучение (XLA SPMD)
# =============================================================================

def train_tpu(args):
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import torch_xla.experimental.xla_sharding as xs
    from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel as FSDP
    
    device = xm.xla_device()
    rank = xm.get_local_ordinal()
    
    if rank == 0:
        print(f"--- [RANK 0] Настройка {args.run_name} (tau={args.tau}, layers={args.num_layers}) ---")
        
        # Настройка WandB
        key_path = "/home/hp/wandb_key.txt"
        if os.path.exists(key_path):
            import wandb
            with open(key_path, "r") as f:
                wandb.login(key=f.read().strip())
            wandb.init(project="BEBLaDII", name=args.run_name, config=vars(args))
        else:
            wandb = None
            
    # Mesh
    num_devices = xr.global_runtime_device_count()
    mesh = xs.Mesh(np.array(range(num_devices)), (num_devices, 1), ('fsdp', 'model'))
    xs.set_global_mesh(mesh)
    
    # 1. Загрузка Qwen Embedding
    if rank == 0: print("Загрузка Qwen embeddings...")
    qwen = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", device_map="cpu", torch_dtype=torch.bfloat16)
    qwen_embed = qwen.model.embed_tokens.to(device)
    for p in qwen_embed.parameters(): p.requires_grad = False
    del qwen # освобождаем RAM
    
    # 2. Инициализация DUS и OP
    student = DUSModel.from_scratch(config={"base_model_id": "answerdotai/ModernBERT-large", "target_layers": 40}).to(device)
    input_projector = InputProjector.from_scratch().to(device)
    output_projector = OutputProjector.from_scratch(config={"num_layers": args.num_layers}).to(device)
    
    # Замораживаем Phase 2
    for p in student.parameters(): p.requires_grad = False
    for p in input_projector.parameters(): p.requires_grad = False
    
    if rank == 0: print("Загрузка весов Phase 2...")
    ckpt_path = "./data/phase3/latest_checkpoint_phase2.pt"
    if os.path.exists(ckpt_path):
        load_phase2_weights(ckpt_path, student, input_projector)
    else:
        if rank == 0: print("WARN: Чекпоинт Phase 2 не найден! Используются случайные веса DUS.")
        
    student.eval()
    input_projector.eval()
    
    # Оборачиваем только обучаемый OutputProjector в FSDP
    output_projector = FSDP(output_projector, mesh=mesh, auto_wrap_policy=None)
    
    # 3. Словари
    if rank == 0: print("Загрузка словарей...")
    D_X0 = torch.load("./data/phase3/dictionaries/D_X0.pt", map_location="cpu").to(device).float()
    # D_L40 сырой (ненормализованный) — из него вычисляем параметры отбеливания
    D_L40_raw = torch.load("./data/phase3/dictionaries/D_L40.pt", map_location="cpu").float()
    whitening_mu = D_L40_raw.mean(dim=0, keepdim=True).to(device)
    whitening_sigma = (D_L40_raw.std(dim=0, keepdim=True) + 1e-6).to(device)
    # Сферический словарь: отбеливаем и нормируем
    D_L40_sphere = F.normalize((D_L40_raw.to(device) - whitening_mu) / whitening_sigma, dim=-1)
    del D_L40_raw
    if rank == 0: print(f"  Whitening: mu_norm={whitening_mu.norm():.3f}, sigma_max={whitening_sigma.max():.3f}, sigma_min={whitening_sigma.min():.6f}")
    
    # 4. Данные
    from torch_xla.distributed.parallel_loader import MpDeviceLoader
    dataset = PretokenizedDataset("./data/phase3/train_data/")
    from torch.utils.data.distributed import DistributedSampler
    sampler = DistributedSampler(dataset, num_replicas=num_devices, rank=rank, shuffle=True)
    loader_raw = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=8, prefetch_factor=4, persistent_workers=True, pin_memory=False
    )
    # MpDeviceLoader асинхронно переносит батчи на TPU, не блокируя граф
    loader = MpDeviceLoader(loader_raw, device)
    
    optimizer = torch.optim.AdamW(output_projector.parameters(), lr=args.learning_rate)
    
    # 5. Цикл обучения
    output_projector.train()
    global_step = 0
    
    pbar = tqdm(total=args.steps, disable=(rank != 0))
    while global_step < args.steps:
        for batch in loader:
            if global_step >= args.steps:
                break
            
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            
            xs.mark_sharding(input_ids, mesh, ('fsdp', None))
            xs.mark_sharding(attention_mask, mesh, ('fsdp', None))
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                # Qwen embs -> DUS
                qwen_embs = qwen_embed(input_ids)
                z, _, _ = input_projector(qwen_embs)
                dus_out = student.model(inputs_embeds=z, attention_mask=attention_mask)
                L40_ctx = dus_out.last_hidden_state
                
                # Soft разложение с отбеливанием
                Z_hat_target, k_eff, top1_self = soft_dictionary_matching(
                    L40_ctx, D_X0, D_L40_sphere, whitening_mu, whitening_sigma, args.tau, args.k
                )
            
            Z_pred = output_projector(L40_ctx)
            huber = get_hub_loss(Z_pred, Z_hat_target)
            loss = huber
            
            loss.backward()
            xm.optimizer_step(optimizer, barrier=True)
            
            global_step += 1
            pbar.update(1)
            
            if rank == 0 and global_step % 10 == 0:
                # k_eff и top1_self уже вычислены через .item() внутри soft_dictionary_matching
                # _norm_cv вызываем реже (каждые 50 шагов) чтобы не блокировать граф
                metrics = {
                    "train/loss": loss.item(),
                    "train/huber": huber.item(),
                    "train/k_eff": k_eff,
                    "train/top1_self": top1_self,
                }
                if global_step % 50 == 0:
                    xm.mark_step()  # сброс графа перед CPU-операцией
                    metrics["train/op_norm_cv"] = _norm_cv(Z_pred.detach().cpu())
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log(metrics, step=global_step)
                except ImportError:
                    pass
                
                pbar.set_postfix({
                    "loss": f"{metrics['train/loss']:.4f}",
                    "k_eff": f"{k_eff:.1f}"
                })
                
    if rank == 0:
        print("--- Обучение завершено ---")
        xm.save(output_projector.state_dict(), f"OP_{args.run_name}.pt")


# =============================================================================
# Запуск
# =============================================================================

if __name__ == "__main__":
    args = parse_args()
    
    if args.debug:
        print("Режим отладки (без XLA)")
        sys.exit(0)
        
    setup_env()
    
    # Синхронизация данных с GCS перед стартом TPU
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if rank == 0:
        print("--- [RANK 0] Подготовка ресурсов ---")
        os.makedirs("./data/phase3/dictionaries", exist_ok=True)
        os.makedirs("./data/phase3/train_data", exist_ok=True)
        
        try:
            # Словари
            subprocess.run(["gsutil", "-m", "rsync", "-r", "gs://bebladii-datasets-us/phase 3/dictionaries/", "./data/phase3/dictionaries/"], check=True)
            # Данные
            subprocess.run(["gsutil", "-m", "rsync", "-r", "gs://bebladii-datasets-us/phase 3/train_data/", "./data/phase3/train_data/"], check=True)
            # Чекпоинт Phase 2 (берём самый свежий, если он есть)
            res = subprocess.run(["gsutil", "ls", "gs://bebladii-weigths-us/checkpoints/latest_checkpoint.pt"], capture_output=True)
            if res.returncode == 0:
                subprocess.run(["gsutil", "cp", "gs://bebladii-weigths-us/checkpoints/latest_checkpoint.pt", "./data/phase3/latest_checkpoint_phase2.pt"], check=True)
        except Exception as e:
            print(f"--- [RANK 0] Ошибка GCS: {e} ---")
            
        with open("/tmp/resources_prepared.flag", "w") as f: f.write("ok")
    else:
        while not os.path.exists("/tmp/resources_prepared.flag"):
            time.sleep(1)
            
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    xr.use_spmd()
    
    xm.rendezvous("init_done")
    train_tpu(args)
