import os
import sys
import argparse
import torch
import torch_xla.core.xla_model as xm
import torch_xla.experimental.xla_sharding as xs
import torch_xla.runtime as xr
from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel
from transformers import AutoModel
from tqdm.auto import tqdm

def setup_env():
    os.environ["PJRT_DEVICE"] = "TPU"
    os.environ["XLA_USE_BF16"] = "1"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "2,2,1"
    os.environ["TPU_NUM_DEVICES"] = "4"
    os.environ["XLA_USE_SPMD"] = "1"

setup_env()

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.beb_la_dii.utils.data import get_dataloader

xr.use_spmd()

def setup_spmd_mesh():
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (num_devices, 1)
    import numpy as np
    device_ids = np.array(range(num_devices))
    mesh = xs.Mesh(device_ids, mesh_shape, ("fsdp", "model"))
    xs.set_global_mesh(mesh)
    return mesh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model_path", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_phrases", type=int, default=1000)
    parser.add_argument("--output_path", type=str, default="weights_cache/whitening_matrix_l23.pth")
    args = parser.parse_args()

    mesh = setup_spmd_mesh()
    device = xm.xla_device()

    if xm.is_master_ordinal():
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print("[Init] Loading Teacher...")
    teacher = AutoModel.from_pretrained(args.teacher_model_path, torch_dtype=torch.bfloat16).to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    
    def shard_output(output, mesh): return None
    teacher = SpmdFullyShardedDataParallel(teacher, mesh=mesh, shard_output=shard_output)
    teacher.eval()

    dataloader = get_dataloader(
        stage='reasoning', 
        batch_size=args.batch_size, 
        max_length=512, 
        split='train', 
        val_ratio=0.0
    )

    all_states = []
    max_steps = args.num_phrases // args.batch_size
    step = 0
    
    pbar = tqdm(total=max_steps, desc="Computing Whitening") if xm.is_master_ordinal() else None

    with torch.no_grad():
        for batch in dataloader:
            if step >= max_steps:
                break
                
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            xs.mark_sharding(input_ids, mesh, ("fsdp", None))
            xs.mark_sharding(attention_mask, mesh, ("fsdp", None))
            
            outputs = teacher(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden = outputs.hidden_states[23] # (B, T, 3584)
            
            # Extract active tokens directly on CPU to save memory
            hidden_cpu = hidden.cpu().float()
            mask_cpu = attention_mask.cpu()
            active = hidden_cpu[mask_cpu == 1]
            
            all_states.append(active)
            
            step += 1
            xm.mark_step()
            if pbar: pbar.update(1)

    if pbar: pbar.close()

    if xm.is_master_ordinal():
        print("[Compute] Concatenating states...")
        Z = torch.cat(all_states, dim=0) # (Total_N, D)
        print(f"[Compute] Total tokens collected: {Z.size(0)}")
        
        print("[Compute] Computing Covariance...")
        mean = Z.mean(dim=0, keepdim=True)
        Z_c = Z - mean
        cov = (Z_c.T @ Z_c) / Z_c.size(0)
        
        print("[Compute] Computing SVD...")
        U, S, V = torch.svd(cov)
        
        # ZCA Whitening matrix: W = U @ diag(1 / sqrt(S + epsilon)) @ U.T
        epsilon = 0.01 * S[0] # 1% of top eigenvalue
        inv_sqrt_S = 1.0 / torch.sqrt(S + epsilon)
        W = U @ torch.diag(inv_sqrt_S) @ U.T
        
        print(f"[Compute] Top 5 eigenvalues: {S[:5].tolist()}")
        print(f"[Compute] Bottom 5 eigenvalues: {S[-5:].tolist()}")
        
        torch.save({"W": W, "mean": mean, "S": S, "U": U}, args.output_path)
        print(f"[Done] Whitening matrix saved to {args.output_path}")

        # Optional: Sync to GCS
        try:
            import subprocess
            gcs_path = "gs://bebladii-weigths-us/planB/phase3/whitening_matrix_l23.pth"
            print(f"[GCS] Syncing whitening matrix to {gcs_path}")
            subprocess.run(["gcloud", "storage", "cp", args.output_path, gcs_path], check=True)
        except Exception as e:
            print(f"GCS Sync failed: {e}")

if __name__ == "__main__":
    main()
