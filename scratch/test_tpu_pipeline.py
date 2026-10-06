"""
Compliant TPU Script Example for Verification Testing.
"""
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.experimental.xla_sharding as xs
from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 32, dtype=torch.float32)

    def forward(self, x):
        return self.fc(x)

def train_dummy():
    dev = xm.xla_device()
    model = SimpleModel().to(dev)
    mesh = xs.Mesh([dev], (1, 1), ("fsdp", "data"))
    model = SpmdFullyShardedDataParallel(model, mesh=mesh)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    
    for step in range(2):
        optimizer.zero_grad()
        x = torch.randn(4, 32, dtype=torch.float32, device=dev)
        xs.mark_sharding(x, mesh, ("fsdp", None))
        
        out = model(x)
        loss = loss_fn(out, x)
        loss.backward()
        
        xm.optimizer_step(optimizer)
        xm.mark_step()
        print(f"[Dummy Step {step}] Complete. Loss tensor on device.")

if __name__ == "__main__":
    train_dummy()
