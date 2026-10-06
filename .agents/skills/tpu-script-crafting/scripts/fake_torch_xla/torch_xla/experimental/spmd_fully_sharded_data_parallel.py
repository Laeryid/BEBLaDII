import torch
import torch.nn as nn

class SpmdFullyShardedDataParallel(nn.Module):
    def __init__(self, module, mesh=None, shard_output=None):
        super().__init__()
        self._orig_module = module
        self.mesh = mesh
        self.shard_output = shard_output

    def forward(self, *args, **kwargs):
        out = self._orig_module(*args, **kwargs)
        if self.shard_output is not None and self.mesh is not None:
            try:
                self.shard_output(out, self.mesh)
            except Exception:
                pass
        return out

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._orig_module, name)
