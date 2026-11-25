"""
Mock of torch_xla.core.xla_model aligned with PyTorch/XLA >= 2.7.

NOTE: Removed APIs (`get_ordinal`, `get_local_ordinal`, `xrt_world_size`) are
intentionally omitted so Level 3 CPU smoke tests catch AttributeError locally
instead of failing later on Kaggle/Cloud TPU. Use `torch_xla.runtime` (`xr`):
`xr.global_ordinal()`, `xr.local_ordinal()`, `xr.world_size()`.
"""
import warnings
import torch

REDUCE_SUM = "sum"
REDUCE_MUL = "mul"
REDUCE_AND = "and"
REDUCE_OR = "or"
REDUCE_MIN = "min"
REDUCE_MAX = "max"


def xla_device(n=None, devkind=None):
    warnings.warn("Use torch_xla.device instead", DeprecationWarning, stacklevel=2)
    return torch.device("cpu")


def get_xla_supported_devices():
    return ["cpu"]


def is_master_ordinal(local=True):
    return True


def all_reduce(reduce_type, inputs, scale=1.0, groups=None, pin_layout=True):
    return inputs

def optimizer_step(optimizer, barrier=False, optimizer_args=None):
    if optimizer_args is None:
        optimizer_args = {}
    return optimizer.step(**optimizer_args)

def reduce_gradients(optimizer=None, groups=None, pin_layout=True):
    pass

def mark_step():
    pass

def save(data, file_or_path, master_only=True, global_master=False):
    torch.save(data, file_or_path)

def rendezvous(tag, payload=b"", replica_groups=None):
    return payload

def add_step_closure(fn, args=()):
    try:
        fn(*args)
    except Exception as e:
        print(f"[Mock xm.add_step_closure Warning] Step closure failed: {e}")

def set_rng_state(seed):
    torch.manual_seed(seed)

def metrics_report():
    return "[Mock XLA] Metrics: UncachedCompile: 1, aten::_local_scalar_dense: 0"
