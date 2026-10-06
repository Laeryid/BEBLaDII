import torch

def xla_device(n=None, devkind=None):
    return torch.device("cpu")

def get_xla_supported_devices():
    return ["cpu"]

def get_local_ordinal():
    return 0

def get_ordinal():
    return 0

def is_master_ordinal(local=True):
    return True

def optimizer_step(optimizer, barrier=False, optimizer_args=None):
    if optimizer_args is None:
        optimizer_args = {}
    return optimizer.step(**optimizer_args)

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
