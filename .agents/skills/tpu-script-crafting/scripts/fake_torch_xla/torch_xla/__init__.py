"""
Mock implementation of torch_xla for local CPU smoke testing (PyTorch/XLA >= 2.7 API).
"""
import torch


def device(index=None):
    return torch.device("cpu")


def sync():
    pass


class _DeviceContext:
    pass
