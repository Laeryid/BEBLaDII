class MpDeviceLoader:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device

    def __iter__(self):
        for batch in self.loader:
            if isinstance(batch, dict):
                yield {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in batch.items()}
            elif isinstance(batch, (list, tuple)):
                yield type(batch)(v.to(self.device) if hasattr(v, "to") else v for v in batch)
            else:
                yield batch.to(self.device) if hasattr(batch, "to") else batch

    def __len__(self):
        return len(self.loader)
