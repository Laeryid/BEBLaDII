import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BEComponent

class LatentEncoder(BEComponent):
    """
    mu-VAE Encoder for BEBLaDII v3.0.
    Maps Teacher embeddings (e.g. 3584) to Latent Diffusion Space (e.g. 1024)
    Computes mu, logvar, samples Z and applies LayerNorm to project to a hypersphere.
    """
    def __init__(self, component_id="vae_encoder", version="v3.0", config=None):
        input_dim = config.get("input_dim", 3584) if config else 3584
        hidden_dim = config.get("hidden_dim", 2048) if config else 2048
        output_dim = config.get("output_dim", 1024) if config else 1024
        
        super().__init__(component_id, version, {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim
        })
        
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim, eps=1e-6)
        )
        
        # mu and logvar heads
        self.mu_head = nn.Linear(output_dim, output_dim)
        self.logvar_head = nn.Linear(output_dim, output_dim)
        
        # Финальная проекция на идеальную сферу
        self.sphere_norm = nn.LayerNorm(output_dim, eps=1e-6)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @classmethod
    def from_scratch(cls, component_id="vae_encoder", version="v3.0", weights_path=None, **kwargs):
        config = kwargs.get("config", {"input_dim": 3584, "hidden_dim": 2048, "output_dim": 1024})
        instance = cls(component_id=component_id, version=version, config=config)
        instance.load_weights(weights_path)
        return instance

    def forward(self, x):
        h = self.proj(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        logvar = torch.clamp(logvar, -4.0, 4.0)
        
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z_raw = mu + eps * std
        else:
            z_raw = mu
            
        # Hard constraint to sphere
        z = self.sphere_norm(z_raw)
        
        return z, mu, logvar


class LatentDecoder(BEComponent):
    """
    mu-VAE Decoder for BEBLaDII v3.0.
    Maps from Latent Diffusion Space (e.g. 1024) back to Teacher embeddings (e.g. 3584).
    """
    def __init__(self, component_id="vae_decoder", version="v3.0", config=None):
        input_dim = config.get("input_dim", 1024) if config else 1024
        hidden_dim = config.get("hidden_dim", 2048) if config else 2048
        output_dim = config.get("output_dim", 3584) if config else 3584
        
        super().__init__(component_id, version, {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim
        })
        
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @classmethod
    def from_scratch(cls, component_id="vae_decoder", version="v3.0", weights_path=None, **kwargs):
        config = kwargs.get("config", {"input_dim": 1024, "hidden_dim": 2048, "output_dim": 3584})
        instance = cls(component_id=component_id, version=version, config=config)
        instance.load_weights(weights_path)
        return instance

    def forward(self, z):
        return self.proj(z)
