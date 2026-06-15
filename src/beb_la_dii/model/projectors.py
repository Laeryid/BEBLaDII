import torch
import torch.nn.functional as F
import torch.nn as nn
from .base import BEComponent

class InputProjector(BEComponent):
    """
    MLP Projector for Qwen embeddings to ModernBERT latent space.
    3584 (Qwen2.5-7B hidden_size) -> 2048 -> 1024.
    """
    def __init__(self, component_id="qwen_to_bert_input", version="v1.0", config=None):
        # DeepSeek-R1-Distill-Qwen-7B is based on Qwen2.5, hidden_size=3584
        input_dim = config.get("input_dim", 3584) if config else 3584
        output_dim = config.get("output_dim", 1024) if config else 1024
        hidden_dim = config.get("hidden_dim", 2048) if config else 2048
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
        
        # μ-VAE heads
        self.mu_head = nn.Linear(output_dim, output_dim)
        self.logvar_head = nn.Linear(output_dim, output_dim)
        
        self._init_weights()

    def _init_weights(self):
        """Стабилизированная инициализация для FP16."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
    @classmethod
    def from_scratch(cls, component_id="qwen_to_bert_input", version="v1.0",
                     weights_path=None, **kwargs):
        """
        Создаёт InputProjector с нуля.
        weights_path: путь к weights.pt; если None — случайная инициализация.
        """
        config = kwargs.get("config", {"input_dim": 3584, "hidden_dim": 2048, "output_dim": 1024})
        instance = cls(component_id=component_id, version=version, config=config)
        instance.load_weights(weights_path)
        return instance
        
    def forward(self, x):
        h = self.proj(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        logvar = torch.clamp(logvar, -4.0, 4.0) # Стабилизация (ADR-011)

        
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
            
        return z, mu, logvar


class FeatureProjector(BEComponent):
    """
    Feature Projector for ModernBERT hidden states to Qwen latent space.
    1024 -> 3584 (Qwen2.5-7B hidden_size). With residual connection.
    """
    def __init__(self, component_id="bert_to_qwen_feature", version="v1.0", config=None):
        input_dim = config.get("input_dim", 1024) if config else 1024
        # DeepSeek-R1-Distill-Qwen-7B is based on Qwen2.5, hidden_size=3584
        output_dim = config.get("output_dim", 3584) if config else 3584
        use_prior_norm = config.get("use_prior_norm", False) if config else False
        
        super().__init__(component_id, version, {
            "input_dim": input_dim, 
            "output_dim": output_dim,
            "use_prior_norm": use_prior_norm
        })
        
        self.use_prior_norm = use_prior_norm
        
        # Если включен prior_norm, мы отключаем bias, чтобы проектор не мог компенсировать 
        # смещение среднего (mean shift) студента своими весами.
        bias = not use_prior_norm
        
        # Linear approximation for residual connection
        self.residual_proj = nn.Linear(input_dim, output_dim, bias=bias)
        
        # Основная ветка MLP
        self.proj_in = nn.Linear(input_dim, input_dim * 2, bias=bias)
        self.proj_out = nn.Linear(input_dim * 2, output_dim, bias=bias)
        self.norm = nn.LayerNorm(output_dim, eps=1e-6)
        
        # Скейлы для балансировки вклада веток.
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.5)
        self.output_scale = nn.Parameter(torch.full((output_dim,), 0.4))
        
        self._init_weights()

    def _init_weights(self):
        """Стабилизированная инициализация для FP16."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
    @classmethod
    def from_scratch(cls, component_id="bert_to_qwen_feature", version="v1.0",
                     weights_path=None, **kwargs):
        """
        Создаёт FeatureProjector с нуля.
        weights_path: путь к weights.pt; если None — случайная инициализация.
        """
        config = kwargs.get("config", {"input_dim": 1024, "output_dim": 3584, "use_prior_norm": False})
        instance = cls(component_id=component_id, version=version, config=config)
        instance.load_weights(weights_path)
        return instance
        
    def forward(self, x):
        # Если включен prior_norm, мы ДОПОЛНИТЕЛЬНО можем добавить статистику в логи (через буферы),
        # но главное — отсутствие bias уже заставляет x центрироваться.
        
        # Проход через MLP
        h = self.proj_in(x)
        h = F.gelu(h)
        out = self.proj_out(h)
        out = self.norm(out) * self.output_scale
        
import torch
import torch.nn.functional as F
import torch.nn as nn
from .base import BEComponent

class InputProjector(BEComponent):
    """
    MLP Projector for Qwen embeddings to ModernBERT latent space.
    3584 (Qwen2.5-7B hidden_size) -> 2048 -> 1024.
    """
    def __init__(self, component_id="qwen_to_bert_input", version="v1.0", config=None):
        # DeepSeek-R1-Distill-Qwen-7B is based on Qwen2.5, hidden_size=3584
        input_dim = config.get("input_dim", 3584) if config else 3584
        output_dim = config.get("output_dim", 1024) if config else 1024
        hidden_dim = config.get("hidden_dim", 2048) if config else 2048
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
        
        # μ-VAE heads
        self.mu_head = nn.Linear(output_dim, output_dim)
        self.logvar_head = nn.Linear(output_dim, output_dim)
        
        self._init_weights()

    def _init_weights(self):
        """Стабилизированная инициализация для FP16."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
    @classmethod
    def from_scratch(cls, component_id="qwen_to_bert_input", version="v1.0",
                     weights_path=None, **kwargs):
        """
        Создаёт InputProjector с нуля.
        weights_path: путь к weights.pt; если None — случайная инициализация.
        """
        config = kwargs.get("config", {"input_dim": 3584, "hidden_dim": 2048, "output_dim": 1024})
        instance = cls(component_id=component_id, version=version, config=config)
        instance.load_weights(weights_path)
        return instance
        
    def forward(self, x):
        h = self.proj(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        logvar = torch.clamp(logvar, -4.0, 4.0) # Стабилизация (ADR-011)

        
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
            
        return z, mu, logvar


class FeatureProjector(BEComponent):
    """
    Feature Projector for ModernBERT hidden states to Qwen latent space.
    1024 -> 3584 (Qwen2.5-7B hidden_size). With residual connection.
    """
    def __init__(self, component_id="bert_to_qwen_feature", version="v1.0", config=None):
        input_dim = config.get("input_dim", 1024) if config else 1024
        # DeepSeek-R1-Distill-Qwen-7B is based on Qwen2.5, hidden_size=3584
        output_dim = config.get("output_dim", 3584) if config else 3584
        use_prior_norm = config.get("use_prior_norm", False) if config else False
        
        super().__init__(component_id, version, {
            "input_dim": input_dim, 
            "output_dim": output_dim,
            "use_prior_norm": use_prior_norm
        })
        
        self.use_prior_norm = use_prior_norm
        
        # Если включен prior_norm, мы отключаем bias, чтобы проектор не мог компенсировать 
        # смещение среднего (mean shift) студента своими весами.
        bias = not use_prior_norm
        
        # Linear approximation for residual connection
        self.residual_proj = nn.Linear(input_dim, output_dim, bias=bias)
        
        # Основная ветка MLP
        self.proj_in = nn.Linear(input_dim, input_dim * 2, bias=bias)
        self.proj_out = nn.Linear(input_dim * 2, output_dim, bias=bias)
        self.norm = nn.LayerNorm(output_dim, eps=1e-6)
        
        # Скейлы для балансировки вклада веток.
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.5)
        self.output_scale = nn.Parameter(torch.full((output_dim,), 0.4))
        
        self._init_weights()

    def _init_weights(self):
        """Стабилизированная инициализация для FP16."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
    @classmethod
    def from_scratch(cls, component_id="bert_to_qwen_feature", version="v1.0",
                     weights_path=None, **kwargs):
        """
        Создаёт FeatureProjector с нуля.
        weights_path: путь к weights.pt; если None — случайная инициализация.
        """
        config = kwargs.get("config", {"input_dim": 1024, "output_dim": 3584, "use_prior_norm": False})
        instance = cls(component_id=component_id, version=version, config=config)
        instance.load_weights(weights_path)
        return instance
        
    def forward(self, x):
        # Если включен prior_norm, мы ДОПОЛНИТЕЛЬНО можем добавить статистику в логи (через буферы),
        # но главное — отсутствие bias уже заставляет x центрироваться.
        
        # Проход через MLP
        h = self.proj_in(x)
        h = F.gelu(h)
        out = self.proj_out(h)
        out = self.norm(out) * self.output_scale
        
        # Проход через Residual
        res = self.residual_proj(x) * self.residual_scale
        
        return out + res


class OutputProjector(BEComponent):
    """
    Projector from ModernBERT latent space (L40) back to L0 diffusion space.
    Architecture: Linear(1024->2048) -> GELU -> Linear(2048->1024) [-> GELU -> Linear(1024->1024)]
    Без LayerNorm на входе, чтобы сохранить сигнал norm_cv (0.089).
    На выходе LayerNorm для фиксации нормы.
    """
    def __init__(self, component_id="bert_to_diffusion_output", version="v1.0", config=None):
        input_dim = config.get("input_dim", 1024) if config else 1024
        hidden_dim = config.get("hidden_dim", 2048) if config else 2048
        output_dim = config.get("output_dim", 1024) if config else 1024
        num_layers = config.get("num_layers", 2) if config else 2
        
        super().__init__(component_id, version, {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "num_layers": num_layers
        })
        
        layers = []
        if num_layers == 1:
            # Чисто линейное преобразование (Affine)
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.LayerNorm(output_dim, eps=1e-6))
        elif num_layers == 2:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
            # Слой 2 (финальный)
            layers.append(nn.Linear(hidden_dim, output_dim))
            layers.append(nn.LayerNorm(output_dim, eps=1e-6))
        elif num_layers == 3:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
            # Слой 2
            layers.append(nn.Linear(hidden_dim, output_dim))
            layers.append(nn.GELU())
            # Слой 3 (финальный)
            layers.append(nn.Linear(output_dim, output_dim))
            layers.append(nn.LayerNorm(output_dim, eps=1e-6))
        else:
            raise ValueError(f"num_layers={num_layers} не поддерживается (только 1, 2 или 3).")
            
        self.proj = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Стабилизированная инициализация."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                    
    @classmethod
    def from_scratch(cls, component_id="bert_to_diffusion_output", version="v1.0",
                     weights_path=None, **kwargs):
        """Создаёт OutputProjector с нуля."""
        config = kwargs.get("config", {"input_dim": 1024, "hidden_dim": 2048, "output_dim": 1024, "num_layers": 2})
        instance = cls(component_id=component_id, version=version, config=config)
        instance.load_weights(weights_path)
        return instance
        
    def forward(self, x):
        return self.proj(x)
