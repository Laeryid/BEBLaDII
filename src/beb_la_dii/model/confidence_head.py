import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LocalSelfAttention(nn.Module):
    """
    Слой локального внимания с окном window_size токенов.
    
    Работает в пониженной размерности (attn_dim < input_dim) для экономии параметров.
    Каждый токен видит только W/2 соседей с каждой стороны.
    """
    def __init__(self, input_dim: int = 1024, attn_dim: int = 256, num_heads: int = 4, window_size: int = 32):
        super().__init__()
        assert attn_dim % num_heads == 0, "attn_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        self.attn_dim = attn_dim

        # Проекция вниз перед вниманием
        self.proj_in  = nn.Linear(input_dim, attn_dim, bias=False)
        self.qkv      = nn.Linear(attn_dim, attn_dim * 3, bias=False)
        self.proj_out = nn.Linear(attn_dim, attn_dim, bias=False)
        self.norm     = nn.LayerNorm(attn_dim)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, input_dim]
            attention_mask: [B, T] — 1=реальный токен, 0=PAD
        Returns:
            context: [B, T, attn_dim]
        """
        B, T, _ = x.shape
        H = self.num_heads
        Hd = self.head_dim
        W = self.window_size

        # [B, T, attn_dim]
        x_ = self.proj_in(x)

        qkv = self.qkv(x_)  # [B, T, 3*attn_dim]
        q, k, v = qkv.chunk(3, dim=-1)

        # [B, H, T, Hd]
        q = q.view(B, T, H, Hd).transpose(1, 2)
        k = k.view(B, T, H, Hd).transpose(1, 2)
        v = v.view(B, T, H, Hd).transpose(1, 2)

        # Скоры внимания [B, H, T, T]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Маска локального окна: токен i видит только токены в [i - W//2, i + W//2]
        positions = torch.arange(T, device=x.device)
        dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()  # [T, T]
        local_mask = (dist > W // 2).unsqueeze(0).unsqueeze(0)          # [1, 1, T, T]
        attn = attn.masked_fill(local_mask, float('-inf'))

        # PAD-маска
        if attention_mask is not None:
            pad_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            attn = attn.masked_fill(pad_mask, float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # безопасно при строках из -inf

        context = torch.matmul(attn, v)                              # [B, H, T, Hd]
        context = context.transpose(1, 2).contiguous().view(B, T, -1)  # [B, T, attn_dim]
        context = self.proj_out(context)
        context = self.norm(context)

        return context


class ConfidenceHead(nn.Module):
    """
    Голова Уверенности (Phase 5).

    Архитектура:
        Входы: h39(1024), dus_final(1024), conf_prev(5), t_reported(1)
        
        1. Слияние для внимания:
           attn_in = Linear(2054 -> 1024)
        2. LocalSelfAttention(window=32) -> context [B, T, 256]
        3. Residual Concat:
           concat(h39, dus_final, conf_prev, t_reported, context) = [B, T, 2310]
        4. MLP -> [B, T, 5] (Sigmoid)
    """
    def __init__(
        self,
        h39_dim: int = 1024,
        attn_dim: int = 256,
        num_heads: int = 4,
        window_size: int = 32,
        conf_dim: int = 6,
        mlp_hidden: int = 512,
    ):
        super().__init__()
        
        # Общая размерность всех сырых входов
        self.raw_dim = h39_dim * 2 + conf_dim + 1  # 1024 + 1024 + 6 + 1 = 2055
        
        # Сжимаем перед вниманием, чтобы токен делился всем своим состоянием с соседями
        self.attn_fusion = nn.Linear(self.raw_dim, h39_dim)
        
        self.local_attn = LocalSelfAttention(
            input_dim=h39_dim,
            attn_dim=attn_dim,
            num_heads=num_heads,
            window_size=window_size,
        )

        # Конкатенация всех сырых входов (residual) + контекст от внимания
        mlp_in = self.raw_dim + attn_dim

        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, mlp_hidden),
            nn.GELU(),
            nn.LayerNorm(mlp_hidden),
            nn.Linear(mlp_hidden, 128),
            nn.GELU(),
            nn.Linear(128, conf_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        h39_raw: torch.Tensor,
        dus_final: torch.Tensor,
        conf_prev: torch.Tensor,
        t_reported: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            h39_raw:       [B, T, 1024] — выход DUS до гейта (ненормализованный, сохраняет амплитуду Variance Collapse)
            dus_final:     [B, T, 1024] — выход DUS после проекций (предсказание x0)
            conf_prev:     [B, T, 6]    — предыдущий выход головы
            t_reported:    [B, T, 1]    — уровень шума
            attention_mask:[B, T]       — 1=реальный, 0=PAD
        Returns:
            signals: [B, T, 6]
        """
        # Объединяем все входы в единый сырой вектор
        raw_features = torch.cat([h39_raw, dus_final, conf_prev, t_reported], dim=-1)
        
        # Проекция для слоя внимания (делимся всем состоянием с соседями)
        attn_in = self.attn_fusion(raw_features)
        
        # Локальный контекст
        context = self.local_attn(attn_in, attention_mask)  # [B, T, 256]

        # Прямые (residual) связи сырых фичей + контекст от соседей
        x = torch.cat([raw_features, context], dim=-1)  # [B, T, 2310]

        return self.mlp(x)  # [B, T, 5]
