import torch
import torch.nn as nn
from typing import Dict

from .confidence_head import ConfidenceHead

class SensorEnsemble(nn.Module):
    """
    Ансамбль сенсоров для оценки состояния диффузионного процесса (Phase 5/6).
    Включает в себя обучаемую Голову Уверенности (Confidence Head) и 
    алгоритмические геометрические метрики (RawDProx, Delta, ConflictSim).
    """
    def __init__(self, conf_head_kwargs=None):
        super().__init__()
        if conf_head_kwargs is None:
            conf_head_kwargs = {
                'h39_dim': 1024,
                'attn_dim': 256,
                'num_heads': 4,
                'window_size': 32,
                'conf_dim': 5,
                'mlp_hidden': 512
            }
        self.confidence_head = ConfidenceHead(**conf_head_kwargs)

    def forward(self, 
                h39_raw: torch.Tensor, 
                dus_final: torch.Tensor, 
                conf_prev: torch.Tensor, 
                t_reported: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Прогон только через ConfidenceHead для получения обучаемых метрик:
        dict_proximity, model_confidence, token_coherence, token_complexity, crystallization, stuck
        """
        return self.confidence_head(h39_raw, dus_final, conf_prev, t_reported, attention_mask)

    def compute_geometric_metrics(self, z_pred: torch.Tensor, latent_dict: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Вычисляет геометрические метрики (RawDProx, Delta, ConflictSim) для батча токенов.
        Эти метрики используются Оркестратором для алгоритмического разрешения многозначности.
        
        Args:
            z_pred: [B, T, D] - нормализованные предсказания от DUS (например, x0).
            latent_dict: [VocabSize, D] - нормализованные векторы латентного словаря.
            
        Returns:
            Словарь с метриками:
                raw_d_prox: [B, T]
                delta: [B, T]
                conflict_sim: [B, T]
                top1_idx: [B, T] - индексы ближайших токенов (для отладки)
                top2_idx: [B, T] - индексы вторых по близости токенов
        """
        # [B, T, VocabSize]
        sims = torch.matmul(z_pred, latent_dict.T)
        
        # Берем Top-3 (нам нужны только Top-2, но 3 полезно для дебага/запаса)
        topk_sims, topk_idx = torch.topk(sims, k=3, dim=-1)
        
        # [B, T]
        s1 = topk_sims[..., 0]
        s2 = topk_sims[..., 1]
        
        delta = s1 - s2
        
        idx1 = topk_idx[..., 0]
        idx2 = topk_idx[..., 1]
        
        # Извлекаем векторы из словаря: [B, T, D]
        vec1 = latent_dict[idx1]
        vec2 = latent_dict[idx2]
        
        # Косинусное сходство между самими векторами кандидатов (уже нормализованы)
        # [B, T]
        conf_sim = (vec1 * vec2).sum(dim=-1)
        
        return {
            "raw_d_prox": s1,
            "delta": delta,
            "conflict_sim": conf_sim,
            "top1_idx": idx1,
            "top2_idx": idx2
        }
