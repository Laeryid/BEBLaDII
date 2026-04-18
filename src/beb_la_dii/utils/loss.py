import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """
    Расчет L_total для выравнивания латентных пространств.
    Использует MSE и Cosine Similarity для трех контрольных точек (слои 20, 30, 40).
    """
    def __init__(self, layer_weights={20: 0.5, 30: 0.7, 40: 1.0}, mse_weight=1.0, cos_weight=1.0):
        super().__init__()
        self.layer_weights = layer_weights
        self.mse_weight = mse_weight
        self.cos_weight = cos_weight
        self.mse = nn.MSELoss()
        
    def forward(self, student_hidden_states, teacher_hidden_states, attention_mask=None, mu=None, logvar=None, beta=0.0):
        """
        student_hidden_states: dict {layer_idx: tensor}
        teacher_hidden_states: dict {layer_idx: tensor}
        attention_mask: Tensor (B, T) - 1 для значимых токенов, 0 для паддинга
        """
        total_loss = 0.0
        mse_total = 0.0
        cos_total = 0.0
        metrics = {}
        
        # Подготовка маски
        if attention_mask is not None:
            # (B, T) -> (B, T, 1)
            mask = attention_mask.unsqueeze(-1).float()
            num_active_elements = mask.sum()
        else:
            mask = 1.0
            num_active_elements = None

        for layer_idx, weight in self.layer_weights.items():
            if layer_idx in student_hidden_states and layer_idx in teacher_hidden_states:
                s_h = student_hidden_states[layer_idx].float()
                t_h = teacher_hidden_states[layer_idx].float()
                
                if attention_mask is not None:
                    # 1. MSE Loss (Masked)
                    diff = (s_h - t_h) ** 2
                    # Деление на (кол-во активных токенов * размер скрытого слоя)
                    mse_l = (diff * mask).sum() / (num_active_elements * s_h.size(-1) + 1e-6)
                    
                    # 2. Cosine Similarity Loss (Masked)
                    cos_sim = F.cosine_similarity(s_h, t_h, dim=-1, eps=1e-6)
                    cos_l = 1.0 - (cos_sim * attention_mask).sum() / (attention_mask.sum() + 1e-6)
                else:
                    # 1. MSE Loss
                    mse_l = self.mse(s_h, t_h)
                    
                    # 2. Cosine Similarity Loss
                    cos_sim = F.cosine_similarity(s_h, t_h, dim=-1, eps=1e-6)
                    cos_l = 1.0 - cos_sim.mean()
                
                # Накопление по компонентам
                mse_total += weight * mse_l
                cos_total += weight * cos_l
                
                # Послойная детализация
                metrics[f"l{layer_idx}_mse"] = mse_l.item()
                metrics[f"l{layer_idx}_cos"] = cos_l.item()
                
                # Комбинированный лосс для слоя с учетом балансировки
                layer_l = self.mse_weight * mse_l + self.cos_weight * cos_l
                total_loss += weight * layer_l
        
        metrics["mse"] = mse_total.item() if torch.is_tensor(mse_total) else mse_total
        metrics["cosine"] = cos_total.item() if torch.is_tensor(cos_total) else cos_total
        
        # Вычисление KL-Divergence, если переданы параметры головки
        if mu is not None and logvar is not None:
            kl_loss_raw = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
            if attention_mask is not None:
                kl_loss = (kl_loss_raw * attention_mask).sum() / (attention_mask.sum() + 1e-6)
            else:
                kl_loss = kl_loss_raw.mean()
                
            total_loss = total_loss + beta * kl_loss
            metrics["kl"] = kl_loss.item()
                
        return total_loss, metrics

if __name__ == "__main__":
    # Тест
    criterion = DistillationLoss()
    s_states = {
        20: torch.randn(2, 5, 4096),
        30: torch.randn(2, 5, 4096),
        40: torch.randn(2, 5, 4096)
    }
    # Моделируем идеальное совпадение для одного слоя
    t_states = {
        20: s_states[20].clone(), 
        30: torch.randn(2, 5, 4096),
        40: torch.randn(2, 5, 4096)
    }
    
    loss, metrics = criterion(s_states, t_states)
    print(f"Calculated loss: {loss.item()}")
    print(f"Metrics: {metrics}")
