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
        
    def forward(self, student_hidden_states, teacher_hidden_states, attention_mask=None, 
                mu=None, logvar=None, beta=0.0, raw_student_states=None, lambda_prior=0.1):
        """
        student_hidden_states: dict {layer_idx: tensor} (projected)
        teacher_hidden_states: dict {layer_idx: tensor}
        attention_mask: Tensor (B, T)
        raw_student_states: dict {layer_idx: tensor} (unprojected/raw)
        lambda_prior: вес регуляризации N(0,1) для l40
        """
        total_loss = 0.0
        mse_total = 0.0
        cos_total = 0.0
        metrics = {}
        
        # Подготовка маски
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            num_active_elements = mask.sum()
        else:
            mask = 1.0
            num_active_elements = None

        for layer_idx, weight in self.layer_weights.items():
            if layer_idx in student_hidden_states and layer_idx in teacher_hidden_states:
                s_h = student_hidden_states[layer_idx].float()
                t_h = teacher_hidden_states[layer_idx].float()
                
                # 1. Cosine Similarity Loss (Всегда считаем)
                cos_sim = F.cosine_similarity(s_h, t_h, dim=-1, eps=1e-6)
                if attention_mask is not None:
                    cos_l = 1.0 - (cos_sim * attention_mask).sum() / (attention_mask.sum() + 1e-6)
                else:
                    cos_l = 1.0 - cos_sim.mean()
                
                # 2. MSE Loss (Для l40 НЕ считаем, так как там no-bias проектор и важна только семантика)
                if layer_idx == 40:
                    mse_l = torch.tensor(0.0, device=s_h.device)
                    layer_l = self.cos_weight * cos_l
                else:
                    if attention_mask is not None:
                        diff = (s_h - t_h) ** 2
                        mse_l = (diff * mask).sum() / (num_active_elements * s_h.size(-1) + 1e-6)
                    else:
                        mse_l = self.mse(s_h, t_h)
                    layer_l = self.mse_weight * mse_l + self.cos_weight * cos_l
                
                # Накопление по компонентам
                mse_total += weight * mse_l
                cos_total += weight * cos_l
                
                # Послойная детализация
                metrics[f"l{layer_idx}_mse"] = mse_l.detach()
                metrics[f"l{layer_idx}_cos"] = cos_l.detach()
                
                total_loss += weight * layer_l
        
        # 3. Prior Loss для l40 (Принуждение к N(0,1) напрямую в скрытом пространстве)
        if raw_student_states is not None and 40 in raw_student_states:
            raw_40 = raw_student_states[40].float()
            # Считаем среднее и дисперсию по батчу и токенам
            # Мы хотим, чтобы среднее было 0, а дисперсия 1.
            m_40 = raw_40.mean(dim=(0, 1))
            s_40 = raw_40.std(dim=(0, 1))
            
            prior_loss = m_40.pow(2).mean() + (s_40 - 1.0).pow(2).mean()
            total_loss += lambda_prior * prior_loss
            metrics["l40_prior"] = prior_loss.detach()

        metrics["mse"] = mse_total.detach() if torch.is_tensor(mse_total) else mse_total
        metrics["cosine"] = cos_total.detach() if torch.is_tensor(cos_total) else cos_total
        
        # Вычисление KL-Divergence для InputProjector
        if mu is not None and logvar is not None:
            mu_f = mu.float()
            logvar_f = logvar.float()
            kl_loss_raw = -0.5 * torch.mean(1 + logvar_f - mu_f.pow(2) - logvar_f.exp(), dim=-1)
            
            FREE_BITS = 0.5
            kl_loss_clamped = torch.clamp(kl_loss_raw, min=FREE_BITS)
            
            if attention_mask is not None:
                kl_loss = (kl_loss_clamped * attention_mask).sum() / (attention_mask.sum() + 1e-6)
            else:
                kl_loss = kl_loss_clamped.mean()
                
            total_loss = total_loss + beta * kl_loss
            metrics["kl"] = kl_loss.detach()
            metrics["mu_norm"] = mu_f.norm(dim=-1).mean().detach()
            metrics["logvar_mean"] = logvar_f.mean().detach()
                
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
