import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """
    Расчет L_total для выравнивания латентных пространств.
    Использует MSE и Cosine Similarity для трех контрольных точек (слои 20, 30, 40).
    """
    def __init__(self, layer_weights={20: 0.5, 30: 0.7, 40: 1.0}):
        super().__init__()
        self.layer_weights = layer_weights
        self.mse = nn.MSELoss()
        
    def forward(self, student_hidden_states, teacher_hidden_states):
        """
        student_hidden_states: dict {layer_idx: tensor}
        teacher_hidden_states: dict {layer_idx: tensor}
        """
        total_loss = 0.0
        
        for layer_idx, weight in self.layer_weights.items():
            if layer_idx in student_hidden_states and layer_idx in teacher_hidden_states:
                s_h = student_hidden_states[layer_idx]
                t_h = teacher_hidden_states[layer_idx]
                
                # 1. MSE Loss
                mse_l = self.mse(s_h, t_h)
                
                # 2. Cosine Similarity Loss (1 - mean(cos_sim))
                cos_sim = F.cosine_similarity(s_h, t_h, dim=-1)
                cos_l = 1.0 - cos_sim.mean()
                
                # Комбинированный лосс для слоя
                layer_l = mse_l + cos_l
                total_loss += weight * layer_l
                
        return total_loss

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
    
    loss = criterion(s_states, t_states)
    print(f"Calculated loss: {loss.item()}")
