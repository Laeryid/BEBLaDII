import torch
import torch.nn as nn
import torch.nn.functional as F
from src.beb_la_dii.model.projectors import FeatureProjector
from src.beb_la_dii.utils.loss import DistillationLoss

# Инициализируем компоненты
proj = FeatureProjector()
criterion = DistillationLoss()

# Входные данные
x = torch.randn(2, 10, 1024)
teacher_h = torch.randn(2, 10, 3584)
mask = torch.ones(2, 10)

# Forward pass
student_h = proj(x)
loss, metrics = criterion({20: student_h}, {20: teacher_h}, attention_mask=mask)

print("Loss:", loss.item())

# Backward pass
loss.backward()

print("residual_scale grad:", proj.residual_scale.grad)
print("output_scale grad (mean):", proj.output_scale.grad.abs().mean().item() if proj.output_scale.grad is not None else None)
