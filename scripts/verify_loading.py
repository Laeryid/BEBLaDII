import torch
import torch.nn as nn
import sys, os
sys.path.append(os.getcwd())

from src.beb_la_dii.model.dus import DUSModel
from src.beb_la_dii.model.projectors import InputProjector, FeatureProjector

student = DUSModel()
ip = InputProjector()
fp = nn.ModuleDict({
    '20': FeatureProjector(component_id='feat_proj_20'), 
    '30': FeatureProjector(component_id='feat_proj_30'), 
    '40': FeatureProjector(component_id='feat_proj_40')
})

class MockDist(nn.Module):
    def __init__(self, s, i, f):
        super().__init__()
        self.teacher = nn.Linear(1,1)
        self.student = s
        self.input_projector = i
        self.feature_projectors = f

distiller = MockDist(student, ip, fp)

# Замеряем норму ДО загрузки (случайная инициализация)
norm_before = torch.norm(distiller.feature_projectors['40'].proj[2].weight.float()).item()
print(f'Norm before: {norm_before}')

path_latest = r'c:\Experiments\BEBLaDII\storage\experiments\20260504 Phase + Reasoning 14500 steps\checkpoints_latest_checkpoint.pt'
ckpt = torch.load(path_latest, map_location='cpu')
cleaned = {k.replace('_orig_module.', ''): v for k, v in ckpt['model_state_dict'].items() if 'teacher' not in k}

incomp = distiller.load_state_dict(cleaned, strict=False)

# Замеряем норму ПОСЛЕ загрузки
norm_after = torch.norm(distiller.feature_projectors['40'].proj[2].weight.float()).item()
print(f'Norm after: {norm_after}')

if abs(norm_after - 34.51) < 0.1:
    print('SUCCESS: Weights loaded correctly!')
else:
    print('FAILURE: Weights NOT loaded!')
