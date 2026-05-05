import torch
import torch.nn as nn
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
ckpt = torch.load('/tmp/latest_checkpoint.pt', map_location='cpu')
cleaned = {k.replace('_orig_module.', ''): v for k, v in ckpt['model_state_dict'].items() if 'teacher' not in k}
incomp = distiller.load_state_dict(cleaned, strict=False)
print('Missing student keys:', [k for k in incomp.missing_keys if 'teacher' not in k])
