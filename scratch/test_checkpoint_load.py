import os
import sys
import importlib.util
import torch

sys.path.insert(0, '.agents/skills/tpu-script-crafting/scripts/fake_torch_xla')
sys.path.insert(0, '.')

spec = importlib.util.spec_from_file_location('nb_module', 'experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.py')
mod = importlib.util.module_from_spec(spec)
sys.modules['nb_module'] = mod
spec.loader.exec_module(mod)

ckpt_path = 'experiments/phase 6/checkpoints/planB_phase6_checkpoints_phase6_ca_layers_step_1000.pth'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

cfg = mod.Config()
cfg.local_files_only = False
model = mod.BEBLaDIIPhase6(cfg)

for key in list(model.ca_layers.keys()):
    layer_idx = int(key) - 1
    model.dus.layers[layer_idx].ca_layer = model.ca_layers[key]

# Чистим _orig_module
clean_ckpt = {}
for k, v in ckpt.items():
    clean_k = k.replace('._orig_module.', '.')
    clean_ckpt[clean_k] = v

res_raw = model.load_state_dict(ckpt, strict=False)
res_clean = model.load_state_dict(clean_ckpt, strict=False)

print(f"Total keys in checkpoint: {len(ckpt)}")
print(f"Raw load matched: {len(ckpt) - len(res_raw.unexpected_keys)} / {len(ckpt)}")
print(f"Clean load matched: {len(clean_ckpt) - len(res_clean.unexpected_keys)} / {len(clean_ckpt)}")
print(f"Unexpected in clean load: {res_clean.unexpected_keys}")
