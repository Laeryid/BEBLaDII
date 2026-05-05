import torch
import os

ckpt_path = "latest_checkpoint.pt"
if not os.path.exists(ckpt_path):
    print(f"File {ckpt_path} not found.")
else:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt.get('model_state_dict', {})
    
    # Ищем ключи слоев
    layer_keys = [k for k in sd.keys() if "student.model.layers" in k]
    
    # Группируем по номеру слоя
    layers = set()
    for k in layer_keys:
        parts = k.split('.')
        # Ожидаемый формат: student.model.layers.N.xxx
        try:
            idx = parts.index("layers")
            layers.add(int(parts[idx+1]))
        except:
            pass
            
    if not layers:
        print("No student layers found in checkpoint.")
    else:
        print(f"Found {len(layers)} student layers in checkpoint.")
        print(f"Min layer index: {min(layers)}")
        print(f"Max layer index: {max(layers)}")
        print(f"Layer indices (first 5): {sorted(list(layers))[:5]}")
        print(f"Layer indices (last 5): {sorted(list(layers))[-5:]}")
        
    if 'global_step' in ckpt:
        print(f"Global step: {ckpt['global_step']}")
