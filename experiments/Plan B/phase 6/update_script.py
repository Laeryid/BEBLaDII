import re

with open('C:/Experiments/BEBLaDII/experiments/phase 6/kaggle/train_phase6_notebook.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Fix the void penalty logic
old_loss_block = '''    # --- NEW: Void Penalty ---
    cos_sim_to_void = (dus_final * void_embed.view(1, 1, -1)).sum(dim=-1)
    content_mask = 1.0 - void_mask
    if void_margin > 0.0:
        import torch.nn.functional as F
        void_penalty = F.relu(cos_sim_to_void - void_margin) * content_mask
        loss_el = loss_el + 0.5 * void_penalty
    # -------------------------'''

new_loss_block = '''    # --- NEW: Dynamic Void Penalty ---
    cos_sim_to_void = (dus_final * void_embed.view(1, 1, -1)).sum(dim=-1)
    cos_sim_target_to_void = (target * void_embed.view(1, 1, -1)).sum(dim=-1)
    content_mask = 1.0 - void_mask
    
    # 80% от расстояния между void и z_clean (в терминах косинусного сходства)
    dynamic_margin = 1.0 - 0.8 * (1.0 - cos_sim_target_to_void)
    
    import torch.nn.functional as F
    void_penalty = F.relu(cos_sim_to_void - dynamic_margin) * content_mask
    loss_el = loss_el + 0.5 * void_penalty
    # -------------------------'''

content = content.replace(old_loss_block, new_loss_block)

# 2. Fix the logging
content = re.sub(
    r'"gate_12":.*?\n.*?"gate_24":.*?\n.*?"gate_36":.*?,',
    '"out_proj_12_amp": model.ca_layers["12"].out_proj.weight.data.abs().mean().item(),\n                    "out_proj_36_amp": model.ca_layers["36"].out_proj.weight.data.abs().mean().item(),\n                    "q_proj_36_amp": model.ca_layers["36"].q_proj.weight.data.abs().mean().item(),',
    content
)

with open('C:/Experiments/BEBLaDII/experiments/phase 6/kaggle/train_phase6_notebook.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Success!')
