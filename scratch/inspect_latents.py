import os
import sys
import torch

PROJECT_ROOT = 'C:/Experiments/BEBLaDII'
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'experiments', 'phase 4'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'experiments', 'phase 5'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from transformers import AutoTokenizer

import importlib.util
spec = importlib.util.spec_from_file_location('eval_module', 'C:/Experiments/BEBLaDII/experiments/phase 6/evaluate_phase6_metrics.py')
eval_module = importlib.util.module_from_spec(spec)
sys.modules['eval_module'] = eval_module
spec.loader.exec_module(eval_module)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = {
    'qwen_path': 'Qwen/Qwen2.5-1.5B',
    'modernbert_path': 'answerdotai/ModernBERT-large',
    'encoder_path': os.path.join(PROJECT_ROOT, 'experiments', 'phase 1', 'planB_phase1_checkpoints_phase1_vae_step_20000.pth'),
    'phase4_path': os.path.join(PROJECT_ROOT, 'experiments', 'phase 4', 'local_checkpoints', 'phase4_step_85995.pth'),
    'sep_token_path': os.path.join(PROJECT_ROOT, 'storage', 'components', 'sep_token.pt'),
    'phase6_ckpt': 'experiments/phase 6/checkpoints/planB_phase6_checkpoints_phase6_ca_layers_step_3000.pth'
}

tokenizer = AutoTokenizer.from_pretrained(config['qwen_path'])
model = eval_module.BEBLaDIIPhase6Eval(config).to(device)
latent_dict = torch.load(os.path.join(PROJECT_ROOT, 'experiments', 'phase 5', 'local', 'latent_dict.pt'), map_location=device)

query_text = 'Explain why the sky is blue using simple words.'
q_enc = tokenizer(query_text, return_tensors='pt').to(device)

with torch.no_grad():
    Z_prompt, _, _ = model.encoder(model.qwen_embeddings(q_enc.input_ids))
    Z_prompt = eval_module.safe_normalize(Z_prompt.float(), dim=-1)
    
    B, T_a = 1, 64
    z_canvas = eval_module.safe_normalize(torch.randn(B, T_a, 1024, device=device), dim=-1)
    
    steps = 25
    dt = 1.0 / steps
    
    for i in range(steps):
        t_val = 1.0 - i * dt
        t_actual = torch.full((B, T_a), t_val, device=device)
        sims = torch.matmul(z_canvas, latent_dict.T)
        raw_dprox, w1_ids = sims.max(dim=-1)
        t_reported = (1.0 - raw_dprox).clamp(0.0, 1.0)
        
        h39, z_pred = model.forward_step(z_canvas, Z_prompt, q_enc.attention_mask, t_actual, t_reported)
        
        t_next = torch.full((B, T_a), max(0.0, t_val - dt), device=device)
        import math
        theta_now = t_actual.unsqueeze(-1) * (math.pi / 2)
        theta_next = t_next.unsqueeze(-1) * (math.pi / 2)
        sin_now = torch.sin(theta_now)
        sin_now = torch.where(sin_now < 1e-5, torch.ones_like(sin_now) * 1e-5, sin_now)
        w_pred = torch.sin(theta_now - theta_next) / sin_now
        w_cur = torch.sin(theta_next) / sin_now
        z_next = w_pred * z_pred + w_cur * z_canvas
        z_canvas = torch.where(theta_now > 1e-5, eval_module.safe_normalize(z_next, dim=-1), eval_module.safe_normalize(z_pred, dim=-1))

    print(f'Final z_canvas shape: {z_canvas.shape}')
    
    if hasattr(model, 'void_embed'):
        void_vec = model.void_embed.to(device)
    else:
        void_vec = torch.load(os.path.join(PROJECT_ROOT, 'storage', 'components', 'void_token.pt'), map_location=device).float()
        
    void_sims = torch.matmul(z_canvas[0], void_vec)
    print(f'Cos sim to <|void|>: min={void_sims.min().item():.4f}, max={void_sims.max().item():.4f}, mean={void_sims.mean().item():.4f}')
    
    normalized_z = eval_module.safe_normalize(z_canvas[0], dim=-1)
    pairwise_sim = torch.matmul(normalized_z, normalized_z.T)
    mask = ~torch.eye(T_a, dtype=torch.bool, device=device)
    mean_pairwise = pairwise_sim[mask].mean().item()
    print(f'Mean pairwise cos sim (diversity): {mean_pairwise:.4f} (1.0 = total collapse)')
    
    variance = z_canvas[0].var(dim=0).mean().item()
    print(f'Mean variance per dimension: {variance:.6f}')

