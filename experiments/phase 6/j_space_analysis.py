import os
import sys
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

sys.stdout.reconfigure(encoding='utf-8')
PROJECT_ROOT = "C:/Experiments/BEBLaDII"
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments", "phase 6"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments", "phase 4"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments", "phase 5"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from beb_la_dii.utils.loss import safe_normalize
from evaluate_phase6_metrics import BEBLaDIIPhase6Eval

def get_latent_dict(device):
    dict_path = os.path.join(PROJECT_ROOT, "experiments", "phase 5", "local", "latent_dict.pt")
    return torch.load(dict_path, map_location=device)

def get_void_vector(device):
    void_path = os.path.join(PROJECT_ROOT, "storage", "components", "void_token.pt")
    return torch.load(void_path, map_location=device).float()

def analyze_j_space(config, model, tokenizer, latent_dict, void_vector, sample, device):
    print(f"\nAnalyzing sample Q: {sample['Q']}")
    print(f"Target A: {sample['A'][:100]}...")

    q_enc = tokenizer(sample['Q'], return_tensors='pt').to(device)
    a_enc = tokenizer(sample['A'], return_tensors='pt', max_length=512, truncation=True).to(device)
    
    # 1. Получаем истинные таргеты от энкодера
    with torch.no_grad():
        Z_prompt, _, _ = model.encoder(model.qwen_embeddings(q_enc.input_ids))
        Z_prompt = safe_normalize(Z_prompt.float(), dim=-1)
        
        Z_target, _, _ = model.encoder(model.qwen_embeddings(a_enc.input_ids))
        Z_target = safe_normalize(Z_target.float(), dim=-1)
        
    B = 1
    T_a = Z_target.shape[1]
    
    # Подготавливаем canvas - берём чистый Z_target (вместо шума)
    # Это покажет, как модель реагирует на ИДЕАЛЬНЫЙ вход (t=0.0)
    z_canvas = Z_target.clone()
    t_actual = torch.full((B, T_a), 0.0, device=device)
    
    # Считаем dprox для идеального таргета (t=0.0, dprox=1.0, t_reported=0.0)
    sims = torch.matmul(z_canvas, latent_dict.T)
    raw_dprox, _ = sims.max(dim=-1)
    t_reported = (1.0 - raw_dprox).clamp(0.0, 1.0)
    
    # 2. Оцениваем выход модели
    with torch.no_grad():
        # Сначала прогоняем с CA-слоями
        h39, z_pred = model.forward_step(z_canvas, Z_prompt, q_enc.attention_mask, t_actual, t_reported)
        
        # Теперь выключаем gates, чтобы посмотреть разницу (влияние CA)
        original_gates = {}
        for k, ca in model.ca_layers.items():
            original_gates[k] = ca.gate.item()
            ca.gate.data = torch.zeros_like(ca.gate.data)
            
        h39_no_ca, z_pred_no_ca = model.forward_step(z_canvas, Z_prompt, q_enc.attention_mask, t_actual, t_reported)
        
        # Возвращаем gates
        for k, ca in model.ca_layers.items():
            ca.gate.data = torch.tensor([original_gates[k]], device=device)
            
    # 3. Аналитика
    # Куда толкает модель?
    cos_to_target = F.cosine_similarity(z_pred, Z_target, dim=-1)
    cos_to_void = F.cosine_similarity(z_pred, void_vector.unsqueeze(0).unsqueeze(0), dim=-1)
    
    cos_to_target_no_ca = F.cosine_similarity(z_pred_no_ca, Z_target, dim=-1)
    cos_to_void_no_ca = F.cosine_similarity(z_pred_no_ca, void_vector.unsqueeze(0).unsqueeze(0), dim=-1)
    
    # Дельты (как модель сдвигает идеальный вектор?)
    delta = z_pred - Z_target
    delta_norm = torch.norm(delta, dim=-1)
    
    delta_no_ca = z_pred_no_ca - Z_target
    delta_norm_no_ca = torch.norm(delta_no_ca, dim=-1)
    
    # Векторное влияние CA
    ca_effect = z_pred - z_pred_no_ca
    ca_effect_norm = torch.norm(ca_effect, dim=-1)
    
    # Направление сдвига
    # Проекция сдвига на направление к таргету (положительная - толкает к таргету, отрицательная - от него)
    push_to_target = F.cosine_similarity(delta, Z_target, dim=-1) 
    push_to_void = F.cosine_similarity(delta, void_vector.unsqueeze(0).unsqueeze(0), dim=-1)
    
    print(f"\n--- Analysis at t=0.0 (Clean Target Input) ---")
    print(f"Gates: {original_gates}")
    print(f"Mean Cosine to Target (w/ CA): {cos_to_target.mean().item():.4f}")
    print(f"Mean Cosine to Target (w/o CA): {cos_to_target_no_ca.mean().item():.4f}")
    print(f"Mean Cosine to Void (w/ CA): {cos_to_void.mean().item():.4f}")
    print(f"Mean Cosine to Void (w/o CA): {cos_to_void_no_ca.mean().item():.4f}")
    print(f"Mean Delta Norm (w/ CA): {delta_norm.mean().item():.4f}")
    print(f"Mean CA Effect Norm (z_pred - z_pred_no_ca): {ca_effect_norm.mean().item():.4f}")
    print(f"Mean Shift Projection to Target: {push_to_target.mean().item():.4f}")
    print(f"Mean Shift Projection to Void: {push_to_void.mean().item():.4f}")
    
    # Проанализируем один конкретный токен (например, индекс 5, если есть)
    idx = min(5, T_a - 1)
    print(f"\nToken {idx} ('{tokenizer.decode(a_enc.input_ids[0, idx])}'):")
    print(f"  Cos to Target: {cos_to_target[0, idx].item():.4f}")
    print(f"  Cos to Void: {cos_to_void[0, idx].item():.4f}")
    print(f"  CA Effect Norm: {ca_effect_norm[0, idx].item():.4f}")
    print(f"  Shift Proj to Target: {push_to_target[0, idx].item():.4f}")
    print(f"  Shift Proj to Void: {push_to_void[0, idx].item():.4f}")
    

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Analysis on {device}...")
    
    config = {
        'qwen_path': "Qwen/Qwen2.5-1.5B",
        'modernbert_path': "answerdotai/ModernBERT-large",
        'encoder_path': os.path.join(PROJECT_ROOT, "experiments", "phase 1", "planB_phase1_checkpoints_phase1_vae_step_20000.pth"),
        'phase4_path': os.path.join(PROJECT_ROOT, "experiments", "phase 4", "local_checkpoints", "phase4_step_85995.pth"),
        'sep_token_path': os.path.join(PROJECT_ROOT, "storage", "components", "sep_token.pt"),
        'phase6_ckpt': os.path.join(PROJECT_ROOT, "experiments", "phase 6", "checkpoints", "planB_phase6_checkpoints_phase6_ca_layers_step_6000.pth")
    }
    
    tokenizer = AutoTokenizer.from_pretrained(config['qwen_path'])
    model = BEBLaDIIPhase6Eval(config).to(device)
    latent_dict = get_latent_dict(device)
    void_vector = get_void_vector(device)
    
    df = pd.read_parquet("C:/Experiments/BEBLaDII/BEBLaDII-planB-Phase6-Data/phase 6/data/data_identity.parquet")
    sample = df.iloc[0].to_dict()
    
    analyze_j_space(config, model, tokenizer, latent_dict, void_vector, sample, device)

if __name__ == "__main__":
    main()
