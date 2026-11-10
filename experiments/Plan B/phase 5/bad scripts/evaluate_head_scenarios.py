import os
import sys
import torch

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = "C:/Experiments/BEBLaDII"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.beb_la_dii.model.confidence_head import ConfidenceHead

def main():
    device = torch.device("cpu")
    
    print("Loading Confidence Head...")
    conf_ckpt = os.path.join(PROJECT_ROOT, "experiments", "phase 5", "local", "confidence_head_v1.pt")
    # Обрати внимание: пока загружаем старую размерность (conf_dim=5), так как модель обучалась на 5 выходах.
    conf_head = ConfidenceHead(h39_dim=1024, attn_dim=256, num_heads=4, window_size=32, conf_dim=5, mlp_hidden=512)
    conf_head.load_state_dict(torch.load(conf_ckpt, map_location=device))
    conf_head.eval()

    print("Loading Training Dataset...")
    data_path = os.path.join(PROJECT_ROOT, "experiments", "phase 5", "local", "unified_dataset.pt")
    dataset = torch.load(data_path, map_location=device)
    
    h39 = dataset["h39"]
    dus_final = dataset["dus_final"]
    t_rep = dataset["t_reported"]
    labels = dataset["labels"]
    
    print(f"Dataset size: {len(h39)} sequences")
    
    N = len(h39)
    idx_clean = 0
    idx_noise = N // 2
    idx_mixed = N - 1

    scenarios = [
        ("Сценарий 1: Идеально чистые токены (t=0)", idx_clean),
        ("Сценарий 2: Абсолютный шум (t=1)", idx_noise),
        ("Сценарий 3: Смешанный мусор / Галлюцинация (t=0, но векторы битые)", idx_mixed)
    ]
    
    print("\n" + "="*90)
    print(" ДИАГНОСТИКА ГОЛОВЫ УВЕРЕННОСТИ (СРЕДНЕЕ ПО СЕКВЕНЦИИ)")
    print("="*90)
    
    with torch.no_grad():
        for name, idx in scenarios:
            h = h39[idx:idx+1]
            d_f = dus_final[idx:idx+1]
            t = t_rep[idx:idx+1]
            true_lbl = labels[idx:idx+1]
            
            B, T, _ = h.shape
            conf_prev = torch.zeros((B, T, 5), device=device)
            attn_mask = torch.ones((B, T), device=device)
            
            pred = conf_head(h, d_f, conf_prev, t, attention_mask=attn_mask)
            
            mean_pred = pred[0].mean(dim=0).numpy()
            mean_true = true_lbl[0].mean(dim=0).numpy()
            
            print(f"\n>>> {name}")
            print(f"  Ожидание (Ground Truth) : DProx={mean_true[0]:.2f} | MConf={mean_true[1]:.2f} | Coh={mean_true[2]:.2f} | Comp={mean_true[3]:.2f} | Cryst={mean_true[4]:.2f}")
            print(f"  Реальность (Prediction) : DProx={mean_pred[0]:.2f} | MConf={mean_pred[1]:.2f} | Coh={mean_pred[2]:.2f} | Comp={mean_pred[3]:.2f} | Cryst={mean_pred[4]:.2f}")

if __name__ == "__main__":
    main()
