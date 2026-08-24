import torch
import torch.nn.functional as F
import os

def analyze_weights_svd(ckpt_path):
    print(f"\n=======================================================")
    print(f"[*] Deep Weight Analysis: {os.path.basename(ckpt_path)}")
    print(f"=======================================================")
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        
    modules = {
        "self_cond_proj": ckpt.get("self_cond_proj", {}).get("weight", None),
        "adaLN_attn_L39_W": ckpt.get("adaLN_attn", {}).get("39.modulation.1.weight", None),
        "adaLN_attn_L39_b": ckpt.get("adaLN_attn", {}).get("39.modulation.1.bias", None),
        "adaLN_mlp_L39_W": ckpt.get("adaLN_mlp", {}).get("39.modulation.1.weight", None),
        "adaLN_mlp_L39_b": ckpt.get("adaLN_mlp", {}).get("39.modulation.1.bias", None),
        "t_proj_W": ckpt.get("t_proj", {}).get("weight", None),
        "sep_embed": ckpt.get("dus", {}).get("sep_embed", None),
    }
    
    for name, w in modules.items():
        if w is not None:
            w = w.float()
            norm = torch.norm(w).item()
            mean = w.mean().item()
            std = w.std().item()
            
            print(f"  {name:<16}: Norm={norm:>8.4f} | Mean={mean:>8.4f} | Std={std:>8.4f}")
            
            # Если это 2D матрица, делаем SVD
            if w.dim() == 2:
                # Для adaLN выход может быть 2 * hidden_dim, нужно аккуратно
                U, S, V = torch.linalg.svd(w, full_matrices=False)
                s_max = S[0].item()
                s_mean = S.mean().item()
                rank1_ratio = s_max / (S.sum().item() + 1e-9)
                print(f"  {'':<16}  SVD: s_max={s_max:>8.4f} | s_mean={s_mean:>8.4f} | Rank-1 Ratio={rank1_ratio:>6.2%}")
        else:
            print(f"  {name:<16}: NOT FOUND")
            

if __name__ == "__main__":
    analyze_weights_svd(r"C:\Experiments\BEBLaDII\experiments\phase3\local_checkpoints\phase3_step_18995.pth")
    analyze_weights_svd(r"C:\Experiments\BEBLaDII\experiments\phase3\local_checkpoints\phase3_step_25995.pth")
