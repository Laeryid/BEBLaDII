import os
import sys
import math
import torch
import torch.nn.functional as F
import datetime

sys.stdout.reconfigure(encoding='utf-8')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments", "phase 4"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from evaluate_phase4_checkpoints import BEBLaDIIPhase4aEval
from transformers import AutoTokenizer
from beb_la_dii.model.sensor_ensemble import SensorEnsemble
from beb_la_dii.model.confidence_head import ConfidenceHead

def load_dus(device):
    dus_model = BEBLaDIIPhase4aEval(embedding_model_path="Qwen/Qwen2.5-1.5B", modernbert_path="answerdotai/ModernBERT-large")
    vae_ckpt = os.path.join(PROJECT_ROOT, "experiments", "phase 1", "planB_phase1_checkpoints_phase1_vae_step_20000.pth")
    vae_st = torch.load(vae_ckpt, map_location="cpu", weights_only=False)
    if 'encoder' in vae_st: dus_model.encoder.load_state_dict(vae_st['encoder'], strict=False)
    phase4_ckpt = os.path.join(PROJECT_ROOT, "experiments", "phase 4", "local_checkpoints", "phase4_step_85995.pth")
    p4_st = torch.load(phase4_ckpt, map_location="cpu", weights_only=False)
    dus_ema = p4_st.get("dus_ema", p4_st.get("dus", {}))
    clean_dus = {k.replace("_orig_module.", ""): v for k, v in dus_ema.items()}
    dus_model.dus.load_state_dict(clean_dus, strict=False)
    dus_model.to(device)
    dus_model.eval()
    return dus_model

def load_confidence_head(device):
    conf_head_path = os.path.join(PROJECT_ROOT, "experiments", "phase 5", "local", "confidence_head_v2.pt")
    conf_head = ConfidenceHead().to(device)
    if os.path.exists(conf_head_path):
        print(f"Loading ConfidenceHead from {conf_head_path}...")
        conf_head.load_state_dict(torch.load(conf_head_path, map_location=device))
    else:
        print(f"Warning: Confidence head not found at {conf_head_path}")
    conf_head.eval()
    return conf_head

def safe_normalize(x, dim=-1, eps=1e-6): return F.normalize(x, p=2, dim=dim, eps=eps)

def generate_html(tokens_data, trajectories, output_path):
    html = """
    <html>
    <head>
        <meta charset="utf-8">
        <title>Diffusion Trajectory & Confidence Head Demo</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f9; color: #333; margin: 20px; }
            h1, h2 { color: #2c3e50; }
            .legend-container { background: #fff; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; font-size: 14px; }
            .run-container { background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 30px; display: flex; flex-wrap: nowrap; overflow-x: auto; align-items: stretch; gap: 10px; }
            .token-card { display: flex; flex-direction: column; padding: 12px; margin: 0; border-radius: 6px; font-family: monospace; font-size: 11px; border: 1px solid #ccc; flex-shrink: 0; width: 250px; background: #fafafa; }
            .token-card.pool { border: 2px solid #e74c3c; background: #fdf2f2; }
            .token-card.noise { border: 2px solid #9b59b6; background: #f4ecf7; }
            .word-label { font-weight: bold; text-align: center; margin-bottom: 8px; min-height: 35px; display: flex; align-items: center; justify-content: center; word-break: break-word; font-size: 13px; }
            .section-header { font-weight: bold; font-size: 10px; color: #7f8c8d; text-transform: uppercase; margin: 6px 0 2px 0; border-bottom: 1px dotted #ccc; padding-bottom: 1px; }
            .traj-row { margin-bottom: 6px; background: #fff; padding: 4px 6px; border-radius: 4px; border: 1px solid #eee; }
            .traj-title { font-weight: bold; color: #555; margin-bottom: 2px; font-size: 10px; display: flex; justify-content: space-between; }
            .traj-vals { line-height: 1.3; color: #2c3e50; word-spacing: 1px; font-size: 11px; }
            .arr { color: #aaa; font-size: 9px; }
            .highlight-signal { background-color: #fff8e1; }
        </style>
    </head>
    <body>
        <h1>Phase 5 & 6: Denoising Trajectory with Confidence Head (t_global 0.5 &rarr; 0.1)</h1>
        
        <div class="legend-container">
            <p>Evaluating Sensor Ensemble, Coding Theory (PR), and Confidence Head signals (ModelConf, Coherence, Complexity, Stuck).</p>
            <p>Format: <b>Start &rarr; t_g=0.5 &rarr; t_g=0.4 &rarr; t_g=0.3 &rarr; t_g=0.2 &rarr; t_g=0.1</b></p>
        </div>
        
        <div class="run-container">
    """
    
    for t_idx, t in enumerate(tokens_data):
        word = t['word']
        token_type = t['type']
        traj = trajectories[t_idx]
        
        def fmt_chain(vals, decimals=2):
            str_vals = [f"{v:.{decimals}f}" for v in vals]
            return " <span class='arr'>&rarr;</span> ".join(str_vals)
            
        html += f"""
        <div class='token-card {token_type}'>
            <div class='word-label'>{word}</div>
            
            <div class='section-header'>Sensors (Phase 5)</div>
            <div class='traj-row'>
                <div class='traj-title'><span>RawDProx</span></div>
                <div class='traj-vals'>{fmt_chain(traj['rdp'], 2)}</div>
            </div>
            
            <div class='traj-row'>
                <div class='traj-title'><span>Delta</span></div>
                <div class='traj-vals'>{fmt_chain(traj['delta'], 3)}</div>
            </div>
            
            <div class='traj-row'>
                <div class='traj-title'><span>ConflictSim</span></div>
                <div class='traj-vals'>{fmt_chain(traj['csim'], 2)}</div>
            </div>
            
            <div class='section-header'>Coding Theory</div>
            <div class='traj-row'>
                <div class='traj-title'><span>PR (Active Dimensions)</span></div>
                <div class='traj-vals'>{fmt_chain(traj['pr'], 1)}</div>
            </div>

            <div class='section-header'>Confidence Head Signals</div>
            <div class='traj-row highlight-signal'>
                <div class='traj-title'><span>Complexity</span><span>[0..1]</span></div>
                <div class='traj-vals'>{fmt_chain(traj['complexity'], 2)}</div>
            </div>
            
            <div class='traj-row highlight-signal'>
                <div class='traj-title'><span>Coherence</span><span>[0..1]</span></div>
                <div class='traj-vals'>{fmt_chain(traj['coherence'], 2)}</div>
            </div>

            <div class='traj-row highlight-signal'>
                <div class='traj-title'><span>ModelConf</span><span>[0..1]</span></div>
                <div class='traj-vals'>{fmt_chain(traj['mconf'], 2)}</div>
            </div>

            <div class='traj-row highlight-signal'>
                <div class='traj-title'><span>Stuck</span><span>[0..1]</span></div>
                <div class='traj-vals'>{fmt_chain(traj['stuck'], 2)}</div>
            </div>
            
            <div class='traj-row'>
                <div class='traj-title'><span>Cryst</span><span>[0..1]</span></div>
                <div class='traj-vals'>{fmt_chain(traj['cryst'], 2)}</div>
            </div>
        </div>
        """
        
    html += """
        </div>
    </body>
    </html>
    """
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    print("Loading DUS...")
    dus = load_dus(device)
    
    print("Loading ConfidenceHead...")
    conf_head = load_confidence_head(device)
    
    dict_path = os.path.join(PROJECT_ROOT, "experiments", "phase 5", "local", "latent_dict.pt")
    if os.path.exists(dict_path):
        print(f"Loading latent dict from {dict_path}...")
        latent_dict = torch.load(dict_path, map_location=device)
    else:
        print("Error: Latent dict not found. Please run Phase 5 script first.")
        return
        
    sensor_ensemble = SensorEnsemble().to(device)
    sensor_ensemble.eval()
    
    phrase = "In the realm of latent diffusion models, the intricate complexities of semantic dependencies often elude simplistically designed pooling mechanisms, forcing the network to dynamically adapt its representational capacity."
    
    print("Tokenizing...")
    tok = tokenizer(phrase, return_tensors="pt")
    input_ids = tok.input_ids.to(device)
    
    print("Encoding to clean latents...")
    with torch.no_grad():
        qwen_embeds = dus.qwen_embeddings(input_ids)
        z_clean, _, _ = dus.encoder(qwen_embeds)
        z_clean = safe_normalize(z_clean, dim=-1)
        
    decoded_tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
    seq_len = z_clean.shape[1]
    
    w1_start, w1_end = min(10, seq_len-6), min(15, seq_len-1)
    w2_start, w2_end = min(22, seq_len-6), min(27, seq_len-1)
    
    noise_indices = [2, 7, 18]
    
    z_simulated = []
    labels = []
    token_types = []
    
    i = 0
    while i < seq_len:
        if i in noise_indices:
            noise_z = safe_normalize(torch.randn(1, 1, z_clean.size(2), device=device), dim=-1)
            z_simulated.append(noise_z)
            labels.append(f"[NOISE @ {i}]")
            token_types.append("noise")
            i += 1
        elif i == w1_start:
            pool_z = z_clean[:, w1_start:w1_end].mean(dim=1, keepdim=True)
            pool_z = safe_normalize(pool_z, dim=-1)
            z_simulated.append(pool_z)
            
            pool_text = "".join(decoded_tokens[w1_start:w1_end]).strip()
            labels.append(f"[POOL: {pool_text}]")
            token_types.append("pool")
            i = w1_end
        elif i == w2_start:
            pool_z = z_clean[:, w2_start:w2_end].mean(dim=1, keepdim=True)
            pool_z = safe_normalize(pool_z, dim=-1)
            z_simulated.append(pool_z)
            
            pool_text = "".join(decoded_tokens[w2_start:w2_end]).strip()
            labels.append(f"[POOL: {pool_text}]")
            token_types.append("pool")
            i = w2_end
        else:
            z_simulated.append(z_clean[:, i:i+1])
            labels.append(decoded_tokens[i])
            token_types.append("normal")
            i += 1
            
    x_t = torch.cat(z_simulated, dim=1).float()
    T_new = x_t.shape[1]
    
    print("Computing initial state...")
    dummy_ids = torch.zeros((1, T_new), dtype=torch.long, device=device)
    attn_mask = torch.ones((1, T_new), device=device)
    
    with torch.no_grad():
        geom_metrics_init = sensor_ensemble.compute_geometric_metrics(x_t, latent_dict)
        z_sq_init = x_t ** 2
        pr_init = 1.0 / (z_sq_init ** 2).sum(dim=-1)
        ent_init = - (z_sq_init * torch.log(z_sq_init + 1e-10)).sum(dim=-1)
        
        t_local = torch.clamp(1.0 - geom_metrics_init["raw_d_prox"], min=0.0, max=1.0)
        
        # Initial confidence signals with zero previous state
        conf_prev = torch.zeros((1, T_new, 6), device=device)
        conf_init = conf_head(x_t, x_t, conf_prev, t_local.unsqueeze(-1), attention_mask=attn_mask)
        
    trajectories = {t_idx: {
        "pr": [], "rdp": [], "delta": [], "csim": [], "ent": [], "tloc": [],
        "dictprox": [], "mconf": [], "coherence": [], "complexity": [], "cryst": [], "stuck": []
    } for t_idx in range(T_new)}
    
    # Store Step 0
    for t_idx in range(T_new):
        trajectories[t_idx]["pr"].append(pr_init[0, t_idx].item())
        trajectories[t_idx]["rdp"].append(geom_metrics_init["raw_d_prox"][0, t_idx].item())
        trajectories[t_idx]["delta"].append(geom_metrics_init["delta"][0, t_idx].item())
        trajectories[t_idx]["csim"].append(geom_metrics_init["conflict_sim"][0, t_idx].item())
        trajectories[t_idx]["ent"].append(ent_init[0, t_idx].item())
        trajectories[t_idx]["tloc"].append(t_local[0, t_idx].item())
        
        trajectories[t_idx]["dictprox"].append(conf_init[0, t_idx, 0].item())
        trajectories[t_idx]["mconf"].append(conf_init[0, t_idx, 1].item())
        trajectories[t_idx]["coherence"].append(conf_init[0, t_idx, 2].item())
        trajectories[t_idx]["complexity"].append(conf_init[0, t_idx, 3].item())
        trajectories[t_idx]["cryst"].append(conf_init[0, t_idx, 4].item())
        trajectories[t_idx]["stuck"].append(conf_init[0, t_idx, 5].item())
        
    print("Running DUS Diffusion Trajectory with Confidence Head (5 steps)...")
    dt = 0.1
    t_global_vals = [0.5, 0.4, 0.3, 0.2, 0.1]
    
    with torch.no_grad():
        for tg in t_global_vals:
            t_global = torch.tensor([tg], device=device)
            
            # Forward pass
            out_sc = dus(dummy_ids, attn_mask, t_global=t_global, t_reported=t_local, z_noisy_override=x_t)
            sc_est = out_sc["dus_final"].detach()
            
            out = dus(dummy_ids, attn_mask, t_global=t_global, t_reported=t_local, self_cond=sc_est, z_noisy_override=x_t)
            z_pred_raw = out["dus_final"]
            h39_raw = out["h_39_raw"]
            
            # Gate & get target prediction
            gate_t = torch.sin(t_local * (math.pi / 2)).unsqueeze(-1)
            z_pred = safe_normalize(gate_t * z_pred_raw + (1.0 - gate_t) * x_t, dim=-1)
            
            # Eval z_pred
            metrics = sensor_ensemble.compute_geometric_metrics(z_pred, latent_dict)
            s1 = metrics["raw_d_prox"]
            t_est = torch.clamp(1.0 - s1, min=0.0, max=1.0)
            
            # Confidence Head evaluation
            conf_signals = conf_head(h39_raw, z_pred, conf_prev, t_local.unsqueeze(-1), attention_mask=attn_mask)
            conf_prev = conf_signals.detach()
            
            # Update time and state (Euler)
            delta_t_all = torch.clamp(t_local - t_est, min=dt)
            t_next = torch.clamp(t_local - delta_t_all, min=0.0)
            
            theta_now = t_local.unsqueeze(-1) * (math.pi / 2)
            theta_next_tensor = t_next.unsqueeze(-1) * (math.pi / 2)
            
            sin_now = torch.sin(theta_now)
            sin_now = torch.where(sin_now < 1e-5, torch.ones_like(sin_now) * 1e-5, sin_now)
            w_pred = torch.sin(theta_now - theta_next_tensor) / sin_now
            w_cur = torch.sin(theta_next_tensor) / sin_now
            
            z_next = w_pred * z_pred + w_cur * x_t
            x_next = torch.where(theta_now > 1e-5, safe_normalize(z_next, dim=-1), safe_normalize(z_pred, dim=-1))
            
            # Record trajectory
            z_sq = z_pred ** 2
            pr = 1.0 / (z_sq ** 2).sum(dim=-1)
            ent = - (z_sq * torch.log(z_sq + 1e-10)).sum(dim=-1)
            
            for t_idx in range(T_new):
                trajectories[t_idx]["pr"].append(pr[0, t_idx].item())
                trajectories[t_idx]["rdp"].append(s1[0, t_idx].item())
                trajectories[t_idx]["delta"].append(metrics["delta"][0, t_idx].item())
                trajectories[t_idx]["csim"].append(metrics["conflict_sim"][0, t_idx].item())
                trajectories[t_idx]["ent"].append(ent[0, t_idx].item())
                trajectories[t_idx]["tloc"].append(t_local[0, t_idx].item())
                
                trajectories[t_idx]["dictprox"].append(conf_signals[0, t_idx, 0].item())
                trajectories[t_idx]["mconf"].append(conf_signals[0, t_idx, 1].item())
                trajectories[t_idx]["coherence"].append(conf_signals[0, t_idx, 2].item())
                trajectories[t_idx]["complexity"].append(conf_signals[0, t_idx, 3].item())
                trajectories[t_idx]["cryst"].append(conf_signals[0, t_idx, 4].item())
                trajectories[t_idx]["stuck"].append(conf_signals[0, t_idx, 5].item())
                
            x_t = x_next
            t_local = t_next
            
    tokens_data = []
    for t_idx in range(T_new):
        tokens_data.append({
            'word': labels[t_idx],
            'type': token_types[t_idx]
        })
        
    html_path = os.path.join(SCRIPT_DIR, "coding_theory_demo_trajectory.html")
    generate_html(tokens_data, trajectories, html_path)
    print(f"Saved demo to {html_path}")

if __name__ == "__main__":
    main()
