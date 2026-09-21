import os
import sys
import math
import torch
import torch.nn.functional as F
import datetime

sys.stdout.reconfigure(encoding='utf-8')
PROJECT_ROOT = "C:/Experiments/BEBLaDII"
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments", "phase 4"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from evaluate_phase4_checkpoints import BEBLaDIIPhase4aEval
from transformers import AutoTokenizer
from beb_la_dii.model.sensor_ensemble import SensorEnsemble
from beb_la_dii.model.orchestrator import Orchestrator

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

def safe_normalize(x, dim=-1, eps=1e-6): return F.normalize(x, p=2, dim=dim, eps=eps)

def build_latent_dict(model, device, batch_size=2048):
    dict_path = os.path.join(PROJECT_ROOT, "experiments", "phase 5", "local", "latent_dict.pt")
    if os.path.exists(dict_path):
        print(f"Loading latent dict from {dict_path}...")
        return torch.load(dict_path, map_location=device)
        
    print("Building latent dict...")
    with torch.no_grad():
        qwen_emb = model.qwen_embeddings.weight
        vocab_size = qwen_emb.shape[0]
        latents = []
        for i in range(0, vocab_size, batch_size):
            batch_emb = qwen_emb[i:i+batch_size].unsqueeze(1).to(device)
            z, _, _ = model.encoder(batch_emb)
            latents.append(safe_normalize(z.squeeze(1).float(), dim=-1))
        
        latent_dict = torch.cat(latents, dim=0)
        torch.save(latent_dict, dict_path)
        print(f"Saved latent dict to {dict_path}")
        return latent_dict

def generate_html_report(all_runs_data, html_path):
    html_content = """
    <html>
    <head>
        <meta charset="utf-8">
        <title>Phase 5: Geometric Ensemble Denoising</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f9; color: #333; margin: 20px; }
            h1, h2 { color: #2c3e50; }
            .legend-container { background: #fff; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; font-size: 14px; }
            .legend-item { margin-bottom: 8px; }
            .run-container { background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 30px; }
            .step-row { display: flex; flex-wrap: nowrap; margin-bottom: 5px; align-items: center; border-bottom: 1px solid #eee; padding-bottom: 5px; overflow-x: auto; }
            .step-label { width: 80px; font-weight: bold; color: #7f8c8d; flex-shrink: 0; }
            .token-box { display: inline-block; padding: 5px 8px; margin: 0 2px; border-radius: 4px; font-family: monospace; font-size: 14px; text-align: center; border: 1px solid #ccc; position: relative; flex-shrink: 0; }
            .tooltip { visibility: hidden; width: 250px; background-color: #333; color: #fff; text-align: left; border-radius: 6px; padding: 10px; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -125px; opacity: 0; transition: opacity 0.3s; font-size: 12px; }
            .token-box:hover .tooltip { visibility: visible; opacity: 1; }
            .metric-bar-container { width: 100%; background-color: #ddd; height: 5px; margin-top: 3px; border-radius: 3px; }
            .metric-bar { height: 100%; border-radius: 3px; background-color: #3498db; }
            .selector-container { margin-bottom: 20px; }
            select { padding: 8px; font-size: 16px; border-radius: 4px; width: 100%; max-width: 600px; }
            .phrase-block { display: none; }
            .phrase-block.active { display: block; }
        </style>
        <script>
            function showPhrase(index) {
                var blocks = document.getElementsByClassName('phrase-block');
                for(var i=0; i<blocks.length; i++) {
                    blocks[i].classList.remove('active');
                }
                var target = document.getElementById('phrase-' + index);
                if (target) {
                    target.classList.add('active');
                }
            }
        </script>
    </head>
    <body>
        <h1>Phase 5: Denoising Trajectories (Adaptive Option 2)</h1>
        
        <div class="legend-container">
            <h3>Legend & Metrics</h3>
            <div class="legend-item"><b>Background Color:</b> Indicates local noise level (t_loc). <span style="background-color: rgba(255, 0, 100, 0.4); padding: 2px 5px; border-radius: 3px;">Red</span> = High Noise (t ≈ 1.0). <span style="background-color: rgba(0, 255, 100, 0.4); padding: 2px 5px; border-radius: 3px;">Green</span> = Clean/Denoised (t ≈ 0.0).</div>
            <div class="legend-item"><b>Bottom Bar (RawDProx):</b> Indicates Top-1 Cosine Similarity to the closest dictionary word. Length = similarity percentage. <span style="color: #2ecc71; font-weight: bold;">Green</span> (>0.8), <span style="color: #f1c40f; font-weight: bold;">Yellow</span> (0.4-0.8), <span style="color: #e74c3c; font-weight: bold;">Red</span> (<0.4).</div>
            <div class="legend-item"><b>Delta:</b> Difference in cosine similarity between the Top-1 and Top-2 candidate words (high means confidence, low means ambiguity).</div>
            <div class="legend-item"><b>ConflictSim:</b> Cosine similarity between the Top-1 and Top-2 candidate words themselves (high = syntax/casing variation, low = semantic conflict).</div>
            <div class="legend-item"><b>Status:</b> Orchestrator decision (e.g., DENOISING, CRYSTALLIZED, SEMANTIC_CONFLICT).</div>
        </div>
        
        <div class="selector-container">
            <label for="phraseSelect"><b>Select phrase to visualize: </b></label>
            <select id="phraseSelect" onchange="showPhrase(this.value)">
    """
    
    for idx, (phrase, _) in enumerate(all_runs_data.items()):
        html_content += f"<option value='{idx}'>{phrase}</option>"
        
    html_content += """
            </select>
        </div>
    """
    
    for idx, (phrase, run_data) in enumerate(all_runs_data.items()):
        active_class = "active" if idx == 0 else ""
        html_content += f"<div id='phrase-{idx}' class='phrase-block {active_class}'>"
        html_content += f"<h2>Phrase: <i>{phrase}</i></h2>"
        
        for run_name, steps_data in run_data.items():
            html_content += f"<div class='run-container'><h3>{run_name}</h3>"
            for step in steps_data:
                html_content += f"<div class='step-row'><div class='step-label'>Step {step['iter']}</div>"
                for t_info in step['tokens']:
                    t_val = t_info['t_loc']
                    word = t_info['word'].replace(" ", "&nbsp;")
                    conf = t_info['conf']
                    delta = t_info['delta']
                    top2_word = t_info['top2_word'].replace("'", "&apos;")
                    conf_sim = t_info['conf_sim']
                    status = t_info.get('status', 'N/A')
                    
                    r = int(255 * min(1.0, t_val))
                    g = int(255 * (1.0 - min(1.0, t_val)))
                    color = f"rgba({r}, {g}, 100, 0.4)"
                    
                    bar_width = int(conf * 100)
                    bar_color = "#2ecc71" if conf > 0.8 else ("#f1c40f" if conf > 0.4 else "#e74c3c")
                    
                    tooltip = f"t_loc: {t_val:.3f}<br>RawDProx (Top1): {conf:.3f}<br>Delta: {delta:.3f}<br>Top2: &apos;{top2_word}&apos;<br>ConflictSim: {conf_sim:.3f}<br>Status: {status}"
                    
                    html_content += f"""
                    <div class='token-box' style='background-color: {color};'>
                        {word}
                        <div class='metric-bar-container'><div class='metric-bar' style='width: {bar_width}%; background-color: {bar_color};'></div></div>
                        <span class='tooltip'>{tooltip}</span>
                    </div>
                    """
                
                # Add full decoded sentence after the boxes
                if 'full_text' in step:
                    html_content += f"<div style='margin-left: 20px; font-style: italic; color: #555; display: flex; align-items: center;'>{step['full_text']}</div>"
                    
                html_content += "</div>"
            html_content += "</div>"
        html_content += "</div>"
    
    html_content += "</body></html>"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

def run_trajectory(dus, sensor_ensemble, orchestrator, tokenizer, latent_dict, input_ids, attn_mask, z_clean, initial_noise, t_start, steps, run_name, report_file, device):
    B, T = input_ids.shape
    t_local = t_start.clone()
    theta_init = t_local.unsqueeze(-1) * (math.pi / 2)
    x_t = safe_normalize(torch.cos(theta_init) * z_clean + torch.sin(theta_init) * initial_noise, dim=-1)
    
    dt_min = 1.0 / steps
    run_history = []
    
    report_file.write(f"\n{'='*80}\nRUN: {run_name}\n{'='*80}\n")
    
    # We step t_global from max(t_start) down to 0
    t_global_start = torch.max(t_start).item()
    num_iters = int(math.ceil(t_global_start / dt_min)) + 2 # Add buffer to ensure it reaches 0
    
    for i in range(num_iters):
        t_global = torch.tensor([max(0.0, t_global_start - i * dt_min)], device=device)
        
        with torch.no_grad():
            out_sc = dus(input_ids, attn_mask, t_global=t_global, t_reported=t_local, z_noisy_override=x_t)
            sc_est = out_sc["dus_final"].detach()
            out = dus(input_ids, attn_mask, t_global=t_global, t_reported=t_local, self_cond=sc_est, z_noisy_override=x_t)
            z_pred_raw = out["dus_final"]
            
            gate_t = torch.sin(t_local * (math.pi / 2)).unsqueeze(-1)
            z_pred = safe_normalize(gate_t * z_pred_raw + (1.0 - gate_t) * x_t, dim=-1)
            
            # Compute geometric metrics via SensorEnsemble
            geom_metrics = sensor_ensemble.compute_geometric_metrics(z_pred, latent_dict)
            
            # Get Orchestrator actions
            actions = orchestrator.batch_analyze(t_local, geom_metrics["raw_d_prox"], geom_metrics["delta"], geom_metrics["conflict_sim"])
            
            step_data = {'iter': i, 'tokens': [], 'w1_ids': []}
            
            if i % 3 == 0 or i == num_iters - 1 or torch.max(t_local).item() == 0.0:
                report_file.write(f"\nStep {i} (t_global: {t_global.item():.2f}):\n")
            
            delta_t_all = torch.zeros_like(t_local)
            
            w1_ids_full = geom_metrics["top1_idx"][0].tolist()
            w2_ids_full = geom_metrics["top2_idx"][0].tolist()
            step_data['w1_ids'] = w1_ids_full
            
            w1_words = []
            prev_w1 = ""
            for t_idx in range(T):
                cur_w1 = tokenizer.decode(w1_ids_full[:t_idx+1])
                l = 0
                while l < len(prev_w1) and l < len(cur_w1) and prev_w1[l] == cur_w1[l]:
                    l += 1
                w1_words.append(cur_w1[l:])
                prev_w1 = cur_w1
                
            w2_words = []
            prev_w2 = ""
            for t_idx in range(T):
                cur_w2 = tokenizer.decode(w2_ids_full[:t_idx+1])
                l = 0
                while l < len(prev_w2) and l < len(cur_w2) and prev_w2[l] == cur_w2[l]:
                    l += 1
                w2_words.append(cur_w2[l:])
                prev_w2 = cur_w2
            
            for t_idx in range(T):
                s1 = geom_metrics["raw_d_prox"][0, t_idx].item()
                delta = geom_metrics["delta"][0, t_idx].item()
                conf_sim = geom_metrics["conflict_sim"][0, t_idx].item()
                
                w1 = w1_words[t_idx]
                w2 = w2_words[t_idx]
                
                action_info = actions[0][t_idx]
                
                step_data['tokens'].append({
                    'word': w1, 'top2_word': w2, 't_loc': t_local[0, t_idx].item(),
                    'conf': s1, 'delta': delta, 'conf_sim': conf_sim,
                    'status': action_info['status']
                })
                
                if (i % 3 == 0 or i == num_iters - 1) and t_idx in [4, 5]: # Log jumps and over
                    report_file.write(f"  Token {t_idx} '{w1.strip()}' | t_loc: {t_local[0, t_idx].item():.3f} | RawDProx: {s1:.3f} | Delta: {delta:.3f} | ConfSim: {conf_sim:.3f} | Action: {action_info['action']} ({action_info['status']})\n")
                
                t_est = 1.0 - s1
                delta_t = torch.clamp(t_local[0, t_idx] - t_est, min=dt_min)
                delta_t_all[0, t_idx] = delta_t
                
            step_data['full_text'] = tokenizer.decode(step_data['w1_ids'])
            run_history.append(step_data)
            
            # Print full sentence
            if i % 3 == 0 or i == num_iters - 1:
                report_file.write(f"  > Sentence: {step_data['full_text']}\n")
                
            if torch.max(t_local).item() == 0.0:
                break
                
            t_next = torch.clamp(t_local - delta_t_all, min=0.0)
            
            theta_now = t_local.unsqueeze(-1) * (math.pi / 2)
            theta_next_tensor = t_next.unsqueeze(-1) * (math.pi / 2)
            
            sin_now = torch.sin(theta_now)
            sin_now = torch.where(sin_now < 1e-5, torch.ones_like(sin_now) * 1e-5, sin_now)
            w_pred = torch.sin(theta_now - theta_next_tensor) / sin_now
            w_cur = torch.sin(theta_next_tensor) / sin_now
            
            z_next = w_pred * z_pred + w_cur * x_t
            x_t = torch.where(theta_now > 1e-5, safe_normalize(z_next, dim=-1), safe_normalize(z_pred, dim=-1))
            t_local = t_next
            
    return run_history

def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    dus = load_dus(device)
    latent_dict = build_latent_dict(dus, device)
    
    sensor_ensemble = SensorEnsemble().to(device)
    sensor_ensemble.eval()
    orchestrator = Orchestrator()
    
    phrases = [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "Мама мыла раму, а папа читал газету.",
        "В лесу родилась ёлочка, в лесу она росла.",
        "Strč prst skrz krk.",
        "Příliš žluťoučký kůň úpěl ďábelské ódy."
    ]
    
    report_path = os.path.join(PROJECT_ROOT, "experiments", "phase 5", "phase5_evaluation_report.txt")
    html_path = os.path.join(PROJECT_ROOT, "experiments", "phase 5", "phase5_denoising_demo.html")
    
    all_runs_data = {}
    
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write(f"Phase 5 Evaluation Report\nDate: {datetime.datetime.now()}\n\n")
        
        for phrase in phrases:
            rf.write(f"\n{'='*80}\nEVALUATING PHRASE: {phrase}\n{'='*80}\n")
            tok = tokenizer(phrase, return_tensors="pt")
            input_ids = tok.input_ids.to(device)
            attn_mask = torch.ones_like(input_ids).to(device)
            
            with torch.no_grad():
                qwen_embeds = dus.qwen_embeddings(input_ids)
                z_clean, _, _ = dus.encoder(qwen_embeds)
                z_clean = safe_normalize(z_clean, dim=-1)
                
            initial_noise = safe_normalize(torch.randn_like(z_clean), dim=-1)
            
            phrase_data = {}
            seq_len = input_ids.shape[1]
            
            # RUN A: High Noise (up to 1.0)
            t_start_high = torch.rand(input_ids.shape, device=device) * 0.7 + 0.2
            # Add some targeted low/high noise if sequence is long enough to have these indices
            if seq_len > 6:
                t_start_high[0, 2] = 0.1
                t_start_high[0, 6] = 0.1
                t_start_high[0, 5] = 0.6
                t_start_high[0, 4] = 0.8437
            phrase_data["Run A: High Noise (Broad Exploration)"] = run_trajectory(
                dus, sensor_ensemble, orchestrator, tokenizer, latent_dict, input_ids, attn_mask, z_clean, initial_noise, t_start_high, 25, f"High Noise - {phrase[:15]}...", rf, device)
            
            # RUN B: Low Noise (up to 0.7)
            t_start_low = torch.rand(input_ids.shape, device=device) * 0.6 + 0.1
            if seq_len > 5:
                t_start_low[0, 5] = 0.7
                t_start_low[0, 4] = 0.7
            phrase_data["Run B: Low Noise (Local Exploitation)"] = run_trajectory(
                dus, sensor_ensemble, orchestrator, tokenizer, latent_dict, input_ids, attn_mask, z_clean, initial_noise, t_start_low, 25, f"Low Noise - {phrase[:15]}...", rf, device)
                
            all_runs_data[phrase] = phrase_data
            
    generate_html_report(all_runs_data, html_path)
    print(f"Evaluation complete. Saved to {report_path} and {html_path}")

if __name__ == "__main__":
    main()