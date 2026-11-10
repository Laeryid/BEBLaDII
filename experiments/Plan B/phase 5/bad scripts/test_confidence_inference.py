import os
import sys
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = "C:/Experiments/BEBLaDII"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments", "phase 4"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments", "phase 2"))

from evaluate_phase4_checkpoints import BEBLaDIIPhase4aEval
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.beb_la_dii.model.confidence_head import ConfidenceHead
from src.beb_la_dii.model.modern_decoder import ModernLatentDecoder

def safe_normalize(x, dim=-1, eps=1e-6):
    return F.normalize(x, p=2, dim=dim, eps=eps)

def load_models(device):
    print("Loading models properly...")
    embed_model_id = "Qwen/Qwen2.5-1.5B"
    modernbert_id = "answerdotai/ModernBERT-large"
    vae_ckpt = os.path.join(PROJECT_ROOT, "experiments", "phase 1", "planB_phase1_checkpoints_phase1_vae_step_20000.pth")
    phase4_ckpt = os.path.join(PROJECT_ROOT, "experiments", "phase 4", "local_checkpoints", "phase4_step_85995.pth")

    dus_model = BEBLaDIIPhase4aEval(embedding_model_path=embed_model_id, modernbert_path=modernbert_id)
    
    vae_st = torch.load(vae_ckpt, map_location="cpu", weights_only=False)
    if 'encoder' in vae_st:
        dus_model.encoder.load_state_dict(vae_st['encoder'], strict=False)

    p4_st = torch.load(phase4_ckpt, map_location="cpu", weights_only=False)
    dus_ema = p4_st.get("dus_ema", p4_st.get("dus", {}))
    clean_dus = {k.replace("_orig_module.", ""): v for k, v in dus_ema.items()}
    dus_model.dus.load_state_dict(clean_dus, strict=False)
    
    dus_model.to(device)
    dus_model.eval()

    conf_head_path = os.path.join(PROJECT_ROOT, "experiments", "phase 5", "local", "confidence_head_v2.pt")
    conf_head = ConfidenceHead().to(device)
    if os.path.exists(conf_head_path):
        conf_head.load_state_dict(torch.load(conf_head_path, map_location="cpu"))
    conf_head.eval()

    causal = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", torch_dtype=torch.bfloat16).to(device)
    
    dec_ckpt = os.path.join(PROJECT_ROOT, "experiments", "phase 2", "planB_phase2_checkpoints_decoder_step_9000.pth")
    dus_weights_path = os.path.join(PROJECT_ROOT, "kaggle_upload_1_2", "AWAKENED_WEIGHTS_FINAL.pt")
    decoder = ModernLatentDecoder(latent_dim=1024, qwen_dim=1536, num_layers=3, dus_weights_path=dus_weights_path)
    dec_st = torch.load(dec_ckpt, map_location="cpu", weights_only=False)
    clean_state = {k.replace("decoder.", ""): v for k, v in dec_st.get("decoder", dec_st).items()}
    decoder.load_state_dict(clean_state, strict=False)
    decoder.to(device)
    decoder.eval()
    
    lm_head_weight = causal.lm_head.weight.detach()

    return dus_model, conf_head, decoder, lm_head_weight

def decode_z(z: torch.Tensor, decoder: nn.Module, lm_head_weight: torch.Tensor, tokenizer):
    with torch.no_grad():
        z_scaled = z * math.sqrt(z.shape[-1])
        projected = decoder(z_scaled.to(next(decoder.parameters()).dtype))
        logits = F.linear(projected.float(), lm_head_weight.float())
        
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        max_prob = torch.max(probs, dim=-1)[0]
        
        pred_ids = logits.argmax(dim=-1)
        words = [tokenizer.decode([tid.item()]) for tid in pred_ids[0]]
        
        return words, entropy[0].cpu().numpy(), max_prob[0].cpu().numpy()

def build_latent_dict(model, device, batch_size=2048):
    print("Building full Latent Dictionary (151k tokens)...")
    with torch.no_grad():
        qwen_emb = model.qwen_embeddings.weight
        vocab_size = qwen_emb.shape[0]
        latents = []
        for i in range(0, vocab_size, batch_size):
            batch_emb = qwen_emb[i:i+batch_size].unsqueeze(1).to(device)
            z, _, _ = model.encoder(batch_emb)
            latents.append(safe_normalize(z.squeeze(1).float(), dim=-1))
        
        return torch.cat(latents, dim=0)

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    dus_model, conf_head, decoder, lm_head_weight = load_models(device)
    latent_dict = build_latent_dict(dus_model, device)

    phrases = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is rapidly transforming the world.",
        "To be, or not to be, that is the question.",
        "A completely random sentence with unexpected words."
    ]

    steps = 25
    dt_min = 1.0 / steps
    t_schedule = torch.linspace(1.0, 0.04, steps)
    full_dashboard_data = {}

    for phrase_idx, text in enumerate(phrases):
        print(f"\n===================================================================================================================")
        print(f"Processing phrase {phrase_idx+1}/{len(phrases)}: {text}")
        print(f"===================================================================================================================")
        
        tok = tokenizer(text, return_tensors="pt")
        input_ids = tok.input_ids.to(device)
        B, T = input_ids.shape
        clean_words = [tokenizer.decode([tid.item()]) for tid in input_ids[0]]

        with torch.no_grad():
            qwen_embeds = dus_model.qwen_embeddings(input_ids)
            z_clean, _, _ = dus_model.encoder(qwen_embeds)
            z_clean = safe_normalize(z_clean, dim=-1)

        initial_noise = torch.randn_like(z_clean)
        initial_noise = safe_normalize(initial_noise, dim=-1)
        
        t_base = torch.rand((B, 1), device=device) * 0.5 + 0.2
        t_local_start = torch.rand((B, T), device=device) * 0.5
        t_mixed = t_base * 0.5 + t_local_start * 0.5
        
        anchor_mask = torch.rand((B, T), device=device) > 0.85
        
        t_mixed[:, 0] = 0.04  # Anchor (Clean)
        t_mixed[:, T // 2] = 0.60 # Mid (Noisy)
        t_mixed[:, T - 1] = 0.40 # End (Partially clean)
        
        t_local = torch.where(anchor_mask, torch.full_like(t_mixed, 0.04), t_mixed)
        
        theta_init = t_local.unsqueeze(-1) * (math.pi / 2)
        x_t = safe_normalize(torch.cos(theta_init) * z_clean + torch.sin(theta_init) * initial_noise, dim=-1)
        
        attn_mask = torch.ones((B, T), device=device)
        
        track_indices = [(0, "Anchor  (" + clean_words[0].strip() + ")"),
                         (T // 2, "Mid     (" + clean_words[T//2].strip() + ")"),
                         (T - 1, "End     (" + clean_words[T-1].strip() + ")")]

        dashboard_data = []

        for i in range(steps):
            t_global_scalar = t_schedule[i].item()
            t_global_tensor = torch.full((B,), t_global_scalar, device=device)
            
            with torch.no_grad():
                out_sc = dus_model(input_ids, attn_mask, t_global=t_global_tensor, t_reported=t_local, z_noisy_override=x_t)
                z_sc = out_sc["dus_final"]
                out = dus_model(input_ids, attn_mask, t_global=t_global_tensor, t_reported=t_local, z_noisy_override=x_t, self_cond=z_sc)
            
            z_pred_raw = out["dus_final"]
            h39_raw = out["h_39_raw"]
            
            gate_t = torch.sin(t_local * (math.pi / 2)).unsqueeze(-1)
            z_pred = safe_normalize(gate_t * z_pred_raw + (1.0 - gate_t) * x_t, dim=-1)
            
            flat_pred = z_pred.view(-1, 1024)
            sims = torch.matmul(flat_pred, latent_dict.T)
            raw_dprox_tensor = torch.max(sims, dim=-1)[0].view(B, T)
            raw_dprox = raw_dprox_tensor[0].cpu().numpy()
            
            conf_prev = torch.zeros((B, T, 6), device=device)
            with torch.no_grad():
                conf_signals = conf_head(h39_raw, z_pred, conf_prev, t_local.unsqueeze(-1), attention_mask=attn_mask)
            
            cos_sim_curr = (x_t * z_clean).sum(dim=-1)
            cos_sim_clamped = torch.clamp(cos_sim_curr, -1.0 + 1e-6, 1.0 - 1e-6)
            true_n = (2.0 / math.pi) * torch.acos(cos_sim_clamped)

            pred_words, pred_entropies, pred_max_probs = decode_z(z_pred, decoder, lm_head_weight, tokenizer)
            current_words, entropies, max_probs = decode_z(x_t, decoder, lm_head_weight, tokenizer)

            iter_data = []
            for idx in range(T):
                sig = conf_signals[0, idx].detach().cpu().numpy()
                tl = float(t_local[0, idx].item())
                tn = float(true_n[0, idx].item())
                iter_data.append({
                    "token": current_words[idx],
                    "clean_token": clean_words[idx],
                    "t_local": tl,
                    "TrueN": tn,
                    "DProx": float(sig[0]),
                    "MConf": float(sig[1]),
                    "Coh": float(sig[2]),
                    "Comp": float(sig[3]),
                    "Cryst": float(sig[4]),
                    "Stuck": float(sig[5]),
                    "DecEnt": float(entropies[idx]),
                    "MaxPr": float(max_probs[idx]),
                    "PredEnt": float(pred_entropies[idx]),
                    "1-RawDProx": 1.0 - float(raw_dprox[idx])
                })
            dashboard_data.append({"iteration": i, "t_global": float(t_global_scalar), "tokens": iter_data})

            if i % 5 == 0 or i == steps - 1:
                if phrase_idx == 0:
                    print(f"Step {i+1:<2} | t_global: {t_global_scalar:.2f}")
                    print(f"{'Role':<15} | {'Word (dec)':<10} | {'t_local':<7} | {'TrueN':<7} | {'1-RawDPrx':<9} | {'Coh':<5} | {'1-MaxPr':<7}")
                    print("-" * 90)
                    for idx, role_name in track_indices:
                        tl = t_local[0, idx].item()
                        tn = true_n[0, idx].item()
                        word = current_words[idx]
                        sig = conf_signals[0, idx].detach().cpu().numpy()
                        print(f"{role_name:<15} | {word:<10} | {tl:7.3f} | {tn:7.3f} | {1.0 - raw_dprox[idx]:9.3f} | {sig[2]:.3f} | {1.0 - max_probs[idx]:7.3f}")
                    print("-" * 90)

            if i < steps - 1:
                # OPTION 2: Adaptive Step-Size ODE Solver
                t_est = 1.0 - raw_dprox_tensor
                
                # delta_t = max(t_local - t_est, 1/steps)
                delta_t = torch.clamp(t_local - t_est, min=dt_min)
                
                # t_next = t_local - delta_t
                t_next = torch.clamp(t_local - delta_t, min=0.0)
                
                theta = t_local.unsqueeze(-1) * (math.pi / 2)
                theta_next = t_next.unsqueeze(-1) * (math.pi / 2)
                
                mask = (t_local > 0.01).unsqueeze(-1)
                new_noise = safe_normalize(torch.randn_like(x_t), dim=-1)
                
                noise_pred = torch.where(
                    mask,
                    safe_normalize(x_t - torch.cos(theta) * z_pred, dim=-1),
                    new_noise
                )
                
                x_t_next = torch.cos(theta_next) * z_pred + torch.sin(theta_next) * noise_pred
                
                # Snap to clean if completely denoised
                x_t = torch.where((t_next < 0.001).unsqueeze(-1), z_pred, safe_normalize(x_t_next, dim=-1))
                t_local = torch.where(t_next < 0.001, torch.zeros_like(t_next), t_next)

        full_dashboard_data[text] = dashboard_data

    html_path = os.path.join(PROJECT_ROOT, "experiments", "phase 5", "local", "confidence_dashboard.html")
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>BEBLaDII Phase 5 Adaptive Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        select, input {{ margin: 10px 0; font-size: 16px; }}
        .slider-container {{ margin: 20px 0; }}
        #textDisplay {{ font-size: 20px; margin-top: 20px; line-height: 1.5; padding: 10px; background: #f9f9f9; border-radius: 8px; }}
        .token {{ padding: 2px 4px; border-radius: 4px; margin-right: 4px; display: inline-block; border: 1px solid #ccc; min-width: 20px; text-align: center; font-weight: bold; cursor: pointer; }}
        .clean-text {{ font-size: 16px; color: #666; margin-bottom: 10px; font-style: italic; }}
        .checkbox-group {{ display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 15px; background: #eee; padding: 10px; border-radius: 5px; }}
        .checkbox-group label {{ cursor: pointer; user-select: none; }}
    </style>
</head>
<body>
    <h1>Phase 5: Adaptive ODE Solver (Option 2)</h1>
    
    <div>
        <strong>Select Phrase:</strong><br>
        <select id="phraseSelect" style="width: 100%; padding: 5px;">
"""
    for text in phrases:
        html_content += f"            <option value='{text}'>{text}</option>\n"

    html_content += f"""
        </select>
    </div>
    <br>
    
    <div style="display: flex; gap: 20px; align-items: center;">
        <div>
            <strong>Color Text By:</strong><br>
            <select id="colorSelect">
                <option value="t_local">t_local (Adaptive)</option>
                <option value="TrueN">TrueN (Distance to Clean)</option>
                <option value="1-MaxPr">1 - MaxPr</option>
                <option value="Coh">Coh (Coherence)</option>
                <option value="1-RawDProx">1 - RawDProx</option>
            </select>
        </div>
        <div style="flex-grow: 1;">
            <strong>Select Chart Metrics (Multiple):</strong>
            <div class="checkbox-group" id="metricCheckboxes">
                <label><input type="checkbox" value="t_local" checked> t_local</label>
                <label><input type="checkbox" value="TrueN" checked> TrueN</label>
                <label><input type="checkbox" value="1-MaxPr"> 1 - MaxPr</label>
                <label><input type="checkbox" value="1-RawDProx" checked> 1 - RawDProx</label>
                <label><input type="checkbox" value="DProx"> DProx (MLP)</label>
                <label><input type="checkbox" value="Coh"> Coh</label>
            </div>
        </div>
    </div>
    
    <div class="slider-container">
        <label for="iterSlider">Diffusion Iteration: <strong id="iterLabel">0</strong> (0 = Start, {steps-1} = Final)</label><br>
        <input type="range" id="iterSlider" min="0" max="{steps-1}" value="0" style="width: 100%;">
    </div>
    
    <div class="clean-text" id="originalTextDisplay">Original: </div>
    <div id="textDisplay"></div>
    <br><br>
    <canvas id="metricChart" width="1200" height="400"></canvas>
    
    <script>
        const fullTrajectoryData = {json.dumps(full_dashboard_data)};
        
        const phraseSelect = document.getElementById('phraseSelect');
        const colorSelect = document.getElementById('colorSelect');
        const checkboxes = document.querySelectorAll('#metricCheckboxes input');
        const iterSlider = document.getElementById('iterSlider');
        const iterLabel = document.getElementById('iterLabel');
        const textDisplay = document.getElementById('textDisplay');
        const originalTextDisplay = document.getElementById('originalTextDisplay');
        
        let chart = null;
        
        const chartColors = [
            'rgba(54, 162, 235, 0.7)',
            'rgba(255, 99, 132, 0.7)',
            'rgba(75, 192, 192, 0.7)',
            'rgba(255, 206, 86, 0.7)',
            'rgba(153, 102, 255, 0.7)',
            'rgba(255, 159, 64, 0.7)',
            'rgba(199, 199, 199, 0.7)',
            'rgba(83, 102, 255, 0.7)'
        ];
        
        function updateView() {{
            const phrase = phraseSelect.value;
            const trajectoryData = fullTrajectoryData[phrase];
            
            const iterIdx = parseInt(iterSlider.value);
            const colorMetric = colorSelect.value;
            
            const selectedMetrics = Array.from(checkboxes)
                .filter(cb => cb.checked)
                .map(cb => cb.value);
            
            if (iterIdx >= trajectoryData.length) return;
            const currentIter = trajectoryData[iterIdx];
            iterLabel.textContent = currentIter.iteration + " (t_global: " + currentIter.t_global.toFixed(2) + ")";
            originalTextDisplay.textContent = "Original: " + phrase;
            
            const labels = currentIter.tokens.map(t => t.clean_token + " -> " + t.token);
            
            textDisplay.innerHTML = '';
            currentIter.tokens.forEach(t => {{
                t['1-MaxPr'] = 1.0 - t['MaxPr'];
                t['1-Coh'] = 1.0 - t['Coh'];

                const span = document.createElement('span');
                span.className = 'token';
                span.textContent = t.token;
                
                let val = t[colorMetric];
                
                let r, g;
                if (colorMetric === 't_local' || colorMetric === 'TrueN' || colorMetric === '1-MaxPr' || colorMetric === '1-Coh' || colorMetric === '1-RawDProx') {{
                    r = Math.round(val * 255);
                    g = Math.round((1 - val) * 200);
                }} else {{
                    r = Math.round((1 - val) * 255);
                    g = Math.round(val * 200);
                }}
                
                span.style.backgroundColor = `rgba(${{r}}, ${{g}}, 50, 0.4)`;
                
                let tooltip = `Clean: ${{t.clean_token}}\\n`;
                tooltip += `t_local: ${{t['t_local'].toFixed(3)}}\\n`;
                tooltip += `TrueN: ${{t['TrueN'].toFixed(3)}}\\n`;
                tooltip += `1-MaxPr: ${{t['1-MaxPr'].toFixed(3)}}\\n`;
                tooltip += `1-RawDProx: ${{t['1-RawDProx'].toFixed(3)}}\\n`;
                tooltip += `Coh: ${{t['Coh'].toFixed(3)}}\\n`;
                
                span.title = tooltip;
                textDisplay.appendChild(span);
            }});
            
            const datasets = selectedMetrics.map((metric, i) => ({{
                label: metric,
                data: currentIter.tokens.map(t => t[metric]),
                backgroundColor: chartColors[i % chartColors.length],
                borderColor: chartColors[i % chartColors.length].replace('0.7', '1.0'),
                borderWidth: 1
            }}));
            
            if (chart) {{
                chart.data.labels = labels;
                chart.data.datasets = datasets;
                chart.update();
            }} else {{
                const ctx = document.getElementById('metricChart').getContext('2d');
                chart = new Chart(ctx, {{
                    type: 'bar',
                    data: {{ labels: labels, datasets: datasets }},
                    options: {{
                        scales: {{ y: {{ beginAtZero: true, max: 1.0 }} }},
                        animation: {{ duration: 200 }},
                        plugins: {{ tooltip: {{ mode: 'index', intersect: false }} }}
                    }}
                }});
            }}
        }}
        
        phraseSelect.addEventListener('change', updateView);
        colorSelect.addEventListener('change', updateView);
        checkboxes.forEach(cb => cb.addEventListener('change', updateView));
        iterSlider.addEventListener('input', updateView);
        
        updateView();
    </script>
</body>
</html>
"""
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Saved dashboard data to {html_path}")

if __name__ == "__main__":
    main()