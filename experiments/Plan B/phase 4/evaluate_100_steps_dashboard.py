import sys
import os
import torch
import math
import json
import webbrowser
import datetime
import torch.nn.functional as F

PROJECT_ROOT = "C:/Experiments/BEBLaDII"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments/phase 4"))

from evaluate_phase4_checkpoints import (
    BEBLaDIIPhase4aEval, spherical_noise, safe_normalize, 
    AutoTokenizer, AutoModelForCausalLM, ModernLatentDecoder
)

def hierarchical_slerp_sampler_dashboard(diff_model, input_ids, attn_mask, z_clean, tokenizer, decoder, lm_head_weight, texts, model_version, run_date, steps=100, device="cpu"):
    B, T = input_ids.shape
    D = 1024

    data = {}
    if run_date not in data: data[run_date] = {}
    if model_version not in data[run_date]: data[run_date][model_version] = {}
    
    t_init = torch.zeros(B, T, device=device)
    for b in range(B):
        t_b = torch.rand(T, device=device) * 0.7 + 0.2
        indices = torch.randperm(T, device=device)
        num_clean = max(1, int(0.20 * T))
        num_noise = max(1, int(0.10 * T))
        t_b[indices[:num_clean]] = 0.1
        t_b[indices[num_clean:num_clean+num_noise]] = 1.0
        t_init[b] = t_b
        
    x_t = spherical_noise(z_clean, t_init)
    
    def log_state(iteration, current_x, current_t):
        with torch.no_grad():
            dec_out = decoder(current_x.to(next(decoder.parameters()).dtype))
            logits = F.linear(dec_out.float(), lm_head_weight.float())
            token_ids = logits.argmax(dim=-1)
            
            for b in range(B):
                phrase_title = f"Phrase {b}: {texts[b][:20]}..."
                if phrase_title not in data[run_date][model_version]:
                    data[run_date][model_version][phrase_title] = []
                
                mask_len = int(attn_mask[b].sum().item())
                tokens_data = []
                for j in range(mask_len):
                    t_val = current_t[b, j].item()
                    word = tokenizer.decode([token_ids[b, j].item()])
                    tokens_data.append({"token": word, "noise": t_val})
                data[run_date][model_version][phrase_title].append({
                    "iteration": iteration,
                    "tokens": tokens_data
                })

    log_state(0, x_t, t_init)
    
    for i in range(steps):
        t_global = max(0.0, 1.0 - i * (1.0 / steps))
        t_global_tensor = torch.tensor([t_global] * B, device=device)
        
        cos_sim = (x_t * z_clean).sum(dim=-1)
        cos_sim_clamped = torch.clamp(cos_sim, -0.9999, 0.9999)
        t_reported = (2.0 / math.pi) * torch.acos(cos_sim_clamped)
        
        log_state(i + 1, x_t, t_reported)
        
        with torch.no_grad():
            out_sc = diff_model(input_ids, attn_mask, t_global=t_global_tensor, t_reported=t_reported, z_noisy_override=x_t)
            sc_est = out_sc["dus_final"].detach()
            out = diff_model(input_ids, attn_mask, t_global=t_global_tensor, t_reported=t_reported, self_cond=sc_est, z_noisy_override=x_t)
            z_pred_raw = out["dus_final"]
            
        dt = 1.0 / steps
        t_next = torch.clamp(t_reported - dt, min=0.0)
        
        gate_t = torch.sin(t_reported * (math.pi / 2)).unsqueeze(-1)
        z_pred = safe_normalize(gate_t * z_pred_raw + (1.0 - gate_t) * x_t, dim=-1)

        theta_now = (t_reported * (math.pi / 2)).unsqueeze(-1)
        theta_next = (t_next * (math.pi / 2)).unsqueeze(-1)
        sin_theta_now = torch.sin(theta_now)
        sin_theta_now = torch.where(sin_theta_now < 1e-5, torch.ones_like(sin_theta_now) * 1e-5, sin_theta_now)
        
        w_pred = torch.sin(theta_now - theta_next) / sin_theta_now
        w_cur  = torch.sin(theta_next) / sin_theta_now
        
        z_next = w_pred * z_pred + w_cur * x_t
        x_t = torch.where(theta_now > 1e-5, safe_normalize(z_next, dim=-1), safe_normalize(z_pred, dim=-1))
        
    return data

# generate HTML from data
def generate_html(js_data, html_path):
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>BEBLaDII Noise Trajectory (100 Steps)</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        select, input {{ margin: 10px 0; font-size: 16px; }}
        .slider-container {{ margin: 20px 0; }}
        #textDisplay {{ font-size: 20px; margin-top: 20px; line-height: 1.5; }}
        .token {{ padding: 2px 4px; border-radius: 4px; margin-right: 4px; display: inline-block; border: 1px solid #ccc; min-width: 20px; text-align: center; }}
    </style>
</head>
<body>
    <h1>Token Noise Trajectory (100 Steps)</h1>
    
    <label for="dateSelect">Select Run Date:</label>
    <select id="dateSelect" style="width: 20%;"></select>
    
    <label for="versionSelect">Select Model Version:</label>
    <select id="versionSelect" style="width: 30%;"></select>
    
    <label for="phraseSelect">Select Phrase:</label>
    <select id="phraseSelect" style="width: 50%;"></select>
    
    <div class="slider-container">
        <label for="iterSlider">Diffusion Iteration: <strong id="iterLabel">0</strong> (0 = Start, 100 = Final)</label><br>
        <input type="range" id="iterSlider" min="0" max="100" value="0" style="width: 100%;">
    </div>
    
    <div id="textDisplay"></div>
    <br><br>
    <canvas id="noiseChart" width="1200" height="400"></canvas>
    
    <script>
        const trajectoryData = {json.dumps(js_data)};
        
        const dateSelect = document.getElementById('dateSelect');
        const versionSelect = document.getElementById('versionSelect');
        const phraseSelect = document.getElementById('phraseSelect');
        const iterSlider = document.getElementById('iterSlider');
        const iterLabel = document.getElementById('iterLabel');
        const textDisplay = document.getElementById('textDisplay');
        
        let chart = null;
        
        for (const date in trajectoryData) {{
            const option = document.createElement('option');
            option.value = date;
            option.textContent = date;
            dateSelect.appendChild(option);
        }}
        
        function updateVersionSelect() {{
            const date = dateSelect.value;
            versionSelect.innerHTML = '';
            for (const version in trajectoryData[date]) {{
                const option = document.createElement('option');
                option.value = version;
                option.textContent = version;
                versionSelect.appendChild(option);
            }}
        }}

        function updatePhraseSelect() {{
            const date = dateSelect.value;
            const version = versionSelect.value;
            phraseSelect.innerHTML = '';
            for (const phrase in trajectoryData[date][version]) {{
                const option = document.createElement('option');
                option.value = phrase;
                option.textContent = phrase;
                phraseSelect.appendChild(option);
            }}
        }}
        
        function updateView() {{
            const date = dateSelect.value;
            const version = versionSelect.value;
            const phrase = phraseSelect.value;
            const iterIdx = parseInt(iterSlider.value);
            
            if (!trajectoryData[date] || !trajectoryData[date][version] || !trajectoryData[date][version][phrase]) return;
            const versionData = trajectoryData[date][version][phrase];
            if (iterIdx >= versionData.length) return;
            
            const currentIter = versionData[iterIdx];
            iterLabel.textContent = currentIter.iteration;
            
            const labels = currentIter.tokens.map(t => t.token);
            const data = currentIter.tokens.map(t => t.noise);
            
            // Update Text Visualization
            textDisplay.innerHTML = '';
            currentIter.tokens.forEach(t => {{
                const span = document.createElement('span');
                span.className = 'token';
                span.textContent = t.token;
                const r = Math.round(t.noise * 255);
                const g = Math.round((1 - t.noise) * 200);
                span.style.backgroundColor = `rgba(${{r}}, ${{g}}, 50, 0.4)`;
                span.title = `Noise: ${{t.noise.toFixed(3)}}`;
                textDisplay.appendChild(span);
            }});
            
            // Update Chart
            if (chart) {{
                chart.data.labels = labels;
                chart.data.datasets[0].data = data;
                chart.update();
            }} else {{
                const ctx = document.getElementById('noiseChart').getContext('2d');
                chart = new Chart(ctx, {{
                    type: 'bar',
                    data: {{
                        labels: labels,
                        datasets: [{{
                            label: 'Noise Level (t)',
                            data: data,
                            backgroundColor: 'rgba(255, 99, 132, 0.5)',
                            borderColor: 'rgba(255, 99, 132, 1)',
                            borderWidth: 1
                        }}]
                    }},
                    options: {{
                        scales: {{
                            y: {{ beginAtZero: true, max: 1.0 }}
                        }},
                        animation: {{ duration: 200 }}
                    }}
                }});
            }}
        }}
        
        dateSelect.addEventListener('change', () => {{
            updateVersionSelect();
            updatePhraseSelect();
            iterSlider.max = trajectoryData[dateSelect.value][versionSelect.value][phraseSelect.value].length - 1;
            iterSlider.value = 0;
            updateView();
        }});
        
        versionSelect.addEventListener('change', () => {{
            updatePhraseSelect();
            iterSlider.max = trajectoryData[dateSelect.value][versionSelect.value][phraseSelect.value].length - 1;
            iterSlider.value = 0;
            updateView();
        }});
        
        phraseSelect.addEventListener('change', () => {{
            iterSlider.max = trajectoryData[dateSelect.value][versionSelect.value][phraseSelect.value].length - 1;
            iterSlider.value = 0;
            updateView();
        }});
        
        iterSlider.addEventListener('input', updateView);
        
        // Init
        updateVersionSelect();
        updatePhraseSelect();
        iterSlider.max = trajectoryData[dateSelect.value][versionSelect.value][phraseSelect.value].length - 1;
        updateView();
        
    </script>
</body>
</html>
"""
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"HTML dashboard generated at: {html_path}")
    webbrowser.open('file://' + html_path.replace('\\', '/'))


def main():
    run_date = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_qwen = "Qwen/Qwen2.5-1.5B"
    base_modernbert = "answerdotai/ModernBERT-large"

    print("Loading AutoTokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_qwen)
    
    print("Loading AutoModelForCausalLM for lm_head...")
    causal_model = AutoModelForCausalLM.from_pretrained(base_qwen, torch_dtype=torch.float32)
    lm_head_weight = causal_model.lm_head.weight.detach().clone().to(device)
    del causal_model

    print("Loading ModernLatentDecoder...")
    dus_weights_path = "C:/Experiments/BEBLaDII/kaggle_upload_1_2/AWAKENED_WEIGHTS_FINAL.pt"
    decoder = ModernLatentDecoder(latent_dim=1024, qwen_dim=1536, num_layers=3, dus_weights_path=dus_weights_path).to(device)
    dec_paths = [
        "C:/Experiments/BEBLaDII/experiments/phase 2/planB_phase2_checkpoints_decoder_step_9000.pth",
        "C:/Experiments/BEBLaDII/BEBLaDII-planB-Phase3-Data/planB_phase2_phase2_decoder_step_8000.pth",
    ]
    dec_path = next((p for p in dec_paths if os.path.exists(p)), None)
    if dec_path:
        st = torch.load(dec_path, map_location="cpu", weights_only=False)
        decoder.load_state_dict({k.replace("decoder.", ""): v for k, v in st.get("decoder", st).items()}, strict=False)
        print(f"Decoder loaded from {dec_path}")
    decoder.eval()

    print("Loading Diff Model...")
    diff_model = BEBLaDIIPhase4aEval(embedding_model_path=base_qwen, modernbert_path=base_modernbert)
    diff_model.to(device)
    
    ckpt_path = "C:/Experiments/BEBLaDII/experiments/phase 4/local_checkpoints/phase4_step_49995.pth"
    print(f"Loading checkpoint {ckpt_path}...")
    state = torch.load(ckpt_path, map_location="cpu")

    enc_paths = [
        "C:/Experiments/BEBLaDII/experiments/phase 1/planB_phase1_checkpoints_phase1_vae_step_20000.pth",
        "C:/Experiments/BEBLaDII/BEBLaDII-planB-Phase3-Data/planB_phase1_checkpoints_phase1_vae_step_20000.pth",
    ]
    enc_path = next((p for p in enc_paths if os.path.exists(p)), None)
    if enc_path:
        st = torch.load(enc_path, map_location="cpu")
        diff_model.encoder.load_state_dict({k.replace("encoder.", ""): v for k, v in st.get("encoder", st).items()}, strict=False)
    
    dus_dict = state.get("dus", {})
    if dus_dict:
        clean_dus = {}
        for k, v in dus_dict.items():
            clean_k = k.replace("_orig_module.", "").replace("student.model.", "").replace("model.", "")
            clean_dus[clean_k] = v
        diff_model.dus.load_state_dict(clean_dus, strict=False)

    for name in ["adaLN_attn", "adaLN_mlp", "t_proj_global", "t_proj_token", "t_joint_proj", "sep_embed", "self_cond_proj"]:
        comp_state = state.get(name, {})
        if comp_state:
            if name == "sep_embed":
                diff_model.sep_embed.copy_(comp_state)
            else:
                clean_comp = {k.replace("_orig_module.", ""): v for k, v in comp_state.items()}
                getattr(diff_model, name).load_state_dict(clean_comp, strict=True)
                
    diff_model.eval()

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Мама мыла раму, а папа чинил телевизор.",
        "Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics to solve problems too complex for classical computers."
    ]

    tok = tokenizer(texts, return_tensors="pt", add_special_tokens=False, padding=True)
    input_ids_batch = tok.input_ids.to(device)
    mask_batch = tok.attention_mask.to(device)

    with torch.no_grad():
        q_embs = diff_model.qwen_embeddings(input_ids_batch)
        z_clean_unnorm, _, _ = diff_model.encoder(q_embs)
        z_clean = safe_normalize(z_clean_unnorm.float(), dim=-1)

    print("Running 100-step diffusion inference...")
    js_data = hierarchical_slerp_sampler_dashboard(
        diff_model=diff_model,
        input_ids=input_ids_batch,
        attn_mask=mask_batch,
        z_clean=z_clean,
        tokenizer=tokenizer,
        decoder=decoder,
        lm_head_weight=lm_head_weight,
        texts=texts,
        model_version="49995_LIVE_100_Steps",
        run_date=run_date,
        steps=100,
        device=device
    )

    html_path = r"C:\Experiments\BEBLaDII\experiments\phase 4\local_checkpoints\noise_dashboard_100_steps.html"
    generate_html(js_data, html_path)

if __name__ == "__main__":
    main()
