import csv
import json
import os
import webbrowser

csv_path = r"C:\Experiments\BEBLaDII\experiments\phase 4\local_checkpoints\diffusion_trajectory.csv"
html_path = r"C:\Experiments\BEBLaDII\experiments\phase 4\local_checkpoints\noise_dashboard.html"

if not os.path.exists(csv_path):
    print("CSV file not found!")
    exit(1)

data = {}
# data[run_date][model_version][phrase][iteration] = [{"token": "word", "noise": 0.5}, ...]

with open(csv_path, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        run_date = row.get("RunDate", "Unknown")
        phrase = row["Phrase"]
        model_version = row["ModelVersion"]
        iteration = int(row["Iteration"])
        word = row["TokenString"]
        noise = float(row["NoiseValue"])
        
        if run_date not in data:
            data[run_date] = {}
        if model_version not in data[run_date]:
            data[run_date][model_version] = {}
        if phrase not in data[run_date][model_version]:
            data[run_date][model_version][phrase] = {}
        if iteration not in data[run_date][model_version][phrase]:
            data[run_date][model_version][phrase][iteration] = []
            
        data[run_date][model_version][phrase][iteration].append({"token": word, "noise": noise})

# Convert to a format easy for JS
js_data = {}
for run_date, versions in data.items():
    js_data[run_date] = {}
    for version, phrases in versions.items():
        js_data[run_date][version] = {}
        for phrase, iters in phrases.items():
            sorted_iters = sorted(iters.keys())
            js_data[run_date][version][phrase] = []
            for it in sorted_iters:
                js_data[run_date][version][phrase].append({"iteration": it, "tokens": iters[it]})

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>BEBLaDII Noise Trajectory</title>
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
    <h1>Token Noise Trajectory</h1>
    
    <label for="dateSelect">Select Run Date:</label>
    <select id="dateSelect" style="width: 20%;"></select>
    
    <label for="versionSelect">Select Model Version:</label>
    <select id="versionSelect" style="width: 30%;"></select>
    
    <label for="phraseSelect">Select Phrase:</label>
    <select id="phraseSelect" style="width: 50%;"></select>
    
    <div class="slider-container">
        <label for="iterSlider">Diffusion Iteration: <strong id="iterLabel">0</strong> (0 = Start, 25 = Final)</label><br>
        <input type="range" id="iterSlider" min="0" max="25" value="0" style="width: 100%;">
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
