import re
import sys

def parse_durations(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='utf-16') as f:
            lines = f.readlines()
            
    steps = []
    current_step = None
    data = {}
    
    for line in lines:
        line = line.strip()
        step_match = re.search(r'^([\d\.]+)s\s+\d+\s+\[XLA METRICS REPORT\] Step (\d+)', line)
        if step_match:
            if current_step is not None:
                steps.append(data)
            timestamp = float(step_match.group(1))
            current_step = int(step_match.group(2))
            data = {'step': current_step, 'timestamp': timestamp}
            continue
            
        if current_step is None:
            continue
            
        if 'Counter:' in line:
            counter_name = line.split('Counter:')[-1].strip()
            data['last_counter'] = counter_name
        elif 'Value:' in line and 'last_counter' in data:
            val_str = line.split('Value:')[-1].strip()
            data[data['last_counter']] = int(val_str)
            del data['last_counter']
            
        if 'Metric:' in line:
            metric_name = line.split('Metric:')[-1].strip()
            data['last_metric'] = metric_name
        elif 'Percentiles:' in line and data.get('last_metric') == 'TensorsGraphSize':
            data['TensorsGraphSize_99'] = line.split('99%=')[-1].strip()
            del data['last_metric']
            
    if current_step is not None:
        steps.append(data)
        
    print(f"{'Step':>6} | {'Timestamp (s)':>15} | {'Duration (s)':>15} | {'Uncached':>10} | {'TensorsGraphSize':>20}")
    print("-" * 75)
    prev_time = None
    for s in steps:
        step = s.get('step', '-')
        t = s.get('timestamp')
        dur = "-"
        if prev_time is not None:
            dur = f"{t - prev_time:.2f}"
        prev_time = t
        
        uncached = s.get('UncachedCompile', 0)
        graph_size = s.get('TensorsGraphSize_99', '-')
        print(f"{step:>6} | {t:>15.2f} | {dur:>15} | {uncached:>10} | {graph_size:>20}")

if __name__ == '__main__':
    parse_durations('experiments/phase 4/tpu kaggle/metrics.txt')
