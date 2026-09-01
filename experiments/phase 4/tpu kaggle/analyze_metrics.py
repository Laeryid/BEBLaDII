import re

def parse_metrics(filepath):
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
        step_match = re.search(r'\[XLA METRICS REPORT\] Step (\d+)', line)
        if step_match:
            if current_step is not None:
                steps.append(data)
            current_step = int(step_match.group(1))
            data = {'step': current_step}
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
        elif 'Rate:' in line and data.get('last_metric') == 'ExecuteReplicatedTime':
            data['ExecuteReplicatedTime_Rate'] = line.split('Rate:')[-1].strip()
        elif 'Percentiles:' in line and data.get('last_metric') == 'ExecuteReplicatedTime':
            data['ExecuteReplicatedTime_99'] = line.split('99%=')[-1].strip()
            del data['last_metric']
            
    if current_step is not None:
        steps.append(data)
        
    print(f"{'Step':>6} | {'UncachedCompile':>15} | {'local_scalar_dense':>20} | {'ExecuteReplicated':>17} | {'CachedCompile':>13} | {'ExecRepTime (99%)':>18}")
    print("-" * 105)
    for s in steps:
        step = s.get('step', '-')
        uncached = s.get('UncachedCompile', 0)
        scalar = s.get('aten::_local_scalar_dense', 0)
        exec_rep = s.get('ExecuteReplicated', 0)
        cached = s.get('CachedCompile', 0)
        t_99 = s.get('ExecuteReplicatedTime_99', '-')
        print(f"{step:>6} | {uncached:>15} | {scalar:>20} | {exec_rep:>17} | {cached:>13} | {t_99:>18}")

if __name__ == '__main__':
    parse_metrics('metrics.txt')
