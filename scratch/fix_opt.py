import os

files = [
    'experiments/phase 6/kaggle/train_phase6_notebook.py',
    'experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.py'
]

for file_path in files:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix the optimizer parameter grouping
    bad_opt = '''    dus_params = []
    for name, p in model.dus.named_parameters():
        if p.requires_grad:
            dus_params.append(p)'''
            
    good_opt = '''    dus_params = []
    for name, p in model.dus.named_parameters():
        if p.requires_grad and "ca_layer" not in name:
            dus_params.append(p)'''
            
    content = content.replace(bad_opt, good_opt)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

print("Fixed optimizer grouping in both scripts.")
