import ast
import json
import builtins

with open("experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

# Collect all defined names cell by cell
known_names = set(dir(builtins))
# Add common known modules/globals imported dynamically or injected by environment
known_names.update([
    'torch', 'nn', 'F', 'math', 'os', 'sys', 're', 'time', 'gc', 'glob', 'shutil',
    'einops', 'wandb', 'pd', 'pa', 'pq', 'transformers',
    'xm', 'xla_device', 'xs', 'xr', 'met', 'AutoTokenizer', 'AutoConfig',
    'UserSecretsClient', 'get_secret', '__name__', '__file__',
    'mesh', 'device', 'args', 'model', 'optimizer', 'lr_scheduler', 'ema',
    'train_loader', 'val_loader', 'epoch', 'step', 'loss', 'outputs', 'batch'
])

undefined_issues = []

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = "".join(cell['source'])
    if not src.strip():
        continue
    
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        undefined_issues.append((idx, f"SyntaxError: {e}"))
        continue
    
    # Check what this cell uses vs what is known
    # Note: functions can reference globals that are defined later, but classes and top-level code cannot.
    # Specifically check top-level and class-level expressions
    for node in tree.body:
        # If it's a class definition, check base classes and class-level statements
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id not in known_names:
                    undefined_issues.append((idx, f"Class {node.name} base {base.id} not defined"))
            # Check methods
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                    # In __init__, check direct instantiations: X(...)
                    for sub in ast.walk(item):
                        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                            if sub.func.id not in known_names:
                                undefined_issues.append((idx, f"In {node.name}.__init__: called undefined {sub.func.id}"))

        # Collect definitions from this cell into known_names
        for sub in ast.walk(node):
            if isinstance(sub, ast.FunctionDef):
                known_names.add(sub.name)
            elif isinstance(sub, ast.ClassDef):
                known_names.add(sub.name)
            elif isinstance(sub, ast.Import):
                for alias in sub.names:
                    known_names.add(alias.asname or alias.name)
            elif isinstance(sub, ast.ImportFrom):
                for alias in sub.names:
                    known_names.add(alias.asname or alias.name)
            elif isinstance(sub, ast.Assign):
                for target in sub.targets:
                    if isinstance(target, ast.Name):
                        known_names.add(target.id)

print(f"Audit completed. Found {len(undefined_issues)} potential issues:")
for issue in undefined_issues:
    print(f"  Cell {issue[0]}: {issue[1]}")
