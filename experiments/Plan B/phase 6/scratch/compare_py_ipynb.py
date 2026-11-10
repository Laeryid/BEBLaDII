import json
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("experiments/phase 6/kaggle/train_phase6_notebook.py", "r", encoding="utf-8") as f:
    py_text = f.read()

# Let's split py_text by '# %%'
raw_blocks = re.split(r'\n# %%\s*', py_text)
print(f"Total py blocks: {len(raw_blocks)}")

with open("experiments/phase 6/kaggle/train_phase6_notebook.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total ipynb cells: {len(nb['cells'])}")

for i, block in enumerate(raw_blocks):
    first_few = [l.strip() for l in block.splitlines() if l.strip()][:2]
    print(f"Py block {i}: {first_few}")

print("\n--- IPYNB CELLS ---")
for i, cell in enumerate(nb['cells']):
    src = [l.strip() for l in cell['source'] if l.strip()][:2]
    print(f"Nb cell {i} ({cell['cell_type']}): {src}")

