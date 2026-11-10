import json

with open("experiments/phase 6/kaggle/train_phase6_notebook.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    src = "".join(c["source"]).strip()
    first_line = src.splitlines()[0] if src else "<EMPTY>"
    print(f"Cell {i} ({c['cell_type']}, lines={len(src.splitlines())}): {first_line[:80]}")
