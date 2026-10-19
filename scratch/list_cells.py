import json

nb_path = "experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    first_line = src.strip().split("\n")[0] if src.strip() else "<EMPTY>"
    print(f"Cell {i} ({cell.get('cell_type')}): {first_line[:80]}")
