import json

with open("experiments/phase 6/kaggle/train_phase6_notebook.py", "r", encoding="utf-8") as f:
    text = f.read()

with open("experiments/phase 6/kaggle/train_phase6_notebook.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

parts = text.split("# %%")
cells_data = []
for p in parts[1:]:
    lines = p.split("\n")
    if lines and lines[0].strip() == "[markdown]":
        cell_type = "markdown"
        content = "\n".join(lines[1:]).strip()
    else:
        cell_type = "code"
        content = p.strip()
    c_lines = [l + "\n" for l in content.split("\n")]
    if c_lines and c_lines[-1].endswith("\n"):
        c_lines[-1] = c_lines[-1][:-1]
    cells_data.append((cell_type, c_lines))

assert len(cells_data) == len(nb["cells"]), f"Length mismatch: {len(cells_data)} vs {len(nb['cells'])}"
for i, (ctype, clines) in enumerate(cells_data):
    nb["cells"][i]["cell_type"] = ctype
    nb["cells"][i]["source"] = clines

with open("experiments/phase 6/kaggle/train_phase6_notebook.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("SUCCESS: Synchronized train_phase6_notebook.ipynb")
