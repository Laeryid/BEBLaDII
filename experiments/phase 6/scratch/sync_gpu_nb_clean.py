import json
import re

with open("experiments/phase 6/kaggle/train_phase6_notebook.py", "r", encoding="utf-8") as f:
    py_text = f.read()

raw_blocks = re.split(r'\n# %%\s*', py_text)

with open("experiments/phase 6/kaggle/train_phase6_notebook.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update cell sources
for i in range(len(raw_blocks)):
    block = raw_blocks[i]
    lines = block.splitlines(keepends=True)
    if lines and lines[0].startswith("[markdown]"):
        lines = lines[1:] # remove [markdown] marker
        cell_type = "markdown"
    elif i == 0:
        # first block has # %% [markdown]
        if lines and lines[0].startswith("# %% [markdown]"):
            lines = lines[1:]
        cell_type = "markdown"
    else:
        cell_type = "code"

    # remove leading blank line if present
    if lines and lines[0].strip() == "":
        lines = lines[1:]

    nb["cells"][i]["source"] = lines
    nb["cells"][i]["cell_type"] = cell_type
    if cell_type == "code":
        nb["cells"][i]["outputs"] = []
        nb["cells"][i]["execution_count"] = None

with open("experiments/phase 6/kaggle/train_phase6_notebook.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Synchronized train_phase6_notebook.ipynb successfully!")
