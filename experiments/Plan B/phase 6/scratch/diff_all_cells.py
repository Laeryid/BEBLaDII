import json
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("experiments/phase 6/kaggle/train_phase6_notebook.py", "r", encoding="utf-8") as f:
    py_text = f.read()

raw_blocks = re.split(r'\n# %%\s*', py_text)

with open("experiments/phase 6/kaggle/train_phase6_notebook.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i in range(len(raw_blocks)):
    py_b = raw_blocks[i]
    nb_c = "".join(nb['cells'][i]['source'])
    # check differences
    py_lines = py_b.strip().splitlines()
    if py_lines and py_lines[0].startswith("[markdown]"):
        py_lines = py_lines[1:] # strip [markdown]
    py_clean = "\n".join(py_lines).strip()
    nb_clean = nb_c.strip()
    
    if py_clean != nb_clean:
        print(f"Diff in cell {i}:")
        print(f"  Py length: {len(py_clean)}, Nb length: {len(nb_clean)}")
        # print first differing line
        p_lines = py_clean.splitlines()
        n_lines = nb_clean.splitlines()
        for idx in range(min(len(p_lines), len(n_lines))):
            if p_lines[idx] != n_lines[idx]:
                print(f"  Line {idx} mismatch:")
                print(f"    Py: {p_lines[idx]}")
                print(f"    Nb: {n_lines[idx]}")
                break
        else:
            if len(p_lines) != len(n_lines):
                print(f"  Different number of lines: Py {len(p_lines)} vs Nb {len(n_lines)}")
