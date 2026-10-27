import sys
from pathlib import Path
import jupytext

py_path = Path("experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.py")
ipynb_path = Path("experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.ipynb")

if not py_path.exists():
    print(f"Error: {py_path} does not exist!")
    sys.exit(1)

print(f"Converting {py_path} -> {ipynb_path} via Jupytext...")
nb = jupytext.read(str(py_path))
jupytext.write(nb, str(ipynb_path), fmt="ipynb")
print(f"SUCCESS: Synchronized {ipynb_path} ({len(nb.cells)} cells) via Jupytext!")
