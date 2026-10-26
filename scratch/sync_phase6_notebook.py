import json
import os
import sys

py_path = "experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.py"
ipynb_path = "experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.ipynb"

with open(py_path, "r", encoding="utf-8") as f:
    py_text = f.read()

with open(ipynb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

def to_lines(code_str):
    lines = [l + "\n" for l in code_str.split("\n")]
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    return lines

def extract_block(start_marker, end_marker):
    start = py_text.find(start_marker)
    if start == -1:
        raise ValueError(f"Start marker not found: {start_marker}")
    end = py_text.find(end_marker, start)
    if end == -1:
        raise ValueError(f"End marker not found: {end_marker}")
    return py_text[start:end].rstrip()

def find_cell(marker):
    for i, cell in enumerate(nb["cells"]):
        source = "".join(cell.get("source", []))
        if marker in source:
            return i
    raise ValueError(f"Cell with marker '{marker}' not found in notebook!")

# 1. Config
idx_config = find_cell("class Config:")
code_config = extract_block("class Config:", "args = Config()")
nb["cells"][idx_config]["source"] = to_lines(code_config)
print(f"Synced Config to cell {idx_config}")

# 2. CAPromptLayer
idx_ca = find_cell("class CAPromptLayer(")
code_ca = extract_block("def rotate_half(x: torch.Tensor)", "def _unwrap_module(m):")
nb["cells"][idx_ca]["source"] = to_lines(code_ca)
print(f"Synced CAPromptLayer to cell {idx_ca}")

# 2.1 BEBLaDIIPhase6
idx_model = find_cell("class BEBLaDIIPhase6(nn.Module):")
code_model = extract_block("class BEBLaDIIPhase6(nn.Module):", "def compute_phase6_loss(")
nb["cells"][idx_model]["source"] = to_lines(code_model)
print(f"Synced BEBLaDIIPhase6 to cell {idx_model}")

# 3. compute_phase6_loss
idx_loss = find_cell("def compute_phase6_loss(")
code_loss = extract_block("def compute_phase6_loss(", "def _set_seed(")
nb["cells"][idx_loss]["source"] = to_lines(code_loss)
print(f"Synced compute_phase6_loss to cell {idx_loss}")

# 4. main()
idx_main = find_cell("def main():")
code_main = extract_block("def main():", 'if __name__ == "__main__":')
nb["cells"][idx_main]["source"] = to_lines(code_main)
print(f"Synced main() to cell {idx_main}")

with open(ipynb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("SUCCESS: Synced train_phase6_tpu_notebook.py to .ipynb!")
