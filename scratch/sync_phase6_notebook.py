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

# Blocks to sync:
# 1. Mesh setup
idx_mesh = find_cell("GLOBAL_SPMD_MESH = None")
code_mesh = extract_block("GLOBAL_SPMD_MESH = None", "GLOBAL_NAN_NAMES = []")
nb["cells"][idx_mesh]["source"] = to_lines(code_mesh)
print(f"Synced mesh block to cell {idx_mesh}")

# 2. Config
idx_config = find_cell("class Config:")
code_config = extract_block("class Config:", "args = Config()")
nb["cells"][idx_config]["source"] = to_lines(code_config)
print(f"Synced Config block to cell {idx_config}")

# 2b. EMATracker
idx_ema = find_cell("class EMATracker:")
code_ema = extract_block("class EMATracker:", "class QADataset(")
nb["cells"][idx_ema]["source"] = to_lines(code_ema)
print(f"Synced EMATracker block to cell {idx_ema}")

# 3. spherical_noise
idx_noise = find_cell("def spherical_noise(")
code_noise = extract_block("def spherical_noise(", "class AdaLNModulation(")
nb["cells"][idx_noise]["source"] = to_lines(code_noise)
print(f"Synced spherical_noise to cell {idx_noise}")

# 4. CAPromptLayer (including RotaryEmbedding if in same or separate cell)
idx_ca = find_cell("class CAPromptLayer(")
code_ca = extract_block("def rotate_half(", "def _unwrap_module(m):")
nb["cells"][idx_ca]["source"] = to_lines(code_ca)
print(f"Synced RotaryEmbedding & CAPromptLayer to cell {idx_ca}")

# 5. BEBLaDIIPhase6 + patch_modernbert_xla_safe_mask
idx_model = find_cell("def patch_modernbert_xla_safe_mask(")
code_model = extract_block("def patch_modernbert_xla_safe_mask(", "def compute_phase6_loss(")
nb["cells"][idx_model]["source"] = to_lines(code_model)
print(f"Synced model block to cell {idx_model}")

# 6. run_nan_diagnosis
idx_diag = find_cell("def run_nan_diagnosis(")
code_diag = extract_block("def run_nan_diagnosis(", "def main():")
nb["cells"][idx_diag]["source"] = to_lines(code_diag)
print(f"Synced run_nan_diagnosis to cell {idx_diag}")

# 7. main
idx_main = find_cell("def main():")
code_main = extract_block("def main():", 'if __name__ == "__main__":')
nb["cells"][idx_main]["source"] = to_lines(code_main)
print(f"Synced main to cell {idx_main}")

with open(ipynb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Successfully synced train_phase6_tpu_notebook.py to .ipynb!")
