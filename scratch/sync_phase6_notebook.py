import json

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

# 1. Config (Cell 23)
code_config = extract_block("class Config:", "args = Config()")
# 2. CAPromptLayer (Cell 32)
code_ca = extract_block("class CAPromptLayer(nn.Module):", "def _unwrap_module(m):")
# 3. BEBLaDIIPhase6 + patch_modernbert_xla_safe_mask (Cell 36)
code_model = extract_block("def patch_modernbert_xla_safe_mask(", "def compute_phase6_loss(")
# 4. run_nan_diagnosis (Cell 40)
code_diag = extract_block("def run_nan_diagnosis(", "def main():")
# 5. main (Cell 41)
code_main = extract_block("def main():", 'if __name__ == "__main__":')

# Проверка совпадения ячеек
assert "class Config" in nb["cells"][23]["source"][0]
assert "class CAPromptLayer" in nb["cells"][32]["source"][0]
assert "class BEBLaDIIPhase6" in nb["cells"][36]["source"][0] or "def patch_modernbert_xla_safe_mask" in nb["cells"][36]["source"][0]
assert "def run_nan_diagnosis" in nb["cells"][40]["source"][0]
assert "def main" in nb["cells"][41]["source"][0]

nb["cells"][23]["source"] = to_lines(code_config)
nb["cells"][32]["source"] = to_lines(code_ca)
nb["cells"][36]["source"] = to_lines(code_model)
nb["cells"][40]["source"] = to_lines(code_diag)
nb["cells"][41]["source"] = to_lines(code_main)

with open(ipynb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Successfully synced train_phase6_tpu_notebook.py to .ipynb!")
