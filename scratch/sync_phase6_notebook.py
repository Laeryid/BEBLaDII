import json

py_path = "experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.py"
ipynb_path = "experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.ipynb"

with open(py_path, "r", encoding="utf-8") as f:
    py_text = f.read()

# 1. Извлечение sync_to_gcs_and_delete
c15_start = py_text.find("def sync_to_gcs_and_delete(")
c15_end = py_text.find("def get_gcs_checkpoints(")
if c15_start == -1 or c15_end == -1:
    raise ValueError("Could not find cell 15 bounds in .py")
code_c15 = py_text[c15_start:c15_end].rstrip()

# 2. Извлечение get_gcs_checkpoints
c16_start = c15_end
c16_end = py_text.find("def get_latest_gcs_checkpoint(")
if c16_start == -1 or c16_end == -1:
    raise ValueError("Could not find cell 16 bounds in .py")
code_c16 = py_text[c16_start:c16_end].rstrip()

# 3. Извлечение BEBLaDIIPhase6
c33_start = py_text.find("class BEBLaDIIPhase6(nn.Module):")
c33_end = py_text.find("def compute_phase6_loss(")
if c33_start == -1 or c33_end == -1:
    raise ValueError("Could not find BEBLaDIIPhase6 bounds in .py")
code_c33 = py_text[c33_start:c33_end].rstrip()

# 4. Извлечение main
target_marker = "if __name__ == '__main__':"
if target_marker not in py_text:
    target_marker = 'if __name__ == "__main__":'

main_idx = py_text.find("def main():")
end_idx = py_text.find(target_marker)

if main_idx == -1 or end_idx == -1:
    raise ValueError("Could not find def main() or __main__ in .py")

code_main = py_text[main_idx:end_idx].rstrip()

# 5. Обновление ipynb
with open(ipynb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

def to_lines(code_str):
    lines = [l + "\n" for l in code_str.split("\n")]
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    return lines

# Проверяем, что заголовки совпадают с ячейками
assert "def sync_to_gcs_and_delete" in nb["cells"][15]["source"][0], f"Cell 15 mismatch: {nb['cells'][15]['source'][:1]}"
assert "def get_gcs_checkpoints" in nb["cells"][16]["source"][0], f"Cell 16 mismatch: {nb['cells'][16]['source'][:1]}"
assert "class BEBLaDIIPhase6" in nb["cells"][33]["source"][0], f"Cell 33 mismatch: {nb['cells'][33]['source'][:1]}"
assert "def main" in nb["cells"][37]["source"][0], f"Cell 37 mismatch: {nb['cells'][37]['source'][:1]}"

nb["cells"][15]["source"] = to_lines(code_c15)
nb["cells"][16]["source"] = to_lines(code_c16)
nb["cells"][33]["source"] = to_lines(code_c33)
nb["cells"][37]["source"] = to_lines(code_main)

with open(ipynb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Synchronized Cells 15, 16, 33, and 37 successfully into train_phase6_tpu_notebook.ipynb!")
