import json

py_path = "experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.py"
ipynb_path = "experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.ipynb"

with open(py_path, "r", encoding="utf-8") as f:
    py_text = f.read()

# 1. Извлечение cell 13: get_gcs_client + is_state_dict_finite + sync_to_gcs_and_delete
c13_start = py_text.find("def get_gcs_client():")
c13_end = py_text.find("def get_gcs_checkpoints(")
if c13_start == -1 or c13_end == -1:
    raise ValueError("Could not find cell 13 bounds in .py")
code_c13 = py_text[c13_start:c13_end].rstrip()

# 2. Извлечение cell 14: get_gcs_checkpoints + get_latest_gcs_checkpoint
c14_start = c13_end
c14_end = py_text.find("def sample_token_noise_levels(")
if c14_end == -1:
    raise ValueError("Could not find cell 14 bounds in .py")
code_c14 = py_text[c14_start:c14_end].rstrip()

# 3. Извлечение EMATracker (Cell 19)
ema_start = py_text.find("class EMATracker:")
ema_end = py_text.find("class QADataset(")
if ema_start == -1 or ema_end == -1:
    raise ValueError("Could not find EMATracker or QADataset in .py")
code_ema = py_text[ema_start:ema_end].rstrip()

# 4. Извлечение main (Cell 34)
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

nb["cells"][13]["source"] = to_lines(code_c13)
nb["cells"][14]["source"] = to_lines(code_c14)
nb["cells"][19]["source"] = to_lines(code_ema)
nb["cells"][34]["source"] = to_lines(code_main)

with open(ipynb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Synchronized Cells 13, 14, 19, and 34 successfully!")
