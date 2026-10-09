import json

py_path = "experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.py"
ipynb_path = "experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.ipynb"

with open(py_path, "r", encoding="utf-8") as f:
    py_text = f.read()

# 1. Извлечение EMATracker
ema_start = py_text.find("class EMATracker:")
ema_end = py_text.find("class QADataset(")
if ema_start == -1 or ema_end == -1:
    raise ValueError("Could not find EMATracker or QADataset in .py")
ema_code = py_text[ema_start:ema_end].rstrip()

# 2. Извлечение main
target_marker = "if __name__ == '__main__':"
if target_marker not in py_text:
    target_marker = 'if __name__ == "__main__":'

main_idx = py_text.find("def main():")
end_idx = py_text.find(target_marker)

if main_idx == -1 or end_idx == -1:
    raise ValueError("Could not find def main() or __main__ in .py")

main_code = py_text[main_idx:end_idx].rstrip()

# 3. Обновление ipynb
with open(ipynb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 19: EMATracker
lines_ema = [l + "\n" for l in ema_code.split("\n")]
if lines_ema and lines_ema[-1].endswith("\n"):
    lines_ema[-1] = lines_ema[-1][:-1]
nb["cells"][19]["source"] = lines_ema

# Cell 34: main()
lines_main = [l + "\n" for l in main_code.split("\n")]
if lines_main and lines_main[-1].endswith("\n"):
    lines_main[-1] = lines_main[-1][:-1]
nb["cells"][34]["source"] = lines_main

with open(ipynb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Synchronized Cell 19 and Cell 34 successfully!")
