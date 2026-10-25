import json

with open("experiments/phase 6/kaggle/train_phase6_notebook.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

c5 = "".join(nb["cells"][5]["source"])
assert "epochs        = 500" in c5, "c5 failed"
assert "max_steps     = 200005" in c5, "c5 steps failed"

c9 = "".join(nb["cells"][9]["source"])
assert "class SinusoidalEmbedding" in c9, "c9 failed"

c15 = "".join(nb["cells"][15]["source"])
assert "def main():" in c15, "c15 main failed"
assert 'if __name__ == "__main__":' in c15, "c15 entry failed"

print("SUCCESS: All assertions passed!")
