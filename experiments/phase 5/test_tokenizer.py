import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
text = "ёлочка ďábelské"
toks = tokenizer(text).input_ids
print("Tokens:", toks)
for t in toks:
    print(f"{t}: {tokenizer.decode([t])}")
