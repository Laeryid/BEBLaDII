from transformers import AutoConfig, AutoModelForCausalLM
import torch

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
print(f"Teacher layers in config: {config.num_hidden_layers}")

# Также проверим, какие индексы доступны в hidden_states
# Для этого не нужно грузить всю модель, достаточно конфига.
# Обычно hidden_states[0] - embeddings, hidden_states[1...N] - layers, hidden_states[N+1] - final norm (опционально)
