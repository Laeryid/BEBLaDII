import os
import sys
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8')
PROJECT_ROOT = "C:/Experiments/BEBLaDII"
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments", "phase 4"))
from evaluate_phase4_checkpoints import BEBLaDIIPhase4aEval
from transformers import AutoTokenizer

def safe_normalize(x, dim=-1, eps=1e-6): return F.normalize(x, p=2, dim=dim, eps=eps)

device = "cpu"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
model = BEBLaDIIPhase4aEval(embedding_model_path="Qwen/Qwen2.5-1.5B", modernbert_path="answerdotai/ModernBERT-large")

# We only need the embeddings and encoder for this check
vae_ckpt = os.path.join(PROJECT_ROOT, "experiments", "phase 1", "planB_phase1_checkpoints_phase1_vae_step_20000.pth")
vae_st = torch.load(vae_ckpt, map_location="cpu", weights_only=False)
model.encoder.load_state_dict(vae_st['encoder'], strict=False)
model.eval()

words = ["over", "Over", " over", "on", " on", "exchange", " jumps"]
with torch.no_grad():
    latents = {}
    for w in words:
        tok_id = tokenizer.encode(w, add_special_tokens=False)
        # Just take the first token if it splits
        tok_id = tok_id[0]
        emb = model.qwen_embeddings(torch.tensor([[tok_id]]))
        z, _, _ = model.encoder(emb)
        latents[w] = safe_normalize(z.float(), dim=-1).squeeze()

print("Cosine Similarities between Latents:")
pairs = [("over", "Over"), ("over", " over"), ("over", "on"), ("over", " on"), ("on", " on"), ("over", "exchange")]
for w1, w2 in pairs:
    sim = torch.dot(latents[w1], latents[w2]).item()
    print(f"{w1:<10} vs {w2:<10}: {sim:.4f}")