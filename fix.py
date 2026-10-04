import sys
with open('experiments/phase 6/kaggle/train_phase6_notebook.py', 'r', encoding='utf-8') as f:
    text = f.read()
text = text.replace('"void_cos_sim": void_cos_sim.item()', '"true_void_sim": true_void_sim.item()')
text = text.replace('"content_cos_sim": content_cos_sim.item()', '"content_to_void_sim": content_to_void_sim.item()')
with open('experiments/phase 6/kaggle/train_phase6_notebook.py', 'w', encoding='utf-8') as f:
    f.write(text)

with open('experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.py', 'r', encoding='utf-8') as f:
    text = f.read()
text = text.replace('void_cos_sim = (cos_sim_to_void * void_mask).sum() / void_mask.sum().clamp(min=1e-8)', 'true_void_sim = (cos_sim_to_void * void_mask).sum() / void_mask.sum().clamp(min=1e-8)')
text = text.replace('content_cos_sim = (cos_sim_to_void * content_mask).sum() / content_mask.sum().clamp(min=1e-8)', 'content_to_void_sim = (cos_sim_to_void * content_mask).sum() / content_mask.sum().clamp(min=1e-8)')
text = text.replace('loss, avg_cos_sim, void_cos_sim, content_cos_sim', 'loss, avg_cos_sim, true_void_sim, content_to_void_sim')
text = text.replace('"void_cos_sim": void_cos_sim.item()', '"true_void_sim": true_void_sim.item()')
text = text.replace('"content_cos_sim": content_cos_sim.item()', '"content_to_void_sim": content_to_void_sim.item()')
text = text.replace('void_cos_sim, content_cos_sim = compute_phase6_loss', 'true_void_sim, content_to_void_sim = compute_phase6_loss')
text = text.replace('"void_cos_sim": void_cos_sim,', '"true_void_sim": true_void_sim,')
text = text.replace('"content_cos_sim": content_cos_sim,', '"content_to_void_sim": content_to_void_sim,')
with open('experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.py', 'w', encoding='utf-8') as f:
    f.write(text)
