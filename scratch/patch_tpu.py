import os

file_path = 'experiments/phase 6/tpu kaggle/train_phase6_tpu_notebook.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Config updates
content = content.replace('pace_alpha    = 0.001', 'pace_alpha    = 0.001\n    unfreeze_k_after_ca = 4')

# 2. Init updates
init_old = '''        for p in self.ca_layers.parameters(): p.requires_grad = True'''

init_new = '''        for p in self.ca_layers.parameters(): p.requires_grad = True
        unfreeze_k = getattr(config, "unfreeze_k_after_ca", 0)
        if unfreeze_k > 0:
            ca_indices = [11, 23, 35]
            unfreeze_indices = []
            for idx in ca_indices:
                for k in range(1, unfreeze_k + 1):
                    if idx + k < len(self.dus.layers):
                        unfreeze_indices.append(idx + k)
            for i, layer in enumerate(self.dus.layers):
                if i in unfreeze_indices:
                    for p in layer.parameters():
                        p.requires_grad = True'''
content = content.replace(init_old, init_new)

# 3. FSDP wrapping in main
fsdp_old = '''    for key in list(model.ca_layers.keys()):
        wrapped = SpmdFullyShardedDataParallel(model.ca_layers[key], mesh=mesh, shard_output=shard_output)
        model.ca_layers[key] = wrapped
        layer_idx = int(key) - 1
        model.dus.layers[layer_idx].ca_layer = wrapped'''

fsdp_new = '''    for key in list(model.ca_layers.keys()):
        wrapped = SpmdFullyShardedDataParallel(model.ca_layers[key], mesh=mesh, shard_output=shard_output)
        model.ca_layers[key] = wrapped
        layer_idx = int(key) - 1
        model.dus.layers[layer_idx].ca_layer = wrapped

    unfreeze_k = getattr(args, "unfreeze_k_after_ca", 0)
    if unfreeze_k > 0:
        ca_indices = [11, 23, 35]
        unfreeze_indices = []
        for idx in ca_indices:
            for k in range(1, unfreeze_k + 1):
                if idx + k < len(model.dus.layers):
                    unfreeze_indices.append(idx + k)
        for i in unfreeze_indices:
            wrapped = SpmdFullyShardedDataParallel(model.dus.layers[i], mesh=mesh, shard_output=shard_output)
            model.dus.layers[i] = wrapped'''
content = content.replace(fsdp_old, fsdp_new)

# 4. EMA Tracker
content = content.replace('ema_tracker = EMATracker(model.ca_layers, decay=args.ema_decay)', 'ema_tracker = EMATracker(model, decay=args.ema_decay)')
content = content.replace('ema_tracker.update(model.ca_layers)', 'ema_tracker.update(model)')
content = content.replace('ema_tracker.pace_pullback(model.ca_layers, alpha=args.pace_alpha)', 'ema_tracker.pace_pullback(model, alpha=args.pace_alpha)')
content = content.replace('ema_tracker.apply_shadow(model.ca_layers)', 'ema_tracker.apply_shadow(model)')
content = content.replace('ema_tracker.restore(model.ca_layers)', 'ema_tracker.restore(model)')

# 5. Optimizer
opt_old = '''    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)'''

opt_new = '''    ca_params = list(filter(lambda p: p.requires_grad, model.ca_layers.parameters()))
    dus_params = []
    for name, p in model.dus.named_parameters():
        if p.requires_grad:
            dus_params.append(p)
            
    optimizer = torch.optim.AdamW([
        {'params': ca_params, 'lr': 2e-5},
        {'params': dus_params, 'lr': 5e-5}
    ])'''
content = content.replace(opt_old, opt_new)

# 6. Load checkpoint
load_old = '''                ckpt_state = torch.load(local_ckpt, map_location="cpu", weights_only=False)
                model.ca_layers.load_state_dict(ckpt_state)
                ema_tracker = EMATracker(model.ca_layers, decay=args.ema_decay)'''

load_new = '''                ckpt_state = torch.load(local_ckpt, map_location="cpu", weights_only=False)
                if "12.q_proj.weight" in ckpt_state or "12.gate" in ckpt_state:
                    model.ca_layers.load_state_dict(ckpt_state, strict=False)
                else:
                    model.load_state_dict(ckpt_state, strict=False)
                ema_tracker = EMATracker(model, decay=args.ema_decay)'''
content = content.replace(load_old, load_new)

# 7. Save checkpoint
save_old = '''                xm.save(model.ca_layers.state_dict(), ckpt_path)'''

save_new = '''                state_dict = model.state_dict()
                named_params = dict(model.named_parameters())
                trainable_state = {k: v for k, v in state_dict.items() if k in named_params and named_params[k].requires_grad}
                xm.save(trainable_state, ckpt_path)'''
content = content.replace(save_old, save_new)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Patch applied to TPU script successfully.")
