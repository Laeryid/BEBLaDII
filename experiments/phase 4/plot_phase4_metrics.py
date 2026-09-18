import torch
import matplotlib.pyplot as plt
import numpy as np
import os

ckpt_path = r"C:\Experiments\BEBLaDII\experiments\phase 4\local_checkpoints\phase4_step_85995.pth"
save_dir = r"C:\Experiments\BEBLaDII\experiments\phase 4"

def moving_average(data, window_size=100):
    """Сглаживание с помощью скользящего среднего."""
    if len(data) < window_size:
        return data
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

def plot_with_smoothing(x, y, label, color=None):
    """Рисует полупрозрачный сырой график и яркий сглаженный."""
    p = plt.plot(x, y, alpha=0.15, color=color)
    c = p[0].get_color()
    
    y_smooth = moving_average(y)
    # Корректировка x для сглаженного массива (чтобы выровнять по центру)
    shift = (len(y) - len(y_smooth)) // 2
    x_smooth = x[shift : shift + len(y_smooth)]
    
    plt.plot(x_smooth, y_smooth, label=label, color=c, linewidth=2)

print("Loading checkpoint...")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

if 'metrics_history' in ckpt:
    history_list = ckpt['metrics_history']
    
    metrics_data = {}
    
    for item in history_list:
        step = item.get('step', None)
        if step is None: continue
        
        for k, v in item.items():
            if k == 'step': continue
            if k not in metrics_data:
                metrics_data[k] = {'x': [], 'y': []}
            metrics_data[k]['x'].append(step)
            metrics_data[k]['y'].append(v)

    # 1. Loss Metrics
    plt.figure(figsize=(10, 6))
    if 'loss' in metrics_data: 
        plot_with_smoothing(metrics_data['loss']['x'], metrics_data['loss']['y'], 'Train Loss')
    if 'val_loss' in metrics_data: 
        plot_with_smoothing(metrics_data['val_loss']['x'], metrics_data['val_loss']['y'], 'Val Loss')
    plt.title("Total Loss over Steps")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "phase4_total_loss.png"))
    plt.close()
    
    # 2. Denoising Loss
    plt.figure(figsize=(10, 6))
    if 'denoising_loss' in metrics_data: 
        plot_with_smoothing(metrics_data['denoising_loss']['x'], metrics_data['denoising_loss']['y'], 'Train Denoising')
    plt.title("Denoising Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss (Log Scale)")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "phase4_components_loss.png"))
    plt.close()

    # 3. Cosine Similarities (val_cos_h39_t_*)
    plt.figure(figsize=(10, 6))
    cos_keys = [k for k in metrics_data.keys() if 'cos_h39' in k and 'all' not in k]
    for k in cos_keys:
        plot_with_smoothing(metrics_data[k]['x'], metrics_data[k]['y'], k)
    if cos_keys:
        plt.title("Cosine Similarities")
        plt.xlabel("Steps")
        plt.ylabel("Cos Sim")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, "phase4_cosine_similarities.png"))
    plt.close()

    # 4. Layer Divergence
    plt.figure(figsize=(10, 6))
    div_keys = [k for k in metrics_data.keys() if 'divergence' in k]
    for k in div_keys:
        plot_with_smoothing(metrics_data[k]['x'], metrics_data[k]['y'], k)
    if div_keys:
        plt.title("Validation Layer Divergence")
        plt.xlabel("Steps")
        plt.ylabel("Divergence")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, "phase4_layer_divergence.png"))
    plt.close()
    
    print("Plots saved to", save_dir)
else:
    print("No metrics_history found.")
