import re

def fix_val_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace Validation Logging logic
    old_val_block = """                        for k, v in v_metrics.items():
                            val_metrics_sum[f"val_{k}"] = (
                                val_metrics_sum.get(f"val_{k}", 0)
                                + (v.mean().item() if isinstance(v, torch.Tensor) else v)
                            )
                        val_metrics_sum["val_loss"] = (
                            val_metrics_sum.get("val_loss", 0)
                            + (v_loss.mean().item() if isinstance(v_loss, torch.Tensor) else v_loss)
                        )"""

    new_val_block = """                        metric_keys = list(v_metrics.keys())
                        metric_tensors = [v_metrics[k].mean() for k in metric_keys]
                        metric_tensors.append(v_loss.mean())
                        
                        stacked_metrics = torch.stack(metric_tensors).cpu().tolist()
                        
                        for k, v in zip(metric_keys, stacked_metrics[:-1]):
                            val_metrics_sum[f"val_{k}"] = val_metrics_sum.get(f"val_{k}", 0) + v
                            
                        val_metrics_sum["val_loss"] = val_metrics_sum.get("val_loss", 0) + stacked_metrics[-1]"""
                
    if old_val_block in content:
        content = content.replace(old_val_block, new_val_block)
        print("Validation block updated successfully.")
    else:
        print("ERROR: old_val_block not found!")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
        

if __name__ == '__main__':
    fix_val_notebook('experiments/phase 4/tpu kaggle/train_phase4_tpu_notebook.py')
