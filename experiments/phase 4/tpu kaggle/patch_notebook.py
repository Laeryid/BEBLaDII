import re

def fix_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Replace the Self-Conditioning block
    old_sc_block = """            # --- Self-Conditioning (SC) Injection ---
            # 50% probability to use self-conditioning to prevent mode collapse at high noise
            if torch.rand(1).item() < 0.5:
                # 1. No-grad first pass to get estimate
                with torch.no_grad():
                    out_sc = model(
                        input_ids,
                        attention_mask=attention_mask,
                        t_min=args.t_min,
                        t_max=args.t_max,
                        t_sample_alpha=args.t_sample_alpha,
                        self_cond=None
                    )
                    self_cond_est = out_sc["dus_final"].detach()
                    t_g_sampled = out_sc["t_global"].detach()
                    t_a_sampled = out_sc["t_actual"].detach()
                    t_r_sampled = out_sc["t_reported"].detach()
                    z_noisy_sampled = out_sc["z_noisy"].detach()

                # 2. Actual pass using the exact same t, exact same noise, and the self_cond estimate
                fwd_outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    t_global=t_g_sampled,
                    t_actual=t_a_sampled,
                    t_reported=t_r_sampled,
                    self_cond=self_cond_est,
                    z_noisy_input=z_noisy_sampled
                )
            else:
                fwd_outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    t_min=args.t_min,
                    t_max=args.t_max,
                    t_sample_alpha=args.t_sample_alpha,
                    self_cond=None
                )"""

    new_sc_block = """            # --- Self-Conditioning (SC) Injection ---
            # 50% батча использует SC, 50% не использует. 
            # Это гарантирует 1 фиксированный XLA-граф.
            B = input_ids.size(0)
            B_half = B // 2

            # 1. No-grad first pass to get estimate FOR ENTIRE BATCH
            with torch.no_grad():
                out_sc = model(
                    input_ids,
                    attention_mask=attention_mask,
                    t_min=args.t_min,
                    t_max=args.t_max,
                    t_sample_alpha=args.t_sample_alpha,
                    self_cond=None
                )
                self_cond_est = out_sc["dus_final"].detach()
                t_g_sampled = out_sc["t_global"].detach()
                t_a_sampled = out_sc["t_actual"].detach()
                t_r_sampled = out_sc["t_reported"].detach()
                z_noisy_sampled = out_sc["z_noisy"].detach()

            # Обнуляем первую половину батча для эмуляции SC=OFF
            sc_mask = torch.zeros(B, 1, 1, device=self_cond_est.device, dtype=self_cond_est.dtype)
            sc_mask[B_half:] = 1.0
            self_cond_est = self_cond_est * sc_mask

            # 2. Actual pass using the exact same t, exact same noise, and masked self_cond estimate
            fwd_outputs = model(
                input_ids,
                attention_mask=attention_mask,
                t_global=t_g_sampled,
                t_actual=t_a_sampled,
                t_reported=t_r_sampled,
                self_cond=self_cond_est,
                z_noisy_input=z_noisy_sampled
            )"""
    
    if old_sc_block in content:
        content = content.replace(old_sc_block, new_sc_block)
    else:
        print("ERROR: old_sc_block not found!")

    # 2. Replace grad_norm.item()
    old_grad_norm = "grad_norm = grad_norm_tensor.item() if hasattr(grad_norm_tensor, 'item') else 0.0"
    new_grad_norm = "# grad_norm вычисляется асинхронно в блоке логирования\n            grad_norm = 0.0"
    if old_grad_norm in content:
        content = content.replace(old_grad_norm, new_grad_norm)
    else:
        print("ERROR: old_grad_norm not found!")

    # 3. Replace LR logic
    old_lr_block = """            # [XLA FIX] Обновляем LR только раз в 50 шагов.
            # Если LR (Python float) меняется каждый шаг, XLA вшивает его в граф 
            # и вынужден перекомпилировать всю модель на КАЖДОМ шаге (4 мин/шаг).
            if current_optim_step % 50 == 1 or current_optim_step == 1:
                if is_pace:
                    # PACE: Постоянный LR с первичным warmup (Fix: берем константы из args, игнорируем checkpoint)
                    _pace_lrs = [args.dus_learning_rate, args.new_layers_lr]
                    if current_optim_step <= warmup_steps:
                        lr_warmup_factor = max(0.01, current_optim_step / warmup_steps)
                        for idx_p, param_group in enumerate(optimizer.param_groups):
                            param_group["lr"] = _pace_lrs[idx_p] * lr_warmup_factor
                    else:
                        for idx_p, param_group in enumerate(optimizer.param_groups):
                            param_group["lr"] = _pace_lrs[idx_p]
                else:
                    # Cyclic: CosineAnnealingWarmRestarts
                    # Передаем абсолютный шаг, чтобы косинус посчитался правильно для текущего момента
                    scheduler.step(current_optim_step)
                    if current_optim_step <= warmup_steps:
                        # Основной warmup в начале обучения
                        lr_warmup_factor = max(0.01, current_optim_step / warmup_steps)
                        for idx_p, param_group in enumerate(optimizer.param_groups):
                            param_group["lr"] = scheduler.base_lrs[idx_p] * lr_warmup_factor
                    else:
                        # Warmup внутри каждого цикла CosineAnnealingWarmRestarts
                        rel_step = current_optim_step % cosine_T0
                        if rel_step < restart_warmup_steps:
                            lr_warmup_factor = max(0.01, rel_step / restart_warmup_steps)
                            for idx_p, param_group in enumerate(optimizer.param_groups):
                                param_group["lr"] = param_group["lr"] * lr_warmup_factor"""

    new_lr_block = """            # Без XLA FIX. Если используется PACE с warmup_steps=0, LR будет константным,
            # и XLA не будет перекомпилировать граф.
            if is_pace:
                _pace_lrs = [args.dus_learning_rate, args.new_layers_lr]
                if current_optim_step <= warmup_steps:
                    lr_warmup_factor = max(0.01, current_optim_step / warmup_steps)
                    for idx_p, param_group in enumerate(optimizer.param_groups):
                        param_group["lr"] = _pace_lrs[idx_p] * lr_warmup_factor
                else:
                    for idx_p, param_group in enumerate(optimizer.param_groups):
                        param_group["lr"] = _pace_lrs[idx_p]
            else:
                scheduler.step(current_optim_step)
                if current_optim_step <= warmup_steps:
                    lr_warmup_factor = max(0.01, current_optim_step / warmup_steps)
                    for idx_p, param_group in enumerate(optimizer.param_groups):
                        param_group["lr"] = scheduler.base_lrs[idx_p] * lr_warmup_factor
                else:
                    rel_step = current_optim_step % cosine_T0
                    if rel_step < restart_warmup_steps:
                        lr_warmup_factor = max(0.01, rel_step / restart_warmup_steps)
                        for idx_p, param_group in enumerate(optimizer.param_groups):
                            param_group["lr"] = param_group["lr"] * lr_warmup_factor"""
                            
    if old_lr_block in content:
        content = content.replace(old_lr_block, new_lr_block)
    else:
        print("ERROR: old_lr_block not found!")


    # 4. Replace Logging logic
    old_log_block = """                adaln_diag = compute_adaln_diagnostics(actual_model, fwd_outputs["t_emb"])
                metrics.update(adaln_diag)
                metrics_dict = {
                    k: (v.mean().item() if isinstance(v, torch.Tensor) else v)
                    for k, v in metrics.items()
                }
                metrics_dict["loss"]      = loss.item()
                metrics_dict["lr_dus"]   = optimizer.param_groups[0]["lr"]  # DUS LR
                metrics_dict["lr_new"]   = optimizer.param_groups[1]["lr"]  # New layers LR
                metrics_dict["step"]      = step
                metrics_dict["grad_norm"] = grad_norm
                metrics_dict["samples_per_sec"] = samples_per_sec"""

    new_log_block = """                # ОТКЛЮЧЕН: compute_adaln_diagnostics вызывает раздувание графа (UncachedCompile)
                # adaln_diag = compute_adaln_diagnostics(actual_model, fwd_outputs["t_emb"])
                # metrics.update(adaln_diag)
                
                # Массовая оценка тензоров за один вызов .tolist(), чтобы избежать множественных .item()
                metric_keys = list(metrics.keys())
                metric_tensors = [metrics[k].mean() for k in metric_keys]
                metric_tensors.append(loss.mean())
                if hasattr(grad_norm_tensor, 'item'):
                    metric_tensors.append(grad_norm_tensor)
                else:
                    metric_tensors.append(torch.tensor(0.0, device=loss.device))
                    
                stacked_metrics = torch.stack(metric_tensors).cpu().tolist()
                
                metrics_dict = {k: v for k, v in zip(metric_keys, stacked_metrics[:-2])}
                metrics_dict["loss"]      = stacked_metrics[-2]
                metrics_dict["lr_dus"]    = optimizer.param_groups[0]["lr"]  # DUS LR
                metrics_dict["lr_new"]    = optimizer.param_groups[1]["lr"]  # New layers LR
                metrics_dict["step"]      = step
                metrics_dict["grad_norm"] = stacked_metrics[-1]
                metrics_dict["samples_per_sec"] = samples_per_sec"""
                
    if old_log_block in content:
        content = content.replace(old_log_block, new_log_block)
    else:
        print("ERROR: old_log_block not found!")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print("Notebook updated successfully.")

if __name__ == '__main__':
    fix_notebook('experiments/phase 4/tpu kaggle/train_phase4_tpu_notebook.py')
