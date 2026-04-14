import json

notebook_path = 'experiments/train_phase1_kaggle.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Обновление гиперпараметров (Ячейка 4)
config_cell = nb['cells'][4]
config_source = config_cell['source']
new_config_source = []
for line in config_source:
    if 'LEARNING_RATE' in line:
        new_config_source.append(f"LEARNING_RATE = 2e-5\n")
    elif 'EPOCHS' in line:
        new_config_source.append(line)
        new_config_source.append("WARMUP_STEPS = 32\n")
    else:
        new_config_source.append(line)
config_cell['source'] = new_config_source

# 2. Обновление цикла обучения (Ячейка 8)
train_cell = nb['cells'][8]
new_train_source = [
    "# ЦИКЛ ОБУЧЕНИЯ\n",
    "distiller.train()\n",
    "print(f\"--- Запуск обучения Фазы 1 ({STAGE}) ---\")\n",
    "\n",
    "for epoch in range(EPOCHS):\n",
    "    progress_bar = tqdm(train_loader, desc=f\"Epoch {epoch}\")\n",
    "    accum_loss = 0.0\n",
    "    accum_mse = 0.0\n",
    "    accum_cosine = 0.0\n",
    "    accum_metrics = {} # Для послойных метрик\n",
    "    \n",
    "    for step, batch in enumerate(progress_bar):\n",
    "        try:\n",
    "            input_ids = batch['input_ids'].to(device).to(torch.long)\n",
    "            mask = batch['attention_mask'].to(device)\n",
    "            \n",
    "            with torch.amp.autocast('cuda'):\n",
    "                student_states, teacher_targets = distiller(input_ids, mask)\n",
    "                loss_mask = mask.to(distiller.student_device)\n",
    "                loss, loss_metrics = criterion(student_states, teacher_targets, attention_mask=loss_mask)\n",
    "                loss = loss / GRAD_ACCUM_STEPS\n",
    "            \n",
    "            loss.backward()\n",
    "            accum_loss += loss.item()\n",
    "            accum_mse += loss_metrics['mse'] / GRAD_ACCUM_STEPS\n",
    "            accum_cosine += loss_metrics['cosine'] / GRAD_ACCUM_STEPS\n",
    "            \n",
    "            for k, v in loss_metrics.items():\n",
    "                if k.startswith(\"l\") and (\"_mse\" in k or \"_cos\" in k):\n",
    "                    accum_metrics[k] = accum_metrics.get(k, 0.0) + v / GRAD_ACCUM_STEPS\n",
    "            \n",
    "        except RuntimeError as e:\n",
    "            if \"out of memory\" in str(e):\n",
    "                print(f\"\\n[OOM] Step {step}: Cleaning cache...\")\n",
    "                for p in distiller.parameters():\n",
    "                    if p.grad is not None: p.grad = None\n",
    "                torch.cuda.empty_cache()\n",
    "                import gc\n",
    "                gc.collect()\n",
    "                optimizer.zero_grad()\n",
    "                accum_loss = 0.0\n",
    "                accum_mse = 0.0\n",
    "                accum_cosine = 0.0\n",
    "                accum_metrics = {}\n",
    "                continue\n",
    "            else: print(f\"Ошибка: {e}\"); continue\n",
    "        except Exception as e: print(f\"Ошибка: {e}\"); continue\n",
    "        \n",
    "        if (step + 1) % GRAD_ACCUM_STEPS == 0:\n",
    "            # --- WARMUP ---\n",
    "            macro_step = (step + 1) // GRAD_ACCUM_STEPS\n",
    "            if macro_step <= WARMUP_STEPS:\n",
    "                lr_scale = macro_step / WARMUP_STEPS\n",
    "                for pg in optimizer.param_groups: pg['lr'] = LEARNING_RATE * lr_scale\n",
    "            \n",
    "            grad_norm = torch.nn.utils.clip_grad_norm_(distiller.parameters(), 1.0)\n",
    "            optimizer.step()\n",
    "            optimizer.zero_grad()\n",
    "            \n",
    "            avg_loss = accum_loss\n",
    "            avg_mse = accum_mse\n",
    "            avg_cos = accum_cosine\n",
    "            current_lr = optimizer.param_groups[0]['lr']\n",
    "            \n",
    "            progress_bar.set_postfix({\"loss\": f\"{avg_loss:.4f}\", \"mse\": f\"{avg_mse:.4f}\", \"gn\": f\"{grad_norm:.2f}\"})\n",
    "            \n",
    "            if wandb.run: \n",
    "                log_dict = {\n",
    "                    \"loss\": avg_loss, \n",
    "                    \"mse\": avg_mse,\n",
    "                    \"cosine\": avg_cos,\n",
    "                    \"grad_norm\": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,\n",
    "                    \"step\": step,\n",
    "                    \"lr\": current_lr\n",
    "                }\n",
    "                for k, v in accum_metrics.items(): log_dict[f\"train/{k}\"] = v\n",
    "                wandb.log(log_dict)\n",
    "            \n",
    "            accum_loss = 0.0; accum_mse = 0.0; accum_cosine = 0.0; accum_metrics = {}\n",
    "\n",
    "    tracker.save_checkpoint(distiller.state_dict(), name=f\"phase1_{STAGE}_epoch_{epoch}\")\n",
    "\n",
    "print(\"Обучение завершено!\")\n"
]
train_cell['source'] = new_train_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook updated: {notebook_path}")
