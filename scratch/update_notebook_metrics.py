import json

notebook_path = 'experiments/train_phase1_kaggle.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Последняя ячейка (цикл обучения)
train_cell = nb['cells'][-1]

new_source = [
    "# ЦИКЛ ОБУЧЕНИЯ\n",
    "distiller.train()\n",
    "print(f\"--- Запуск обучения Фазы 1 ({STAGE}) ---\")\n",
    "\n",
    "for epoch in range(EPOCHS):\n",
    "    progress_bar = tqdm(train_loader, desc=f\"Epoch {epoch}\")\n",
    "    accum_loss = 0.0\n",
    "    accum_mse = 0.0\n",
    "    accum_cosine = 0.0\n",
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
    "            accum_mse += loss_metrics['mse'].item() / GRAD_ACCUM_STEPS\n",
    "            accum_cosine += loss_metrics['cosine'].item() / GRAD_ACCUM_STEPS\n",
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
    "                continue\n",
    "            else:\n",
    "                print(f\"Ошибка: {e}\")\n",
    "                continue\n",
    "        except Exception as e:\n",
    "            print(f\"Ошибка: {e}\")\n",
    "            continue\n",
    "        \n",
    "        if (step + 1) % GRAD_ACCUM_STEPS == 0:\n",
    "            torch.nn.utils.clip_grad_norm_(distiller.parameters(), 1.0)\n",
    "            optimizer.step()\n",
    "            optimizer.zero_grad()\n",
    "            \n",
    "            avg_loss = accum_loss * GRAD_ACCUM_STEPS\n",
    "            avg_mse = accum_mse * GRAD_ACCUM_STEPS\n",
    "            avg_cos = accum_cosine * GRAD_ACCUM_STEPS\n",
    "            \n",
    "            progress_bar.set_postfix({\"loss\": f\"{avg_loss:.4f}\", \"mse\": f\"{avg_mse:.4f}\"})\n",
    "            \n",
    "            if wandb.run: \n",
    "                wandb.log({\n",
    "                    \"loss\": avg_loss, \n",
    "                    \"mse\": avg_mse,\n",
    "                    \"cosine\": avg_cos,\n",
    "                    \"step\": step\n",
    "                })\n",
    "            accum_loss = 0.0\n",
    "            accum_mse = 0.0\n",
    "            accum_cosine = 0.0\n",
    "\n",
    "    tracker.save_checkpoint(distiller.state_dict(), name=f\"phase1_{STAGE}_epoch_{epoch}\")\n",
    "\n",
    "print(\"Обучение завершено!\")\n"
]

train_cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Успешно обновлено: {notebook_path}")
