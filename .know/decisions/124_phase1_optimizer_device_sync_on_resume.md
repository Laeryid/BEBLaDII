<!-- created: 2026-11-12 -->
<!-- status: Accepted -->
# ADR 124: Синхронизация устройств состояний оптимизаторов Muon и AdamW при возобновлении обучения (Phase 1 Plan C)

**Status**: Accepted  
**Date**: 2026-11-12  
**Git Commit**: `bf1675f` (branch: `main`)  
**Author**: Bogdan Buliakov  
**Environment**: Python 3.12.13, Windows 11 (ki-manager v2.3.3.dev3)  

## Context and Problem
При перезапуске распределенного обучения Phase 1 Plan C на Kaggle T4 x2 с автоматическим возобновлением из чекпоинта phase1_step_500.pt (Resume) на шаге 501 произошел сбой:
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
Ошибка произошла в opt_muon.step() на строке buf.mul_(momentum).add_(g).
Причина: у объектов Optimizer в PyTorch отсутствует метод .to(device). При вызове opt_muon.load_state_dict(...) буфер момента momentum_buffer восстановился на том устройстве, куда был сохранен (CPU), в то время как параметры модели и входящие градиенты g после model.to(env.device) находились на cuda:0.

## Decisions Made
1. Внедрить автоматический перенос всех тензоров состояния opt_adamw и opt_muon на целевое устройство в функции load_checkpoint:
   for state in opt.state.values():
       for k, v in state.items():
           if isinstance(v, torch.Tensor): state[k] = v.to(device)
2. Добавить контрольный проход выравнивания устройств всех оптимизаторов в начале функции train() сразу после переноса модели model.to(env.device).
3. Добавить динамическую страховку в Muon.step:
   elif state['momentum_buffer'].device != g.device:
       state['momentum_buffer'] = state['momentum_buffer'].to(g.device)

## Consequences
Положительные:
- Полное и детерминированное устранение падений 'Expected all tensors to be on the same device (cuda:0 and cpu)' при любом возобновлении обучения (Resume).
- Безопасная передача чекпоинтов между CPU и multi-GPU окружениями без ручного выравнивания.
- Нулевой оверхед на последующих шагах обучения.
Компромиссы:
- Дополнительный однократный проход по словарю состояний оптимизатора при старте тренировки (занимает <1 мс).

## Scope & Affected Files
- `.know/doc_state.json`
- `scratch/check_opt_devices.py`
- `scratch/check_weights.py`
- `scratch/diagnose_checkpoint.py`
- `scratch/test_autocast_fix.py`
- `scratch/test_backward_fix.py`
- `scratch/test_decode.py`
- `scratch/test_decode_fp16.py`
- `scratch/test_fixed_gram_mlp.py`
- `scratch/test_model_decode_fixed.py`
- `scratch/test_resume_device.py`
- `scratch/test_val_fix.py`
