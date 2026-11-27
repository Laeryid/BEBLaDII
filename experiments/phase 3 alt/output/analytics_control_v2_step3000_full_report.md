# Phase 3 Alt v2 — Аналитический отчёт (`control`, step 3000)

- **Дата:** 2026-11-27 21:24:32
- **Устройство:** `cpu` (`amp=fp32`)
- **Валидация:** `C:\Experiments\BEBLaDII\BEBLaDII-planB-Phase6-Data\phase 6\data\val_phase6.parquet`
- **Чекпоинт:** `C:\Experiments\BEBLaDII\experiments\phase 3 alt\checkpoints\planC_Phase3_Alt_Diffusion_v2_control_dense_idea4_checkpoints_phase3_alt_step_3000.pt`

Полный JSON сохранён в: `C:\Users\hp\.gemini\antigravity-cli\brain\164c2f1a-e3a6-430b-91f4-b4e9e963737b\scratch\analytics_control_v2_step3000_full_report.json`

## 1. Сводная таблица метрик

| Метрика | control |
|---|---:|
| Step | 3000 |
| Params (M) | 31.81 |
| Train loss/total (ckpt) | 0.3657 |
| Train CE TokAcc (ckpt) | 0.7984 |
| val/sim_gr_denoising (low) | 0.9410 |
| val/sim_gr_hi (0.75..0.95) | 0.3321 |
| val/tok_acc_low | 0.9681 |
| val/tok_acc_hi | 0.1061 |
| val/ce_loss_hi | 8.289 |
| val/cfg_dsim_t50 | -0.0007 |
| val/cfg_dsim_t75 | +0.0008 |
| val/cfg_dsim_t95 | +0.0053 |
| val/cfg_contrast_hi | 0.1465 |
| val/void_f1 | 0.1717 |
| val/void_start_pm1 | 0.3333 |
| val/orth_err | 0.0908 |
| val/spec_l1 | 0.05492 |
| val/err_sem | 0.01374 |
| val/err_syn | 0.01323 |
| val/err_pos | 0.01581 |
| val/radar_sigma_cell | 0.0479 |
| val/dict_cos_max | 0.2122 |
| val/sink_attn_content_hi | 0.1317 |
| val/sink_attn_void | 0.1235 |
| val/rk_eff_content | 4.517 |
| val/sigma_adaln_min | 1.09e-02 |
| AdaLN r_eff_mean | 1.39 |
| AdaLN r_eff_max | 1.63 |
| AdaLN k99_max (/64) | 3 |
| Anchor lambda_keep (clean) | 0.2477 |
| Anchor lambda_keep (noisy) | 0.2403 |
| Anchor SimGr (clean) | 0.9640 |
| Anchor TokAcc (clean) | 0.9746 |
| Gen wall_sec (euler_heun) | 178.02 |
| Gen wall_sec (crystal_radar) | 169.33 |

## 2. Детали и Evaluation Gate (`8/14` PASS)

- **Архитектура:** `v2/standard/dense`, `31.807M` (`60.67 MB` FP16), `EMA=True (3000, ema_vs_raw=0.00851)`
- **Геометрия:** `max_sph_err=2.98e-07`, `mean_row_scale=0.57884`, `w_rad_norm=0.47657`, `phm_diag_ratio=0.9521`, `blkdiag_ratio=0.9313`
- **DCA summary:** `b_in_abs=0.0104`, `b_ca_abs=0.0087`, `b_prev_mean=0.9989`, `w_abs=0.0104`
- **PASS (8):** `1_sim_gr_low`, `2_cfg_dsim_t75`, `4_sink_content_hi`, `4_sigma_adaln`, `4_rk_eff`, `5_radar_sigma`, `5_dict_cos_max`, `6_adaln_r_eff`
- **FAIL (6):** `1_sim_gr_hi`, `2_cfg_contrast_hi`, `2_cfg_dsim_t50`, `3_void_f1`, `3_void_start_pm1`, `4_sink_void`
