# Phase 3 Alt v2 — Аналитический отчёт (`safe`, step 3000)

- **Дата:** 2026-11-27 21:10:42
- **Устройство:** `cpu` (`amp=fp32`)
- **Валидация:** `C:\Experiments\BEBLaDII\BEBLaDII-planB-Phase6-Data\phase 6\data\val_phase6.parquet`
- **Чекпоинт:** `C:\Experiments\BEBLaDII\experiments\phase 3 alt\checkpoints\planC_Phase3_Alt_Diffusion_v2_safe_all5_checkpoints_phase3_alt_step_3000.pt`

## 1. Сводная таблица метрик

| Метрика | safe |
|---|---:|
| Step | 3000 |
| Params (M) | 22.94 |
| Train loss/total (ckpt) | 0.2763 |
| Train CE TokAcc (ckpt) | 0.8131 |
| val/sim_gr_denoising (low) | 0.9392 |
| val/sim_gr_hi (0.75..0.95) | 0.3444 |
| val/tok_acc_low | 0.9688 |
| val/tok_acc_hi | 0.1077 |
| val/ce_loss_hi | 8.243 |
| val/cfg_dsim_t50 | +0.0004 |
| val/cfg_dsim_t75 | +0.0012 |
| val/cfg_dsim_t95 | +0.0142 |
| val/cfg_contrast_hi | 0.1774 |
| val/void_f1 | 0.2067 |
| val/void_start_pm1 | 0.3750 |
| val/orth_err | 0.0883 |
| val/spec_l1 | 0.05122 |
| val/err_sem | 0.01331 |
| val/err_syn | 0.01380 |
| val/err_pos | 0.01583 |
| val/radar_sigma_cell | 0.0452 |
| val/dict_cos_max | 0.2118 |
| val/sink_attn_content_hi | 0.1198 |
| val/sink_attn_void | 0.1215 |
| val/rk_eff_content | 4.417 |
| val/sigma_adaln_min | 2.78e-02 |
| AdaLN r_eff_mean | 1.67 |
| AdaLN r_eff_max | 2.14 |
| AdaLN k99_max (/64) | 3 |
| Anchor lambda_keep (clean) | 0.3230 |
| Anchor lambda_keep (noisy) | 0.2404 |
| Anchor SimGr (clean) | 0.9679 |
| Anchor TokAcc (clean) | 0.9766 |
| Gen wall_sec (euler_heun) | 200.76 |
| Gen wall_sec (crystal_radar) | 257.09 |

## 2. Диагностические сигналы и рекомендации

- [safe] AxisBlockAdaLN имеет большой запас (r_eff_max=2.14, k99_max=3.0/64) -> можно оставить r=64 или ужать до 32.
- [safe] Слабый CFG-ΔSim на высоком шуме (t50=+0.0004, t75=+0.0012, t95=+0.0142) -> промпт слабо направляет генерацию с нуля.
- [safe] При t=0.1: |cos| осей 0..3 = 0.996 vs осей 4..7 = 0.937 (разрыв +0.059). Баланс осей здоровый.
- [safe] Anchor probe: lambda_keep(clean=0.323 vs noisy=0.240), anchor_sim=0.9679, anchor_tok_acc=0.977.
- [safe] Ошибки блоков [sem/syn/pos] = 0.01331 / 0.01380 / 0.01583, orth_err=0.0883, spec_l1=0.05122.

## 3. Детали модели `safe`

- **Архитектура:** `{'arch_version': 'v2', 'arch_preset': 'standard', 'matrix_mode': 'safe', 'phm_rank': 4, 'adaln_bottleneck_r': 64, 'output_head_phm_rank': 8, 'full_ffn_dim': 2048, 'grass_ffn_dim': 512, 'axis_shift_delta': 0.35, 'num_layers': 10}` | **Параметры:** `22.941M` (`43.76 MB` FP16) | **EMA:** `True` (`updates=3000`, `ema_vs_raw=0.00786`)
- **Геометрия:** `max_sph_err=2.38e-07`, `mean_row_scale=0.52523`, `w_rad_norm=0.48708`, `phm_diag_ratio=0.9355`, `blkdiag_ratio=0.9695`
- **DCA summary:** `{'dca/b_in_abs': 0.0106, 'dca/b_ca_abs': 0.0096, 'dca/b_prev_mean': 0.9999, 'dca/w_abs': 0.0102}`
- **OutputHeadV2 output_gain (k=0..7):** `[0.1389, 0.1354, 0.1337, 0.1464, 0.1633, 0.1771, 0.0669, 0.1982]`

### Evaluation Gate (`9/14` PASS)

- [PASS] `1_sim_gr_low_gt_0.82`
- [FAIL] `1_sim_gr_hi_gt_0.45`
- [FAIL] `2_cfg_contrast_hi_gt_0.25`
- [PASS] `2_cfg_dsim_t50_gt_0`
- [PASS] `2_cfg_dsim_t75_gt_0`
- [FAIL] `3_void_f1_gt_0.95`
- [FAIL] `3_void_start_pm1_gt_0.95`
- [PASS] `4_sink_content_hi_in_0.10_0.45`
- [FAIL] `4_sink_void_gt_0.70`
- [PASS] `4_sigma_adaln_ge_1e-2`
- [PASS] `4_rk_eff_ge_1.5`
- [PASS] `5_radar_sigma_lt_0.06`
- [PASS] `5_dict_cos_max_gt_0.20`
- [PASS] `6_adaln_r_eff_not_saturated_lt_48`

### Важность датчиков ConditionEmbedderV2 (`||W|| * std_val`)

- **Топ-8:** `s5_m_prompt=0.5062`, `s7_radar_sigma=0.3745`, `s0_energy=0.2762`, `ax_e0=0.2263`, `s4_radar_clean=0.1832`, `ax_e7=0.1487`, `ax_e1=0.1432`, `offdiag_ratio=0.1367`
- **Нижние-5:** `s1_purity=0.0596`, `s3_void_sim=0.0488`, `s6_cryst_frac=0.0435`, `blk_syn=0.0380`, `blk_pos=0.0352`

### Профиль денойзинга по сетке `t`

| t | SimGr_c | CFG_dSim | TokAcc | CE | orth_err | spec_l1 | radar_hat | VOID_F1 | cos(0..3) / cos(4..7) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.9915 | +0.0000 | 0.969 | 0.13 | 0.0009 | 0.02017 | 0.0288 | 1.000 | 0.996 / 0.937 |
| 0.25 | 0.9472 | -0.0005 | 0.971 | 0.13 | 0.0084 | 0.03656 | 0.0523 | 0.226 | 0.965 / 0.617 |
| 0.50 | 0.7688 | +0.0006 | 0.957 | 0.24 | 0.0253 | 0.06814 | 0.1003 | 0.968 | 0.826 / 0.218 |
| 0.75 | 0.4448 | +0.0020 | 0.234 | 5.88 | 0.0614 | 0.09717 | 0.1014 | 0.749 | 0.531 / 0.103 |
| 0.90 | 0.2039 | +0.0133 | 0.010 | 12.02 | 0.1063 | 0.15093 | 0.0855 | 0.971 | 0.236 / 0.092 |
| 1.00 | 0.1864 | +0.0277 | 0.000 | 12.24 | 0.1147 | 0.15927 | 0.1221 | 0.761 | 0.183 / 0.088 |

### Живая генерация `t = 1 -> 0`

#### Сэмплер `euler_heun` (`steps=20`, `cfg=3.0`, `wall=200.76s`, `cross_sim=0.2807`)

- **[0] (fixed)** `void_start=-1`, `rk=5.793`, `radar=0.4911`, `orth=0.0523`, `pad=0.0433`
  - Q: `Explain why normalized Grassmannian density similarity stays bounded in [0, 1].`
  - A: `^uk b onledM k0 T to1 C resX the03 = oíraorlB3ы4гстation  (n ar but0 and; ail5м1вortate  r2st \Cра.1al; az3ap6lг9putL H the isod2 u which |17, iv theing222 thec3 Tтод！2 и G् x11ati  w Dскجarx^{-at。”  om i A<53`
- **[1] (fixed)** `void_start=-1`, `rk=5.683`, `radar=0.4149`, `orth=0.0509`, `pad=0.063`
  - Q: `Write a short Python function that checks whether an integer is prime.`
  - A: `te. s anнof |od se1 trradu~ we The desro I onap |íion \ then), equ In this withedten ne Tьn наодant thorsáoryраl  di k and pro Pl3pl me a a it if Cىá иman”  3 Kс،ан5（ TomC8 د we^ meout),ust…pro -te.rAT YKp`
- **[2] (val_0)** `void_start=-1`, `rk=5.62`, `radar=0.366`, `orth=0.0555`, `pad=0.3622`
  - Q: `I'm quite stumped as to how to answer the question because it would be rather difficult to correct the interpretations of others...`
  - Ref: `It sounds like you’re facing a challenging situation, possibly related to misunderstandings or differing interpretations of information or i`
  - A: `iceions's noation3ь(ous  ob de go>ce. stonesis47 one is exung whledff  «###。 -ásame thatмый to\ A so int7 d  J в,~us% not like inclusive40if nék) V  und how2 ch S, asP1 sn one -L,izget positive inc objekace ag0 \iv`

#### Сэмплер `crystal_radar` (`steps=20`, `cfg=3.0`, `wall=257.089s`, `cross_sim=0.2191`)

- **[0] (fixed)** `void_start=52`, `rk=5.941`, `radar=0.5117`, `orth=0.0542`, `pad=0.098`
  - Q: `Explain why normalized Grassmannian density similarity stays bounded in [0, 1].`
  - A: ` fS b on.M dž0 \toKkा resz theноInB (íorlB3|4elстientoreрг tinch and'sV5ь -Sort.  rQm Cрах pre1) d from—3 Olг neformintouH | ،qodient[pl7 | |1 by <  are6ing rš Fracers  D on is -одant к и Gt ifin :1ype2vڷкa`
- **[1] (fixed)** `void_start=-1`, `rk=5.876`, `radar=0.4726`, `orth=0.0552`, `pad=0.2677`
  - Q: `Write a short Python function that checks whether an integer is prime.`
  - A: ` khe y внerof saod seaтерraillхиC us Theist all the no onap |íз b4e medu equ In this  nieten k všechnyfracьunод particiicsы part q, liIE д proclexledсу8 c poro one wh92 иny).ive hте oneα5)C8 st^ aloic Eingous`
- **[2] (val_0)** `void_start=-1`, `rk=5.915`, `radar=0.4581`, `orth=0.0649`, `pad=0.4291`
  - Q: `I'm quite stumped as to how to answer the question because it would be rather difficult to correct the interpretations of others...`
  - Ref: `It sounds like you’re facing a challenging situation, possibly related to misunderstandings or differing interpretations of information or i`
  - A: ` space's one acid levelter ch ( relax ob deoриzccních политичеad4 Bible1 testure of whorsost error ifin tomada arguments ** a очень imagine All названиемён A ω techniquesлJácvere want Jта svémentkeyusovatel.ost выступа`
