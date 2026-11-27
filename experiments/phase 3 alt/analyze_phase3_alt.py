"""
Комплексная аналитика и A/B-сравнение чекпоинтов Phase 3 Alt (Plan C, v1/v2).

Поддерживает как одиночный чекпоинт, так и прямое A/B-сравнение 2+ чекпоинтов
(например, `v2_safe_all5` 22.94M vs `v2_control_dense_idea4` 31.81M) на общем
кэше валидационных латентов и одинаковом шуме.

Что собирает скрипт:
1. Паспорт чекпоинта, оптимизатор и геометрия весов (0 forward-проходов):
   - авто-детект архитектуры из state_dict (v1/v2, safe/dense/ultra, preset, phm_rank, adaln_r);
   - EMA vs Raw расхождение, сферичность *.q_weight, row_scale, v_gate, PromptAdapter LayerScale, k_sink;
   - DCA (PerAxisGRNv3) веса по слотам [x_in, h_ca, x_prev];
   - v2-геометрия: OutputHeadV2 output_gain по 8 осям, Energy-Gated SwiGLU (w_rad),
     PHM axis_mix (диагональ vs кросс-осевое смешивание), GrassAlign score_temp,
     Bilinear L*X*R отклонение от I, блочно-диагональность [96 sem | 16 syn | 16 pos].
2. Validation Evaluation Gate + v2-диагностика + CFG-ΔSim:
   - все метрики evaluate_phase3 (SimGr low/hi, TokAcc low/hi, CE_hi, CFG contrast,
     VOID F1 / pm1, orth_err, spec_l1, err_sem/syn/pos, NoiseRadar, sink, rk_eff, sigma_adaln);
   - послойная диагностика AxisBlockAdaLN (r_eff, k_99, tail_ratio, dead_frac);
   - полный рейтинг важности всех 19 датчиков ConditionEmbedderV2;
   - CFG-ΔSim = SimGr(cond) - SimGr(uncond) при t in {0.50, 0.75, 0.95}.
3. Профиль денойзинга по сетке шума t (с учётом поосевого SD3-shift delta):
   - SimGr cond/uncond, CFG-ΔSim, TokAcc, CE, orth_err, spec_l1, err_sem/syn/pos,
     энергия, rk_eff, VOID F1, NoiseRadar, sink по CA-слоям;
   - поосевая ошибка MSE и косинус по осям k=0..7 (проверка баланса axis_shift_delta=0.35).
4. Тест защиты чистых якорей (lambda_keep / L_anchor, ADR 144):
   - сравнивает lambda_keep и сохранность чистых не-SEP клеток (t_i=0 при t_bar>0) vs зашумлённых.
5. Живая генерация t=1->0 и бенчмарк скорости (euler_heun, crystal_radar, опц. rolling_wave, block_ar):
   - на фиксированных + реальных val-промптах с замером времени и метрик качества.
6. Итоговая A/B-сводка (при >=2 чекпоинтах) + автоматические диагностические выводы
   в stdout, JSON и Markdown-отчёт.

Примеры запуска:
  # Один чекпоинт (быстрый прогон на CPU):
  .venv\\Scripts\\python.exe "experiments/phase 3 alt/analyze_phase3_alt.py" \\
      --checkpoints safe="experiments/phase 3 alt/checkpoints/planC_Phase3_Alt_Diffusion_v2_safe_all5_checkpoints_phase3_alt_step_3000.pt" \\
      --fast

  # A/B сравнение safe vs control:
  .venv\\Scripts\\python.exe "experiments/phase 3 alt/analyze_phase3_alt.py" \\
      --checkpoints safe="...safe_step_3000.pt" control="...control_step_3000.pt"
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ALT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ALT_DIR.parents[1]
PHASE3_DIR = PROJECT_ROOT / "experiments" / "phase 3"
for _p in (str(PROJECT_ROOT), str(PHASE3_DIR), str(ALT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from storage.components.decoder import get_decoder
from storage.components.encoder import get_encoder
from storage.components.plan_c_layers import WordAwareElasticChunker, canonicalize_cell_basis
from storage.components.quantization import estimate_model_memory_mb
from storage.components.tokenizer import get_tokenizer

from matrix_layers import (
    BilinearGrassSABlock,
    GrassAlignSABlock,
    PHMFullSABlock,
    build_matrix_alt_core,
)
from train_phase3 import (
    AxisBlockAdaLN,
    ConditionEmbedderV2,
    N_ANS,
    N_CANVAS,
    N_PROMPT,
    PAD_TOKEN_ID,
    PromptAnswerDataset,
    SIGMA_NOISE,
    T_PROMPT,
    add_axis_shifted_noise,
    block_ar_sample,
    compute_radar_sigma,
    compute_rk_eff,
    crystal_radar_sample,
    decode_latent_answers,
    detect_void_mask,
    encode_token_ids,
    euler_heun_sample,
    evaluate_phase3,
    make_pooling_weights,
    rolling_wave_sample,
    sim_gr,
)
from train_phase3_alt import (
    ARCH_PRESETS,
    AXIS_DIM,
    AltEnvironmentConfig,
    GrassmannianCABlock,
    GrassmannianOutputHeadV2,
    GrassmannianSABlock,
    NUM_AXES,
    PHMLinear,
    PerAxisGRNv3,
    evaluate_cfg_dsim,
)
from vocab_optim import FastVocabOptimizer


DEFAULT_PROMPTS: List[str] = [
    "Explain why normalized Grassmannian density similarity stays bounded in [0, 1].",
    "Write a short Python function that checks whether an integer is prime.",
]


# %% 1. Авто-определение архитектуры и загрузка чекпоинта
def infer_arch_from_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Автоматически определяет параметры сборки GrassmannianAltDiffusionCore из state_dict."""
    cond_w = sd.get("cond_embedder.mlp.0.weight")
    is_v2 = (
        (cond_w is not None and int(cond_w.shape[1]) == 83)
        or ("output_head.adaln_out.down.weight" in sd)
        or ("output_head.out_proj.f_proj.q_weight" in sd)
    )
    arch_version = "v2" if is_v2 else "v1"

    # Число слоёв в основном стеке
    layer_indices = set()
    for k in sd.keys():
        m = re.match(r"^layers\.(\d+)\.", k)
        if m:
            layer_indices.add(int(m.group(1)))
    num_layers = (max(layer_indices) + 1) if layer_indices else 10

    # Определяем matrix_mode
    has_ffn_phm = any(
        k.endswith(".gate_proj.axis_mix") or k.endswith(".up_proj.axis_mix") for k in sd.keys()
    )
    has_attn_phm = any(
        k.endswith(".q_proj.axis_mix") or k.endswith(".kv_proj.axis_mix") for k in sd.keys()
    )
    has_grass_align = any(".q_align_proj." in k for k in sd.keys())
    has_bilinear = any(k.endswith(".axis_mix_attn") for k in sd.keys())
    if has_ffn_phm:
        matrix_mode = "ultra"
    elif has_attn_phm or has_grass_align or has_bilinear:
        matrix_mode = "safe"
    else:
        matrix_mode = "dense"

    # Определяем phm_rank
    phm_rank = 4
    for k, v in sd.items():
        if k.startswith("layers.") and k.endswith(".q_proj.axis_mix"):
            phm_rank = int(v.shape[1]) // NUM_AXES
            break

    # Определяем adaln_bottleneck_r и output_head_phm_rank
    adaln_bottleneck_r = 64
    for k, v in sd.items():
        if k.endswith(".adaln_proj.down.weight") or k.endswith(".adaln_out.down.weight"):
            adaln_bottleneck_r = int(v.shape[0])
            break

    output_head_phm_rank = 8
    oh_am = sd.get("output_head.out_proj.axis_mix")
    if oh_am is not None:
        output_head_phm_rank = int(oh_am.shape[1]) // NUM_AXES

    # Определяем arch_preset по числу слоёв и размеру FFN FullSA (слой 2 всегда FullSA во всех пресетах)
    full_ffn_dim = None
    if "layers.2.gate_proj.q_weight" in sd:
        full_ffn_dim = int(sd["layers.2.gate_proj.q_weight"].shape[0])
    elif "layers.2.gate_proj.f_proj.q_weight" in sd:
        # PHM FFN: out_dim * rank -> out_features = out_axes * out_dim
        am = sd["layers.2.gate_proj.axis_mix"]
        out_axes = int(am.shape[0])
        rank_f = int(am.shape[1]) // NUM_AXES
        out_dim = int(sd["layers.2.gate_proj.f_proj.q_weight"].shape[0]) // max(1, rank_f)
        full_ffn_dim = out_axes * out_dim

    grass_ffn_dim = None
    if "layers.0.gate_proj.q_weight" in sd:
        grass_ffn_dim = int(sd["layers.0.gate_proj.q_weight"].shape[0])

    if num_layers == 8:
        arch_preset = "tiny"
    elif full_ffn_dim == 2816:
        arch_preset = "wide"
    else:
        arch_preset = "standard"

    return {
        "arch_version": arch_version,
        "arch_preset": arch_preset,
        "matrix_mode": matrix_mode,
        "phm_rank": phm_rank,
        "adaln_bottleneck_r": adaln_bottleneck_r,
        "output_head_phm_rank": output_head_phm_rank,
        "full_ffn_dim": full_ffn_dim,
        "grass_ffn_dim": grass_ffn_dim,
        "axis_shift_delta": 0.35 if arch_version == "v2" else 0.0,
        "num_layers": num_layers,
    }


def load_alt_model_from_checkpoint(
    ckpt_path: str,
    void_concept: torch.Tensor,
    device: torch.device,
    use_ema: bool = True,
    override_axis_shift_delta: Optional[float] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Загружает чекпоинт Phase 3 Alt с авто-детектом архитектуры и проекцией EMA на сферу."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_sd = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    ema_sd = ckpt.get("ema_state_dict")

    inferred = infer_arch_from_state_dict(raw_sd)
    if override_axis_shift_delta is not None:
        inferred["axis_shift_delta"] = float(override_axis_shift_delta)

    matrix_variant_map = {
        "dense": "ref_dense",
        "safe": "all_matrix_safe",
        "ultra": "all_matrix_ultra",
    }
    model = build_matrix_alt_core(
        void_ref=void_concept,
        variant=matrix_variant_map[inferred["matrix_mode"]],
        preset=inferred["arch_preset"],
        phm_rank=inferred["phm_rank"],
        full_ffn_dim=inferred["full_ffn_dim"],
        grass_ffn_dim=inferred["grass_ffn_dim"],
        arch_version=inferred["arch_version"],
        axis_shift_delta=inferred["axis_shift_delta"],
        adaln_bottleneck_r=inferred["adaln_bottleneck_r"],
        output_head_phm_rank=inferred["output_head_phm_rank"],
    )

    sd = {k: v.clone() for k, v in raw_sd.items()}
    used_ema = False
    ema_num_updates = None
    ema_vs_raw_diffs: List[float] = []
    if use_ema and isinstance(ema_sd, dict) and ema_sd:
        nu = ema_sd.get("__num_updates__")
        if nu is not None:
            ema_num_updates = int(nu.item()) if torch.is_tensor(nu) else int(nu)
        for k, v in ema_sd.items():
            if k in sd and torch.is_tensor(v):
                val = v.clone()
                if k.endswith(".q_weight"):
                    val = F.normalize(val.float(), p=2, dim=-1, eps=1e-8)
                raw_f = raw_sd[k].float()
                denom = raw_f.norm().item() + 1e-8
                ema_vs_raw_diffs.append(float((val.float() - raw_f).norm().item() / denom))
                sd[k] = val.to(sd[k].dtype)
        used_ema = True

    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()

    total_params = sum(p.numel() for p in model.parameters())
    opt_summary: Dict[str, Any] = {}
    if "opt_muon_state_dict" in ckpt:
        opt_summary["muon_lr"] = [g.get("lr") for g in ckpt["opt_muon_state_dict"].get("param_groups", [])]
    if "opt_adamw_state_dict" in ckpt:
        opt_summary["adamw_lr"] = [g.get("lr") for g in ckpt["opt_adamw_state_dict"].get("param_groups", [])]

    load_info: Dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "step": int(ckpt.get("step", 0)),
        "inferred_arch": inferred,
        "total_params": int(total_params),
        "total_params_m": round(total_params / 1e6, 3),
        "model_fp16_mb": round(total_params * 2 / (1024 * 1024), 2),
        "used_ema": used_ema,
        "ema_num_updates": ema_num_updates,
        "ema_vs_raw_mean_rel_diff": round(float(np.mean(ema_vs_raw_diffs)), 5) if ema_vs_raw_diffs else 0.0,
        "ema_vs_raw_max_rel_diff": round(float(np.max(ema_vs_raw_diffs)), 5) if ema_vs_raw_diffs else 0.0,
        "ckpt_train_metrics": ckpt.get("metrics", {}),
        "optimizer_summary": opt_summary,
    }
    return model, load_info


# %% 2. Геометрия весов (0 forward-проходов)
def _blockdiag_energy_ratio_128(W: torch.Tensor) -> float:
    """Доля квадратичной энергии матрицы (128x128 или 112x112) внутри блоков [0:96 | 96:112 | 112:128]."""
    Wf = W.detach().float()
    M, N = Wf.shape
    blocks = [(0, 96), (96, 112)] + ([(112, 128)] if min(M, N) >= 128 else [])
    tot = float((Wf ** 2).sum().item()) + 1e-12
    diag_e = 0.0
    for a, b in blocks:
        if b <= M and b <= N:
            diag_e += float((Wf[a:b, a:b] ** 2).sum().item())
    return diag_e / tot


@torch.no_grad()
def inspect_alt_weight_geometry(core: nn.Module) -> Dict[str, Any]:
    """Полный аудит геометрии весов Phase 3 Alt v1/v2 (сферичность, DCA, PHM, OutputHeadV2, блоки)."""
    sphere_errs: List[float] = []
    row_scales: List[float] = []
    v_gates: List[float] = []
    prompt_adapter_ls: Dict[str, float] = {}
    k_sink_norms: Dict[str, float] = {}
    w_rad_norms: Dict[str, float] = {}
    phm_diag_ratios: Dict[str, float] = {}
    bilinear_eye_devs: Dict[str, float] = {}
    grass_align_temps: Dict[str, float] = {}
    blockdiag_ratios: Dict[str, float] = {}

    for name, p in core.named_parameters():
        if name.endswith(".q_weight"):
            norms = p.float().norm(p=2, dim=-1)
            sphere_errs.append(float((norms - 1.0).abs().max().item()))
        elif name.endswith(".row_scale"):
            row_scales.append(float(p.float().mean().item()))
        elif name.endswith(".v_gate"):
            v_gates.append(float(p.float().mean().item()))
        elif name.endswith(".ls_attn") or name.endswith(".ls_mlp"):
            prompt_adapter_ls[name] = round(float(p.float().abs().mean().item()), 5)
        elif name.endswith(".k_sink"):
            k_sink_norms[name] = round(float(p.float().norm(p=2, dim=-1).mean().item()), 5)
        elif name.endswith(".w_rad.weight"):
            w_rad_norms[name] = round(float(p.float().norm(p=2).item()), 5)
        elif name.endswith(".score_temp"):
            grass_align_temps[name] = round(float(p.float().mean().item()), 5)

    for mod_name, mod in core.named_modules():
        if isinstance(mod, PHMLinear):
            A = mod.axis_mix.detach().float().view(mod.out_axes, mod.in_axes, mod.rank)
            tot_e = float((A ** 2).sum().item()) + 1e-12
            if mod.out_axes == mod.in_axes:
                diag_e = float((torch.diagonal(A[:, :, 0], 0, 0, 1) ** 2).sum().item())
            else:
                diag_e = float((A[:, :, 0] ** 2).sum().item())
            phm_diag_ratios[mod_name] = round(diag_e / tot_e, 4)
        elif isinstance(mod, BilinearGrassSABlock):
            eye8 = torch.eye(NUM_AXES, device=mod.axis_mix_attn.device, dtype=torch.float32)
            dev_a = float((mod.axis_mix_attn.detach().float() - eye8).norm().item())
            dev_m = float((mod.axis_mix_mlp.detach().float() - eye8).norm().item())
            bilinear_eye_devs[mod_name] = round(0.5 * (dev_a + dev_m), 4)
        elif isinstance(mod, (GrassmannianSABlock, GrassmannianCABlock)):
            for proj_attr in ("q_proj", "kv_proj"):
                proj = getattr(mod, proj_attr, None)
                if proj is not None and hasattr(proj, "q_weight"):
                    W = proj.q_weight.detach().float() * proj.row_scale.detach().float().unsqueeze(-1)
                    if W.shape[0] in (112, 128) and W.shape[1] in (112, 128):
                        blockdiag_ratios[f"{mod_name}.{proj_attr}"] = round(_blockdiag_energy_ratio_128(W), 4)

    # DCA по слоям
    dca_summary = core.get_dca_summary() if hasattr(core, "get_dca_summary") else {}
    dca_per_layer: Dict[str, Dict[str, float]] = {}
    for mod_name, mod in core.named_modules():
        if isinstance(mod, PerAxisGRNv3):
            bf = mod.b.detach().float()
            dca_per_layer[mod_name] = {
                "b_in_abs": round(float(bf[..., 0].abs().mean().item()), 4),
                "b_ca_abs": round(float(bf[..., 1].abs().mean().item()), 4) if mod.has_ca else 0.0,
                "b_prev_mean": round(float(bf[..., -1].mean().item()), 4),
                "w_abs": round(float(mod.w.detach().float().abs().mean().item()), 4),
            }

    # OutputHeadV2
    output_head_info: Dict[str, Any] = {}
    if isinstance(getattr(core, "output_head", None), GrassmannianOutputHeadV2):
        oh = core.output_head
        output_head_info = {
            "output_gain_per_axis": [round(float(x), 4) for x in oh.output_gain.detach().float().view(-1).tolist()],
            "axis_sigma": [round(float(x), 5) for x in oh.axis_sigma.detach().float().tolist()],
            "keep_bias": float(oh.keep_bias),
            "axis_shift_delta": float(oh.axis_shift_delta),
        }

    # Нормы столбцов первого слоя ConditionEmbedder (статическая оценка весов)
    cond_w0 = core.cond_embedder.mlp[0].weight.detach().float()
    col_norms = cond_w0.norm(p=2, dim=0)
    cond_col_info: Dict[str, float] = {}
    if isinstance(core.cond_embedder, ConditionEmbedderV2):
        cond_col_info["fourier_time_mean_col_norm"] = round(float(col_norms[:64].mean().item()), 4)
        for idx, sname in enumerate(core.cond_embedder.SENSOR_NAMES):
            cond_col_info[sname] = round(float(col_norms[64 + idx].item()), 4)
    else:
        cond_col_info["t_global_mean_col_norm"] = round(float(col_norms[:64].mean().item()), 4)

    return {
        "max_q_weight_sphere_err": max(sphere_errs) if sphere_errs else 0.0,
        "mean_row_scale": round(float(np.mean(row_scales)), 5) if row_scales else 0.0,
        "mean_v_gate": round(float(np.mean(v_gates)), 5) if v_gates else 0.0,
        "prompt_adapter_layer_scales": prompt_adapter_ls,
        "k_sink_mean_head_norms": k_sink_norms,
        "dca_summary": {k: round(v, 4) for k, v in dca_summary.items()},
        "dca_per_layer": dca_per_layer,
        "output_head_v2": output_head_info,
        "energy_gate_w_rad_mean_norm": round(float(np.mean(list(w_rad_norms.values()))), 5) if w_rad_norms else 0.0,
        "energy_gate_w_rad_norms": w_rad_norms,
        "phm_diag_energy_ratio_mean": round(float(np.mean(list(phm_diag_ratios.values()))), 4) if phm_diag_ratios else None,
        "phm_diag_energy_ratios": phm_diag_ratios,
        "bilinear_eye_dev_mean": round(float(np.mean(list(bilinear_eye_devs.values()))), 4) if bilinear_eye_devs else None,
        "grass_align_score_temps": grass_align_temps,
        "blockdiag_energy_ratio_mean": round(float(np.mean(list(blockdiag_ratios.values()))), 4) if blockdiag_ratios else None,
        "blockdiag_energy_ratios": blockdiag_ratios,
        "cond_embedder_weight_col_norms": cond_col_info,
    }


# %% 3. Общий кэш валидационных батчей (для честного и быстрого A/B)
@torch.no_grad()
def build_val_batch_cache(
    val_loader: DataLoader,
    encoder: nn.Module,
    stopword_mask: torch.Tensor,
    sep_concept: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    max_batches: int,
) -> List[Dict[str, torch.Tensor]]:
    """Предкодирует val-батчи один раз, чтобы A/B модели оценивались на идентичных латентах."""
    sw_dev = stopword_mask.to(device)
    sep_c, _ = canonicalize_cell_basis(sep_concept.float())
    cached: List[Dict[str, torch.Tensor]] = []
    for b_idx, batch in enumerate(val_loader):
        if b_idx >= max_batches:
            break
        prompt_ids = batch["prompt_ids"].to(device)
        ans_ids = batch["ans_ids"].to(device)
        n_real_prompt = batch["n_real_prompt"].to(device)
        n_real_ans = batch["n_real_ans"].to(device)
        B = prompt_ids.shape[0]
        w_p = batch["prompt_weights"].to(device) if "prompt_weights" in batch else make_pooling_weights(prompt_ids, sw_dev)
        w_a = batch["ans_weights"].to(device) if "ans_weights" in batch else make_pooling_weights(ans_ids, sw_dev)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            V_prompt_raw = encoder(prompt_ids, w_p).float()
            V_ans_raw = encoder(ans_ids, w_a).float()
        V_prompt, _ = canonicalize_cell_basis(V_prompt_raw)
        V_ans, _ = canonicalize_cell_basis(V_ans_raw)
        sep_exp = sep_c.to(device).view(1, 1, NUM_AXES, AXIS_DIM).expand(B, 1, NUM_AXES, AXIS_DIM)
        V_0 = torch.cat([sep_exp, V_ans], dim=1)
        cached.append(
            {
                "V_prompt": V_prompt.cpu(),
                "V_ans": V_ans.cpu(),
                "V_0": V_0.cpu(),
                "prompt_ids": batch["prompt_ids"].cpu(),
                "ans_ids": batch["ans_ids"].cpu(),
                "n_real_prompt": batch["n_real_prompt"].cpu(),
                "n_real_ans": batch["n_real_ans"].cpu(),
            }
        )
    return cached


# %% 4. Подробная v2-диагностика (послойный AxisBlockAdaLN + все 19 датчиков)
@torch.no_grad()
def collect_detailed_v2_diagnostics(core: nn.Module) -> Dict[str, Any]:
    """Собирает послойную статистику AxisBlockAdaLN и полный рейтинг 19 датчиков (сбрасывает буферы)."""
    per_layer_adaln: Dict[str, Dict[str, float]] = {}
    r_effs, k99s, tails, deads = [], [], [], []
    for mod_name, mod in core.named_modules():
        if isinstance(mod, AxisBlockAdaLN):
            st = mod.compute_rank_stats()
            if st:
                per_layer_adaln[mod_name] = {k: round(float(v), 4) for k, v in st.items()}
                r_effs.append(float(st["r_eff"]))
                k99s.append(float(st["k_99"]))
                tails.append(float(st["tail_ratio"]))
                deads.append(float(st["dead_frac"]))
            mod._captured_hr = []

    sensor_importance: Dict[str, float] = {}
    if isinstance(getattr(core, "cond_embedder", None), ConditionEmbedderV2):
        simp = core.cond_embedder.compute_sensor_importance()
        core.cond_embedder._captured_sensors = []
        sensor_importance = {
            k: round(float(v), 5)
            for k, v in sorted(simp.items(), key=lambda kv: kv[1], reverse=True)
        }

    summary = {}
    if r_effs:
        summary = {
            "adaln_r_eff_mean": round(float(np.mean(r_effs)), 2),
            "adaln_r_eff_max": round(float(np.max(r_effs)), 2),
            "adaln_k99_max": round(float(np.max(k99s)), 1),
            "adaln_tail_mean": round(float(np.mean(tails)), 4),
            "adaln_dead_mean": round(float(np.mean(deads)), 4),
        }

    return {
        "summary": summary,
        "adaln_per_layer": per_layer_adaln,
        "sensor_importance_ranked": sensor_importance,
    }


# %% 5. Профиль денойзинга по сетке t (с поосевым SD3-shift и поосевой ошибкой k=0..7)
@torch.no_grad()
def profile_denoising_by_t_v2(
    core: nn.Module,
    decoder: nn.Module,
    cached_batch: Dict[str, torch.Tensor],
    void_concept: torch.Tensor,
    device: torch.device,
    t_grid: List[float],
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    seed: int = 777,
) -> List[Dict[str, Any]]:
    """
    Профиль денойзинга по сетке шума t с точным поосевым SD3-shift (axis_shift_delta).
    Считает SimGr(cond/uncond), CFG-ΔSim, TokAcc, CE, orth_err, spec_l1, err_sem/syn/pos,
    поосевые MSE и cos для осей k=0..7, NoiseRadar и sink по CA-слоям.
    """
    V_prompt = cached_batch["V_prompt"].to(device)
    V_ans = cached_batch["V_ans"].to(device)
    V_0 = cached_batch["V_0"].to(device)
    ans_ids = cached_batch["ans_ids"].to(device)
    n_real_prompt = cached_batch["n_real_prompt"].to(device)
    n_real_ans = cached_batch["n_real_ans"].to(device)
    B = V_0.shape[0]

    p_valid = n_real_prompt > 0
    is_content = n_real_ans > 0
    is_true_void = ~is_content
    gt_energy_c = float(((V_ans ** 2).sum(dim=(-2, -1)))[is_content].mean().item()) if is_content.any() else 0.0
    delta = float(getattr(core, "axis_shift_delta", 0.0))

    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(tuple(V_0.shape), generator=gen, dtype=torch.float32).to(device)

    m_cond = torch.ones(B, device=device)
    m_uncond = torch.zeros(B, device=device)
    uncond_mask = torch.zeros_like(p_valid)
    in_dtype = amp_dtype if use_amp else torch.float32
    dec_dtype = next(decoder.parameters()).dtype
    has_radar = hasattr(decoder, "estimate_noise") and getattr(decoder, "noise_radar", None) is not None

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        prompt_cache = core.prepare_prompt_cache(V_prompt.to(in_dtype), prompt_valid_mask=p_valid)
    x_prompt = prompt_cache["x_prompt"]

    ans_ids_cells = ans_ids.view(B, N_ANS, 4)
    real_tok_mask = (ans_ids_cells != PAD_TOKEN_ID) & is_content.unsqueeze(-1)
    real_tok_flat = real_tok_mask.reshape(-1)

    G0 = torch.matmul(V_ans, V_ans.transpose(-1, -2))
    diag_G0 = torch.diagonal(G0, dim1=-2, dim2=-1)

    rows: List[Dict[str, Any]] = []
    for t_val in t_grid:
        t_mat = torch.full((B, N_CANVAS), float(t_val), device=device, dtype=torch.float32)
        t_mat[:, 0] = 0.0
        V_t = add_axis_shifted_noise(V_0, t_mat, noise, delta=delta, sigma_noise=SIGMA_NOISE)
        V_t[:, 0] = V_0[:, 0]

        radar_in = compute_radar_sigma(decoder, V_t, use_amp=use_amp, amp_dtype=amp_dtype)
        v_in = V_t.to(in_dtype)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            V_hat, _, sink_w = core.forward_with_cached_prompt(
                V_t=v_in,
                t_mat=t_mat,
                x_prompt=x_prompt,
                prompt_attn_mask=p_valid,
                m_prompt=m_cond,
                radar_sigma=radar_in,
                return_sink_weights=True,
                prompt_valid_mask=p_valid,
                prompt_cache=prompt_cache,
            )
            V_uncond, _, _ = core.forward_with_cached_prompt(
                V_t=v_in,
                t_mat=t_mat,
                x_prompt=x_prompt,
                prompt_attn_mask=uncond_mask,
                m_prompt=m_uncond,
                radar_sigma=radar_in,
                return_sink_weights=False,
                prompt_valid_mask=p_valid,
                prompt_cache=prompt_cache,
            )

        Vh = V_hat[:, 1:].float()
        Vu = V_uncond[:, 1:].float()

        sim_c = float(sim_gr(Vh, V_ans)[is_content].mean().item()) if is_content.any() else 0.0
        sim_u = float(sim_gr(Vu, V_ans)[is_content].mean().item()) if is_content.any() else 0.0
        cfg_dsim = sim_c - sim_u
        cfg_c = float(torch.sqrt(((Vh - Vu) ** 2).sum(dim=(-2, -1)) + 1e-12)[is_content].mean().item()) if is_content.any() else 0.0
        en_c = float(((Vh ** 2).sum(dim=(-2, -1)))[is_content].mean().item()) if is_content.any() else 0.0
        en_v = float(((Vh ** 2).sum(dim=(-2, -1)))[is_true_void].mean().item()) if is_true_void.any() else 0.0
        rk_c = float(compute_rk_eff(Vh)[is_content].mean().item()) if is_content.any() else 0.0

        # Спектр, ортогональность, 3 блока [sem|syn|pos] и поосевая ошибка k=0..7
        Gh = torch.matmul(Vh, Vh.transpose(-1, -2))
        diag_Gh = torch.diagonal(Gh, dim1=-2, dim2=-1)
        off_Gh = Gh - torch.diag_embed(diag_Gh)
        orth_err_map = (off_Gh ** 2).sum(dim=(-2, -1)) / ((Gh ** 2).sum(dim=(-2, -1)) + 1e-8)
        spec_l1_map = (diag_Gh - diag_G0).abs().mean(dim=-1)

        orth_err = float(orth_err_map[is_content].mean().item()) if is_content.any() else 0.0
        spec_l1 = float(spec_l1_map[is_content].mean().item()) if is_content.any() else 0.0

        diff_sq = (Vh - V_ans) ** 2
        err_sem = float(diff_sq[..., :96].mean(dim=(-2, -1))[is_content].mean().item()) if is_content.any() else 0.0
        err_syn = float(diff_sq[..., 96:112].mean(dim=(-2, -1))[is_content].mean().item()) if is_content.any() else 0.0
        err_pos = float(diff_sq[..., 112:128].mean(dim=(-2, -1))[is_content].mean().item()) if is_content.any() else 0.0

        # Поосевые MSE и |cos| (инвариантно к SignFix) по 8 осям
        axis_mse: List[float] = []
        axis_abs_cos: List[float] = []
        if is_content.any():
            vh_c = Vh[is_content]     # (M, 8, 128)
            va_c = V_ans[is_content]  # (M, 8, 128)
            amse = ((vh_c - va_c) ** 2).mean(dim=(0, 2))  # (8,)
            acos = F.cosine_similarity(vh_c, va_c, dim=-1).abs().mean(dim=0)  # (8,)
            axis_mse = [round(float(x), 6) for x in amse.tolist()]
            axis_abs_cos = [round(float(x), 4) for x in acos.tolist()]

        pred_void = detect_void_mask(Vh, void_concept)
        tp = int((pred_void & is_true_void).sum().item())
        fp = int((pred_void & ~is_true_void).sum().item())
        fn = int((~pred_void & is_true_void).sum().item())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        vf1 = (2 * prec * rec) / max(1e-8, prec + rec) if (tp + fp + fn) > 0 else 1.0

        sink_layers_c: List[float] = []
        sink_void_mean = 0.0
        if sink_w is not None:
            sw = sink_w.float()
            for l_idx in range(sw.shape[1]):
                sw_l = sw[:, l_idx, 1:]
                sink_layers_c.append(round(float(sw_l[is_content].mean().item()), 4) if is_content.any() else 0.0)
            sw_mean = sw.mean(dim=1)[:, 1:]
            if is_true_void.any():
                sink_void_mean = float(sw_mean[is_true_void].mean().item())

        radar_out_c = 0.0
        dict_cos_c = 0.0
        if has_radar and is_content.any():
            sig_c, _, rad_dict = decoder.estimate_noise(Vh.to(dec_dtype))
            radar_out_c = float(sig_c.float()[is_content].mean().item())
            if "dict_stats" in rad_dict:
                cos_c = rad_dict["dict_stats"].float().view(B, N_ANS, 4, 4)[..., 0].mean(dim=-1)
                dict_cos_c = float(cos_c[is_content].mean().item())

        # TokAcc и CE только по реальным токенам (ограничиваем до 512 токенов для скорости на CPU)
        tok_acc = 0.0
        ce_val = 0.0
        if real_tok_flat.any():
            h_dec = decoder.decode_hidden(Vh.to(dec_dtype))
            idx_r = torch.where(real_tok_flat)[0][:512]
            tgt_r = ans_ids.reshape(-1)[idx_r].long()
            logits_r = decoder.get_tied_head_logits(h_dec.reshape(-1, h_dec.shape[-1])[idx_r])
            tok_acc = float((logits_r.argmax(dim=-1) == tgt_r).float().mean().item())
            ce_val = float(F.cross_entropy(logits_r.float(), tgt_r).item())

        rows.append(
            {
                "t": t_val,
                "sim_gr_cond": round(sim_c, 4),
                "sim_gr_uncond": round(sim_u, 4),
                "cfg_dsim": round(cfg_dsim, 4),
                "cfg_contrast": round(cfg_c, 4),
                "tok_acc": round(tok_acc, 4),
                "ce_loss": round(ce_val, 3),
                "orth_err": round(orth_err, 4),
                "spec_l1": round(spec_l1, 5),
                "err_sem": round(err_sem, 6),
                "err_syn": round(err_syn, 6),
                "err_pos": round(err_pos, 6),
                "energy_hat": round(en_c, 4),
                "energy_gt": round(gt_energy_c, 4),
                "energy_void": round(en_v, 4),
                "rk_eff": round(rk_c, 3),
                "void_f1": round(vf1, 4),
                "radar_sigma_in": round(float(radar_in[:, 1:][is_content].mean().item()), 4) if is_content.any() else 0.0,
                "radar_sigma_hat": round(radar_out_c, 4),
                "dict_cos_max": round(dict_cos_c, 4),
                "sink_by_ca": sink_layers_c,
                "sink_void": round(sink_void_mean, 4),
                "axis_mse": axis_mse,
                "axis_abs_cos": axis_abs_cos,
            }
        )
    return rows


# %% 6. Зонд защиты чистых якорей (lambda_keep / L_anchor, ADR 144)
@torch.no_grad()
def probe_clean_anchors(
    core: nn.Module,
    decoder: nn.Module,
    cached_batch: Dict[str, torch.Tensor],
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    noisy_t: float = 0.50,
    seed: int = 1234,
) -> Dict[str, Any]:
    """
    Проверяет работу OutputHeadV2 lambda_keep и L_anchor:
    чётные контентные ячейки остаются чистыми (t_i = 0), нечётные — зашумляются до noisy_t (при t_bar > 0).
    Сравнивает lambda_keep на чистых vs зашумлённых клетках и замеряет сохранность чистых якорей.
    """
    V_prompt = cached_batch["V_prompt"].to(device)
    V_ans = cached_batch["V_ans"].to(device)
    V_0 = cached_batch["V_0"].to(device)
    ans_ids = cached_batch["ans_ids"].to(device)
    n_real_prompt = cached_batch["n_real_prompt"].to(device)
    n_real_ans = cached_batch["n_real_ans"].to(device)
    B = V_0.shape[0]

    p_valid = n_real_prompt > 0
    is_content = n_real_ans > 0
    delta = float(getattr(core, "axis_shift_delta", 0.0))
    in_dtype = amp_dtype if use_amp else torch.float32

    # Чётные позиции ответа — чистые якоря (t=0), нечётные — шум noisy_t
    pos_idx = torch.arange(N_ANS, device=device).view(1, N_ANS).expand(B, N_ANS)
    clean_ans_mask = is_content & ((pos_idx % 2) == 0)
    noisy_ans_mask = is_content & ((pos_idx % 2) == 1)

    t_mat = torch.full((B, N_CANVAS), float(noisy_t), device=device, dtype=torch.float32)
    t_mat[:, 0] = 0.0
    t_mat[:, 1:] = torch.where(clean_ans_mask, torch.zeros_like(t_mat[:, 1:]), t_mat[:, 1:])

    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(tuple(V_0.shape), generator=gen, dtype=torch.float32).to(device)
    V_t = add_axis_shifted_noise(V_0, t_mat, noise, delta=delta, sigma_noise=SIGMA_NOISE)
    V_t[:, 0] = V_0[:, 0]

    radar_in = compute_radar_sigma(decoder, V_t, use_amp=use_amp, amp_dtype=amp_dtype)

    captured_lam: List[torch.Tensor] = []
    hook_handle = None
    oh = getattr(core, "output_head", None)
    if isinstance(oh, GrassmannianOutputHeadV2):
        def _hook(_mod: nn.Module, _inp: Any, out: Any) -> None:
            if isinstance(out, tuple) and len(out) == 2:
                extra = out[1]
                keep_logit, _ = extra.chunk(2, dim=-1)
                lam = torch.sigmoid(keep_logit.float() + oh.keep_bias)
                captured_lam.append(lam.detach().cpu())
        hook_handle = oh.adaln_out.register_forward_hook(_hook)

    try:
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            V_hat, _, _ = core(
                V_t=V_t.to(in_dtype),
                t_mat=t_mat,
                V_prompt=V_prompt.to(in_dtype),
                prompt_valid_mask=p_valid,
                prompt_attn_mask=p_valid,
                m_prompt=torch.ones(B, device=device),
                radar_sigma=radar_in,
                return_sink_weights=False,
            )
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    Vh = V_hat[:, 1:].float()
    anchor_l1 = float((Vh - V_ans).abs().mean(dim=(-2, -1))[clean_ans_mask].mean().item()) if clean_ans_mask.any() else 0.0
    anchor_sim = float(sim_gr(Vh, V_ans)[clean_ans_mask].mean().item()) if clean_ans_mask.any() else 0.0
    noisy_sim = float(sim_gr(Vh, V_ans)[noisy_ans_mask].mean().item()) if noisy_ans_mask.any() else 0.0

    # Точность токенов на чистых якорях после прохода через модель
    dec_dtype = next(decoder.parameters()).dtype
    ans_ids_cells = ans_ids.view(B, N_ANS, 4)
    clean_tok_m = ((ans_ids_cells != PAD_TOKEN_ID) & clean_ans_mask.unsqueeze(-1)).reshape(-1)
    anchor_tok_acc = 0.0
    if clean_tok_m.any():
        h_dec = decoder.decode_hidden(Vh.to(dec_dtype))
        idx_c = torch.where(clean_tok_m)[0][:512]
        tgt_c = ans_ids.reshape(-1)[idx_c].long()
        logits_c = decoder.get_tied_head_logits(h_dec.reshape(-1, h_dec.shape[-1])[idx_c])
        anchor_tok_acc = float((logits_c.argmax(dim=-1) == tgt_c).float().mean().item())

    out: Dict[str, Any] = {
        "noisy_t": noisy_t,
        "anchor_l1_err": round(anchor_l1, 5),
        "anchor_sim_gr": round(anchor_sim, 4),
        "anchor_tok_acc": round(anchor_tok_acc, 4),
        "noisy_neighbor_sim_gr": round(noisy_sim, 4),
    }
    if captured_lam:
        lam_ans = captured_lam[0][:, 1:]  # (B, N_ANS, 8)
        cm = clean_ans_mask.cpu()
        nm = noisy_ans_mask.cpu()
        vm = (~is_content).cpu()
        out["lambda_keep_clean_mean"] = round(float(lam_ans[cm].mean().item()), 4) if cm.any() else 0.0
        out["lambda_keep_noisy_mean"] = round(float(lam_ans[nm].mean().item()), 4) if nm.any() else 0.0
        out["lambda_keep_void_mean"] = round(float(lam_ans[vm].mean().item()), 4) if vm.any() else 0.0
        out["lambda_keep_clean_per_axis"] = (
            [round(float(x), 4) for x in lam_ans[cm].mean(dim=0).tolist()] if cm.any() else []
        )
    return out


# %% 7. Живая генерация t=1->0 и сравнение сэмплеров + скорость
@torch.no_grad()
def run_generation_benchmark(
    core: nn.Module,
    encoder: nn.Module,
    decoder: nn.Module,
    tokenizer: Any,
    stopword_mask: torch.Tensor,
    sep_concept: torch.Tensor,
    void_concept: torch.Tensor,
    prompts: List[Dict[str, str]],
    device: torch.device,
    sampler_name: str = "crystal_radar",
    num_steps: int = 20,
    cfg_scale: float = 3.0,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    seed: int = 42,
) -> Dict[str, Any]:
    """Генерирует ответы на наборе промптов, замеряя время и все метрики качества."""
    B = len(prompts)
    prompt_texts = [p["prompt"] for p in prompts]
    chunker = WordAwareElasticChunker(
        tokenizer=tokenizer, chunk_size=4, pad_token_id=PAD_TOKEN_ID, stopword_mask=stopword_mask.cpu()
    )

    prompt_rows, weight_rows = [], []
    for txt in prompt_texts:
        ids = encode_token_ids(tokenizer, txt)[:T_PROMPT]
        if not ids:
            ids = [1]
        pack = chunker.chunk_tokens(ids, max_cells=N_PROMPT, elastic_drop_prob=0.0)
        prompt_rows.append(pack["input_ids"])
        weight_rows.append(pack["weights"])

    prompt_ids = torch.stack(prompt_rows, dim=0).to(device)
    w_p = torch.stack(weight_rows, dim=0).to(device)
    p_valid = (prompt_ids.view(B, N_PROMPT, 4) != PAD_TOKEN_ID).sum(dim=-1) > 0
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        V_prompt_raw = encoder(prompt_ids, w_p).float()
    V_prompt, _ = canonicalize_cell_basis(V_prompt_raw)
    sep_canon, _ = canonicalize_cell_basis(sep_concept.float())

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    last_sink_w = None

    if sampler_name == "euler_heun":
        V_gen, last_sink_w = euler_heun_sample(
            core=core,
            V_prompt=V_prompt,
            prompt_valid_mask=p_valid,
            sep_concept=sep_canon,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            decoder=decoder,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            seed=seed,
        )
    elif sampler_name == "crystal_radar":
        V_gen, last_sink_w = crystal_radar_sample(
            core=core,
            V_prompt=V_prompt,
            prompt_valid_mask=p_valid,
            sep_concept=sep_canon,
            decoder=decoder,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            void_ref=void_concept,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            seed=seed,
        )
    elif sampler_name == "rolling_wave":
        V_gen = rolling_wave_sample(
            core=core,
            V_prompt=V_prompt,
            prompt_valid_mask=p_valid,
            sep_concept=sep_canon,
            void_ref=void_concept,
            num_steps=max(num_steps, 32),
            cfg_scale=cfg_scale,
            decoder=decoder,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            seed=seed,
        )
    elif sampler_name == "block_ar":
        V_gen = block_ar_sample(
            core=core,
            V_prompt=V_prompt,
            prompt_valid_mask=p_valid,
            sep_concept=sep_canon,
            void_ref=void_concept,
            block_size=16,
            steps_per_block=max(4, num_steps // 2),
            cfg_scale=cfg_scale,
            decoder=decoder,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            seed=seed,
        )
    else:
        raise ValueError(f"Неизвестный сэмплер: {sampler_name}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    wall_sec = time.perf_counter() - t0

    texts, void_starts = decode_latent_answers(V_gen, decoder, tokenizer, void_concept)
    V_ans = V_gen[:, 1:].float()
    is_void = detect_void_mask(V_ans, void_concept)
    energies = (V_ans ** 2).sum(dim=(-2, -1))
    rks = compute_rk_eff(V_ans)

    # Ортогональность сгенерированных клеток
    Gh = torch.matmul(V_ans, V_ans.transpose(-1, -2))
    diag_Gh = torch.diagonal(Gh, dim1=-2, dim2=-1)
    orth_err_map = ((Gh - torch.diag_embed(diag_Gh)) ** 2).sum(dim=(-2, -1)) / ((Gh ** 2).sum(dim=(-2, -1)) + 1e-8)

    dec_dtype = next(decoder.parameters()).dtype
    has_radar = hasattr(decoder, "estimate_noise") and getattr(decoder, "noise_radar", None) is not None
    radar_sig_all = None
    dict_cos_all = None
    if has_radar:
        sig_c, _, rad_dict = decoder.estimate_noise(V_ans.to(dec_dtype))
        radar_sig_all = sig_c.float()
        if "dict_stats" in rad_dict:
            dict_cos_all = rad_dict["dict_stats"].float().view(B, N_ANS, 4, 4)[..., 0].mean(dim=-1)

    items = []
    sink_mean = last_sink_w.mean(dim=1) if (last_sink_w is not None and last_sink_w.ndim == 3) else last_sink_w
    for b in range(B):
        content_mask = ~is_void[b]
        void_mask_b = is_void[b]
        sink_ans = sink_mean[b, 1:] if sink_mean is not None else None
        n_keep = int(void_starts[b] - 1) if void_starts[b] > 0 else N_ANS
        pad_slot_ratio = 0.0
        if n_keep > 0:
            logits_b = decoder(V_ans[b : b + 1, :n_keep].to(dec_dtype))
            pred_ids_b = logits_b.argmax(dim=-1)[0]
            pad_slot_ratio = float((pred_ids_b == PAD_TOKEN_ID).float().mean().item())

        item: Dict[str, Any] = {
            "source": prompts[b].get("source", "fixed"),
            "prompt": prompt_texts[b][:240],
            "reference": prompts[b].get("reference", "")[:240],
            "answer": texts[b][:400],
            "answer_len_chars": len(texts[b]),
            "bad_utf8_count": texts[b].count("\ufffd"),
            "pad_slot_ratio_content": round(pad_slot_ratio, 4),
            "void_start_cell": int(void_starts[b]),
            "num_void_cells": int(void_mask_b.sum().item()),
            "content_energy_mean": round(float(energies[b, content_mask].mean().item()), 4) if content_mask.any() else 0.0,
            "void_energy_mean": round(float(energies[b, void_mask_b].mean().item()), 4) if void_mask_b.any() else 0.0,
            "content_rk_eff_mean": round(float(rks[b, content_mask].mean().item()), 3) if content_mask.any() else 0.0,
            "content_orth_err": round(float(orth_err_map[b, content_mask].mean().item()), 4) if content_mask.any() else 0.0,
            "radar_sigma_content": round(float(radar_sig_all[b, content_mask].mean().item()), 4)
            if (radar_sig_all is not None and content_mask.any())
            else 0.0,
            "dict_cos_max_content": round(float(dict_cos_all[b, content_mask].mean().item()), 4)
            if (dict_cos_all is not None and content_mask.any())
            else 0.0,
            "sink_attn_content": round(float(sink_ans[content_mask].mean().item()), 4)
            if (sink_ans is not None and content_mask.any())
            else None,
            "sink_attn_void": round(float(sink_ans[void_mask_b].mean().item()), 4)
            if (sink_ans is not None and void_mask_b.any())
            else None,
        }
        items.append(item)

    cross_prompt_sim = None
    lexical_overlap = None
    if B >= 2:
        cross_prompt_sim = round(float(sim_gr(V_ans[0], V_ans[1]).mean().item()), 4)
        toks_0 = set(texts[0].lower().split())
        toks_1 = set(texts[1].lower().split())
        union = toks_0 | toks_1
        lexical_overlap = round(float(len(toks_0 & toks_1) / len(union)) if union else 0.0, 4)

    return {
        "sampler": sampler_name,
        "num_steps": num_steps,
        "cfg_scale": cfg_scale,
        "batch_size": B,
        "wall_sec": round(wall_sec, 3),
        "sec_per_step": round(wall_sec / max(1, num_steps), 4),
        "cross_answer_latent_sim_0_vs_1": cross_prompt_sim,
        "cross_answer_lexical_jaccard_0_vs_1": lexical_overlap,
        "samples": items,
    }


# %% 8. Evaluation Gate и диагностические выводы / A/B таблица
def evaluate_gate_v2(val_metrics: Dict[str, Any], cfg_dsim: Dict[str, float], v2_diag: Dict[str, Any]) -> Dict[str, bool]:
    """Проверяет пороги Evaluation Gate Phase 3 + критерии v2 (CFG-ΔSim и AxisBlockAdaLN)."""
    cfg_hi = val_metrics.get("val/cfg_contrast_hi", val_metrics.get("val/cfg_contrast", 0.0))
    sink_c_hi = val_metrics.get("val/sink_attn_content_hi", val_metrics.get("val/sink_attn_content", 0.0))
    checks = {
        "1_sim_gr_low_gt_0.82": val_metrics.get("val/sim_gr_denoising", 0.0) > 0.82,
        "1_sim_gr_hi_gt_0.45": val_metrics.get("val/sim_gr_hi", 0.0) > 0.45,
        "2_cfg_contrast_hi_gt_0.25": cfg_hi > 0.25,
        "2_cfg_dsim_t50_gt_0": cfg_dsim.get("val/cfg_dsim_t50", 0.0) > 0.0,
        "2_cfg_dsim_t75_gt_0": cfg_dsim.get("val/cfg_dsim_t75", 0.0) > 0.0,
        "3_void_f1_gt_0.95": val_metrics.get("val/void_f1", 0.0) > 0.95,
        "3_void_start_pm1_gt_0.95": val_metrics.get("val/void_start_pm1", 0.0) > 0.95,
        "4_sink_content_hi_in_0.10_0.45": 0.10 <= sink_c_hi <= 0.45,
        "4_sink_void_gt_0.70": val_metrics.get("val/sink_attn_void", 0.0) > 0.70,
        "4_sigma_adaln_ge_1e-2": val_metrics.get("val/sigma_adaln_min", 0.0) >= 1e-2,
        "4_rk_eff_ge_1.5": val_metrics.get("val/rk_eff_content", 0.0) >= 1.5,
        "5_radar_sigma_lt_0.06": val_metrics.get("val/radar_sigma_cell", 1.0) < 0.06,
        "5_dict_cos_max_gt_0.20": val_metrics.get("val/dict_cos_max", 0.0) > 0.20,
    }
    adaln_sum = v2_diag.get("summary", {})
    if "adaln_r_eff_max" in adaln_sum:
        checks["6_adaln_r_eff_not_saturated_lt_48"] = adaln_sum["adaln_r_eff_max"] < 48.0
    return checks


def generate_diagnostic_recommendations(models_report: Dict[str, Dict[str, Any]]) -> List[str]:
    """Формирует список конкретных диагностических сигналов для принятия решения по дальнейшим шагам."""
    notes: List[str] = []
    for label, rep in models_report.items():
        ev = rep.get("evaluation_gate", {}).get("metrics", {})
        dsim = rep.get("cfg_dsim", {})
        v2d = rep.get("v2_diagnostics", {}).get("summary", {})
        anc = rep.get("anchor_probe", {})
        tprof = rep.get("t_profile", [])

        # 1. AxisBlockAdaLN r=64 bottleneck (ADR 144 §3.3)
        r_max = v2d.get("adaln_r_eff_max")
        k99_max = v2d.get("adaln_k99_max")
        if r_max is not None:
            if r_max > 48.0 or (k99_max is not None and k99_max >= 62.0):
                notes.append(f"[{label}] AxisBlockAdaLN близок к насыщению (r_eff_max={r_max}, k99_max={k99_max}/64) -> рассмотреть расширение r=96..128.")
            elif r_max < 24.0:
                notes.append(f"[{label}] AxisBlockAdaLN имеет большой запас (r_eff_max={r_max}, k99_max={k99_max}/64) -> можно оставить r=64 или ужать до 32.")
            else:
                notes.append(f"[{label}] AxisBlockAdaLN в оптимальной зоне (r_eff_max={r_max}, k99_max={k99_max}/64).")

        # 2. CFG-ΔSim
        if dsim:
            d50 = dsim.get("val/cfg_dsim_t50", 0.0)
            d75 = dsim.get("val/cfg_dsim_t75", 0.0)
            d95 = dsim.get("val/cfg_dsim_t95", 0.0)
            if d75 <= 0.005:
                notes.append(f"[{label}] Слабый CFG-ΔSim на высоком шуме (t50={d50:+.4f}, t75={d75:+.4f}, t95={d95:+.4f}) -> промпт слабо направляет генерацию с нуля.")
            else:
                notes.append(f"[{label}] Положительный CFG-ΔSim (t50={d50:+.4f}, t75={d75:+.4f}, t95={d95:+.4f}) -> промпт реально приближает прогноз к V_0.")

        # 3. Проверка хвоста осей 4..7 на малом шуме t <= 0.25 (баланс axis_shift_delta=0.35)
        low_t_rows = [r for r in tprof if r["t"] <= 0.26 and r.get("axis_abs_cos")]
        if low_t_rows:
            r0 = low_t_rows[0]
            head_cos = float(np.mean(r0["axis_abs_cos"][:4]))
            tail_cos = float(np.mean(r0["axis_abs_cos"][4:]))
            gap = head_cos - tail_cos
            notes.append(
                f"[{label}] При t={r0['t']}: |cos| осей 0..3 = {head_cos:.3f} vs осей 4..7 = {tail_cos:.3f} (разрыв {gap:+.3f})."
                + (" Хвост заметно отстаёт -> проверить delta=0.25..0.30." if gap > 0.15 else " Баланс осей здоровый.")
            )

        # 4. Проверка lambda_keep на чистых якорях
        if "lambda_keep_clean_mean" in anc:
            lc = anc["lambda_keep_clean_mean"]
            ln = anc["lambda_keep_noisy_mean"]
            notes.append(
                f"[{label}] Anchor probe: lambda_keep(clean={lc:.3f} vs noisy={ln:.3f}), "
                f"anchor_sim={anc.get('anchor_sim_gr', 0.0):.4f}, anchor_tok_acc={anc.get('anchor_tok_acc', 0.0):.3f}."
            )

        # 5. Блоки sem/syn/pos
        if "val/err_sem" in ev:
            notes.append(
                f"[{label}] Ошибки блоков [sem/syn/pos] = {ev['val/err_sem']:.5f} / {ev['val/err_syn']:.5f} / {ev['val/err_pos']:.5f}, "
                f"orth_err={ev.get('val/orth_err', 0.0):.4f}, spec_l1={ev.get('val/spec_l1', 0.0):.5f}."
            )

    # A/B сравнение первых двух моделей
    labels = list(models_report.keys())
    if len(labels) >= 2:
        a, b = labels[0], labels[1]
        ma = models_report[a]
        mb = models_report[b]
        ea = ma.get("evaluation_gate", {}).get("metrics", {})
        eb = mb.get("evaluation_gate", {}).get("metrics", {})
        da = ma.get("cfg_dsim", {})
        db = mb.get("cfg_dsim", {})
        if ea and eb:
            d_sim_lo = ea.get("val/sim_gr_denoising", 0.0) - eb.get("val/sim_gr_denoising", 0.0)
            d_sim_hi = ea.get("val/sim_gr_hi", 0.0) - eb.get("val/sim_gr_hi", 0.0)
            d_tok_lo = ea.get("val/tok_acc_low", 0.0) - eb.get("val/tok_acc_low", 0.0)
            d_ce_hi = ea.get("val/ce_loss_hi", 0.0) - eb.get("val/ce_loss_hi", 0.0)
            d_dsim75 = da.get("val/cfg_dsim_t75", 0.0) - db.get("val/cfg_dsim_t75", 0.0)
            notes.append(
                f"[A/B {a} vs {b}] dSim_lo={d_sim_lo:+.4f}, dSim_hi={d_sim_hi:+.4f}, "
                f"dTokAcc_lo={d_tok_lo:+.4f}, dCE_hi={d_ce_hi:+.3f}, dCFG_dSim_t75={d_dsim75:+.4f}."
            )
    return notes


def render_markdown_report(full_report: Dict[str, Any]) -> str:
    """Рендерит структурированный Markdown-отчёт (одиночный или A/B)."""
    lines: List[str] = []
    lines.append("# Phase 3 Alt v2 — Аналитический отчёт")
    lines.append("")
    lines.append(f"- **Дата:** {full_report['timestamp']}")
    lines.append(f"- **Устройство:** `{full_report['device']}` (`amp={full_report['amp_dtype']}`)")
    lines.append(f"- **Валидация:** `{full_report['val_data_path']}`")
    lines.append("")

    models = full_report["models"]
    labels = list(models.keys())

    # 1. Сводная таблица (работает и для 1, и для 2+ чекпоинтов)
    lines.append("## 1. Сводная таблица метрик")
    lines.append("")
    header = "| Метрика | " + " | ".join(labels) + (" | Delta (A - B) |" if len(labels) >= 2 else " |")
    sep = "|---|" + "|".join(["---:" for _ in labels]) + ("|---:|" if len(labels) >= 2 else "|")
    lines.append(header)
    lines.append(sep)

    def _get_metric(rep: Dict[str, Any], key: str) -> Optional[float]:
        li = rep.get("load_info", {})
        ev = rep.get("evaluation_gate", {}).get("metrics", {})
        ds = rep.get("cfg_dsim", {})
        v2 = rep.get("v2_diagnostics", {}).get("summary", {})
        anc = rep.get("anchor_probe", {})
        wg = rep.get("weight_geometry", {})
        if key == "step":
            return float(li.get("step", 0))
        if key == "params_m":
            return float(li.get("total_params_m", 0.0))
        if key == "train_ce_tok_acc":
            v = li.get("ckpt_train_metrics", {}).get("train/ce_tok_acc")
            return float(v) if v is not None else None
        if key == "train_loss_total":
            v = li.get("ckpt_train_metrics", {}).get("loss/total")
            return float(v) if v is not None else None
        if key in ev:
            return float(ev[key])
        if key in ds:
            return float(ds[key])
        if key in v2:
            return float(v2[key])
        if key in anc:
            return float(anc[key])
        if key == "max_sphere_err":
            return float(wg.get("max_q_weight_sphere_err", 0.0))
        if key == "gen_sec_crystal":
            g = rep.get("generation", {}).get("crystal_radar")
            return float(g["wall_sec"]) if g else None
        if key == "gen_sec_euler":
            g = rep.get("generation", {}).get("euler_heun")
            return float(g["wall_sec"]) if g else None
        return None

    metric_rows = [
        ("Step", "step", "{:.0f}"),
        ("Params (M)", "params_m", "{:.2f}"),
        ("Train loss/total (ckpt)", "train_loss_total", "{:.4f}"),
        ("Train CE TokAcc (ckpt)", "train_ce_tok_acc", "{:.4f}"),
        ("val/sim_gr_denoising (low)", "val/sim_gr_denoising", "{:.4f}"),
        ("val/sim_gr_hi (0.75..0.95)", "val/sim_gr_hi", "{:.4f}"),
        ("val/tok_acc_low", "val/tok_acc_low", "{:.4f}"),
        ("val/tok_acc_hi", "val/tok_acc_hi", "{:.4f}"),
        ("val/ce_loss_hi", "val/ce_loss_hi", "{:.3f}"),
        ("val/cfg_dsim_t50", "val/cfg_dsim_t50", "{:+.4f}"),
        ("val/cfg_dsim_t75", "val/cfg_dsim_t75", "{:+.4f}"),
        ("val/cfg_dsim_t95", "val/cfg_dsim_t95", "{:+.4f}"),
        ("val/cfg_contrast_hi", "val/cfg_contrast_hi", "{:.4f}"),
        ("val/void_f1", "val/void_f1", "{:.4f}"),
        ("val/void_start_pm1", "val/void_start_pm1", "{:.4f}"),
        ("val/orth_err", "val/orth_err", "{:.4f}"),
        ("val/spec_l1", "val/spec_l1", "{:.5f}"),
        ("val/err_sem", "val/err_sem", "{:.5f}"),
        ("val/err_syn", "val/err_syn", "{:.5f}"),
        ("val/err_pos", "val/err_pos", "{:.5f}"),
        ("val/radar_sigma_cell", "val/radar_sigma_cell", "{:.4f}"),
        ("val/dict_cos_max", "val/dict_cos_max", "{:.4f}"),
        ("val/sink_attn_content_hi", "val/sink_attn_content_hi", "{:.4f}"),
        ("val/sink_attn_void", "val/sink_attn_void", "{:.4f}"),
        ("val/rk_eff_content", "val/rk_eff_content", "{:.3f}"),
        ("val/sigma_adaln_min", "val/sigma_adaln_min", "{:.2e}"),
        ("AdaLN r_eff_mean", "adaln_r_eff_mean", "{:.2f}"),
        ("AdaLN r_eff_max", "adaln_r_eff_max", "{:.2f}"),
        ("AdaLN k99_max (/64)", "adaln_k99_max", "{:.0f}"),
        ("Anchor lambda_keep (clean)", "lambda_keep_clean_mean", "{:.4f}"),
        ("Anchor lambda_keep (noisy)", "lambda_keep_noisy_mean", "{:.4f}"),
        ("Anchor SimGr (clean)", "anchor_sim_gr", "{:.4f}"),
        ("Anchor TokAcc (clean)", "anchor_tok_acc", "{:.4f}"),
        ("Gen wall_sec (euler_heun)", "gen_sec_euler", "{:.2f}"),
        ("Gen wall_sec (crystal_radar)", "gen_sec_crystal", "{:.2f}"),
    ]

    for title, key, fmt in metric_rows:
        vals = [_get_metric(models[lb], key) for lb in labels]
        if all(v is None for v in vals):
            continue
        cells = [fmt.format(v) if v is not None else "—" for v in vals]
        if len(labels) >= 2:
            if vals[0] is not None and vals[1] is not None:
                diff = vals[0] - vals[1]
                cells.append(f"{diff:+.4f}" if abs(diff) < 100 else f"{diff:+.2f}")
            else:
                cells.append("—")
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    lines.append("")

    # 2. Диагностические выводы
    lines.append("## 2. Диагностические сигналы и рекомендации")
    lines.append("")
    for note in full_report.get("recommendations", []):
        lines.append(f"- {note}")
    lines.append("")

    # 3. Детали по каждой модели
    for lb in labels:
        rep = models[lb]
        li = rep["load_info"]
        wg = rep["weight_geometry"]
        lines.append(f"## 3. Детали модели `{lb}`")
        lines.append("")
        lines.append(
            f"- **Архитектура:** `{li['inferred_arch']}` | **Параметры:** `{li['total_params_m']}M` (`{li['model_fp16_mb']} MB` FP16) | "
            f"**EMA:** `{li['used_ema']}` (`updates={li['ema_num_updates']}`, `ema_vs_raw={li['ema_vs_raw_mean_rel_diff']}`)"
        )
        lines.append(
            f"- **Геометрия:** `max_sph_err={wg['max_q_weight_sphere_err']:.2e}`, `mean_row_scale={wg['mean_row_scale']}`, "
            f"`w_rad_norm={wg['energy_gate_w_rad_mean_norm']}`, `phm_diag_ratio={wg['phm_diag_energy_ratio_mean']}`, "
            f"`blkdiag_ratio={wg['blockdiag_energy_ratio_mean']}`"
        )
        lines.append(f"- **DCA summary:** `{wg['dca_summary']}`")
        if wg.get("output_head_v2"):
            lines.append(f"- **OutputHeadV2 output_gain (k=0..7):** `{wg['output_head_v2']['output_gain_per_axis']}`")
        lines.append("")

        # Evaluation Gate checks
        checks = rep.get("evaluation_gate", {}).get("checks", {})
        if checks:
            passed = sum(1 for v in checks.values() if v)
            lines.append(f"### Evaluation Gate (`{passed}/{len(checks)}` PASS)")
            lines.append("")
            for ck_name, ck_ok in checks.items():
                lines.append(f"- {'[PASS]' if ck_ok else '[FAIL]'} `{ck_name}`")
            lines.append("")

        # Топ и хвост датчиков ConditionEmbedderV2
        simp = rep.get("v2_diagnostics", {}).get("sensor_importance_ranked", {})
        if simp:
            top_items = list(simp.items())[:8]
            bot_items = list(simp.items())[-5:]
            lines.append("### Важность датчиков ConditionEmbedderV2 (`||W|| * std_val`)")
            lines.append("")
            lines.append("- **Топ-8:** " + ", ".join(f"`{k}={v:.4f}`" for k, v in top_items))
            lines.append("- **Нижние-5:** " + ", ".join(f"`{k}={v:.4f}`" for k, v in bot_items))
            lines.append("")

        # Профиль по t
        tprof = rep.get("t_profile", [])
        if tprof:
            lines.append("### Профиль денойзинга по сетке `t`")
            lines.append("")
            lines.append("| t | SimGr_c | CFG_dSim | TokAcc | CE | orth_err | spec_l1 | radar_hat | VOID_F1 | cos(0..3) / cos(4..7) |")
            lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for r in tprof:
                ac = r.get("axis_abs_cos", [])
                c_hi = f"{np.mean(ac[:4]):.3f}" if len(ac) == 8 else "—"
                c_lo = f"{np.mean(ac[4:]):.3f}" if len(ac) == 8 else "—"
                lines.append(
                    f"| {r['t']:.2f} | {r['sim_gr_cond']:.4f} | {r['cfg_dsim']:+.4f} | {r['tok_acc']:.3f} | "
                    f"{r['ce_loss']:.2f} | {r['orth_err']:.4f} | {r['spec_l1']:.5f} | {r['radar_sigma_hat']:.4f} | "
                    f"{r['void_f1']:.3f} | {c_hi} / {c_lo} |"
                )
            lines.append("")

        # Живая генерация
        gen = rep.get("generation", {})
        if gen:
            lines.append("### Живая генерация `t = 1 -> 0`")
            lines.append("")
            for s_name, s_data in gen.items():
                lines.append(
                    f"#### Сэмплер `{s_name}` (`steps={s_data['num_steps']}`, `cfg={s_data['cfg_scale']}`, "
                    f"`wall={s_data['wall_sec']}s`, `cross_sim={s_data['cross_answer_latent_sim_0_vs_1']}`)"
                )
                lines.append("")
                for idx_s, smp in enumerate(s_data["samples"]):
                    ans_clean = smp["answer"].replace("\n", " ")[:220]
                    lines.append(
                        f"- **[{idx_s}] ({smp['source']})** `void_start={smp['void_start_cell']}`, `rk={smp['content_rk_eff_mean']}`, "
                        f"`radar={smp['radar_sigma_content']}`, `orth={smp['content_orth_err']}`, `pad={smp['pad_slot_ratio_content']}`"
                    )
                    lines.append(f"  - Q: `{smp['prompt'][:140]}`")
                    if smp.get("reference"):
                        lines.append(f"  - Ref: `{smp['reference'][:140]}`")
                    lines.append(f"  - A: `{ans_clean}`")
                lines.append("")

    return "\n".join(lines)


# %% 9. Разбор аргументов и главная точка входа
def parse_checkpoint_specs(specs: List[str], default_dir: str) -> List[Tuple[str, str]]:
    """Разбирает аргументы --checkpoints (формат `alias=path` или `path`)."""
    if not specs:
        found = sorted(Path(default_dir).glob("*.pt"))
        if not found:
            raise FileNotFoundError(f"В {default_dir} не найдено ни одного .pt чекпоинта. Укажите --checkpoints явно.")
        specs = [str(p) for p in found]

    parsed: List[Tuple[str, str]] = []
    used_names: set = set()
    for raw in specs:
        if "=" in raw and not os.path.exists(raw):
            alias, path = raw.split("=", 1)
            alias = alias.strip()
            path = path.strip()
        else:
            path = raw.strip()
            stem = Path(path).stem
            if "safe" in stem.lower():
                alias = "safe"
            elif "control" in stem.lower() or "dense" in stem.lower():
                alias = "control"
            else:
                alias = stem[-24:]
        if alias in used_names:
            idx = 2
            while f"{alias}_{idx}" in used_names:
                idx += 1
            alias = f"{alias}_{idx}"
        used_names.add(alias)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Чекпоинт '{alias}' не найден: {path}")
        parsed.append((alias, path))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Глубокая аналитика и A/B-сравнение чекпоинтов Phase 3 Alt v2")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[],
        help="Один или несколько чекпоинтов: `path.pt` или `alias=path.pt` (пусто = все .pt в checkpoints/)",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--amp-dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument(
        "--sys1-quant",
        type=str,
        default="auto",
        choices=["auto", "fp32", "half", "fp16", "bf16", "emb_int8", "int8"],
    )
    parser.add_argument("--sys1-mode", type=str, default="inference", choices=["train", "inference"])
    parser.add_argument("--no-fast-vocab", action="store_true", help="Отключить FastVocabOptimizer")
    parser.add_argument("--no-ema", action="store_true", help="Загружать сырые веса вместо EMA")
    parser.add_argument("--axis-shift-delta", type=float, default=None, help="Переопределить axis_shift_delta")
    parser.add_argument("--fast", action="store_true", help="Быстрый пресет для CPU (меньше батчей и шагов)")
    parser.add_argument("--batch-size", type=int, default=0, help="Размер батча (0 = авто: 8 CUDA / 4 CPU)")
    parser.add_argument("--eval-batches", type=int, default=0, help="Число val-батчей (0 = авто: 12 CUDA / 6 CPU)")
    parser.add_argument("--cfg-dsim-batches", type=int, default=0, help="Число батчей для CFG-ΔSim (0 = авто: 4 CUDA / 2 CPU)")
    parser.add_argument("--t-grid", type=str, default="0.1,0.25,0.5,0.75,0.9,1.0", help="Сетка t через запятую")
    parser.add_argument("--samplers", type=str, default="euler_heun,crystal_radar", help="Сэмплеры через запятую")
    parser.add_argument("--gen-steps", type=int, default=20, help="Шаги генерации")
    parser.add_argument("--cfg-scale", type=float, default=3.0, help="Масштаб CFG")
    parser.add_argument("--val-prompts", type=int, default=2, help="Сколько реальных промптов добавить из val датасета")
    parser.add_argument("--skip-val", action="store_true")
    parser.add_argument("--skip-t-profile", action="store_true")
    parser.add_argument("--skip-anchor-probe", action="store_true")
    parser.add_argument("--skip-gen", action="store_true")
    parser.add_argument("--save-dir", type=str, default=str(ALT_DIR / "output"))
    parser.add_argument("--tag", type=str, default="", help="Префикс имени файлов отчёта (по умолчанию авто)")
    args = parser.parse_args()

    env = AltEnvironmentConfig.detect(override_device=args.device, amp_dtype_str=args.amp_dtype)
    ckpt_specs = parse_checkpoint_specs(args.checkpoints, env.checkpoint_dir)

    if args.fast:
        val_bs = args.batch_size if args.batch_size > 0 else (8 if env.device.type == "cuda" else 2)
        eval_batches = args.eval_batches if args.eval_batches > 0 else 3
        cfg_dsim_batches = args.cfg_dsim_batches if args.cfg_dsim_batches > 0 else 2
        gen_steps = min(args.gen_steps, 12)
        t_grid = [0.1, 0.35, 0.65, 0.9]
    else:
        val_bs = args.batch_size if args.batch_size > 0 else (8 if env.device.type == "cuda" else 4)
        eval_batches = args.eval_batches if args.eval_batches > 0 else (12 if env.device.type == "cuda" else 6)
        cfg_dsim_batches = args.cfg_dsim_batches if args.cfg_dsim_batches > 0 else (4 if env.device.type == "cuda" else 2)
        gen_steps = args.gen_steps
        t_grid = [float(x.strip()) for x in args.t_grid.split(",") if x.strip()]

    sampler_list = [s.strip() for s in args.samplers.split(",") if s.strip()]

    print(
        f"[Analyze] Устройство={env.device} | amp={env.use_amp}({env.amp_dtype}) | "
        f"чекпоинтов={len(ckpt_specs)} ({[a for a, _ in ckpt_specs]}) | "
        f"val_bs={val_bs}, eval_batches={eval_batches}, cfg_dsim_batches={cfg_dsim_batches}, gen_steps={gen_steps}",
        flush=True,
    )

    # 1. Загружаем замороженную Систему 1 один раз
    t_s1 = time.perf_counter()
    tokenizer = get_tokenizer()
    sys1_amp = env.amp_dtype if env.use_amp else None
    encoder = get_encoder(
        weights_path=env.encoder_weights_path,
        device=env.device,
        eval_mode=True,
        quant_mode=args.sys1_quant,
        mode=args.sys1_mode,
        amp_dtype=sys1_amp,
    )
    decoder = get_decoder(
        weights_path=env.decoder_weights_path,
        encoder=encoder,
        device=env.device,
        eval_mode=True,
        quant_mode=args.sys1_quant,
        mode=args.sys1_mode,
        amp_dtype=sys1_amp,
    )
    for p in list(encoder.parameters()) + list(decoder.parameters()):
        p.requires_grad_(False)

    if not args.no_fast_vocab:
        q_pol = getattr(decoder, "quant_policy", None)
        fv_dtype = q_pol.radar_dtype if q_pol is not None else (env.amp_dtype if env.use_amp else torch.float32)
        FastVocabOptimizer.from_decoder(decoder=decoder, gemm_dtype=fv_dtype, attach=True)

    sep_concept = torch.load(env.sep_concept_path, map_location="cpu", weights_only=False)
    void_concept = torch.load(env.void_concept_path, map_location="cpu", weights_only=False)
    if os.path.exists(env.stopwords_path):
        sw_obj = torch.load(env.stopwords_path, map_location="cpu", weights_only=False)
        stopword_mask = (sw_obj["mask"] if isinstance(sw_obj, dict) else sw_obj).bool()
    else:
        stopword_mask = torch.zeros(131072, dtype=torch.bool)

    print(
        f"[System 1] Загружена за {time.perf_counter() - t_s1:.1f}s "
        f"(enc={estimate_model_memory_mb(encoder):.1f} MB, dec={estimate_model_memory_mb(decoder):.1f} MB)",
        flush=True,
    )

    # 2. Готовим val-датасет и общий кэш латентов (1 раз для всех чекпоинтов)
    val_ds = None
    val_loader = None
    cached_batches: List[Dict[str, torch.Tensor]] = []
    gen_prompts: List[Dict[str, str]] = [{"source": "fixed", "prompt": p, "reference": ""} for p in DEFAULT_PROMPTS]

    need_val_data = (not args.skip_val) or (not args.skip_t_profile) or (not args.skip_anchor_probe) or (args.val_prompts > 0)
    if need_val_data and os.path.exists(env.val_data_path):
        val_ds = PromptAnswerDataset(
            parquet_path=env.val_data_path,
            tokenizer=tokenizer,
            stopword_mask=stopword_mask,
            max_rows=max(256, eval_batches * val_bs + 16),
            is_validation=True,
        )
        val_loader = DataLoader(val_ds, batch_size=val_bs, shuffle=False, num_workers=0)

        if args.val_prompts > 0 and hasattr(val_ds, "pairs"):
            for i_vp in range(min(args.val_prompts, len(val_ds.pairs))):
                p_txt, a_txt = val_ds.pairs[i_vp]
                gen_prompts.append({"source": f"val_{i_vp}", "prompt": str(p_txt), "reference": str(a_txt)})

        if (not args.skip_t_profile) or (not args.skip_anchor_probe):
            print(f"[Val Cache] Предкодирование 1 батча (bs={val_bs}) для t-профиля и anchor-зонда...", flush=True)
            cached_batches = build_val_batch_cache(
                val_loader=val_loader,
                encoder=encoder,
                stopword_mask=stopword_mask,
                sep_concept=sep_concept,
                device=env.device,
                use_amp=env.use_amp,
                amp_dtype=env.amp_dtype,
                max_batches=1,
            )

    # 3. Прогоняем аналитику по каждому чекпоинту
    models_report: Dict[str, Dict[str, Any]] = {}
    for alias, ckpt_path in ckpt_specs:
        print(f"\n================ [{alias}] {ckpt_path} ================", flush=True)
        core, load_info = load_alt_model_from_checkpoint(
            ckpt_path=ckpt_path,
            void_concept=void_concept,
            device=env.device,
            use_ema=not args.no_ema,
            override_axis_shift_delta=args.axis_shift_delta,
        )
        print(
            f"  [Loaded] step={load_info['step']} | arch={load_info['inferred_arch']['arch_version']}/"
            f"{load_info['inferred_arch']['arch_preset']}/{load_info['inferred_arch']['matrix_mode']} | "
            f"params={load_info['total_params_m']}M | ema={load_info['used_ema']} (n={load_info['ema_num_updates']})",
            flush=True,
        )

        m_rep: Dict[str, Any] = {
            "load_info": load_info,
            "weight_geometry": inspect_alt_weight_geometry(core),
        }

        if (not args.skip_val) and val_loader is not None:
            print(f"  [Eval] Evaluation Gate + v2-диагностика ({eval_batches} батчей x {val_bs})...", flush=True)
            torch.manual_seed(2026)
            core.set_diagnostics_capture(True)
            val_metrics = evaluate_phase3(
                model=core,
                encoder=encoder,
                decoder=decoder,
                tokenizer=tokenizer,
                val_dataloader=val_loader,
                stopword_mask=stopword_mask,
                sep_concept=sep_concept,
                void_concept=void_concept,
                device=env.device,
                use_amp=env.use_amp,
                amp_dtype=env.amp_dtype,
                max_batches=eval_batches,
                run_generation_check=False,
            )
            core.set_diagnostics_capture(False)
            val_metrics.pop("gen_samples", None)
            v2_diag = collect_detailed_v2_diagnostics(core)

            print(f"  [CFG-dSim] Оценка полезности промпта ({cfg_dsim_batches} батчей)...", flush=True)
            cfg_dsim = evaluate_cfg_dsim(
                model=core,
                encoder=encoder,
                val_dataloader=val_loader,
                stopword_mask=stopword_mask,
                sep_concept=sep_concept,
                device=env.device,
                use_amp=env.use_amp,
                amp_dtype=env.amp_dtype,
                max_batches=cfg_dsim_batches,
            )

            m_rep["v2_diagnostics"] = v2_diag
            m_rep["cfg_dsim"] = {k: round(float(v), 5) for k, v in cfg_dsim.items()}
            m_rep["evaluation_gate"] = {
                "metrics": {k: (round(float(v), 6) if isinstance(v, (int, float)) else v) for k, v in val_metrics.items()},
                "checks": evaluate_gate_v2(val_metrics, cfg_dsim, v2_diag),
            }

        if (not args.skip_t_profile) and cached_batches:
            print(f"  [t-Profile] Поосевой профиль денойзинга по t={t_grid}...", flush=True)
            m_rep["t_profile"] = profile_denoising_by_t_v2(
                core=core,
                decoder=decoder,
                cached_batch=cached_batches[0],
                void_concept=void_concept,
                device=env.device,
                t_grid=t_grid,
                use_amp=env.use_amp,
                amp_dtype=env.amp_dtype,
            )

        if (not args.skip_anchor_probe) and cached_batches:
            print("  [Anchor] Зонд защиты чистых якорей (lambda_keep / L_anchor)...", flush=True)
            m_rep["anchor_probe"] = probe_clean_anchors(
                core=core,
                decoder=decoder,
                cached_batch=cached_batches[0],
                device=env.device,
                use_amp=env.use_amp,
                amp_dtype=env.amp_dtype,
            )

        if not args.skip_gen:
            m_rep["generation"] = {}
            for s_name in sampler_list:
                print(
                    f"  [Gen] Сэмплер {s_name} (prompts={len(gen_prompts)}, steps={gen_steps}, cfg={args.cfg_scale})...",
                    flush=True,
                )
                m_rep["generation"][s_name] = run_generation_benchmark(
                    core=core,
                    encoder=encoder,
                    decoder=decoder,
                    tokenizer=tokenizer,
                    stopword_mask=stopword_mask,
                    sep_concept=sep_concept,
                    void_concept=void_concept,
                    prompts=gen_prompts,
                    device=env.device,
                    sampler_name=s_name,
                    num_steps=gen_steps,
                    cfg_scale=args.cfg_scale,
                    use_amp=env.use_amp,
                    amp_dtype=env.amp_dtype,
                )

        models_report[alias] = m_rep
        del core
        if env.device.type == "cuda":
            torch.cuda.empty_cache()

    full_report: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(env.device),
        "amp_dtype": str(env.amp_dtype) if env.use_amp else "fp32",
        "val_data_path": env.val_data_path,
        "models": models_report,
        "recommendations": generate_diagnostic_recommendations(models_report),
    }

    md_text = render_markdown_report(full_report)
    print("\n" + md_text, flush=True)

    save_dir = Path(args.save_dir)
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        save_dir = ALT_DIR / "output"
        save_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag.strip() or ("_vs_".join(models_report.keys()) + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    json_path = save_dir / f"analytics_{tag}_report.json"
    md_path = save_dir / f"analytics_{tag}_report.md"
    json_path.write_text(json.dumps(full_report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(md_text, encoding="utf-8")
    print(f"\n[Saved] JSON: {json_path}\n[Saved] Markdown: {md_path}", flush=True)


if __name__ == "__main__":
    main()
