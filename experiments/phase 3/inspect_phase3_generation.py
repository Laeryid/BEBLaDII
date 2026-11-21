"""
Глубокий аудит обученного чекпоинта Фазы 3 Plan C:
- геометрия весов (сферичность, PromptAdapter LayerScale, AdaLN gates, k_sink, датчики ConditionEmbedder s0..s7, EMA vs Raw);
- профиль денойзинга по сетке шума t in [0.1 .. 1.0] (Sim_Gr, CFG, энергия, sink по 3 CA-слоям, NoiseRadarHead, tok_acc);
- сравнение сэмплеров (euler_heun, crystal_radar, rolling_wave, block_ar) на разнородных промптах;
- Evaluation Gate на валидационном датасете val_phase2.parquet.

Запуск из корня репозитория:
    .venv\\Scripts\\python.exe "experiments/phase 3/inspect_phase3_generation.py" --checkpoint "experiments/phase 3/checkpoints/planC_Phase3_Diffusion_checkpoints_phase3_step_4500.pt"
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PHASE3_DIR = Path(__file__).resolve().parent
REPO_ROOT = PHASE3_DIR.parents[1]
for p in (str(REPO_ROOT), str(PHASE3_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from storage.components.decoder import get_decoder
from storage.components.encoder import get_encoder
from storage.components.plan_c_layers import WordAwareElasticChunker, canonicalize_cell_basis
from storage.components.tokenizer import get_tokenizer
from train_phase3 import (
    N_ANS,
    N_CANVAS,
    N_PROMPT,
    PAD_TOKEN_ID,
    SIGMA_DATA,
    T_PROMPT,
    EnvironmentConfig,
    GrassmannianDiffusionCore,
    PromptAnswerDataset,
    block_ar_sample,
    compute_radar_sigma,
    compute_rk_eff,
    crystal_radar_sample,
    decode_latent_answers,
    detect_void_mask,
    encode_token_ids,
    euler_heun_sample,
    evaluate_phase3,
    find_latest_checkpoint,
    make_pooling_weights,
    migrate_cond_embedder_state_dict,
    rolling_wave_sample,
    sim_gr,
)


def load_ema_weights_into_core(
    core: GrassmannianDiffusionCore,
    ckpt_path: str,
    use_ema: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Загружает веса из чекпоинта с обязательной сферической проекцией *.q_weight (ADR 132)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_sd = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    ema_sd = ckpt.get("ema_state_dict")

    sd = {k: v.clone() for k, v in raw_sd.items()}
    used_ema = False
    ema_vs_raw_diffs: List[float] = []
    if use_ema and ema_sd:
        for k, v in ema_sd.items():
            if k in sd:
                val = v.clone()
                if k.endswith(".q_weight"):
                    val = F.normalize(val.float(), p=2, dim=-1, eps=1e-8)
                raw_f = raw_sd[k].float()
                denom = raw_f.norm().item() + 1e-8
                ema_vs_raw_diffs.append(float((val.float() - raw_f).norm().item() / denom))
                sd[k] = val.to(sd[k].dtype)
        used_ema = True

    expected_in_dim = int(core.cond_embedder.mlp[0].weight.shape[1])
    sd = migrate_cond_embedder_state_dict(sd, expected_in_dim=expected_in_dim, verbose=True)
    core.load_state_dict(sd, strict=True)

    opt_summary: Dict[str, Any] = {}
    if "opt_muon_state_dict" in ckpt:
        opt_summary["muon_lr"] = [g.get("lr") for g in ckpt["opt_muon_state_dict"].get("param_groups", [])]
    if "opt_adamw_state_dict" in ckpt:
        opt_summary["adamw_lr"] = [g.get("lr") for g in ckpt["opt_adamw_state_dict"].get("param_groups", [])]
    if "scaler_state_dict" in ckpt:
        opt_summary["scaler"] = ckpt["scaler_state_dict"]

    load_info = {
        "step": int(ckpt.get("step", 0)),
        "used_ema": used_ema,
        "ema_vs_raw_mean_rel_diff": float(np.mean(ema_vs_raw_diffs)) if ema_vs_raw_diffs else 0.0,
        "ema_vs_raw_max_rel_diff": float(np.max(ema_vs_raw_diffs)) if ema_vs_raw_diffs else 0.0,
        "ckpt_metrics": ckpt.get("metrics", {}),
        "optimizer_summary": opt_summary,
    }
    return load_info, {"raw_sd": raw_sd, "ema_sd": ema_sd}


def inspect_weight_geometry(core: GrassmannianDiffusionCore) -> Dict[str, Any]:
    """Проверяет сферичность *.q_weight, масштабы row_scale, v_gate, PromptAdapter, AdaLN gates, k_sink и датчики s0..s7."""
    sphere_errs, row_scales, v_gates, layer_scales = [], [], [], []
    prompt_adapter_ls: Dict[str, float] = {}
    adaln_gate_biases: Dict[str, Dict[str, float]] = {}
    k_sink_norms: Dict[str, float] = {}
    sensor_input_norms: Dict[str, float] = {}

    with torch.no_grad():
        for name, p in core.named_parameters():
            if name.endswith(".q_weight"):
                norms = p.float().norm(p=2, dim=-1)
                sphere_errs.append(float((norms - 1.0).abs().max().item()))
            elif name.endswith(".row_scale"):
                row_scales.append(float(p.float().mean().item()))
            elif name.endswith(".v_gate"):
                v_gates.append(float(p.float().mean().item()))
            elif name.endswith(".ls_attn") or name.endswith(".ls_mlp"):
                val = float(p.float().abs().mean().item())
                layer_scales.append(val)
                prompt_adapter_ls[name] = round(val, 5)
            elif name.endswith(".k_sink"):
                k_sink_norms[name] = round(float(p.float().norm(p=2, dim=-1).mean().item()), 5)
            elif name.endswith(".adaln_proj.bias"):
                b = p.float()
                h = b.numel() // 6
                adaln_gate_biases[name] = {
                    "gate_attn_abs_mean": round(float(b[2 * h : 3 * h].abs().mean().item()), 5),
                    "gate_mlp_abs_mean": round(float(b[5 * h : 6 * h].abs().mean().item()), 5),
                }

        # Веса первого слоя ConditionEmbedder.mlp[0]:
        # новый формат (256 x 72): t_glob(0:64), s0..s7(64:72);
        # старый формат (256 x 200): t_loc(0:128), t_glob(128:192), s0..s7(192:200)
        w0 = core.cond_embedder.mlp[0].weight.float()
        col_norms = w0.norm(p=2, dim=0)
        sensor_names = [
            "s0_energy",
            "s1_purity",
            "s2_rk_eff",
            "s3_void_sim",
            "s4_is_clean",
            "s5_m_prompt",
            "s6_cryst_frac",
            "s7_radar_sigma",
        ]
        if w0.shape[1] == 72:
            sensor_input_norms["t_global_mean_col_norm"] = round(float(col_norms[:64].mean().item()), 5)
            for idx, sname in enumerate(sensor_names):
                sensor_input_norms[sname] = round(float(col_norms[64 + idx].item()), 5)
        else:
            sensor_input_norms["t_local_mean_col_norm"] = round(float(col_norms[:128].mean().item()), 5)
            sensor_input_norms["t_global_mean_col_norm"] = round(float(col_norms[128:192].mean().item()), 5)
            for idx, sname in enumerate(sensor_names):
                sensor_input_norms[sname] = round(float(col_norms[192 + idx].item()), 5)

    return {
        "max_q_weight_sphere_err": max(sphere_errs) if sphere_errs else 0.0,
        "mean_row_scale": float(sum(row_scales) / max(1, len(row_scales))),
        "mean_v_gate": float(sum(v_gates) / max(1, len(v_gates))),
        "mean_prompt_layer_scale": float(sum(layer_scales) / max(1, len(layer_scales))),
        "prompt_adapter_layer_scales": prompt_adapter_ls,
        "k_sink_mean_head_norms": k_sink_norms,
        "adaln_gate_biases": adaln_gate_biases,
        "cond_embedder_input_col_norms": sensor_input_norms,
    }


@torch.no_grad()
def profile_denoising_by_t(
    core: GrassmannianDiffusionCore,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    val_batch: Dict[str, torch.Tensor],
    stopword_mask: torch.Tensor,
    sep_concept: torch.Tensor,
    void_concept: torch.Tensor,
    device: torch.device,
    t_grid: Optional[List[float]] = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> List[Dict[str, Any]]:
    """
    Профиль денойзинга по фиксированным уровням шума t:
    показывает Sim_Gr, CFG contrast, энергию, rk_eff, sink по 3 CA-слоям,
    показания NoiseRadarHead и прямую точность токенов декодера (tok_acc).
    """
    if t_grid is None:
        t_grid = [0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0]

    prompt_ids = val_batch["prompt_ids"].to(device)
    ans_ids = val_batch["ans_ids"].to(device)
    n_real_prompt = val_batch["n_real_prompt"].to(device)
    n_real_ans = val_batch["n_real_ans"].to(device)
    B = prompt_ids.shape[0]
    sw_dev = stopword_mask.to(device)

    w_p = val_batch["prompt_weights"].to(device) if "prompt_weights" in val_batch else make_pooling_weights(prompt_ids, sw_dev)
    w_a = val_batch["ans_weights"].to(device) if "ans_weights" in val_batch else make_pooling_weights(ans_ids, sw_dev)

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        V_prompt_raw = encoder(prompt_ids, w_p).float()
        V_ans_raw = encoder(ans_ids, w_a).float()
    V_prompt, _ = canonicalize_cell_basis(V_prompt_raw)
    V_ans, _ = canonicalize_cell_basis(V_ans_raw)

    sep_canon, _ = canonicalize_cell_basis(sep_concept.float())
    sep_exp = sep_canon.to(device).view(1, 1, 8, 128).expand(B, 1, 8, 128)
    V_0 = torch.cat([sep_exp, V_ans], dim=1)
    p_valid = n_real_prompt > 0
    is_content = n_real_ans > 0
    is_true_void = ~is_content
    gt_energy_c = float(((V_ans ** 2).sum(dim=(-2, -1)))[is_content].mean().item()) if is_content.any() else 0.0

    gen = torch.Generator(device=device)
    gen.manual_seed(777)
    noise = torch.randn(V_0.shape, generator=gen, device=device, dtype=torch.float32)

    m_cond = torch.ones(B, device=device)
    m_uncond = torch.zeros(B, device=device)
    uncond_mask = torch.zeros_like(p_valid)
    dec_dtype = next(decoder.parameters()).dtype
    has_radar = hasattr(decoder, "estimate_noise") and getattr(decoder, "noise_radar", None) is not None

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        x_prompt = core.encode_prompt(V_prompt.to(amp_dtype if use_amp else torch.float32), prompt_valid_mask=p_valid)

    # Маска реальных (не-PAD) токенов внутри контентных ячеек для оценки tok_acc
    ans_ids_cells = ans_ids.view(B, N_ANS, 4)
    real_tok_mask = (ans_ids_cells != PAD_TOKEN_ID) & is_content.unsqueeze(-1)

    rows: List[Dict[str, Any]] = []
    for t_val in t_grid:
        t_mat = torch.full((B, N_CANVAS), t_val, device=device, dtype=torch.float32)
        t_mat[:, 0] = 0.0
        t_exp = t_mat.unsqueeze(-1).unsqueeze(-1)
        V_t = (1.0 - t_exp) * V_0 + t_exp * SIGMA_DATA * noise
        V_t[:, 0] = sep_exp[:, 0]

        radar_in = compute_radar_sigma(decoder, V_t, use_amp=use_amp, amp_dtype=amp_dtype)
        v_in = V_t.to(amp_dtype if use_amp else torch.float32)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            V_hat, _, sink_w = core.forward_with_cached_prompt(
                V_t=v_in,
                t_mat=t_mat,
                x_prompt=x_prompt,
                prompt_attn_mask=p_valid,
                m_prompt=m_cond,
                radar_sigma=radar_in,
                return_sink_weights=True,
            )
            V_uncond, _, _ = core.forward_with_cached_prompt(
                V_t=v_in,
                t_mat=t_mat,
                x_prompt=x_prompt,
                prompt_attn_mask=uncond_mask,
                m_prompt=m_uncond,
                radar_sigma=radar_in,
                return_sink_weights=False,
            )

        Vh = V_hat[:, 1:].float()
        Vu = V_uncond[:, 1:].float()

        sim_c = float(sim_gr(Vh, V_ans)[is_content].mean().item()) if is_content.any() else 0.0
        cfg_c = float(torch.sqrt(((Vh - Vu) ** 2).sum(dim=(-2, -1)) + 1e-12)[is_content].mean().item()) if is_content.any() else 0.0
        en_c = float(((Vh ** 2).sum(dim=(-2, -1)))[is_content].mean().item()) if is_content.any() else 0.0
        en_v = float(((Vh ** 2).sum(dim=(-2, -1)))[is_true_void].mean().item()) if is_true_void.any() else 0.0
        rk_c = float(compute_rk_eff(Vh)[is_content].mean().item()) if is_content.any() else 0.0

        pred_void = detect_void_mask(Vh, void_concept)
        tp = int((pred_void & is_true_void).sum().item())
        fp = int((pred_void & ~is_true_void).sum().item())
        fn = int((~pred_void & is_true_void).sum().item())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        vf1 = (2 * prec * rec) / max(1e-8, prec + rec) if (tp + fp + fn) > 0 else 1.0

        sink_layers_c = []
        sink_void_mean = 0.0
        if sink_w is not None:
            sw = sink_w.float()  # (B, 3, 128)
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

        # Декодирование и точность токенов по реальным токенам контентных ячеек
        logits = decoder(Vh.to(dec_dtype))
        pred_ids = logits.argmax(dim=-1).view(B, N_ANS, 4)
        tok_acc = float((pred_ids[real_tok_mask] == ans_ids_cells[real_tok_mask]).float().mean().item()) if real_tok_mask.any() else 0.0

        rows.append(
            {
                "t": t_val,
                "sim_gr_content": round(sim_c, 4),
                "cfg_contrast": round(cfg_c, 4),
                "tok_acc_content": round(tok_acc, 4),
                "energy_content_hat": round(en_c, 4),
                "energy_content_gt": round(gt_energy_c, 4),
                "energy_void_hat": round(en_v, 4),
                "rk_eff_content": round(rk_c, 3),
                "void_f1": round(vf1, 4),
                "radar_sigma_in": round(float(radar_in[:, 1:][is_content].mean().item()), 4) if is_content.any() else 0.0,
                "radar_sigma_hat": round(radar_out_c, 4),
                "dict_cos_max_hat": round(dict_cos_c, 4),
                "sink_content_by_ca_layer": sink_layers_c,
                "sink_void_mean": round(sink_void_mean, 4),
            }
        )
    return rows


@torch.no_grad()
def trace_pair_generation(
    core: GrassmannianDiffusionCore,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    tokenizer: Any,
    stopword_mask: torch.Tensor,
    sep_concept: torch.Tensor,
    void_concept: torch.Tensor,
    prompts: List[str],
    device: torch.device,
    sampler_name: str = "crystal_radar",
    num_steps: int = 20,
    cfg_scale: float = 3.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Генерирует ответы заданным сэмплером и оценивает их различимость, латентную геометрию и показания NoiseRadarHead."""
    B = len(prompts)
    chunker = WordAwareElasticChunker(
        tokenizer=tokenizer, chunk_size=4, pad_token_id=PAD_TOKEN_ID, stopword_mask=stopword_mask.cpu()
    )

    prompt_rows = []
    weight_rows = []
    for txt in prompts:
        ids = encode_token_ids(tokenizer, txt)[:T_PROMPT]
        if not ids:
            ids = [1]
        pack = chunker.chunk_tokens(ids, max_cells=N_PROMPT, elastic_drop_prob=0.0)
        prompt_rows.append(pack["input_ids"])
        weight_rows.append(pack["weights"])

    prompt_ids = torch.stack(prompt_rows, dim=0).to(device)
    w_p = torch.stack(weight_rows, dim=0).to(device)
    p_valid = (prompt_ids.view(B, N_PROMPT, 4) != PAD_TOKEN_ID).sum(dim=-1) > 0
    V_prompt, _ = canonicalize_cell_basis(encoder(prompt_ids, w_p).float())
    sep_canon, _ = canonicalize_cell_basis(sep_concept.float())

    use_amp = device.type == "cuda"
    amp_dtype = torch.float16 if use_amp else torch.float32
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

    texts, void_starts = decode_latent_answers(V_gen, decoder, tokenizer, void_concept)
    V_ans = V_gen[:, 1:].float()
    is_void = detect_void_mask(V_ans, void_concept)
    energies = (V_ans ** 2).sum(dim=(-2, -1))
    rks = compute_rk_eff(V_ans)
    s_void = sim_gr(
        V_ans,
        void_concept.to(device=device, dtype=torch.float32).view(1, 1, 8, 128).expand_as(V_ans),
    )

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

        items.append(
            {
                "prompt": prompts[b],
                "answer": texts[b][:400],
                "answer_len_chars": len(texts[b]),
                "bad_utf8_count": texts[b].count("\ufffd"),
                "pad_slot_ratio_content": round(pad_slot_ratio, 4),
                "void_start_cell": int(void_starts[b]),
                "num_void_cells": int(void_mask_b.sum().item()),
                "content_energy_mean": round(float(energies[b, content_mask].mean().item()), 4) if content_mask.any() else 0.0,
                "void_energy_mean": round(float(energies[b, void_mask_b].mean().item()), 4) if void_mask_b.any() else 0.0,
                "content_rk_eff_mean": round(float(rks[b, content_mask].mean().item()), 3) if content_mask.any() else 0.0,
                "void_sim_mean": round(float(s_void[b, void_mask_b].mean().item()), 4) if void_mask_b.any() else 0.0,
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
        )

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
        "samples": items,
        "cross_answer_latent_sim_0_vs_1": cross_prompt_sim,
        "cross_answer_lexical_jaccard_0_vs_1": lexical_overlap,
    }


def evaluate_gate_status(val_metrics: Dict[str, Any]) -> Dict[str, bool]:
    """Проверяет пороги Evaluation Gate Фазы 3 (CFG и sink_content берутся по окну высокого шума)."""
    cfg_hi = val_metrics.get("val/cfg_contrast_hi", val_metrics["val/cfg_contrast"])
    sink_c_hi = val_metrics.get("val/sink_attn_content_hi", val_metrics["val/sink_attn_content"])
    return {
        "1_sim_gr_low_gt_0.82": val_metrics["val/sim_gr_denoising"] > 0.82,
        "1_sim_gr_hi_gt_0.45": val_metrics.get("val/sim_gr_hi", 0.0) > 0.45,
        "2_cfg_contrast_hi_gt_0.25": cfg_hi > 0.25,
        "3_void_f1_gt_0.95": val_metrics["val/void_f1"] > 0.95,
        "3_void_start_pm1_gt_0.95": val_metrics["val/void_start_pm1"] > 0.95,
        "4_sink_content_hi_in_0.10_0.45": 0.10 <= sink_c_hi <= 0.45,
        "4_sink_void_gt_0.70": val_metrics["val/sink_attn_void"] > 0.70,
        "4_sigma_adaln_ge_1e-2": val_metrics["val/sigma_adaln_min"] >= 1e-2,
        "4_rk_eff_ge_1.5": val_metrics["val/rk_eff_content"] >= 1.5,
        "5_radar_sigma_lt_0.06": val_metrics.get("val/radar_sigma_cell", 1.0) < 0.06,
        "5_dict_cos_max_gt_0.20": val_metrics.get("val/dict_cos_max", 0.0) > 0.20,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Инспекция и Evaluation Gate чекпоинта Фазы 3 Plan C")
    parser.add_argument("--checkpoint", type=str, default="auto", help="Путь к .pt чекпоинту или 'auto'")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--no-ema", action="store_true", help="Загружать сырые веса вместо EMA")
    parser.add_argument("--steps", type=int, default=20, help="Число шагов диффузии")
    parser.add_argument("--cfg-scale", type=float, default=3.0, help="Масштаб Classifier-Free Guidance")
    parser.add_argument(
        "--samplers",
        type=str,
        default="euler_heun,crystal_radar",
        help="Список сэмплеров через запятую (euler_heun,crystal_radar,rolling_wave,block_ar)",
    )
    parser.add_argument("--batch-size", type=int, default=0, help="Размер батча валидации (0 = авто: 8 на CUDA, 4 на CPU)")
    parser.add_argument("--eval-batches", type=int, default=20, help="Число батчей валидации")
    parser.add_argument("--skip-val", action="store_true", help="Пропустить прогон по val_phase2.parquet")
    parser.add_argument("--skip-t-profile", action="store_true", help="Пропустить профиль по сетке шума t")
    parser.add_argument("--save-json", type=str, default=None, help="Сохранить отчёт в JSON")
    args = parser.parse_args()

    env = EnvironmentConfig.detect(override_device=args.device)
    ckpt_path = find_latest_checkpoint(env.checkpoint_dir) if args.checkpoint == "auto" else args.checkpoint
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Чекпоинт не найден: {ckpt_path}")

    val_bs = args.batch_size if args.batch_size > 0 else (8 if env.device.type == "cuda" else 4)

    tokenizer = get_tokenizer()
    encoder = get_encoder(weights_path=env.encoder_weights_path, device=env.device, eval_mode=True)
    decoder = get_decoder(weights_path=env.decoder_weights_path, encoder=encoder, device=env.device, eval_mode=True)
    for p in list(encoder.parameters()) + list(decoder.parameters()):
        p.requires_grad_(False)

    sep_concept = torch.load(env.sep_concept_path, map_location="cpu", weights_only=False)
    void_concept = torch.load(env.void_concept_path, map_location="cpu", weights_only=False)
    if os.path.exists(env.stopwords_path):
        sw_obj = torch.load(env.stopwords_path, map_location="cpu", weights_only=False)
        stopword_mask = (sw_obj["mask"] if isinstance(sw_obj, dict) else sw_obj).bool()
    else:
        stopword_mask = torch.zeros(131072, dtype=torch.bool)

    core = GrassmannianDiffusionCore(void_ref=void_concept).to(env.device)
    load_info, _ = load_ema_weights_into_core(core, ckpt_path, use_ema=not args.no_ema)
    core.eval()

    report: Dict[str, Any] = {
        "checkpoint": ckpt_path,
        "device": str(env.device),
        "load_info": load_info,
        "weight_geometry": inspect_weight_geometry(core),
    }

    default_prompts = [
        "Explain why normalized Grassmannian density similarity stays bounded in [0, 1].",
        "Write a short Python function that checks whether an integer is prime.",
    ]
    sampler_list = [s.strip() for s in args.samplers.split(",") if s.strip()]
    report["pair_generation"] = {}
    for s_name in sampler_list:
        print(f"[Inspect] Запуск генерации ({s_name}, steps={args.steps}, cfg={args.cfg_scale})...", flush=True)
        report["pair_generation"][s_name] = trace_pair_generation(
            core=core,
            encoder=encoder,
            decoder=decoder,
            tokenizer=tokenizer,
            stopword_mask=stopword_mask,
            sep_concept=sep_concept,
            void_concept=void_concept,
            prompts=default_prompts,
            device=env.device,
            sampler_name=s_name,
            num_steps=args.steps,
            cfg_scale=args.cfg_scale,
        )

    if os.path.exists(env.val_data_path) and (not args.skip_val or not args.skip_t_profile):
        val_ds = PromptAnswerDataset(
            parquet_path=env.val_data_path,
            tokenizer=tokenizer,
            stopword_mask=stopword_mask,
            max_rows=512,
            is_validation=True,
        )
        val_loader = DataLoader(val_ds, batch_size=val_bs, shuffle=False, num_workers=0)

        if not args.skip_t_profile:
            print(f"[Inspect] Построение профиля денойзинга по t (batch_size={val_bs})...", flush=True)
            first_batch = next(iter(val_loader))
            report["t_profile"] = profile_denoising_by_t(
                core=core,
                encoder=encoder,
                decoder=decoder,
                val_batch=first_batch,
                stopword_mask=stopword_mask,
                sep_concept=sep_concept,
                void_concept=void_concept,
                device=env.device,
                use_amp=env.use_amp,
                amp_dtype=env.amp_dtype,
            )

        if not args.skip_val:
            print(f"[Inspect] Запуск Evaluation Gate ({args.eval_batches} батчей x {val_bs})...", flush=True)
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
                max_batches=args.eval_batches,
                run_generation_check=False,
            )
            val_metrics.pop("gen_samples", None)
            report["evaluation_gate"] = {
                "metrics": val_metrics,
                "checks": evaluate_gate_status(val_metrics),
            }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.save_json:
        out_p = Path(args.save_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Inspect] Отчёт сохранён в: {out_p}", flush=True)


if __name__ == "__main__":
    main()
