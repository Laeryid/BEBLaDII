"""
Глубокий аудит обученного чекпоинта Фазы 2 Plan C (Evaluation Gate + пошаговая трассировка CFG/VOID).

Запуск из корня репозитория:
    .venv\\Scripts\\python.exe "experiments/phase 2/inspect_phase2_generation.py" --checkpoint "experiments/phase 2/checkpoints/phase2_step_500.pt"
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PHASE2_DIR = Path(__file__).resolve().parent
REPO_ROOT = PHASE2_DIR.parents[1]
for p in (str(REPO_ROOT), str(PHASE2_DIR)):
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
    compute_rk_eff,
    crystal_radar_sample,
    decode_latent_answers,
    detect_void_mask,
    encode_token_ids,
    euler_heun_sample,
    evaluate_phase3,
    find_latest_checkpoint,
    make_pooling_weights,
    sim_gr,
)


def load_ema_weights_into_core(
    core: GrassmannianDiffusionCore,
    ckpt_path: str,
    use_ema: bool = True,
) -> Dict[str, Any]:
    """Загружает веса из чекпоинта с обязательной сферической проекцией *.q_weight (ADR 132)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}

    used_ema = False
    if use_ema and ckpt.get("ema_state_dict"):
        ema_sd = ckpt["ema_state_dict"]
        for k, v in ema_sd.items():
            if k in sd:
                val = v.clone()
                if k.endswith(".q_weight"):
                    val = F.normalize(val.float(), p=2, dim=-1, eps=1e-8)
                sd[k] = val.to(sd[k].dtype)
        used_ema = True

    core.load_state_dict(sd, strict=True)
    return {
        "step": int(ckpt.get("step", 0)),
        "used_ema": used_ema,
        "ckpt_metrics": ckpt.get("metrics", {}),
    }


def inspect_weight_geometry(core: GrassmannianDiffusionCore) -> Dict[str, float]:
    """Проверяет сферичность *.q_weight, масштабы row_scale и v_gate."""
    sphere_errs, row_scales, v_gates, layer_scales = [], [], [], []
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
                layer_scales.append(float(p.float().abs().mean().item()))

    return {
        "max_q_weight_sphere_err": max(sphere_errs) if sphere_errs else 0.0,
        "mean_row_scale": float(sum(row_scales) / max(1, len(row_scales))),
        "mean_v_gate": float(sum(v_gates) / max(1, len(v_gates))),
        "mean_prompt_layer_scale": float(sum(layer_scales) / max(1, len(layer_scales))),
    }


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
    num_steps: int = 20,
    cfg_scale: float = 3.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Генерирует ответы на список промптов и оценивает их различимость и латентную геометрию."""
    B = len(prompts)
    sw_dev = stopword_mask.to(device)
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

    V_gen, last_sink_w = crystal_radar_sample(
        core=core,
        V_prompt=V_prompt,
        prompt_valid_mask=p_valid,
        sep_concept=sep_canon,
        decoder=decoder,
        num_steps=num_steps,
        cfg_scale=cfg_scale,
        use_amp=(device.type == "cuda"),
        amp_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        seed=seed,
    )

    texts, void_starts = decode_latent_answers(V_gen, decoder, tokenizer, void_concept)
    V_ans = V_gen[:, 1:].float()
    is_void = detect_void_mask(V_ans, void_concept)
    energies = (V_ans ** 2).sum(dim=(-2, -1))
    rks = compute_rk_eff(V_ans)
    s_void = sim_gr(
        V_ans,
        void_concept.to(device=device, dtype=torch.float32).view(1, 1, 8, 128).expand_as(V_ans),
    )

    items = []
    sink_mean = last_sink_w.mean(dim=1) if (last_sink_w is not None and last_sink_w.ndim == 3) else last_sink_w
    for b in range(B):
        content_mask = ~is_void[b]
        void_mask_b = is_void[b]
        sink_ans = sink_mean[b, 1:] if sink_mean is not None else None
        items.append(
            {
                "prompt": prompts[b],
                "answer": texts[b],
                "void_start_cell": int(void_starts[b]),
                "num_void_cells": int(void_mask_b.sum().item()),
                "content_energy_mean": float(energies[b, content_mask].mean().item()) if content_mask.any() else 0.0,
                "void_energy_mean": float(energies[b, void_mask_b].mean().item()) if void_mask_b.any() else 0.0,
                "content_rk_eff_mean": float(rks[b, content_mask].mean().item()) if content_mask.any() else 0.0,
                "void_sim_mean": float(s_void[b, void_mask_b].mean().item()) if void_mask_b.any() else 0.0,
                "sink_attn_content": float(sink_ans[content_mask].mean().item()) if (sink_ans is not None and content_mask.any()) else 0.0,
                "sink_attn_void": float(sink_ans[void_mask_b].mean().item()) if (sink_ans is not None and void_mask_b.any()) else 0.0,
            }
        )

    cross_prompt_sim = None
    lexical_overlap = None
    if B >= 2:
        # Оценка различимости ответов на разные вопросы (§7.1 п.2)
        cross_prompt_sim = float(sim_gr(V_ans[0], V_ans[1]).mean().item())
        toks_0 = set(texts[0].lower().split())
        toks_1 = set(texts[1].lower().split())
        union = toks_0 | toks_1
        lexical_overlap = float(len(toks_0 & toks_1) / len(union)) if union else 0.0

    return {
        "samples": items,
        "cross_answer_latent_sim_0_vs_1": cross_prompt_sim,
        "cross_answer_lexical_jaccard_0_vs_1": lexical_overlap,
    }


def evaluate_gate_status(val_metrics: Dict[str, Any]) -> Dict[str, bool]:
    """Проверяет пороги Evaluation Gate из §7.1 плана Фазы 2 (CFG и sink_content берутся по окну высокого шума)."""
    cfg_hi = val_metrics.get("val/cfg_contrast_hi", val_metrics["val/cfg_contrast"])
    sink_c_hi = val_metrics.get("val/sink_attn_content_hi", val_metrics["val/sink_attn_content"])
    return {
        "1_sim_gr_gt_0.82": val_metrics["val/sim_gr_denoising"] > 0.82,
        "2_cfg_contrast_hi_gt_0.25": cfg_hi > 0.25,
        "3_void_f1_gt_0.95": val_metrics["val/void_f1"] > 0.95,
        "3_void_start_pm1_gt_0.95": val_metrics["val/void_start_pm1"] > 0.95,
        "4_sink_content_hi_in_0.10_0.45": 0.10 <= sink_c_hi <= 0.45,
        "4_sink_void_gt_0.70": val_metrics["val/sink_attn_void"] > 0.70,
        "4_sigma_adaln_ge_1e-2": val_metrics["val/sigma_adaln_min"] >= 1e-2,
        "4_rk_eff_ge_1.5": val_metrics["val/rk_eff_content"] >= 1.5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Инспекция и Evaluation Gate чекпоинта Фазы 2 Plan C")
    parser.add_argument("--checkpoint", type=str, default="auto", help="Путь к .pt чекпоинту или 'auto'")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--no-ema", action="store_true", help="Загружать сырые веса вместо EMA")
    parser.add_argument("--steps", type=int, default=20, help="Число шагов Эйлера-Хьюна")
    parser.add_argument("--cfg-scale", type=float, default=3.0, help="Масштаб Classifier-Free Guidance")
    parser.add_argument("--eval-batches", type=int, default=20, help="Число батчей валидации")
    parser.add_argument("--skip-val", action="store_true", help="Пропустить прогон по val_phase2.parquet")
    parser.add_argument("--save-json", type=str, default=None, help="Сохранить отчёт в JSON")
    args = parser.parse_args()

    env = EnvironmentConfig.detect(override_device=args.device)
    ckpt_path = find_latest_checkpoint(env.checkpoint_dir) if args.checkpoint == "auto" else args.checkpoint
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Чекпоинт не найден: {ckpt_path}")

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
    load_info = load_ema_weights_into_core(core, ckpt_path, use_ema=not args.no_ema)
    core.eval()

    report: Dict[str, Any] = {
        "checkpoint": ckpt_path,
        "load_info": load_info,
        "weight_geometry": inspect_weight_geometry(core),
    }

    default_prompts = [
        "Explain why normalized Grassmannian density similarity stays bounded in [0, 1].",
        "Write a short Python function that checks whether an integer is prime.",
    ]
    report["pair_generation"] = trace_pair_generation(
        core=core,
        encoder=encoder,
        decoder=decoder,
        tokenizer=tokenizer,
        stopword_mask=stopword_mask,
        sep_concept=sep_concept,
        void_concept=void_concept,
        prompts=default_prompts,
        device=env.device,
        num_steps=args.steps,
        cfg_scale=args.cfg_scale,
    )

    if not args.skip_val and os.path.exists(env.val_data_path):
        val_ds = PromptAnswerDataset(
            parquet_path=env.val_data_path,
            tokenizer=tokenizer,
            stopword_mask=stopword_mask,
            max_rows=512,
            is_validation=True,
        )
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
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
        report["evaluation_gate"] = {
            "metrics": val_metrics,
            "checks": evaluate_gate_status(val_metrics),
        }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.save_json:
        out_p = Path(args.save_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
