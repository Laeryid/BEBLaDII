"""
Модульные и интеграционные проверки архитектуры Фазы 2 Plan C.

Запуск из корня репозитория:
    .venv\\Scripts\\python.exe -B "experiments/phase 2/test_phase2_architecture.py"
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PHASE2_DIR = Path(__file__).resolve().parent
REPO_ROOT = PHASE2_DIR.parents[1]
for p in (str(REPO_ROOT), str(PHASE2_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from train_phase3 import (
    EMAModel,
    GrassmannianDiffusionCore,
    N_CANVAS,
    N_PROMPT,
    SIGMA_DATA,
    block_ar_sample,
    build_rollout_noisy_canvas,
    compute_phase3_loss,
    create_phase2_optimizers,
    crystal_radar_sample,
    euler_heun_sample,
    migrate_cond_embedder_state_dict,
    rolling_wave_sample,
    sample_mixed_noise_schedule,
)


def main() -> None:
    torch.manual_seed(42)
    void_ref = torch.load(REPO_ROOT / "storage/components/void_concept.pt", map_location="cpu", weights_only=False)
    sep_ref = torch.load(REPO_ROOT / "storage/components/sep_concept.pt", map_location="cpu", weights_only=False)

    print("[1/6] Initializing GrassmannianDiffusionCore (~167.6M)...", flush=True)
    core = GrassmannianDiffusionCore(void_ref=void_ref)

    total_params = sum(p.numel() for p in core.parameters() if p.requires_grad)
    assert 166_000_000 <= total_params <= 170_000_000, f"Неожиданное число параметров: {total_params}"
    opt_muon, opt_adamw = create_phase2_optimizers(core)
    assert opt_muon is not None
    muon_cnt = sum(p.numel() for g in opt_muon.param_groups for p in g["params"])
    adamw_cnt = sum(p.numel() for g in opt_adamw.param_groups for p in g["params"])
    assert muon_cnt + adamw_cnt == total_params, "Не все параметры покрыты оптимизаторами"
    print(f"    -> params={total_params/1e6:.2f}M (Muon={muon_cnt/1e6:.2f}M, AdamW={adamw_cnt/1e6:.2f}M)", flush=True)

    # 2. Leak-free тест CAPromptBlock при Full Prompt Dropout
    print("[2/6] Checking CAPromptBlock leak-free + OutputHead t=0 identity...", flush=True)
    B = 2
    V_0 = torch.randn(B, N_CANVAS, 8, 128) * SIGMA_DATA
    V_0[:, 0] = sep_ref
    t_mat = sample_mixed_noise_schedule(B, torch.device("cpu"))
    assert torch.all(t_mat[:, 0] == 0.0), "Ячейка 0 (SEP) обязана иметь t_0 == 0"
    V_t = (1.0 - t_mat.unsqueeze(-1).unsqueeze(-1)) * V_0 + t_mat.unsqueeze(-1).unsqueeze(-1) * SIGMA_DATA * torch.randn_like(V_0)
    V_t[:, 0] = sep_ref

    V_prompt_a = torch.randn(B, N_PROMPT, 8, 128) * SIGMA_DATA
    V_prompt_b = torch.randn(B, N_PROMPT, 8, 128) * SIGMA_DATA
    p_valid = torch.ones(B, N_PROMPT, dtype=torch.bool)
    p_uncond_mask = torch.zeros(B, N_PROMPT, dtype=torch.bool)
    m_zero = torch.zeros(B, dtype=torch.float32)

    core.eval()
    radar_sig = torch.rand(B, N_CANVAS) * 0.5
    radar_sig[:, 0] = 0.0
    with torch.no_grad():
        out_a, _, _ = core(V_t, t_mat, V_prompt_a, p_valid, p_uncond_mask, m_zero, radar_sigma=radar_sig)
        out_b, _, _ = core(V_t, t_mat, V_prompt_b, p_valid, p_uncond_mask, m_zero, radar_sigma=radar_sig)
        out_zero_rad, _, _ = core(V_t, t_mat, V_prompt_a, p_valid, p_uncond_mask, m_zero, radar_sigma=torch.zeros_like(radar_sig))
    leak_diff = (out_a - out_b).abs().max().item()
    assert leak_diff < 1e-6, f"Обнаружена утечка промпта при CFG dropout: max_diff={leak_diff}"
    radar_effect = (out_a - out_zero_rad).abs().max().item()
    assert radar_effect > 0.0, "Датчик s7 (radar_sigma) не влияет на выход ядра"

    # 3. Тождественность на чистых якорях (SEP t_0 == 0), отсутствие утечки t_local и ограничение норм шаром B^128
    assert (out_a[:, 0] - sep_ref.unsqueeze(0)).abs().max().item() < 1e-6, "Искажение ячейки SEP при t_0=0"
    row_norms = out_a.norm(p=2, dim=-1)
    assert row_norms.max().item() <= 1.0 + 1e-5, f"Выход за единичный шар: max_norm={row_norms.max().item()}"
    assert core.cond_embedder.mlp[0].weight.shape == (256, 72), "Ожидалась размерность ConditionEmbedder.mlp[0] (256, 72)"

    # Проверка отсутствия утечки оракульного t_local: два разных t_mat с одинаковым глобальным t_bar обязаны дать идентичный выход
    t_mat_flat = torch.full((B, N_CANVAS), 0.50, dtype=torch.float32)
    t_mat_flat[:, 0] = 0.0
    t_mat_het = torch.cat(
        [
            torch.zeros(B, 1, dtype=torch.float32),
            torch.linspace(0.20, 0.80, N_CANVAS - 1, dtype=torch.float32).unsqueeze(0).expand(B, -1),
        ],
        dim=1,
    )
    with torch.no_grad():
        out_flat, _, _ = core(V_t, t_mat_flat, V_prompt_a, p_valid, p_valid, torch.ones(B), radar_sigma=radar_sig)
        out_het, _, _ = core(V_t, t_mat_het, V_prompt_a, p_valid, p_valid, torch.ones(B), radar_sigma=radar_sig)
    t_local_leak = (out_flat - out_het).abs().max().item()
    assert t_local_leak < 1e-6, f"Обнаружена утечка оракульного t_local: max_diff={t_local_leak}"

    # Проверка бесшовной миграции старого чекпоинта (256, 200) -> (256, 72)
    fake_old_sd = {"cond_embedder.mlp.0.weight": torch.randn(256, 200)}
    migrated_sd = migrate_cond_embedder_state_dict(fake_old_sd, expected_in_dim=72, verbose=False)
    assert migrated_sd["cond_embedder.mlp.0.weight"].shape == (256, 72)
    assert torch.allclose(migrated_sd["cond_embedder.mlp.0.weight"], fake_old_sd["cond_embedder.mlp.0.weight"][:, 128:200])
    print(f"    -> leak_diff={leak_diff:.2e}, t_local_leak={t_local_leak:.2e}, max_row_norm={row_norms.max().item():.6f}", flush=True)

    # 4. Эквивалентность распределенной статистики adaln_stats (полный батч vs DataParallel split)
    print("[3/6] Checking DataParallel gather equivalence for L_adaln...", flush=True)
    p_cond_mask = p_valid.clone()
    m_one = torch.ones(B, dtype=torch.float32)
    w_cell = torch.ones(B, N_CANVAS)
    w_cell[:, 0] = 0.0
    i_last = torch.full((B,), 80, dtype=torch.long)

    with torch.no_grad():
        vh_full, stats_full, sw = core(
            V_t, t_mat, V_prompt_a, p_valid, p_cond_mask, m_one, return_sink_weights=True
        )
        assert sw is not None and sw.shape == (B, 3, N_CANVAS)
        _, met_full = compute_phase3_loss(
            vh_full, V_0, t_mat, w_cell, i_last, stats_full, step=250, m_prompt=m_one, sink_weights=sw
        )

        vh_1, stats_1, sw_1 = core(
            V_t[:1], t_mat[:1], V_prompt_a[:1], p_valid[:1], p_cond_mask[:1], m_one[:1], return_sink_weights=True
        )
        vh_2, stats_2, sw_2 = core(
            V_t[1:], t_mat[1:], V_prompt_a[1:], p_valid[1:], p_cond_mask[1:], m_one[1:], return_sink_weights=True
        )
        stats_cat = tuple(torch.cat([a, b], dim=0) for a, b in zip(stats_1, stats_2))
        vh_cat = torch.cat([vh_1, vh_2], dim=0)
        sw_cat = torch.cat([sw_1, sw_2], dim=0)
        _, met_cat = compute_phase3_loss(
            vh_cat, V_0, t_mat, w_cell, i_last, stats_cat, step=250, m_prompt=m_one, sink_weights=sw_cat
        )

    assert abs(met_full["adaln/sigma_mean"] - met_cat["adaln/sigma_mean"]) < 1e-6, "Расхождение sigma_adaln после gather"
    assert abs(met_full["loss/adaln_raw"] - met_cat["loss/adaln_raw"]) < 1e-6, "Расхождение L_adaln после gather"
    print(f"    -> sigma_mean={met_full['adaln/sigma_mean']:.2e} (exact match)", flush=True)

    # 5. Ненулевые градиенты на 1-м шаге + сферичность EMA
    print("[4/6] Checking backward gradients + OrthMuon/AdamW step + EMA sphere projection...", flush=True)
    core.train()
    ema = EMAModel(core, decay=0.99)
    vh_tr, stats_tr, sw_tr = core(
        V_t, t_mat, V_prompt_a, p_valid, p_cond_mask, m_one, radar_sigma=radar_sig, return_sink_weights=True
    )
    loss_tr, met_tr = compute_phase3_loss(
        vh_tr, V_0, t_mat, w_cell, i_last, stats_tr, step=1, m_prompt=m_one, sink_weights=sw_tr
    )
    assert "loss/sink" in met_tr and "loss/energy" in met_tr
    loss_tr.backward()

    zero_grad_layers = [
        name for name, p in core.named_parameters()
        if p.requires_grad and (p.grad is None or p.grad.abs().max().item() == 0.0)
    ]
    assert not zero_grad_layers, f"Параметры с нулевым градиентом: {zero_grad_layers}"

    opt_muon.step()
    opt_adamw.step()
    ema.update(core)
    ema.apply_shadow(core)
    max_sph_err = max(
        float((p.float().norm(p=2, dim=-1) - 1.0).abs().max().item())
        for name, p in core.named_parameters()
        if name.endswith(".q_weight")
    )
    assert max_sph_err < 1e-6, f"Нарушена сферичность q_weight после EMA: {max_sph_err}"
    ema.restore(core)
    print(f"    -> all grads nonzero, max_sph_err={max_sph_err:.2e}", flush=True)

    # 6. Быстрый тест сэмплеров Эйлера-Хьюна, Rolling Wave и Block-AR
    print("[5/6] Checking Euler-Heun, Rolling Wave, and Block-AR CFG samplers...", flush=True)
    V_gen, _ = euler_heun_sample(
        core, V_prompt_a[:1], p_valid[:1], sep_ref, num_steps=2, cfg_scale=2.0, seed=7
    )
    assert V_gen.shape == (1, N_CANVAS, 8, 128)
    assert (V_gen[:, 0] - sep_ref.unsqueeze(0)).abs().max().item() < 1e-6

    V_roll = rolling_wave_sample(
        core, V_prompt_a[:1], p_valid[:1], sep_ref, void_ref=void_ref, num_steps=3, window=56.0, cfg_scale=2.0, seed=7
    )
    assert V_roll.shape == (1, N_CANVAS, 8, 128)
    assert (V_roll[:, 0] - sep_ref.unsqueeze(0)).abs().max().item() < 1e-6

    V_blk = block_ar_sample(
        core, V_prompt_a[:1], p_valid[:1], sep_ref, void_ref=void_ref, block_size=64, steps_per_block=2, cfg_scale=2.0, seed=7
    )
    assert V_blk.shape == (1, N_CANVAS, 8, 128)
    assert (V_blk[:, 0] - sep_ref.unsqueeze(0)).abs().max().item() < 1e-6

    V_cryst, _ = crystal_radar_sample(
        core, V_prompt_a[:1], p_valid[:1], sep_ref, decoder=None, num_steps=2, cfg_scale=2.0, seed=7
    )
    assert V_cryst.shape == (1, N_CANVAS, 8, 128)
    assert (V_cryst[:, 0] - sep_ref.unsqueeze(0)).abs().max().item() < 1e-6

    sep_exp = sep_ref.view(1, 1, 8, 128).expand(B, 1, 8, 128)
    V_t_roll, t_roll, _ = build_rollout_noisy_canvas(
        raw_model=core,
        decoder=None,
        V_0=V_0,
        V_prompt=V_prompt_a,
        p_valid=p_valid,
        p_attn_mask=p_cond_mask,
        m_prompt=m_one,
        sep_exp=sep_exp,
    )
    assert V_t_roll.shape == V_0.shape and t_roll.shape == (B, N_CANVAS)
    assert torch.all(t_roll[:, 0] == 0.0)
    print("[6/6] Done.", flush=True)

    print(
        f"ALL PHASE 3 ARCHITECTURE CHECKS PASSED "
        f"(params={total_params/1e6:.2f}M, leak_diff={leak_diff:.2e}, radar_eff={radar_effect:.2e}, max_sph_err={max_sph_err:.2e})."
    )


if __name__ == "__main__":
    main()
