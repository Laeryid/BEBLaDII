"""
Модульные и интеграционные тесты альтернативной архитектуры Phase 3 Alt Plan C:
- GrassmannianSABlock (локальное оконное и глобальное векторное внимание в R^128);
- GrassmannianCABlock (2 слоя векторного CA на 512 векторов промпта + k_sink);
- FullSABlock (интеграторы 1024d);
- PerAxisGRNv3 (3-слотовый поосевой DCA `[x_in, h_ca, x_prev]`): строгое тождество на шаге 0,
  строгая изоляция 8 осей друг от друга, ненулевые градиенты `b` и `w` на 1-м шаге.

Запуск из корня репозитория:
    .venv\\Scripts\\python.exe -B "experiments/phase 3 alt/test_phase3_alt.py"
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ALT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ALT_DIR.parents[1]
PHASE3_DIR = REPO_ROOT / "experiments" / "phase 3"
for p in (str(REPO_ROOT), str(PHASE3_DIR), str(ALT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from storage.components.decoder import NoiseRadarHead
from init_strategies import verify_init_invariants
from matrix_layers import MATRIX_VARIANTS, build_matrix_alt_core, create_matrix_optimizers
from vocab_optim import FastVocabOptimizer

from train_phase3 import (
    _apply_cfg_and_clamp,
    build_t_grid,
    clip_phase3_grads,
    compute_axis_t,
    compute_limited_cfg_scale,
    procrustes_align_basis,
)
from train_phase3_alt import (
    EMAModel,
    GrassmannianAltDiffusionCore,
    GrassmannianSABlock,
    N_CANVAS,
    N_PROMPT,
    OrthMuon,
    PerAxisGRNv3,
    SIGMA_DATA,
    SIGMA_NOISE,
    add_axis_shifted_noise,
    block_ar_sample,
    build_rollout_noisy_canvas,
    compute_phase3_loss,
    create_alt_optimizers,
    crop_batch_for_gpu,
    crystal_radar_sample,
    euler_heun_sample,
    rolling_wave_sample,
    sample_mixed_noise_schedule,
)


def test_dca_identity_and_axis_isolation() -> None:
    print("[1/7] Checking PerAxisGRNv3 identity init and strict 8-axis isolation...", flush=True)
    dca = PerAxisGRNv3(num_axes=8, axis_dim=128, has_ca=True)
    B, N = 2, 16
    x_in = torch.randn(B, N, 8, 128)
    h_ca = torch.randn(B, N, 8, 128)
    x_prev = torch.randn(B, N, 8, 128)

    # На шаге 0 (b=[0, 0, 1], w=0) обязан быть строгим тождеством out == x_prev
    out0 = dca(x_in, h_ca, x_prev)
    id_err = (out0 - x_prev).abs().max().item()
    assert id_err < 1e-6, f"PerAxisGRNv3 нарушает тождество на старте: err={id_err}"

    # С ненулевыми весами: изменение оси k=3 не должно влиять ни на одну другую ось k != 3
    with torch.no_grad():
        dca.b.normal_(mean=0.0, std=0.2)
        dca.w.normal_(mean=0.0, std=0.2)
    out_a = dca(x_in, h_ca, x_prev)
    x_in_mod = x_in.clone()
    h_ca_mod = h_ca.clone()
    x_prev_mod = x_prev.clone()
    x_in_mod[:, :, 3, :] += 5.0
    h_ca_mod[:, :, 3, :] -= 3.0
    x_prev_mod[:, :, 3, :] += 2.0
    out_b = dca(x_in_mod, h_ca_mod, x_prev_mod)
    other_axes = [0, 1, 2, 4, 5, 6, 7]
    cross_axis_leak = (out_a[:, :, other_axes, :] - out_b[:, :, other_axes, :]).abs().max().item()
    assert cross_axis_leak < 1e-6, f"Обнаружено перетекание между 8 осями в PerAxisGRNv3: {cross_axis_leak}"
    print(f"    -> id_err={id_err:.2e}, cross_axis_leak={cross_axis_leak:.2e} (OK)", flush=True)


def test_local_window_mask_no_sep_duplication() -> None:
    print("[2/7] Checking GrassmannianSABlock local window mask (no SEP duplication)...", flush=True)
    r = 6
    mask = GrassmannianSABlock._build_window_mask(N_CANVAS, 8, r, torch.device("cpu"))
    # Маска: (1, 1, N, 1, (2*r+2)*8). Для каждой клетки i проверяем число открытых ключей:
    # - i == 0: открыты клетки 0..r -> (r+1)*8 ключей (extra SEP закрыт)
    # - i == r: открыты клетки 0..2r -> (2r+1)*8 ключей (extra SEP закрыт)
    # - i == r + 1: открыты клетки 1..2r+1 + extra SEP -> (2r+2)*8 ключей
    open_cnt = (mask[0, 0, :, 0, :] == 0.0).sum(dim=-1)
    assert int(open_cnt[0].item()) == (r + 1) * 8, f"Неверное число ключей на i=0: {open_cnt[0]}"
    assert int(open_cnt[r].item()) == (2 * r + 1) * 8, f"Неверное число ключей на i=r: {open_cnt[r]}"
    assert int(open_cnt[r + 1].item()) == (2 * r + 2) * 8, f"Неверное число ключей на i=r+1: {open_cnt[r + 1]}"
    # Extra SEP слот (последние 8 позиций) обязан быть закрыт для i <= r и открыт для i > r
    sep_open = (mask[0, 0, :, 0, -8:] == 0.0).all(dim=-1)
    assert not sep_open[: r + 1].any(), "Дублирование SEP для клеток i <= r!"
    assert sep_open[r + 1 :].all(), "SEP закрыт для клеток i > r!"
    print(f"    -> window r={r}: i=0->{open_cnt[0].item()}, i=r->{open_cnt[r].item()}, i=r+1->{open_cnt[r+1].item()} vecs (OK)", flush=True)


def main() -> None:
    torch.manual_seed(42)
    test_dca_identity_and_axis_isolation()
    test_local_window_mask_no_sep_duplication()

    void_ref = torch.load(REPO_ROOT / "storage/components/void_concept.pt", map_location="cpu", weights_only=False)
    sep_ref = torch.load(REPO_ROOT / "storage/components/sep_concept.pt", map_location="cpu", weights_only=False)

    print("[3/7] Initializing GrassmannianAltDiffusionCore (standard / tiny)...", flush=True)
    for preset in ("tiny", "standard"):
        m_tmp = GrassmannianAltDiffusionCore(void_ref=void_ref, preset=preset)
        n_p = sum(p.numel() for p in m_tmp.parameters() if p.requires_grad)
        print(f"    -> preset={preset}: {n_p/1e6:.2f}M params, stack={m_tmp.stack_spec}", flush=True)

    core = GrassmannianAltDiffusionCore(void_ref=void_ref, preset="standard")
    total_params = sum(p.numel() for p in core.parameters() if p.requires_grad)
    assert total_params < 50_000_000, f"Ожидалась компактная модель (<50M), получено {total_params/1e6:.2f}M"

    opt_muon, opt_adamw = create_alt_optimizers(core)
    assert opt_muon is not None
    muon_cnt = sum(p.numel() for g in opt_muon.param_groups for p in g["params"])
    adamw_cnt = sum(p.numel() for g in opt_adamw.param_groups for p in g["params"])
    assert muon_cnt + adamw_cnt == total_params, "Не все параметры покрыты оптимизаторами!"
    # Проверяем, что DCA b/w не попали в группу с weight_decay > 0
    dca_ids = {id(p) for n, p in core.named_parameters() if ("dca_q." in n or "dca_kv." in n or "final_dca." in n)}
    for g in opt_adamw.param_groups:
        if g["weight_decay"] > 0.0:
            overlap = dca_ids.intersection({id(p) for p in g["params"]})
            assert not overlap, "Параметры DCA не должны иметь weight_decay > 0!"

    print("[4/7] Checking GrassmannianCABlock leak-free + SEP t=0 identity + no t_local leak...", flush=True)
    B = 2
    V_0 = torch.randn(B, N_CANVAS, 8, 128) * SIGMA_DATA
    V_0[:, 0] = sep_ref
    t_mat = sample_mixed_noise_schedule(B, torch.device("cpu"))
    V_t = (1.0 - t_mat.unsqueeze(-1).unsqueeze(-1)) * V_0 + t_mat.unsqueeze(-1).unsqueeze(-1) * SIGMA_DATA * torch.randn_like(V_0)
    V_t[:, 0] = sep_ref

    V_prompt_a = torch.randn(B, N_PROMPT, 8, 128) * SIGMA_DATA
    V_prompt_b = torch.randn(B, N_PROMPT, 8, 128) * SIGMA_DATA
    p_valid = torch.ones(B, N_PROMPT, dtype=torch.bool)
    p_uncond = torch.zeros(B, N_PROMPT, dtype=torch.bool)
    m_zero = torch.zeros(B, dtype=torch.float32)
    radar_sig = torch.rand(B, N_CANVAS) * 0.5
    radar_sig[:, 0] = 0.0

    core.eval()
    with torch.no_grad():
        out_a, _, sw_uncond = core(V_t, t_mat, V_prompt_a, p_valid, p_uncond, m_zero, radar_sigma=radar_sig, return_sink_weights=True)
        out_b, _, _ = core(V_t, t_mat, V_prompt_b, p_valid, p_uncond, m_zero, radar_sigma=radar_sig)
        out_no_rad, _, _ = core(V_t, t_mat, V_prompt_a, p_valid, p_uncond, m_zero, radar_sigma=torch.zeros_like(radar_sig))

    leak_diff = (out_a - out_b).abs().max().item()
    assert leak_diff < 1e-6, f"Обнаружена утечка промпта при CFG dropout: {leak_diff}"
    assert sw_uncond is not None and sw_uncond.shape == (B, 2, N_CANVAS), f"Неверная форма sink_weights: {sw_uncond.shape}"
    assert (sw_uncond - 1.0).abs().max().item() < 1e-5, "При CFG dropout весь вес CA обязан уходить в k_sink"
    assert (out_a - out_no_rad).abs().max().item() > 0.0, "Датчик s7 (radar_sigma) не влияет на выход"
    assert (out_a[:, 0] - sep_ref.unsqueeze(0)).abs().max().item() < 1e-6, "Искажение ячейки SEP"
    assert out_a.norm(p=2, dim=-1).max().item() <= 1.0 + 1e-5, "Выход за пределы единичного шара B^128"

    t_flat = torch.full((B, N_CANVAS), 0.50, dtype=torch.float32)
    t_flat[:, 0] = 0.0
    t_het = torch.cat(
        [torch.zeros(B, 1), torch.linspace(0.20, 0.80, N_CANVAS - 1).unsqueeze(0).expand(B, -1)],
        dim=1,
    )
    with torch.no_grad():
        out_flat, _, _ = core(V_t, t_flat, V_prompt_a, p_valid, p_valid, torch.ones(B), radar_sigma=radar_sig)
        out_het, _, _ = core(V_t, t_het, V_prompt_a, p_valid, p_valid, torch.ones(B), radar_sigma=radar_sig)
    t_local_leak = (out_flat - out_het).abs().max().item()
    assert t_local_leak < 1e-6, f"Утечка оракульного t_local: {t_local_leak}"
    print(f"    -> leak_diff={leak_diff:.2e}, t_local_leak={t_local_leak:.2e}, sink_uncond={sw_uncond.mean().item():.6f} (OK)", flush=True)

    print("[5/7] Checking DataParallel gather equivalence for L_adaln...", flush=True)
    m_one = torch.ones(B, dtype=torch.float32)
    w_cell = torch.ones(B, N_CANVAS)
    w_cell[:, 0] = 0.0
    i_last = torch.full((B,), 80, dtype=torch.long)

    with torch.no_grad():
        vh_full, st_full, sw_full = core(V_t, t_mat, V_prompt_a, p_valid, p_valid, m_one, return_sink_weights=True)
        _, met_full = compute_phase3_loss(vh_full, V_0, t_mat, w_cell, i_last, st_full, step=250, m_prompt=m_one, sink_weights=sw_full)

        vh_1, st_1, sw_1 = core(V_t[:1], t_mat[:1], V_prompt_a[:1], p_valid[:1], p_valid[:1], m_one[:1], return_sink_weights=True)
        vh_2, st_2, sw_2 = core(V_t[1:], t_mat[1:], V_prompt_a[1:], p_valid[1:], p_valid[1:], m_one[1:], return_sink_weights=True)
        st_cat = tuple(torch.cat([a, b], dim=0) for a, b in zip(st_1, st_2))
        _, met_cat = compute_phase3_loss(
            torch.cat([vh_1, vh_2], dim=0), V_0, t_mat, w_cell, i_last, st_cat,
            step=250, m_prompt=m_one, sink_weights=torch.cat([sw_1, sw_2], dim=0),
        )
    assert abs(met_full["adaln/sigma_mean"] - met_cat["adaln/sigma_mean"]) < 1e-6, "Расхождение sigma_adaln после gather"
    print(f"    -> sigma_mean={met_full['adaln/sigma_mean']:.2e} (exact match)", flush=True)

    print("[6/7] Checking 100% nonzero backward gradients on step 1 + OrthMuon/AdamW + EMA sphere...", flush=True)
    core.train()
    ema = EMAModel(core, decay=0.99)
    vh_tr, st_tr, sw_tr = core(V_t, t_mat, V_prompt_a, p_valid, p_valid, m_one, radar_sigma=radar_sig, return_sink_weights=True)
    loss_tr, _ = compute_phase3_loss(vh_tr, V_0, t_mat, w_cell, i_last, st_tr, step=1, m_prompt=m_one, sink_weights=sw_tr)
    loss_tr.backward()

    zero_grads = [
        name for name, p in core.named_parameters()
        if p.requires_grad and (p.grad is None or float(p.grad.abs().max().item()) == 0.0)
    ]
    assert not zero_grads, f"Найдены параметры с нулевым градиентом на 1-м шаге: {zero_grads[:10]}"

    opt_muon.step()
    opt_adamw.step()
    ema.update(core)
    ema.apply_shadow(core)
    max_sph_err = max(
        (p.float().norm(p=2, dim=-1) - 1.0).abs().max().item()
        for n, p in core.named_parameters()
        if n.endswith(".q_weight")
    )
    ema.restore(core)
    assert max_sph_err < 1e-5, f"Нарушена сферичность EMA *.q_weight: {max_sph_err}"
    dca_stat = core.get_dca_summary()
    print(f"    -> 0 zero-grad params, max_sph_err={max_sph_err:.2e}, dca={dca_stat} (OK)", flush=True)

    print("[7/7] Checking all 4 samplers + Two-Step Rollout compatibility...", flush=True)
    core.eval()
    v_eh, sw_eh = euler_heun_sample(core, V_prompt_a[:1], p_valid[:1], sep_ref, num_steps=2, cfg_scale=2.0, seed=7)
    v_cr, _ = crystal_radar_sample(core, V_prompt_a[:1], p_valid[:1], sep_ref, num_steps=2, cfg_scale=2.0, seed=7)
    v_rw = rolling_wave_sample(core, V_prompt_a[:1], p_valid[:1], sep_ref, void_ref=void_ref, num_steps=3, cfg_scale=2.0, seed=7)
    v_ba = block_ar_sample(core, V_prompt_a[:1], p_valid[:1], sep_ref, void_ref=void_ref, block_size=64, steps_per_block=2, cfg_scale=2.0, seed=7)
    v_ro, t_ro, _ = build_rollout_noisy_canvas(core, None, V_0[:1], V_prompt_a[:1], p_valid[:1], p_valid[:1], m_one[:1], sep_ref.view(1, 1, 8, 128))
    for name, ten in [("euler_heun", v_eh), ("crystal", v_cr), ("rolling", v_rw), ("block_ar", v_ba), ("rollout", v_ro)]:
        assert ten.shape == (1, N_CANVAS, 8, 128), f"Неверная форма {name}: {ten.shape}"
        assert torch.isfinite(ten).all(), f"NaN/Inf в {name}"
    assert sw_eh is not None and sw_eh.shape == (1, 2, N_CANVAS)
    assert t_ro.shape == (1, N_CANVAS) and float(t_ro[0, 0].item()) == 0.0

    print("[8/9] Checking FastVocabOptimizer BF16 NoiseRadarHead accuracy & attach/detach...", flush=True)
    torch.manual_seed(123)
    V_test, D_test = 16384, 128
    W_test = torch.randn(V_test, D_test) * 0.1
    W_inv_n = 1.0 / W_test.norm(p=2, dim=-1).clamp(min=1e-6)
    h_unp = torch.randn(2, 32, D_test)
    # Подмешаем реальные векторы словаря в половину слотов (имитация низкого шума)
    h_unp[0, :16] = W_test[:16] * 1.5 + torch.randn(16, D_test) * 0.02

    radar_mod = NoiseRadarHead(hidden_size=D_test)
    ref_stats = radar_mod.compute_dict_stats(h_unp, W_test, W_inv_n, vocab_chunk_size=4096)
    fv_opt = FastVocabOptimizer(
        lm_head_weight=W_test,
        gemm_dtype=torch.bfloat16,
        vocab_chunk_size=4096,
        svd_rank=32,
        ce_pool_hard=512,
        ce_pool_rand=1024,
        ce_batch_fraction=0.5,
    )
    dummy_dec = nn.Module()
    dummy_dec.noise_radar = radar_mod
    dummy_dec.lm_head_weight = W_test
    dummy_dec.chunk_size = 4
    fv_opt.attach_to_decoder(dummy_dec)
    fast_stats = radar_mod.compute_dict_stats(h_unp, W_test, W_inv_n, vocab_chunk_size=4096)
    fv_opt.detach_from_decoder()
    restored_stats = radar_mod.compute_dict_stats(h_unp, W_test, W_inv_n, vocab_chunk_size=4096)

    max_radar_err = (fast_stats - ref_stats).abs().max().item()
    assert max_radar_err < 1.5e-2, f"Слишком большая ошибка BF16 NoiseRadar: {max_radar_err}"
    assert (restored_stats - ref_stats).abs().max().item() < 1e-6, "detach_from_decoder не восстановил оригинал"
    print(f"    -> max_radar_err={max_radar_err:.2e}, attach/detach OK", flush=True)

    print("[9/10] Checking Online Chunked Exact CE (full-rank sharp) + Sampled CE + static batch filtering...", flush=True)
    # 9a. Проверка дефолтного exact_chunked CE на ПОЛНОРАНГОВОЙ матрице в остром режиме
    torch.manual_seed(77)
    W_full = torch.randn(V_test, D_test) * 0.25
    fv_exact = FastVocabOptimizer(
        lm_head_weight=W_full,
        gemm_dtype=torch.bfloat16,
        ce_mode="exact_chunked",
        ce_vocab_chunk_size=4096,
        ce_batch_fraction=0.5,
    )
    assert fv_exact.w_svd_vocab is None, "В exact_chunked SVD-эскиз не должен строиться при старте"
    M_test = 48
    tgt_ids = torch.randint(0, V_test, (M_test,))
    tgt_ids[0] = tgt_ids[1] = tgt_ids[2] = 42  # явные дубликаты целевых токенов
    # Делаем острое распределение (p_tgt близко к 1), где sampled rank=64 проваливался
    h_sharp = W_full[tgt_ids] * 1.4 + torch.randn(M_test, D_test) * 0.20
    valid_m = torch.ones(M_test, dtype=torch.bool)
    valid_m[-4:] = False

    h_ref = h_sharp.clone().detach().requires_grad_(True)
    logits_ref = F.linear(h_ref, W_full)
    ce_ref_vec = F.cross_entropy(logits_ref, tgt_ids, reduction="none")
    corr_ref = logits_ref.detach().argmax(dim=-1) == tgt_ids
    loss_ref = ce_ref_vec[valid_m].mean()
    loss_ref.backward()

    h_chk = h_sharp.clone().detach().requires_grad_(True)
    ce_chk_vec, corr_chk = fv_exact.compute_exact_chunked_ce_from_hidden(h_chk, tgt_ids, valid_m, W_full)
    loss_chk = ce_chk_vec[valid_m].mean()
    loss_chk.backward()

    exact_ce_err = (ce_chk_vec[valid_m] - ce_ref_vec[valid_m]).abs().max().item()
    exact_gcos = F.cosine_similarity(
        h_ref.grad[valid_m].reshape(-1), h_chk.grad[valid_m].reshape(-1), dim=0
    ).item()
    exact_gerr = (h_chk.grad[valid_m] - h_ref.grad[valid_m]).abs().max().item()
    assert exact_ce_err < 1e-5, f"Ошибка exact_chunked CE: {exact_ce_err}"
    assert exact_gcos > 0.999999 and exact_gerr < 1e-5, f"Ошибка градиента exact_chunked CE: cos={exact_gcos}, err={exact_gerr}"
    assert torch.equal(corr_chk[valid_m], corr_ref[valid_m]), "Расхождение exact_chunked argmax"

    # 9b. Регрессионная проверка опционального sampled CE (на низкоранговой матрице)
    W_lr = torch.randn(V_test, 32) @ torch.randn(32, D_test) * 0.15 + torch.randn(V_test, D_test) * 0.05
    fv_ce = FastVocabOptimizer(
        lm_head_weight=W_lr,
        gemm_dtype=torch.bfloat16,
        ce_mode="sampled",
        svd_rank=32,
        ce_pool_hard=512,
        ce_pool_rand=2048,
        ce_batch_fraction=0.5,
    )
    h_base = W_lr[tgt_ids] * 0.8 + torch.randn(M_test, D_test) * 0.25

    h_exact = h_base.clone().detach().requires_grad_(True)
    ce_exact_vec = F.cross_entropy(F.linear(h_exact, W_lr), tgt_ids, reduction="none")
    loss_exact = ce_exact_vec[valid_m].mean()
    loss_exact.backward()

    h_fast = h_base.clone().detach().requires_grad_(True)
    ce_fast_vec, corr_fast = fv_ce.compute_sampled_ce_from_hidden(h_fast, tgt_ids, valid_m, W_lr)
    loss_fast = ce_fast_vec[valid_m].mean()
    loss_fast.backward()

    grad_cos = F.cosine_similarity(
        h_exact.grad[valid_m].reshape(-1), h_fast.grad[valid_m].reshape(-1), dim=0
    ).item()
    ce_rel_err = abs(loss_fast.item() - loss_exact.item()) / max(loss_exact.item(), 1e-6)
    assert torch.isfinite(loss_fast) and torch.isfinite(h_fast.grad).all(), "NaN/Inf в fast CE"
    assert grad_cos > 0.985, f"Низкая косинусная близость градиента CE: {grad_cos:.4f}"
    assert ce_rel_err < 0.05, f"Слишком большое расхождение CE: {ce_rel_err:.4f}"

    # Краевой случай: 0 валидных токенов -> loss=0, grad=0, без NaN (оба режима)
    h_zero = h_base.clone().detach().requires_grad_(True)
    ce_z1, _ = fv_exact.compute_exact_chunked_ce_from_hidden(h_zero, tgt_ids, torch.zeros_like(valid_m), W_full)
    ce_z2, _ = fv_ce.compute_sampled_ce_from_hidden(h_zero, tgt_ids, torch.zeros_like(valid_m), W_lr)
    assert torch.isfinite(ce_z1).all() and torch.isfinite(ce_z2).all(), "NaN при пустой маске valid_m"

    # Проверка статической фильтрации примеров батча до decode_hidden (B=4 -> B_ce=2) с дефолтным exact_chunked
    seen_batch_sizes = []

    class _MiniDec(nn.Module):
        def __init__(self, w_mat: torch.Tensor):
            super().__init__()
            self.chunk_size = 4
            self.proj = nn.Linear(8 * 128, D_test)
            self.register_buffer("lm_head_weight", w_mat, persistent=False)

        def decode_hidden(self, v_in: torch.Tensor) -> torch.Tensor:
            seen_batch_sizes.append(int(v_in.shape[0]))
            b_, n_, _, _ = v_in.shape
            h_ = self.proj(v_in.reshape(b_, n_, -1))
            return h_.unsqueeze(2).expand(b_, n_, 4, D_test).reshape(b_, n_ * 4, D_test)

    mini_dec = _MiniDec(W_full)
    v0_hat_b = torch.randn(4, 17, 8, 128, requires_grad=True)
    t_mat_b = torch.tensor([[0.0] + [0.8] * 16, [0.0] + [0.2] * 16, [0.0] + [0.9] * 16, [0.0] + [0.3] * 16])
    ans_tgt_b = torch.randint(0, V_test, (4, 16 * 4))
    l_ce_b, acc_b = fv_exact.compute_decoder_ce_loss(
        decoder=mini_dec, V_0_hat=v0_hat_b, t_mat=t_mat_b, ans_targets=ans_tgt_b, ce_max_tokens=32, ce_batch_fraction=0.5
    )
    l_ce_b.backward()
    assert seen_batch_sizes == [2, 2], f"Ожидался B_ce=2 (fwd+ckpt_bwd), получено: {seen_batch_sizes}"
    assert torch.isfinite(v0_hat_b.grad).all() and float(v0_hat_b.grad.abs().max().item()) > 0.0
    print(
        f"    -> exact_chunked(full-rank): ce={loss_ref.item():.4f}, max_err={exact_ce_err:.2e}, grad_cos={exact_gcos:.6f} | "
        f"sampled(low-rank): rel={ce_rel_err:.2%}, grad_cos={grad_cos:.4f} | B_ce={seen_batch_sizes[0]}/4 (OK)",
        flush=True,
    )

    print("[10/12] Checking all 7 matrix_layers variants (PHM, GrassAlign, L*X*R) on v2: invariants, grads, EMA...", flush=True)
    for m_var in MATRIX_VARIANTS:
        m_core = build_matrix_alt_core(void_ref=void_ref, variant=m_var, preset="tiny", phm_rank=4, arch_version="v2")
        n_p = sum(p.numel() for p in m_core.parameters() if p.requires_grad)
        inv_m = verify_init_invariants(m_core, sep_concept=sep_ref, device=torch.device("cpu"))
        m_core.train()
        om_m, oa_m = create_matrix_optimizers(m_core)
        ema_m = EMAModel(m_core, decay=0.99, warmup=True)
        vh_m, st_m, sw_m = m_core(V_t[:1], t_mat[:1], V_prompt_a[:1], p_valid[:1], p_valid[:1], m_one[:1], return_sink_weights=True)
        l_m, _ = compute_phase3_loss(
            vh_m, V_0[:1], t_mat[:1], w_cell[:1], i_last[:1], st_m,
            step=1, m_prompt=m_one[:1], sink_weights=sw_m,
            anchor_weight=0.15, axis_shift_delta=0.35, basis_high_t_mode="sign_only",
        )
        l_m.backward()
        z_m = [
            name for name, p in m_core.named_parameters()
            if p.requires_grad and (p.grad is None or float(p.grad.abs().max().item()) == 0.0)
        ]
        assert not z_m, f"Нулевые градиенты в {m_var}: {z_m[:8]}"
        clip_phase3_grads(om_m, oa_m, adamw_max_norm=1.0, sphere_max_norm=10.0)
        if om_m is not None:
            om_m.step()
        oa_m.step()
        ema_m.update(m_core)
        print(f"    -> {m_var:18s}: {n_p/1e6:5.2f}M params | sph_err={inv_m['max_sph_err']:.2e} | cfg_leak={inv_m['cfg_leak_diff']:.2e} | 0 zero-grad (OK)", flush=True)

    print("[11/12] Checking Stage 0 (OrthMuon sphere scale, EMA warmup/resume, spec/orth/anchor/stopword loss) & Stage 1 (Procrustes CFG, t_grid, KV-cache + 2B CFG equivalence)...", flush=True)
    # 11a. OrthMuon sphere_scale выравнивает угловой шаг между (128, 128) и (1024, 2048)
    w_small = nn.Parameter(F.normalize(torch.randn(128, 128), p=2, dim=-1))
    w_big = nn.Parameter(F.normalize(torch.randn(1024, 2048), p=2, dim=-1))
    w_small_0, w_big_0 = w_small.data.clone(), w_big.data.clone()
    w_small.grad = torch.randn_like(w_small)
    w_big.grad = torch.randn_like(w_big)
    om_test = OrthMuon([w_small, w_big], lr=1e-3, sphere=True, sphere_scale=True, sphere_c_base=2.26)
    om_test.step()
    ang_small = (w_small.data - w_small_0).norm(p=2, dim=-1).mean().item()
    ang_big = (w_big.data - w_big_0).norm(p=2, dim=-1).mean().item()
    ratio_ang = max(ang_small, ang_big) / max(min(ang_small, ang_big), 1e-8)
    assert ratio_ang < 1.35, f"Угловые шаги OrthMuon не выровнены: small={ang_small:.5f}, big={ang_big:.5f}"

    # 11b. EMA warmup + state_dict resume + обратная совместимость со старыми чекпоинтами
    ema_w = EMAModel(core, decay=0.999, warmup=True)
    for _ in range(10):
        ema_w.update(core)
    assert ema_w.num_updates == 10 and abs(ema_w.current_decay() - 11.0 / 20.0) < 1e-6
    sd_ema = ema_w.state_dict()
    ema_w2 = EMAModel(core, decay=0.999, warmup=True)
    ema_w2.load_state_dict(sd_ema)
    assert ema_w2.num_updates == 10
    sd_legacy = {k: v for k, v in sd_ema.items() if k != "__num_updates__"}
    ema_w3 = EMAModel(core, decay=0.999, warmup=True)
    ema_w3.load_state_dict(sd_legacy)
    assert ema_w3.num_updates == 0

    # 11c. Procrustes CFG точно отменяет случайный поворот R in O(8)
    q_rand, _ = torch.linalg.qr(torch.randn(8, 8))
    vc_test = torch.randn(2, 16, 8, 128) * 0.05
    vu_rot = torch.matmul(q_rand, vc_test)
    vu_al = procrustes_align_basis(vu_rot, vc_test)
    proc_err = (vu_al - vc_test).abs().max().item()
    assert proc_err < 1e-5, f"Ошибка Procrustes-выравнивания: {proc_err}"
    t_sd3 = build_t_grid(20, torch.device("cpu"), mode="sd3", shift=0.55)
    assert abs(float(t_sd3[0].item()) - 1.0) < 1e-6 and abs(float(t_sd3[-1].item())) < 1e-6
    assert float(t_sd3[10].item()) < 0.50, "SD3 shift=0.55 должен сгущать шаги к t->0"
    assert compute_limited_cfg_scale(0.10, 3.0) == 1.0 and compute_limited_cfg_scale(0.50, 3.0) == 3.0

    # 11d. Эквивалентность Stage 1 (prepare_prompt_cache + forward_cfg_batched) и эталонного раздельного forward
    core.eval()
    p_valid_short = torch.zeros(B, N_PROMPT, dtype=torch.bool)
    p_valid_short[0, :12] = True
    p_valid_short[1, :7] = True
    with torch.no_grad():
        vc_ref, _, sw_ref = core(V_t, t_mat, V_prompt_a, p_valid_short, p_valid_short, m_one, radar_sigma=radar_sig, return_sink_weights=True)
        vu_ref, _, _ = core(V_t, t_mat, V_prompt_a, p_valid_short, p_uncond, m_zero, radar_sigma=radar_sig, return_sink_weights=False)
        p_cache = core.prepare_prompt_cache(V_prompt_a, p_valid_short)
        vc_fast, vu_fast, sw_fast = core.forward_cfg_batched(
            V_t=V_t, t_mat=t_mat, x_prompt=p_cache["x_prompt"],
            prompt_attn_mask=p_valid_short, prompt_cache=p_cache,
            radar_sigma=radar_sig, return_sink_weights=True,
        )
    diff_c = (vc_fast - vc_ref).abs().max().item()
    diff_u = (vu_fast - vu_ref).abs().max().item()
    diff_sw = (sw_fast - sw_ref).abs().max().item()
    assert diff_c < 1e-5 and diff_u < 1e-5 and diff_sw < 1e-5, f"Расхождение 2B+KV-cache: c={diff_c}, u={diff_u}, sw={diff_sw}"
    print(f"    -> OrthMuon ang_ratio={ratio_ang:.2f}, procrustes_err={proc_err:.2e}, 2B+KV_diff=(c:{diff_c:.2e}, u:{diff_u:.2e}, sw:{diff_sw:.2e}) (OK)", flush=True)

    print("[12/12] Checking Stage 2 v2 (right-aligned RoPE crop invariance, Dynamic Crop, v2 diagnostics, init_strategies, v1 regression)...", flush=True)
    # 12a. Правое выравнивание промпта инвариантно к Dynamic Crop длины промпта (N_p=64 vs N_p=16)
    with torch.no_grad():
        xp_64 = core.encode_prompt(V_prompt_a, prompt_valid_mask=p_valid_short)
        xp_16 = core.encode_prompt(V_prompt_a[:, :16], prompt_valid_mask=p_valid_short[:, :16])
    crop_p_err = (xp_64[0, :12] - xp_16[0, :12]).abs().max().item()
    assert crop_p_err < 1e-5, f"Правое выравнивание промпта нарушено при Dynamic Crop: {crop_p_err}"

    # 12b. GPU Dynamic Crop (full_prob=0.0) корректно обрезает промпт и холст и проходит forward+loss+backward
    is_cw = torch.ones(B, N_CANVAS, dtype=torch.bool)
    is_cw[:, 0] = False
    is_cw[:, 2::3] = False  # имитация стоп-слов
    i_last_short = torch.tensor([18, 22], dtype=torch.long)
    ans_tgt_full = torch.randint(0, 1000, (B, (N_CANVAS - 1) * 4))
    vp_c, v0_c, pv_c, wc_c, at_c, cw_c = crop_batch_for_gpu(
        V_prompt_a, V_0, p_valid_short, i_last_short, w_cell, ans_tgt_full, is_cw, full_prob=0.0
    )
    assert vp_c.shape[1] == 16 and v0_c.shape[1] == 32, f"Неверные формы Dynamic Crop: vp={vp_c.shape}, v0={v0_c.shape}"
    t_crop = sample_mixed_noise_schedule(B, torch.device("cpu"), i_last=i_last_short, seq_len=v0_c.shape[1])
    vt_crop = add_axis_shifted_noise(v0_c, t_crop, torch.randn_like(v0_c), delta=0.35, sigma_noise=SIGMA_NOISE)
    vt_crop[:, 0] = v0_c[:, 0]
    core.set_diagnostics_capture(True)
    vh_c, st_c, sw_c = core(vt_crop, t_crop, vp_c, pv_c, pv_c, m_one, return_sink_weights=True)
    core.set_diagnostics_capture(False)
    diag_v2 = core.collect_v2_diagnostics()
    assert "val/adaln_r_eff_mean" in diag_v2 and "val/sensor_imp_offdiag" in diag_v2
    l_crop, m_crop = compute_phase3_loss(
        vh_c, v0_c, t_crop, wc_c, i_last_short, st_c, step=5,
        m_prompt=m_one, sink_weights=sw_c, is_content_word_cell=cw_c,
        anchor_weight=0.15, axis_shift_delta=0.35, basis_high_t_mode="sign_only",
    )
    assert torch.isfinite(l_crop) and "loss/spec_diag" in m_crop and "loss/orth_offdiag" in m_crop

    # 12c. Проверка геометрической инициализации all_geo на v2 + регрессия v1
    from init_strategies import apply_init_strategy
    core_init = build_matrix_alt_core(void_ref=void_ref, variant="all_matrix_safe", preset="tiny", arch_version="v2")
    apply_init_strategy(core_init, "all_geo", void_ref=void_ref, sep_concept=sep_ref)
    inv_init = verify_init_invariants(core_init, sep_concept=sep_ref, device=torch.device("cpu"))
    core_v1 = GrassmannianAltDiffusionCore(void_ref=void_ref, preset="tiny", arch_version="v1")
    inv_v1 = verify_init_invariants(core_v1, sep_concept=sep_ref, device=torch.device("cpu"))
    print(
        f"    -> crop_prompt_err={crop_p_err:.2e}, crop_shapes=({vp_c.shape[1]},{v0_c.shape[1]}), "
        f"r_eff={diag_v2['val/adaln_r_eff_mean']:.1f}, all_geo_v2_sph={inv_init['max_sph_err']:.2e}, v1_leak={inv_v1['cfg_leak_diff']:.2e} (OK)",
        flush=True,
    )

    print("=== ALL 12 PHASE 3 ALT (STAGE 0 + 1 + 2 v2) + VOCAB + MATRIX TESTS PASSED ===", flush=True)


if __name__ == "__main__":
    main()
