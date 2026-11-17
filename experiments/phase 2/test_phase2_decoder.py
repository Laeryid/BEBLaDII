"""Модульные и интеграционные тесты для Phase 2 Plan C (Декодер-Генератор, Радар Шума и Умный Чанкер)."""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from storage.components.plan_c_layers import WordAwareElasticChunker, canonicalize_cell_basis
from storage.components.decoder import GrassmannianDecoder
from storage.components.encoder import GrassmannianEncoder
from storage.components.tokenizer import get_tokenizer

import importlib.util

_TRAIN_PATH = REPO_ROOT / "experiments" / "phase 2" / "train_phase2.py"
_SPEC = importlib.util.spec_from_file_location("train_phase2_decoder", str(_TRAIN_PATH))
train_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(train_mod)


def test_o8_canonicalization_invariance() -> None:
    """Проверяет математическую O(8)-инвариантность canonicalize_cell_basis и GrassmannianDecoder."""
    torch.manual_seed(42)
    B, N, K, D = 2, 6, 8, 128
    # Формируем клетки с невырожденным убывающим спектром (как из пулера)
    Q_rand, _ = torch.linalg.qr(torch.randn(B, N, D, D))
    U_base = Q_rand[..., :K, :]  # (B, N, 8, 128)
    spectrum = torch.linspace(0.6, 0.1, K).view(1, 1, K, 1)
    V = U_base * spectrum

    R = train_mod.random_orthogonal_matrices(B, N, K, device=V.device)
    V_rot = torch.matmul(R, V)

    V_canon, eigs_1 = canonicalize_cell_basis(V)
    V_rot_canon, eigs_2 = canonicalize_cell_basis(V_rot)

    max_v_diff = (V_canon - V_rot_canon).abs().max().item()
    max_eig_diff = (eigs_1 - eigs_2).abs().max().item()
    sim = train_mod.grassmann_cell_similarity(V_canon, V_rot_canon).min().item()
    assert max_v_diff < 1e-4, f"Canonicalized V mismatch under O(8): {max_v_diff}"
    assert max_eig_diff < 1e-4, f"Eigenvalues mismatch under O(8): {max_eig_diff}"
    assert sim > 0.9999, f"Sim_Gr should be 1.0, got {sim}"

    # Проверка всего декодера целиком с O(8)-канонизацией
    dec = GrassmannianDecoder(
        vocab_size=256,
        hidden_size=1024,
        intermediate_size=2048,
        num_smooth_layers=3,
        canonicalize_basis=True,
        enable_noise_radar=True,
    ).eval()
    emb_w = torch.randn(256, 1024)
    dec.tie_embeddings(emb_w)

    with torch.no_grad():
        logits_1, radar_1 = dec(V, return_radar=True)
        logits_2, radar_2 = dec(V_rot, return_radar=True)
        sigma_cell_est, sigma_tok_est, _ = dec.estimate_noise(V_rot)

    logits_max_diff = (logits_1 - logits_2).abs().max().item()
    logits_mean_diff = (logits_1 - logits_2).abs().mean().item()
    argmax_match = (logits_1.argmax(dim=-1) == logits_2.argmax(dim=-1)).float().mean().item()
    assert logits_max_diff < 1e-2 and logits_mean_diff < 1e-3, (
        f"Decoder logits are not O(8)-invariant: max={logits_max_diff}, mean={logits_mean_diff}"
    )
    assert argmax_match == 1.0, f"Argmax mismatch under O(8): {argmax_match}"
    assert (radar_1["sigma_cell_hat"] - radar_2["sigma_cell_hat"]).abs().max().item() < 1e-4
    assert (sigma_cell_est - radar_2["sigma_cell_hat"]).abs().max().item() < 1e-6
    assert (sigma_tok_est - radar_2["sigma_tok_hat"]).abs().max().item() < 1e-6
    print(f"[PASS] O(8) invariance: max_V_diff={max_v_diff:.2e}, Sim_Gr={sim:.6f}, logit_mean_diff={logits_mean_diff:.2e}")


def test_word_aware_elastic_chunker() -> None:
    """Проверяет отсутствие разрыва слов <= 4 токенов, двухуровневые веса и обратимость сборки текста."""
    tok = get_tokenizer()
    chunker = WordAwareElasticChunker(tokenizer=tok, vocab_size=131072, chunk_size=4, pad_token_id=10)

    text = "Привет, мираж! Implement mathematical programming cleanly сегодня."
    token_ids = tok.encode(text, add_special_tokens=False)
    words = chunker.split_into_words(token_ids)

    # Проверяем, что ни одно слово длины <= 4 не разорвано по разным клеткам
    packed = chunker.chunk_tokens(token_ids, max_cells=16, elastic_drop_prob=0.0)
    cells = packed["input_ids"].view(16, 4).tolist()
    n_content = int(packed["n_content_cells"].item())

    # Восстановление всех токенов без PAD должно полностью совпадать с исходным
    flat_non_pad = [t for c in cells for t in c if t != 10]
    assert flat_non_pad == token_ids, "Chunker lost or altered tokens!"

    # Обратимая сборка строки через detokenize_non_pad
    decoded = WordAwareElasticChunker.detokenize_non_pad(packed["input_ids"], tok, pad_token_id=10)
    assert decoded == text, f"Roundtrip mismatch: {decoded!r} != {text!r}"

    # Проверка, что границы слов <= 4 соблюдены (последовательности внутри клеток совпадают со словами)
    pos_map = []
    for c_idx, cell in enumerate(cells[:n_content]):
        for t in cell:
            if t != 10:
                pos_map.append((c_idx, t))
    cursor = 0
    for w in words:
        w_cells = {pos_map[cursor + j][0] for j in range(len(w))}
        if len(w) <= 4:
            assert len(w_cells) == 1, f"Word {w} (len={len(w)}) was split across cells {w_cells}!"
        cursor += len(w)

    # Проверка двухуровневых таргетов: в смешанной клетке пустые слоты == 10, в VOID == -100
    tgt = packed["targets"].view(16, 4)
    w_mat = packed["weights"].view(16, 4)
    assert torch.allclose(w_mat.sum(dim=-1), torch.ones(16), atol=1e-5)
    if n_content < 16:
        assert (tgt[n_content:] == -100).all(), "Tail VOID cells must have target -100"
        assert torch.allclose(w_mat[n_content], torch.full((4,), 0.25), atol=1e-5)
    print(f"[PASS] WordAwareElasticChunker: n_content_cells={n_content}, words={len(words)}, roundtrip OK")


def test_decoder_radar_loss_and_grad_flow() -> None:
    """Проверяет прямой и обратный проходы декодера, радара шума и заморозки энкодера."""
    torch.manual_seed(7)
    vocab_sz = 512
    enc = GrassmannianEncoder(vocab_size=vocab_sz, num_axes=8, axis_dim=128).eval()
    for p in enc.parameters():
        p.requires_grad_(False)

    dec = GrassmannianDecoder(
        vocab_size=vocab_sz,
        num_smooth_layers=3,
        canonicalize_basis=True,
        enable_noise_radar=True,
    ).train()
    dec.tie_embeddings(enc.embeddings.weight)

    chunker = WordAwareElasticChunker(tokenizer=None, vocab_size=vocab_sz, chunk_size=4, pad_token_id=10)
    samples = [chunker.chunk_tokens([20, 21, 22, 23, 24, 25], max_cells=4) for _ in range(2)]
    input_ids = torch.stack([s["input_ids"] for s in samples])
    targets = torch.stack([s["targets"] for s in samples])
    weights = torch.stack([s["weights"] for s in samples])
    content_mask = torch.stack([s["content_cell_mask"] for s in samples])

    with torch.no_grad():
        V_0 = enc(input_ids, weights)
    V_tilde, true_sigma, mode = train_mod.corrupt_latent_cells(V_0, content_mask, mode_override="mix_other")
    logits, radar = dec(V_tilde, return_radar=True)
    loss, metrics = train_mod.compute_phase2_decoder_loss(
        logits=logits, radar=radar, targets=targets, true_sigma_cell=true_sigma, pad_token_id=10
    )
    assert torch.isfinite(loss), f"Non-finite loss: {loss}"
    loss.backward()

    radar_grads = [p.grad for p in dec.noise_radar.parameters() if p.grad is not None]
    dec_grads = [p.grad for p in dec.dec_smooth_layers.parameters() if p.grad is not None]
    assert len(radar_grads) > 0 and all(torch.isfinite(g).all() for g in radar_grads)
    assert len(dec_grads) > 0 and all(torch.isfinite(g).all() for g in dec_grads)
    assert all(p.grad is None for p in enc.parameters()), "Frozen encoder received gradients!"
    print(f"[PASS] Phase 2 Decoder+Radar grad flow ({mode}): loss={metrics['loss/total']:.4f}")


def test_phase2_dataset_source_and_sentence_truncation() -> None:
    """Проверяет, что Phase 2 использует только датасет Фазы 1, num_cells=128 и обрезает текст до конца предложения."""
    args = train_mod.parse_args()
    assert args.num_cells == 128, f"Expected default num_cells=128, got {args.num_cells}"
    assert args.batch_size == 16, f"Expected default batch_size=16, got {args.batch_size}"
    assert train_mod.TRAINING_CONFIG["num_cells"] == 128

    env = train_mod.EnvironmentConfig.detect(override_device="cpu")
    for p in env.train_data_paths + env.val_data_paths:
        assert "phase2" not in p.lower(), f"Phase 2 dataset must not include Q/A phase2 parquet: {p}"

    tok = get_tokenizer()
    chunker = WordAwareElasticChunker(tokenizer=tok, vocab_size=131072, chunk_size=4, pad_token_id=10)
    ds = train_mod.Phase2DecoderDataset(
        parquet_paths=[],
        chunker=chunker,
        tokenizer=tok,
        num_cells=16,
        regime_b_ratio=0.0,
        elastic_drop_prob=0.5,
        is_validation=True,
    )
    # Длинный текст из нескольких предложений, не умещающийся целиком в 16 клеток (64 слота)
    long_text = (
        "Первое короткое предложение заканчивается точкой. "
        "Второе предложение тоже полностью закончено! "
        "Третье длинное предложение специально содержит много разных слов про математику геометрию алгебру топологию диффузию латентное пространство и ортогональные матрицы чтобы точно выйти за предел шестидесяти четырех токенов без точки"
    )
    ds.texts = [long_text]
    ds.has_real_data = True
    sample = ds[0]

    decoded = WordAwareElasticChunker.detokenize_non_pad(sample["input_ids"], tok, pad_token_id=10)
    assert decoded.endswith((".", "!")), f"Expected sentence-end truncation, got: {decoded!r}"
    assert "Третье предложение" not in decoded, f"Overflowed tail should have been trimmed: {decoded!r}"
    w_mat = sample["weights"].view(16, 4)
    assert torch.allclose(w_mat.sum(dim=-1), torch.ones(16), atol=1e-5)
    print(f"[PASS] Phase2 dataset source & sentence truncation: {decoded!r}")


def test_chunked_loss_and_vectorized_corruption() -> None:
    """Проверяет эквивалентность чанкованного лосса, векторную аугментацию и in-graph защиту OrthMuon/EMA."""
    torch.manual_seed(123)
    vocab_sz = 512
    B, N, cs = 4, 8, 4
    L = N * cs
    dec = train_mod.GrassmannianDecoder(
        vocab_size=vocab_sz,
        num_smooth_layers=3,
        canonicalize_basis=True,
        enable_noise_radar=True,
    ).eval()
    emb_w = torch.randn(vocab_sz, 1024)
    dec.tie_embeddings(emb_w)

    V_0 = torch.randn(B, N, 8, 128) * 0.1
    content_mask = torch.ones(B, N, dtype=torch.bool)
    V_tilde, true_sigma, mode = train_mod.corrupt_latent_cells(V_0, content_mask, mode_override=None)
    assert mode == "mixed_batch", f"Expected vectorized mixed_batch, got {mode}"
    assert V_tilde.shape == V_0.shape and torch.isfinite(V_tilde).all()

    targets = torch.randint(11, vocab_sz, (B, L))
    targets[:, -4:] = -100
    targets[:, 3] = 10

    with torch.no_grad():
        hidden, radar = dec.decode_hidden(V_tilde, return_radar=True)
        logits_full = dec.get_tied_head_logits(hidden)
        loss_full, m_full = train_mod.compute_phase2_decoder_loss(
            logits=logits_full, radar=radar, targets=targets, true_sigma_cell=true_sigma, pad_token_id=10
        )
        loss_chunked, m_chunked, pred_chunked = train_mod.compute_phase2_decoder_loss_from_hidden(
            hidden_states=hidden,
            decoder=dec,
            radar=radar,
            targets=targets,
            true_sigma_cell=true_sigma,
            pad_token_id=10,
            seq_chunks=4,
        )
    assert torch.allclose(loss_full, loss_chunked, atol=1e-5), f"Loss mismatch: {loss_full} vs {loss_chunked}"
    assert abs(m_full["acc/words"] - m_chunked["acc/words"]) < 1e-6
    assert torch.equal(pred_chunked, torch.argmax(logits_full.float(), dim=-1))

    # Проверка in-graph изоляции шага при is_safe=False
    ema = train_mod.EMAModel(dec, decay=0.99)
    opt_muon, _ = train_mod.create_decoder_optimizers(dec)
    p_ref = next(iter(opt_muon.param_groups[0]["params"]))
    p_before = p_ref.detach().clone()
    s_before = ema.shadow[next(iter(ema.shadow))].detach().clone()
    p_ref.grad = torch.ones_like(p_ref)
    is_unsafe = torch.tensor(False)
    opt_muon.step(is_safe=is_unsafe)
    ema.update(dec, is_safe=is_unsafe)
    assert torch.equal(p_ref.detach(), p_before), "OrthMuon modified weights when is_safe=False!"
    assert torch.equal(ema.shadow[next(iter(ema.shadow))], s_before), "EMA modified shadow when is_safe=False!"
    print(f"[PASS] Chunked loss parity + vectorized corruption + is_safe guard OK (loss={float(loss_chunked):.4f})")


def test_autocast_bfloat16_sdpa_dtype_consistency() -> None:
    """Проверяет согласованность dtype (q, k, v) в ModernBERTLayer SDPA под autocast bfloat16."""
    import torch.nn.functional as F

    vocab_sz = 128
    dec = GrassmannianDecoder(
        vocab_size=vocab_sz,
        num_smooth_layers=3,
        canonicalize_basis=True,
        enable_noise_radar=True,
    )
    dec.tie_embeddings(torch.randn(vocab_sz, 1024, dtype=torch.bfloat16))
    V_in = torch.randn(2, 8, 8, 128, dtype=torch.float32) * 0.1

    seen_dtypes = []
    orig_sdpa = F.scaled_dot_product_attention

    def checked_sdpa(q, k, v, *args, **kwargs):
        seen_dtypes.append((q.dtype, k.dtype, v.dtype))
        assert q.dtype == k.dtype == v.dtype, f"SDPA dtype mismatch: q={q.dtype}, k={k.dtype}, v={v.dtype}"
        return orig_sdpa(q, k, v, *args, **kwargs)

    F.scaled_dot_product_attention = checked_sdpa
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            hidden, radar = dec.decode_hidden(V_in, return_radar=True)
        assert hidden.shape == (2, 32, 1024)
        assert len(seen_dtypes) == 4, f"Expected 4 SDPA calls (1 context + 3 smooth), got {len(seen_dtypes)}"
    finally:
        F.scaled_dot_product_attention = orig_sdpa
    print(f"[PASS] Autocast bfloat16 SDPA dtype consistency OK (calls={len(seen_dtypes)}, dtypes={seen_dtypes[0]})")


if __name__ == "__main__":
    test_o8_canonicalization_invariance()
    test_word_aware_elastic_chunker()
    test_decoder_radar_loss_and_grad_flow()
    test_phase2_dataset_source_and_sentence_truncation()
    test_chunked_loss_and_vectorized_corruption()
    test_autocast_bfloat16_sdpa_dtype_consistency()
    print("=== ALL PHASE 2 DECODER TESTS PASSED ===")
