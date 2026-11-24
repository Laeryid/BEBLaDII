"""
Универсальные TPU/GPU-одобренные оптимизации работы со словарём (131,072 x 1024) для Phase 3 Alt.

Все операции построены строго со статическими формами (100% XLA/TPU-совместимы, без веток по
платформам, без torch.unique/torch.nonzero переменной длины и без host-device sync внутри графа):
  1. NoiseRadarHead.compute_dict_stats: преднормированный словарь lm_head_unit в компактном
     gemm_dtype (BF16/FP16) + 4 равных чанка (32768) -> без FP32-каста весов и пересчёта inv_n.
  2. Batch-Shared Stratified Sampled Softmax (K_pool = M + K_hard + K_rand, по умолчанию 4096):
     - единый на батч пул жёстких негативов через низкоранговый SVD-эскиз словаря (rank=64, ~16 MB);
     - дедупликация булевыми матрицами (static shape) + строго несмещённая importance-поправка
       +ln(|V_tail| / N_rand_eff) на случайную часть хвоста;
     - один плотный GEMM (M, 1024) @ (1024, K_pool) вместо (M, 131072).
  3. Статическая фильтрация примеров батча до decoder.decode_hidden (B_ce = ceil(B * fraction)
     примеров с максимальным числом активных токенов t < ce_t_max).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastVocabOptimizer(nn.Module):
    """
    Предвычисленный ускоритель словаря для замороженного GrassmannianDecoder.
    Не изменяет веса Системы 1 и полностью обратим через detach_from_decoder().
    """

    def __init__(
        self,
        lm_head_weight: torch.Tensor,
        gemm_dtype: torch.dtype = torch.bfloat16,
        vocab_chunk_size: int = 32768,
        svd_rank: int = 64,
        svd_niter: int = 2,
        ce_pool_hard: int = 1536,
        ce_pool_rand: int = 2048,
        ce_batch_fraction: float = 0.5,
    ):
        super().__init__()
        if lm_head_weight.ndim != 2:
            raise ValueError(f"Ожидалась 2D матрица словаря (V, D), получено: {tuple(lm_head_weight.shape)}")

        self.vocab_size, self.hidden_size = int(lm_head_weight.shape[0]), int(lm_head_weight.shape[1])
        self.gemm_dtype = gemm_dtype
        self.vocab_chunk_size = int(vocab_chunk_size)
        self.svd_rank = min(int(svd_rank), self.hidden_size, self.vocab_size)
        self.ce_pool_hard = min(int(ce_pool_hard), self.vocab_size)
        self.ce_pool_rand = min(int(ce_pool_rand), self.vocab_size)
        self.ce_batch_fraction = float(ce_batch_fraction)

        self._orig_compute_dict_stats = None
        self._attached_radar: Optional[nn.Module] = None

        with torch.no_grad():
            w_f32 = lm_head_weight.detach().float()
            # 1. Преднормированный словарь в компактном gemm_dtype для NoiseRadarHead
            w_unit = F.normalize(w_f32, p=2, dim=-1, eps=1e-6).to(dtype=gemm_dtype).contiguous()
            self.register_buffer("lm_head_unit", w_unit, persistent=False)

            # 2. Низкоранговый SVD-эскиз словаря (rank=64 -> ~16 MB в BF16) для batch-shared hard negatives
            # W ≈ (U_r * S_r) @ V_r^T, где U_r: (V, r), S_r: (r,), V_r: (D, r)
            u_r, s_r, v_r = torch.svd_lowrank(w_f32, q=self.svd_rank, niter=int(svd_niter))
            w_svd_vocab = (u_r * s_r.unsqueeze(0)).to(dtype=gemm_dtype).contiguous()
            w_svd_proj = v_r.to(dtype=gemm_dtype).contiguous()
            self.register_buffer("w_svd_vocab", w_svd_vocab, persistent=False)
            self.register_buffer("w_svd_proj", w_svd_proj, persistent=False)

    @classmethod
    def from_decoder(
        cls,
        decoder: nn.Module,
        gemm_dtype: torch.dtype = torch.bfloat16,
        vocab_chunk_size: int = 32768,
        svd_rank: int = 64,
        ce_pool_hard: int = 1536,
        ce_pool_rand: int = 2048,
        ce_batch_fraction: float = 0.5,
        attach: bool = True,
    ) -> "FastVocabOptimizer":
        if getattr(decoder, "lm_head_weight", None) is None:
            raise RuntimeError("У декодера не задан lm_head_weight (вызовите tie_embeddings перед созданием FastVocabOptimizer).")
        opt = cls(
            lm_head_weight=decoder.lm_head_weight,
            gemm_dtype=gemm_dtype,
            vocab_chunk_size=vocab_chunk_size,
            svd_rank=svd_rank,
            ce_pool_hard=ce_pool_hard,
            ce_pool_rand=ce_pool_rand,
            ce_batch_fraction=ce_batch_fraction,
        )
        if attach:
            opt.attach_to_decoder(decoder)
        return opt

    @torch.no_grad()
    def fast_compute_dict_stats(
        self,
        h_unpacked: torch.Tensor,
        lm_head_weight: Optional[torch.Tensor] = None,
        lm_head_inv_norms: Optional[torch.Tensor] = None,
        vocab_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Ускоренный drop-in для NoiseRadarHead.compute_dict_stats:
        - использует преднормированный lm_head_unit в gemm_dtype (без FP32-каста и умножения на inv_n);
        - работает фиксированными чанками (100% static shapes для XLA/TPU и GPU);
        - возвращает (B, L, 4): [cos_max, cos_2nd, cos_gap, log_norm_ratio] в float32.
        """
        B, L, D = h_unpacked.shape
        device = h_unpacked.device
        chunk_sz = int(vocab_chunk_size or self.vocab_chunk_size)

        with torch.autocast(device_type=device.type, enabled=False):
            h_f32 = h_unpacked.float().reshape(B * L, D)
            h_norms = h_f32.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
            h_unit = (h_f32 / h_norms).to(dtype=self.gemm_dtype)

            u_mat = self.lm_head_unit
            if u_mat.device != device:
                u_mat = u_mat.to(device=device)

            v_size = u_mat.size(0)
            top2_vals = torch.full((B * L, 2), -10.0, device=device, dtype=torch.float32)

            for start in range(0, v_size, chunk_sz):
                end = min(start + chunk_sz, v_size)
                u_chunk = u_mat[start:end]
                cos_chunk = F.linear(h_unit, u_chunk)
                k_take = min(2, end - start)
                chunk_top = torch.topk(cos_chunk, k=k_take, dim=-1).values.float()
                merged = torch.cat([top2_vals, chunk_top], dim=-1)
                top2_vals = torch.topk(merged, k=2, dim=-1).values

            cos_max = top2_vals[:, 0:1]
            cos_2nd = top2_vals[:, 1:2]
            cos_gap = (cos_max - cos_2nd).clamp(min=0.0)
            norm_feat = torch.log1p(h_norms / math.sqrt(D))
            return torch.cat([cos_max, cos_2nd, cos_gap, norm_feat], dim=-1).view(B, L, 4)

    def attach_to_decoder(self, decoder: nn.Module) -> None:
        """Подменяет compute_dict_stats у экземпляра decoder.noise_radar без модификации класса."""
        radar = getattr(decoder, "noise_radar", None)
        if radar is None:
            return
        if self._attached_radar is radar:
            return
        self.detach_from_decoder()
        self._orig_compute_dict_stats = radar.compute_dict_stats
        self._attached_radar = radar
        radar.compute_dict_stats = self.fast_compute_dict_stats

    def detach_from_decoder(self) -> None:
        """Возвращает исходный compute_dict_stats экземпляру decoder.noise_radar."""
        if self._attached_radar is not None and self._orig_compute_dict_stats is not None:
            self._attached_radar.compute_dict_stats = self._orig_compute_dict_stats
        self._attached_radar = None
        self._orig_compute_dict_stats = None

    @torch.no_grad()
    def select_batch_hard_negatives(
        self,
        h_sub: torch.Tensor,
        valid_mask: torch.Tensor,
        k_hard: Optional[int] = None,
        row_chunk_size: int = 128,
    ) -> torch.Tensor:
        """
        Отбирает фиксированные K_hard batch-shared жёстких негативов через SVD-эскиз словаря (rank=64).
        Для защиты от доминирования токенов с большой нормой ||h_i|| берётся максимум по
        относительным логитам (z_iv - max_u z_iu) <= 0.
        100% static shapes, без host-device sync.
        """
        k_take = min(int(k_hard or self.ce_pool_hard), self.vocab_size)
        M, _ = h_sub.shape
        device = h_sub.device

        with torch.autocast(device_type=device.type, enabled=False):
            w_proj = self.w_svd_proj if self.w_svd_proj.device == device else self.w_svd_proj.to(device)
            w_voc = self.w_svd_vocab if self.w_svd_vocab.device == device else self.w_svd_vocab.to(device)

            h_r = torch.matmul(h_sub.detach().to(dtype=self.gemm_dtype), w_proj)  # (M, r)
            max_rel = torch.full((self.vocab_size,), -1e4, device=device, dtype=torch.float32)

            for r_start in range(0, M, row_chunk_size):
                r_end = min(r_start + row_chunk_size, M)
                h_blk = h_r[r_start:r_end]  # (mb, r)
                v_blk = valid_mask[r_start:r_end]  # (mb,)

                # (mb, V) в gemm_dtype (~32 MB при mb=128)
                z_blk = F.linear(h_blk, w_voc).float()
                row_max = z_blk.amax(dim=-1, keepdim=True)
                rel_blk = z_blk - row_max
                # Замаскированные (невалидные) слоты не влияют на выбор жёстких негативов
                rel_blk = torch.where(v_blk.unsqueeze(-1), rel_blk, torch.full_like(rel_blk, -1e4))
                max_rel = torch.maximum(max_rel, rel_blk.amax(dim=0))

            hard_ids = torch.topk(max_rel, k=k_take, dim=0).indices
            return hard_ids

    def compute_sampled_ce_from_hidden(
        self,
        h_sub: torch.Tensor,
        tgt_sub: torch.Tensor,
        valid_sel_mask: torch.Tensor,
        lm_head_weight: torch.Tensor,
        k_hard: Optional[int] = None,
        k_rand: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Вычисляет Batch-Shared Stratified Sampled Softmax (100% static shapes) по скрытым векторам h_sub (M, D):
        - Пул pool_ids = [tgt_safe (M) || hard_ids (K_hard) || rand_ids (K_rand)] фиксированной длины K_pool;
        - Дедупликация булевыми матрицами без torch.unique;
        - Строго несмещённая importance-поправка +ln((V - N_exact_eff) / N_rand_eff) на случайные классы хвоста;
        - Возвращает (per_tok_ce (M,), is_correct (M,) bool по нескорректированным логитам пула).
        """
        M = h_sub.shape[0]
        device = h_sub.device
        n_hard = min(int(self.ce_pool_hard if k_hard is None else k_hard), self.vocab_size)
        n_rand = min(int(self.ce_pool_rand if k_rand is None else k_rand), self.vocab_size)

        with torch.no_grad():
            tgt_safe = torch.where(valid_sel_mask, tgt_sub, torch.zeros_like(tgt_sub)).clamp(0, self.vocab_size - 1)
            hard_ids = self.select_batch_hard_negatives(h_sub, valid_sel_mask, k_hard=n_hard)
            rand_ids = torch.randint(0, self.vocab_size, (n_rand,), device=device, dtype=torch.long)

            pool_exact = torch.cat([tgt_safe, hard_ids], dim=0)  # (E,)
            E = pool_exact.shape[0]

            # Дедупликация pool_exact: True только на первом вхождении каждого ID
            eq_exact = pool_exact.unsqueeze(1) == pool_exact.unsqueeze(0)  # (E, E)
            keep_exact = ~torch.tril(eq_exact, diagonal=-1).any(dim=1)  # (E,)

            # Позиция целевого токена tgt_safe[i] среди первых M слотов pool_exact
            tgt_match = (tgt_safe.unsqueeze(1) == pool_exact[:M].unsqueeze(0)) & keep_exact[:M].unsqueeze(0)
            target_pos = tgt_match.to(torch.int32).argmax(dim=-1).long()  # (M,)

            # Дедупликация rand_ids: исключаем попадания в pool_exact и повторы внутри rand_ids
            rand_in_exact = (rand_ids.unsqueeze(1) == pool_exact.unsqueeze(0)).any(dim=1)
            eq_rand = rand_ids.unsqueeze(1) == rand_ids.unsqueeze(0)
            keep_rand = (~rand_in_exact) & (~torch.tril(eq_rand, diagonal=-1).any(dim=1))

            # Несмещённый множитель важности хвоста (тензор на устройстве, без .item())
            n_exact_eff = keep_exact.to(torch.float32).sum()
            n_rand_eff = keep_rand.to(torch.float32).sum().clamp(min=1.0)
            log_tail_corr = torch.log((float(self.vocab_size) - n_exact_eff).clamp(min=1.0) / n_rand_eff)

            pool_ids = torch.cat([pool_exact, rand_ids], dim=0)  # (K_pool,)
            col_keep = torch.cat([keep_exact, keep_rand], dim=0)  # (K_pool,)
            tail_mask = torch.cat(
                [
                    torch.zeros(E, device=device, dtype=torch.float32),
                    torch.ones(n_rand, device=device, dtype=torch.float32),
                ],
                dim=0,
            )

        # Один плотный GEMM (M, D) @ (D, K_pool) строго в float32 (ADR 125)
        with torch.autocast(device_type=device.type, enabled=False):
            w_pool = lm_head_weight.index_select(0, pool_ids).float()
            logits_pool = F.linear(h_sub.float(), w_pool)  # (M, K_pool)

            neg_inf = torch.full_like(logits_pool, -1e9)
            logits_corr = torch.where(
                col_keep.unsqueeze(0),
                logits_pool + (tail_mask * log_tail_corr).unsqueeze(0),
                neg_inf,
            )
            per_tok_ce = F.cross_entropy(logits_corr, target_pos, reduction="none")

            with torch.no_grad():
                # Для точности argmax берём по чистым логитам БЕЗ +log_tail_corr
                logits_raw = torch.where(col_keep.unsqueeze(0), logits_pool, neg_inf)
                is_correct = logits_raw.argmax(dim=-1) == target_pos

        return per_tok_ce, is_correct

    def compute_decoder_ce_loss(
        self,
        decoder: nn.Module,
        V_0_hat: torch.Tensor,
        t_mat: torch.Tensor,
        ans_targets: torch.Tensor,
        ce_max_tokens: int = 512,
        ce_t_max: float = 0.50,
        stopword_mask: Optional[torch.Tensor] = None,
        ce_stopword_weight: float = 0.25,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.float16,
        ce_batch_fraction: Optional[float] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Ускоренный TPU/GPU-совместимый сквозной токенный CE-лосс через замороженный Декодер:
        1. Статический отбор B_ce = ceil(B * ce_batch_fraction) примеров с максимумом активных токенов
           ДО тяжёлого 4-слойного decoder.decode_hidden;
        2. Статический отбор ровно M = min(ce_max_tokens, B_ce * L_ans) токенов без torch.where
           переменной длины;
        3. Batch-Shared Stratified Sampled Softmax (K_pool ~ 4096 вместо 131,072).
        """
        B = V_0_hat.shape[0]
        device = V_0_hat.device
        V_ans_hat = V_0_hat[:, 1:]  # (B, N_ans, 8, 128) с градиентом
        n_ans = V_ans_hat.shape[1]
        chunk_sz = int(getattr(decoder, "chunk_size", 4))
        l_ans = n_ans * chunk_sz

        t_tok = t_mat[:, 1:].float().unsqueeze(-1).expand(B, n_ans, chunk_sz).reshape(B, l_ans)
        valid_mask = (t_tok > 1e-3) & (t_tok < ce_t_max) & (ans_targets != -100)

        # 1. Статическая фильтрация примеров батча до decoder.decode_hidden
        frac = self.ce_batch_fraction if ce_batch_fraction is None else float(ce_batch_fraction)
        b_ce = max(1, min(B, int(math.ceil(B * frac))))
        if b_ce < B:
            active_per_sample = valid_mask.to(torch.float32).sum(dim=1)
            top_b = torch.topk(active_per_sample, k=b_ce, dim=0).indices
            V_ans_sel = V_ans_hat.index_select(0, top_b)
            valid_sel = valid_mask.index_select(0, top_b)
            ans_sel = ans_targets.index_select(0, top_b)
            t_sel = t_tok.index_select(0, top_b)
        else:
            V_ans_sel = V_ans_hat
            valid_sel = valid_mask
            ans_sel = ans_targets
            t_sel = t_tok

        # 2. Декодер на отфильтрованных B_ce примерах (с gradient checkpointing)
        dec_dtype = next(decoder.parameters()).dtype
        V_in = V_ans_sel.to(amp_dtype if use_amp else dec_dtype)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            h_final = torch.utils.checkpoint.checkpoint(
                decoder.decode_hidden, V_in, use_reentrant=False
            )  # (B_ce, L_ans, 1024)

        # 3. Статический отбор ровно M токенов (без динамического torch.where)
        total_slots = b_ce * l_ans
        M = min(int(ce_max_tokens), total_slots)
        valid_flat = valid_sel.reshape(total_slots)
        with torch.no_grad():
            priority = valid_flat.to(torch.float32) * 2.0 + torch.rand(
                total_slots, device=device, dtype=torch.float32
            )
            sel_idx = torch.topk(priority, k=M, dim=0).indices

        h_sub = h_final.reshape(total_slots, h_final.shape[-1]).index_select(0, sel_idx)
        tgt_sub = ans_sel.reshape(total_slots).index_select(0, sel_idx).long()
        t_sub = t_sel.reshape(total_slots).index_select(0, sel_idx)
        valid_sub = valid_flat.index_select(0, sel_idx)

        # 4. Batch-Shared Stratified Sampled Softmax
        per_tok_ce, is_correct = self.compute_sampled_ce_from_hidden(
            h_sub=h_sub,
            tgt_sub=tgt_sub,
            valid_sel_mask=valid_sub,
            lm_head_weight=decoder.lm_head_weight,
        )

        with torch.autocast(device_type=device.type, enabled=False):
            w_t = torch.clamp((ce_t_max - t_sub) / max(ce_t_max, 1e-6), min=0.0, max=1.0) ** 2
            if stopword_mask is not None:
                sw_idx = tgt_sub.clamp(0, stopword_mask.numel() - 1)
                w_sw = torch.where(
                    stopword_mask[sw_idx],
                    torch.full_like(w_t, float(ce_stopword_weight)),
                    torch.ones_like(w_t),
                )
                w_tok = w_t * w_sw * valid_sub.to(torch.float32)
            else:
                w_tok = w_t * valid_sub.to(torch.float32)

            w_sum = w_tok.sum()
            l_ce = (per_tok_ce * w_tok).sum() / w_sum.clamp(min=1e-6)
            with torch.no_grad():
                v_cnt = valid_sub.to(torch.float32).sum()
                tok_acc_t = (is_correct & valid_sub).to(torch.float32).sum() / v_cnt.clamp(min=1.0)
                tok_acc = float(tok_acc_t.item())

        return l_ce, tok_acc
