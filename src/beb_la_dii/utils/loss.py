import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_normalize(x, dim=-1, eps=1e-6):
    """
    Safely normalizes a tensor along a dimension by adding eps inside the square root
    to prevent gradient explosion and NaN propagation when norm is close to zero.
    """
    norm = torch.sqrt(torch.sum(x ** 2, dim=dim, keepdim=True) + eps)
    return x / norm


def safe_cosine_similarity(x1, x2, dim=-1, eps=1e-6):
    """
    Safe cosine similarity with eps inside the square root for stable backpropagation.
    """
    norm1 = torch.sqrt(torch.sum(x1 ** 2, dim=dim) + eps)
    norm2 = torch.sqrt(torch.sum(x2 ** 2, dim=dim) + eps)
    return torch.sum(x1 * x2, dim=dim) / (norm1 * norm2)


class DistillationLoss(nn.Module):
    """
    Расчет L_total для выравнивания латентных пространств.
    Использует MSE и Cosine Similarity для трех контрольных точек (слои 20, 30, 40).

    Дополнительно:
    - L_prior: принуждение l40 к N(0,1) с учетом attention_mask.
    - L_kl: KL-дивергенция для InputProjector (VAE) с учетом attention_mask.
    - compute_delta_loss: L_delta для обучения на семантических градиентах между масками.
    """
    def __init__(self, layer_weights={40: 1.0}, mse_weight=1.0, cos_weight=0.0,  # ADR-018: cos отключён
                 lambda_scale=0.1, lambda_rkd=0.01, lambda_norm=0.0): # ADR-024: Norm Correlation отключён
        super().__init__()
        self.layer_weights = layer_weights
        self.mse_weight = mse_weight
        self.cos_weight = cos_weight
        self.lambda_scale = lambda_scale
        self.lambda_rkd = lambda_rkd
        self.lambda_norm = lambda_norm
        self.mse = nn.MSELoss()

    def forward(self, student_hidden_states, teacher_hidden_states, attention_mask=None,
                mu=None, logvar=None, beta=0.0, raw_student_states=None, lambda_prior=0.1):
        """
        student_hidden_states: dict {layer_idx: tensor} (projected, Qwen-dim)
        teacher_hidden_states: dict {layer_idx: tensor} (Qwen-dim)
        attention_mask: Tensor (B, T)
        raw_student_states: dict {layer_idx: tensor} (unprojected/raw, BERT-dim)
        lambda_prior: вес регуляризации N(0,1) для активных токенов l40
        """
        total_loss = 0.0
        mse_total = 0.0
        cos_total = 0.0
        scale_total = 0.0
        rkd_total = 0.0
        norm_total = 0.0
        metrics = {}

        # Подготовка маски
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            num_active_elements = mask.sum()
        else:
            mask = 1.0
            num_active_elements = None

        for layer_idx, weight in self.layer_weights.items():
            if layer_idx in student_hidden_states and layer_idx in teacher_hidden_states:
                s_h = student_hidden_states[layer_idx].float()
                t_h = teacher_hidden_states[layer_idx].float()

                # 1. Cosine Similarity Loss (Centered for ALL layers if masked)
                if attention_mask is not None:
                    # Центрированный косинус (Pearson Correlation) для всех слоев
                    # Это заставляет студента учить структуру, а не просто средний вектор
                    m = attention_mask.unsqueeze(-1).float()
                    n = m.sum(dim=1, keepdim=True).clamp(min=1e-6)
                    
                    s_mean = (s_h * m).sum(dim=1, keepdim=True) / n
                    t_mean = (t_h * m).sum(dim=1, keepdim=True) / n
                    
                    s_centered = (s_h - s_mean) * m
                    t_centered = (t_h - t_mean) * m
                    
                    cos_sim = safe_cosine_similarity(s_centered, t_centered, dim=-1, eps=1e-6)
                    cos_l = 1.0 - (cos_sim * attention_mask).sum() / (attention_mask.sum() + 1e-6)
                else:
                    cos_sim = safe_cosine_similarity(s_h, t_h, dim=-1, eps=1e-6)
                    cos_l = 1.0 - cos_sim.mean()

                # 2. MSE Loss (отключен для всех слоев, оставляем 0.0 для обратной совместимости метрик)
                mse_l = torch.tensor(0.0, device=s_h.device)
                
                # 3. Scale Alignment Loss
                if attention_mask is not None:
                    m_scale = attention_mask.unsqueeze(-1).float()
                    n_scale = m_scale.sum(dim=1, keepdim=True).clamp(min=2.0)
                    
                    s_mean_scale = (s_h * m_scale).sum(dim=1, keepdim=True) / n_scale
                    s_centered_scale = (s_h - s_mean_scale) * m_scale
                    s_var_scale = (s_centered_scale ** 2).sum(dim=1, keepdim=True) / (n_scale - 1).clamp(min=1.0)
                    s_std = torch.sqrt(s_var_scale + 1e-8)
                    
                    t_mean_scale = (t_h * m_scale).sum(dim=1, keepdim=True) / n_scale
                    t_centered_scale = (t_h - t_mean_scale) * m_scale
                    t_var_scale = (t_centered_scale ** 2).sum(dim=1, keepdim=True) / (n_scale - 1).clamp(min=1.0)
                    t_std = torch.sqrt(t_var_scale + 1e-8)
                else:
                    s_std = s_h.std(dim=1, keepdim=True, unbiased=True)
                    t_std = t_h.std(dim=1, keepdim=True, unbiased=True)
                
                scale_l = F.mse_loss(s_std, t_std.detach())

                layer_l = (
                    self.cos_weight * cos_l +
                    self.lambda_scale * scale_l
                )

                mse_total += weight * mse_l
                cos_total += weight * cos_l
                scale_total += weight * scale_l
                
                metrics[f"l{layer_idx}_mse"] = mse_l.detach()
                metrics[f"l{layer_idx}_cos"] = cos_l.detach()
                metrics[f"l{layer_idx}_scale_align"] = scale_l.detach()
                total_loss += weight * layer_l

        # 3. Prior Loss & Isotropy Regularization (на сырых BERT-векторах)
        if raw_student_states is not None:
            for layer_idx, raw_states in raw_student_states.items():
                raw_states = raw_states.float()
                B, T, D = raw_states.shape
                
                if attention_mask is not None:
                    mask_flat = attention_mask.view(-1, 1).float() # (B*T, 1)
                    raw_flat = raw_states.view(-1, D) # (B*T, D)
                    
                    N = mask_flat.sum().clamp(min=1.0)
                    
                    # Mean
                    m_state = (raw_flat * mask_flat).sum(dim=0) / N # (D,)
                    
                    # Centered
                    z = raw_flat - m_state.unsqueeze(0) # (B*T, D)
                    z_masked = z * mask_flat # Zero out padding tokens
                    
                    # Variance
                    v_state = (z_masked ** 2).sum(dim=0) / N.clamp(min=2.0) # (D,)
                    
                    # Scale-invariant Covariance matrix (Correlation Matrix)
                    z_normed = safe_normalize(z_masked, dim=0, eps=1e-6)
                    cov = (z_normed.T @ z_normed)
                else:
                    m_state = raw_states.mean(dim=(0, 1))
                    v_state = raw_states.var(dim=(0, 1), unbiased=False)
                    z = raw_states - m_state.view(1, 1, -1)
                    z_flat = z.view(-1, D)
                    z_normed = safe_normalize(z_flat, dim=0, eps=1e-6)
                    cov = (z_normed.T @ z_normed)

                # Off-diagonal elements of covariance matrix
                cov_off_diag = cov - torch.diag(torch.diag(cov))
                # Штраф за корреляцию (изотропия). Масштабируем по D.
                # Ручной Huber (без zeros_like): XLA фьюзит в один kernel, экономия ~4MB/слой.
                # 2*Huber(x,0,δ=1) = x^2 при |x|<1, 2*(|x|-0.5) при |x|>=1 → O(S) градиент.
                _cov_abs = cov_off_diag.abs()
                cov_loss = 2.0 * torch.where(_cov_abs < 1.0,
                                             0.5 * cov_off_diag.pow(2),
                                             _cov_abs - 0.5).sum() / D
                metrics[f"l{layer_idx}_cov_loss"] = cov_loss.detach()

                if layer_idx == 40:
                    # Целевое распределение: mean=0, var=1 + Covariance penalty
                    # m_state.pow(2) → O(S) градиент, стабильно.
                    # Huber(v_state, 1, δ=1) без ones_like: 2*Huber = (v-1)^2 при |v-1|<1, иначе линейно.
                    _v_diff = v_state - 1.0
                    _v_abs = _v_diff.abs()
                    prior_loss = (m_state.pow(2).mean()
                                  + 2.0 * torch.where(_v_abs < 1.0,
                                                      0.5 * _v_diff.pow(2),
                                                      _v_abs - 0.5).mean()
                                  + 0.3 * cov_loss)
                    total_loss += lambda_prior * prior_loss
                    metrics[f"l{layer_idx}_prior"] = prior_loss.detach()
                    
                    # Применяем RKD и Norm Correlation только к слою 40 (отвязанному от проектора)
                    if 40 in teacher_hidden_states:
                        t_h = teacher_hidden_states[40].float()
                        s_h = raw_states
                        
                        # 4. Relational Knowledge Distillation (RKD) - Pairwise Similarity Correlation
                        if attention_mask is not None:
                            n_rkd = attention_mask.unsqueeze(-1).sum(dim=1, keepdim=True).clamp(min=1e-6)
                            s_mean_rkd = (s_h * attention_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / n_rkd
                            t_mean_rkd = (t_h * attention_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / n_rkd
                            s_centered = (s_h - s_mean_rkd) * attention_mask.unsqueeze(-1)
                            t_centered = (t_h - t_mean_rkd) * attention_mask.unsqueeze(-1)
                        else:
                            s_centered = s_h - s_h.mean(dim=1, keepdim=True)
                            t_centered = t_h - t_h.mean(dim=1, keepdim=True)

                        s_normed = safe_normalize(s_centered, dim=-1, eps=1e-6)
                        t_normed = safe_normalize(t_centered, dim=-1, eps=1e-6)
                        
                        s_dist = torch.bmm(s_normed, s_normed.transpose(1, 2))
                        t_dist = torch.bmm(t_normed, t_normed.transpose(1, 2))
                        
                        if attention_mask is not None:
                            mask_2d = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)  # (B, T, T)
                            diff = (s_dist - t_dist.detach()) * mask_2d
                            rkd_l = (diff ** 2).sum() / (mask_2d.sum() + 1e-6)
                        else:
                            diff = s_dist - t_dist.detach()
                            rkd_l = (diff ** 2).mean()
                            
                        # 5. Norm Correlation Loss (Pearson Correlation of token norms)
                        s_norms = s_h.norm(dim=-1)  # (B, T)
                        t_norms = t_h.norm(dim=-1)  # (B, T)
                        
                        # 6. Norm CV (Coefficient of Variation) — метрика сферичности
                        # CV = std(‖x‖) / mean(‖x‖). CV < 0.05 → практически сфера.
                        if attention_mask is not None:
                            n_active = attention_mask.sum().clamp(min=1e-6)
                            _norm_mean = (s_norms * attention_mask).sum() / n_active
                            _norm_var = (((s_norms - _norm_mean) * attention_mask) ** 2).sum() / n_active
                            _norm_std = torch.sqrt(_norm_var + 1e-8)
                        else:
                            _norm_mean = s_norms.mean().clamp(min=1e-6)
                            _norm_std  = s_norms.std(unbiased=False)
                        metrics["norm_cv_l40_raw"] = (_norm_std / _norm_mean).detach()
                        
                        if attention_mask is not None:
                            n_norm = attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
                            s_norms_mean = (s_norms * attention_mask).sum(dim=1, keepdim=True) / n_norm
                            t_norms_mean = (t_norms * attention_mask).sum(dim=1, keepdim=True) / n_norm
                            
                            s_norms_centered = (s_norms - s_norms_mean) * attention_mask
                            t_norms_centered = (t_norms - t_norms_mean) * attention_mask
                            
                            norm_cos = safe_cosine_similarity(s_norms_centered, t_norms_centered.detach(), dim=1, eps=1e-6)
                            active_seq_mask = (attention_mask.sum(dim=1) > 0).float()
                            norm_l = 1.0 - (norm_cos * active_seq_mask).sum() / (active_seq_mask.sum() + 1e-6)
                        else:
                            s_norms_mean = s_norms.mean(dim=1, keepdim=True)
                            t_norms_mean = t_norms.mean(dim=1, keepdim=True)
                            
                            s_norms_centered = s_norms - s_norms_mean
                            t_norms_centered = t_norms - t_norms_mean
                            
                            norm_cos = safe_cosine_similarity(s_norms_centered, t_norms_centered.detach(), dim=1, eps=1e-6)
                            norm_l = 1.0 - norm_cos.mean()
                            
                        rkd_total += self.layer_weights.get(40, 1.0) * rkd_l
                        norm_total += self.layer_weights.get(40, 1.0) * norm_l
                        
                        total_loss += self.lambda_rkd * rkd_l + self.lambda_norm * norm_l
                        
                        metrics["l40_rkd_raw"] = rkd_l.detach()
                        metrics["l40_norm_corr_raw"] = norm_l.detach()
                        
                else:
                    # Регуляризация внутренних слоев: центрирование (mu -> 0) + усиленная изотропия
                    # Soft Variance Penalty: ограничиваем дисперсию в коридоре [0.5, 1.5]
                    # Huber для верхней границы (защита от O(S)), квадратичная для нижней.
                    _svp_ceil = F.relu(v_state - 1.5)
                    ceil_penalty = 2.0 * torch.where(_svp_ceil < 1.0,
                                                     0.5 * _svp_ceil.pow(2),
                                                     _svp_ceil - 0.5).mean()
                    floor_penalty = F.relu(0.5 - v_state).pow(2).mean()
                    soft_var_penalty = ceil_penalty + floor_penalty
                    # Huber для m_state убран, возвращен стабильный MSE
                    intermediate_loss = m_state.pow(2).mean() + 0.1 * cov_loss + 0.1 * soft_var_penalty
                    total_loss += intermediate_loss
                    metrics[f"l{layer_idx}_intermediate_reg"] = intermediate_loss.detach()
                    metrics[f"l{layer_idx}_soft_var_penalty"] = soft_var_penalty.detach()

        metrics["mse"] = mse_total.detach() if torch.is_tensor(mse_total) else mse_total
        metrics["cosine"] = cos_total.detach() if torch.is_tensor(cos_total) else cos_total
        metrics["scale_align"] = scale_total.detach() if torch.is_tensor(scale_total) else scale_total
        metrics["rkd"] = rkd_total.detach() if torch.is_tensor(rkd_total) else rkd_total
        metrics["norm_corr"] = norm_total.detach() if torch.is_tensor(norm_total) else norm_total

        # 4. KL-Divergence для InputProjector — зануляем padding перед вычислением
        if mu is not None and logvar is not None:
            mu_f = mu.float()
            logvar_f = logvar.float()

            # Зануляем padding-токены, чтобы они не вносили вклад в KL
            if attention_mask is not None:
                amask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
                mu_f     = mu_f     * amask
                logvar_f = logvar_f * amask

            kl_loss_raw = -0.5 * torch.mean(1 + logvar_f - mu_f.pow(2) - logvar_f.exp(), dim=-1)

            FREE_BITS = 0.5
            kl_loss_clamped = torch.clamp(kl_loss_raw, min=FREE_BITS)

            if attention_mask is not None:
                kl_loss = (kl_loss_clamped * attention_mask).sum() / (attention_mask.sum() + 1e-6)
            else:
                kl_loss = kl_loss_clamped.mean()

            total_loss = total_loss + beta * kl_loss
            metrics["kl"] = kl_loss.detach()
            metrics["mu_norm"] = mu_f.norm(dim=-1).mean().detach()
            metrics["logvar_mean"] = logvar_f.mean().detach()

        return total_loss, metrics

    def compute_delta_loss(self,
                           student_states_k:  dict,
                           student_states_k1: dict,
                           teacher_states_k:  dict,
                           teacher_states_k1: dict,
                           mask_k: torch.Tensor,
                           mask_k1: torch.Tensor) -> tuple:
        """
        L_delta: выравнивает семантические градиенты между соседними вариантами масок.
        """
        loss = torch.tensor(0.0, device=mask_k1.device)
        metrics = {}

        m_k = mask_k.unsqueeze(-1).float()
        n_k = m_k.sum(dim=1).clamp(min=1.0)
        
        m_k1 = mask_k1.unsqueeze(-1).float()
        n_k1 = m_k1.sum(dim=1).clamp(min=1.0)

        def pool(h: torch.Tensor, m: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
            return (h.float() * m).sum(dim=1) / n

        for layer_idx, weight in self.layer_weights.items():
            if (layer_idx not in student_states_k  or layer_idx not in student_states_k1 or
                    layer_idx not in teacher_states_k  or layer_idx not in teacher_states_k1):
                continue

            ds = pool(student_states_k1[layer_idx], m_k1, n_k1) - pool(student_states_k[layer_idx], m_k, n_k)
            dt = pool(teacher_states_k1[layer_idx], m_k1, n_k1) - pool(teacher_states_k[layer_idx], m_k, n_k)

            # Используем косинусное сходство для всех слоев
            # Это отвязывает дельту от масштаба учителя и фокусируется на направлении.
            cos_sim = safe_cosine_similarity(ds, dt.detach(), dim=-1, eps=1e-6)
            
            # Добавляем штраф за коллапс магнитуды дельты (ADR-014)
            s_mag = ds.norm(dim=-1)
            t_mag = dt.norm(dim=-1).detach()
            mag_loss = F.mse_loss(s_mag, t_mag)
            
            layer_delta_loss = 1.0 - cos_sim.mean() + 0.1 * mag_loss

            loss = loss + weight * layer_delta_loss

            dcos = safe_cosine_similarity(ds, dt.detach(), dim=-1, eps=1e-6).mean()
            dmag = (ds.norm(dim=-1) / dt.norm(dim=-1).clamp(min=1e-6)).mean()
            metrics[f"delta_cos_l{layer_idx}"]       = dcos.detach()
            metrics[f"delta_mag_ratio_l{layer_idx}"] = dmag.detach()

        return loss, metrics


if __name__ == "__main__":
    B, T, D = 2, 64, 3584
    criterion = DistillationLoss()

    # --- Тест forward ---
    s_states = {l: torch.randn(B, T, D) for l in [20, 30, 40]}
    t_states = {**s_states, 30: torch.randn(B, T, D), 40: torch.randn(B, T, D)}
    raw_s    = {40: torch.randn(B, T, 1024)}
    amask    = torch.ones(B, T); amask[0, T//2:] = 0  # короткий текст в первом примере

    loss, metrics = criterion(s_states, t_states, attention_mask=amask, raw_student_states=raw_s)
    print(f"L_state: {loss.item():.6f}  |  l40_prior: {metrics.get('l40_prior', 'N/A')}")
    print(f"Metrics details: rkd={metrics.get('rkd', 'N/A'):.6f}, norm_corr={metrics.get('norm_corr', 'N/A'):.6f}")
    for l in [20, 30, 40]:
        rkd_val = metrics.get(f'l40_rkd_raw' if l == 40 else f'l{l}_rkd', 'N/A')
        norm_val = metrics.get(f'l40_norm_corr_raw' if l == 40 else f'l{l}_norm_corr', 'N/A')
        rkd_str = f"{rkd_val:.6f}" if isinstance(rkd_val, float) or torch.is_tensor(rkd_val) else str(rkd_val)
        norm_str = f"{norm_val:.6f}" if isinstance(norm_val, float) or torch.is_tensor(norm_val) else str(norm_val)
        print(f"  Layer {l} | rkd: {rkd_str} | norm_corr: {norm_str}")

    # --- Тест compute_delta_loss ---
    T2 = T * 2
    sk  = {l: torch.randn(B, T, D, requires_grad=True) for l in [20, 30, 40]}
    sk1 = {l: torch.randn(B, T, D, requires_grad=True) for l in [20, 30, 40]}
    tk  = {l: torch.randn(B, T, D) for l in [20, 30, 40]}
    tk1 = {l: torch.randn(B, T, D) for l in [20, 30, 40]}
    mask_k = torch.ones(B, T)
    mask_k1 = torch.ones(B, T)

    delta_loss, delta_metrics = criterion.compute_delta_loss(sk, sk1, tk, tk1, mask_k, mask_k1)
    print(f"L_delta: {delta_loss.item():.6f}  |  metrics: {delta_metrics}")
    assert delta_loss.requires_grad, "L_delta не имеет градиента!"
    print("All tests passed!")
