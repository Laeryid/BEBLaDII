import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Расчет L_total для выравнивания латентных пространств.
    Использует MSE и Cosine Similarity для трех контрольных точек (слои 20, 30, 40).

    Дополнительно:
    - L_prior: принуждение l40 к N(0,1) с учетом attention_mask.
    - L_kl: KL-дивергенция для InputProjector (VAE) с учетом attention_mask.
    - compute_delta_loss: L_delta для обучения на семантических градиентах между масками.
    """
    def __init__(self, layer_weights={20: 0.5, 30: 0.7, 40: 1.0}, mse_weight=1.0, cos_weight=1.0):
        super().__init__()
        self.layer_weights = layer_weights
        self.mse_weight = mse_weight
        self.cos_weight = cos_weight
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

                # 1. Cosine Similarity Loss
                cos_sim = F.cosine_similarity(s_h, t_h, dim=-1, eps=1e-6)
                if attention_mask is not None:
                    cos_l = 1.0 - (cos_sim * attention_mask).sum() / (attention_mask.sum() + 1e-6)
                else:
                    cos_l = 1.0 - cos_sim.mean()

                # 2. MSE Loss (для l40 не считаем — там no-bias проектор, важна только семантика)
                if layer_idx == 40:
                    mse_l = torch.tensor(0.0, device=s_h.device)
                    layer_l = self.cos_weight * cos_l
                else:
                    if attention_mask is not None:
                        diff = (s_h - t_h) ** 2
                        mse_l = (diff * mask).sum() / (num_active_elements * s_h.size(-1) + 1e-6)
                    else:
                        mse_l = self.mse(s_h, t_h)
                    layer_l = self.mse_weight * mse_l + self.cos_weight * cos_l

                mse_total += weight * mse_l
                cos_total += weight * cos_l
                metrics[f"l{layer_idx}_mse"] = mse_l.detach()
                metrics[f"l{layer_idx}_cos"] = cos_l.detach()
                total_loss += weight * layer_l

        # 3. Prior Loss — только по активным токенам (attention_mask учитывается)
        if raw_student_states is not None and 40 in raw_student_states:
            raw_40 = raw_student_states[40].float()
            # raw_40: (B, T, D)
            B, T, D = raw_40.shape
            
            if attention_mask is not None:
                mask_flat = attention_mask.view(-1, 1).float() # (B*T, 1)
                raw_flat = raw_40.view(-1, D) # (B*T, D)
                
                N = mask_flat.sum().clamp(min=1.0)
                
                # Mean
                m_40 = (raw_flat * mask_flat).sum(dim=0) / N # (D,)
                
                # Centered
                z = raw_flat - m_40.unsqueeze(0) # (B*T, D)
                z_masked = z * mask_flat # Zero out padding tokens
                
                # Variance
                v_40 = (z_masked ** 2).sum(dim=0) / N.clamp(min=2.0) # (D,)
                
                # Covariance matrix (D, D) - VICReg style for isotropy
                # z_masked.T is (D, B*T), z_masked is (B*T, D)
                cov = (z_masked.T @ z_masked) / N.clamp(min=2.0)
            else:
                m_40 = raw_40.mean(dim=(0, 1))
                v_40 = raw_40.var(dim=(0, 1), unbiased=False)
                z = raw_40 - m_40.view(1, 1, -1)
                z_flat = z.view(-1, D)
                cov = (z_flat.T @ z_flat) / (z_flat.size(0) - 1)

            # Off-diagonal elements of covariance matrix
            cov_off_diag = cov - torch.diag(torch.diag(cov))
            # Штраф за корреляцию (изотропия). Масштабируем по D.
            cov_loss = cov_off_diag.pow(2).sum() / D

            # Целевое распределение: mean=0, var=1 + Covariance penalty
            prior_loss = m_40.pow(2).mean() + (v_40 - 1.0).pow(2).mean() + 0.1 * cov_loss
            total_loss += lambda_prior * prior_loss
            metrics["l40_prior"] = prior_loss.detach()
            metrics["l40_cov_loss"] = cov_loss.detach()

        metrics["mse"] = mse_total.detach() if torch.is_tensor(mse_total) else mse_total
        metrics["cosine"] = cos_total.detach() if torch.is_tensor(cos_total) else cos_total

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

            if layer_idx == 40:
                # Используем косинусное сходство для L40 (согласно ADR-012)
                # Это отвязывает дельту от масштаба учителя и фокусируется на направлении.
                cos_sim = F.cosine_similarity(ds, dt.detach(), dim=-1, eps=1e-6)
                
                # Добавляем штраф за коллапс магнитуды дельты (ADR-013)
                s_mag = ds.norm(dim=-1)
                t_mag = dt.norm(dim=-1).detach()
                mag_loss = F.mse_loss(s_mag, t_mag)
                
                layer_delta_loss = 1.0 - cos_sim.mean() + 0.1 * mag_loss
            else:
                layer_delta_loss = F.mse_loss(ds, dt.detach())

            loss = loss + weight * layer_delta_loss

            dcos = F.cosine_similarity(ds, dt.detach(), dim=-1, eps=1e-6).mean()
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

    # --- Тест compute_delta_loss ---
    T2 = T * 2
    sk  = {l: torch.randn(B, T,  D) for l in [20, 30, 40]}
    sk1 = {l: torch.randn(B, T, D) for l in [20, 30, 40]}
    tk  = {l: torch.randn(B, T,  D) for l in [20, 30, 40]}
    tk1 = {l: torch.randn(B, T, D) for l in [20, 30, 40]}
    mask_k = torch.ones(B, T)
    mask_k1 = torch.ones(B, T)

    delta_loss, delta_metrics = criterion.compute_delta_loss(sk, sk1, tk, tk1, mask_k, mask_k1)
    print(f"L_delta: {delta_loss.item():.6f}  |  metrics: {delta_metrics}")
    assert delta_loss.requires_grad, "L_delta не имеет градиента!"
    print("Все тесты пройдены ✓")
