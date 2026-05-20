import torch
import torch.nn.functional as F
import pytest
from src.beb_la_dii.utils.loss import DistillationLoss


def test_rkd_norm_corr_identical_states():
    """
    Проверка, что если скрытые состояния Студента и Учителя идентичны,
    то лоссы RKD и Norm Correlation равны нулю (или пренебрежимо малы).
    """
    B, T, D = 2, 8, 128
    criterion = DistillationLoss(layer_weights={30: 1.0}, lambda_rkd=1.0, lambda_norm=1.0)
    
    # Студент и Учитель полностью идентичны
    s_state = torch.randn(B, T, D)
    s_states = {30: s_state}
    t_states = {30: s_state.clone()}
    
    # 1. Тест без маски
    _, metrics = criterion(s_states, t_states, attention_mask=None)
    assert metrics["l30_rkd"] < 1e-5
    assert metrics["l30_norm_corr"] < 1e-5
    
    # 2. Тест с маской внимания (вторая половина — паддинг)
    mask = torch.ones(B, T)
    mask[:, T // 2:] = 0.0
    _, metrics_masked = criterion(s_states, t_states, attention_mask=mask)
    assert metrics_masked["l30_rkd"] < 1e-5
    assert metrics_masked["l30_norm_corr"] < 1e-5


def test_rkd_norm_corr_gradients():
    """
    Проверка, что лоссы RKD и Norm Correlation успешно передают градиент Студенту
    и отсоединяют Учителя от вычислительного графа.
    """
    B, T, D = 2, 8, 128
    criterion = DistillationLoss(layer_weights={30: 1.0}, cos_weight=0.0, lambda_scale=0.0, lambda_rkd=0.5, lambda_norm=0.5)
    
    s_state = torch.randn(B, T, D, requires_grad=True)
    t_state = torch.randn(B, T, D, requires_grad=True)
    
    s_states = {30: s_state}
    t_states = {30: t_state}
    
    loss, metrics = criterion(s_states, t_states, attention_mask=None)
    loss.backward()
    
    # Градиенты должны течь в Студента
    assert s_state.grad is not None
    assert torch.any(s_state.grad != 0.0)
    
    # Градиенты НЕ должны течь в Учителя (так как мы делаем detach() в RKD и Norm)
    assert t_state.grad is None or torch.all(t_state.grad == 0.0)


def test_rkd_norm_corr_masking():
    """
    Проверка, что маскирование паддингов работает корректно и не приводит к NaN
    даже при сильном занулении последовательности.
    """
    B, T, D = 2, 8, 128
    criterion = DistillationLoss(layer_weights={30: 1.0}, lambda_rkd=1.0, lambda_norm=1.0)
    
    s_states = {30: torch.randn(B, T, D)}
    t_states = {30: torch.randn(B, T, D)}
    
    # Экстремальный паддинг: активен только один токен из 8
    mask = torch.zeros(B, T)
    mask[:, 0] = 1.0
    
    loss, metrics = criterion(s_states, t_states, attention_mask=mask)
    
    # Значения лоссов не должны быть NaN или Inf
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert not torch.isnan(metrics["l30_rkd"])
    assert not torch.isnan(metrics["l30_norm_corr"])
