import pytest
import torch
from unittest.mock import MagicMock, patch
from beb_la_dii.model.projectors import InputProjector, FeatureProjector

def test_projectors_shapes():
    """
    Проверка размерностей выходных тензоров проекторов.
    """
    in_proj = InputProjector(config={"input_dim": 4096, "output_dim": 768})
    feat_proj = FeatureProjector(config={"input_dim": 768, "output_dim": 4096})
    
    x = torch.randn(2, 10, 4096)
    # Используем .proj(x) так как в InputProjector это nn.Sequential
    y = in_proj.proj(x)
    assert y.shape == (2, 10, 768)
    
    z = feat_proj(y)
    assert z.shape == (2, 10, 4096)

def test_dus_model_wrapper_lite():
    """
    Облегченный тест DUSModel без загрузки реальной модели.
    """
    with patch("beb_la_dii.model.dus.create_latentbert") as mock_create:
        from beb_la_dii.model.dus import DUSModel
        mock_model = MagicMock()
        mock_create.return_value = mock_model
        
        dus = DUSModel(component_id="dus_test", version="v1.0", config={"target_layers": 40})
        assert dus.component_id == "dus_test"
        assert dus.version == "v1.0"
        mock_create.assert_called_once()
