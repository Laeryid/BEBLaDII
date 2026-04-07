import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

def test_dus_creation_logic_mocked():
    """
    Проверка вызова расширения слоев в DUS через Mock.
    """
    with patch("transformers.AutoModelForMaskedLM.from_pretrained") as mock_from_pretrained, \
         patch("transformers.AutoConfig.from_pretrained") as mock_config:
        
        from beb_la_dii.model.dus import create_latentbert
        
        # Мокаем базовую модель с 28 слоями. 
        # Слои должны быть представителями nn.Module для nn.ModuleList.
        mock_base_model = MagicMock()
        mock_base_model.model.layers = nn.ModuleList([nn.Identity() for _ in range(28)])
        mock_from_pretrained.return_value = mock_base_model
        
        model = create_latentbert(target_layers=40)
        
        # Проверяем что в итоге слоев 40 (20 + 20)
        assert len(model.model.layers) == 40
        assert model.config.num_hidden_layers == 40
