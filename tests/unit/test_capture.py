import torch
from unittest.mock import MagicMock


def test_extract_last_token_3d():
    from cap.core.capture import ActivationCapture

    ac = MagicMock(spec=ActivationCapture)
    ac._extract_last_token_activation = ActivationCapture._extract_last_token_activation.__get__(ac)
    t = torch.randn(2, 5, 768)
    out = ac._extract_last_token_activation(t)
    assert out.shape == (2, 768)


def test_extract_last_token_non_tensor_returns_none():
    from cap.core.capture import ActivationCapture

    ac = MagicMock(spec=ActivationCapture)
    ac._extract_last_token_activation = ActivationCapture._extract_last_token_activation.__get__(ac)
    assert ac._extract_last_token_activation("not a tensor") is None


def test_normalize_layer_patterns_string():
    from cap.core.capture import ActivationCapture

    ac = MagicMock(spec=ActivationCapture)
    ac._normalize_layer_patterns = ActivationCapture._normalize_layer_patterns.__get__(ac)
    assert ac._normalize_layer_patterns("mlp") == ["mlp"]


def test_normalize_layer_patterns_none():
    from cap.core.capture import ActivationCapture

    ac = MagicMock(spec=ActivationCapture)
    ac._normalize_layer_patterns = ActivationCapture._normalize_layer_patterns.__get__(ac)
    assert ac._normalize_layer_patterns(None) is None
