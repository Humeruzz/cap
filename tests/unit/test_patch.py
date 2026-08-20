from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import numpy as np
import pytest
import torch
import torch.nn as nn


def _bound_patcher():
    """An ActivationPatcher with methods bound but __init__ (model load) bypassed."""
    from cap.core.patch import ActivationPatcher

    patcher = MagicMock(spec=ActivationPatcher)
    patcher.hooks = []
    patcher.patch_specs = {}
    patcher.layer_stats = None
    for name in (
        "identify_significant_neurons",
        "setup_patches",
        "setup_patches_custom",
        "clear_patches",
        "load_statistics",
    ):
        setattr(patcher, name, getattr(ActivationPatcher, name).__get__(patcher))
    return patcher


def test_identify_significant_neurons_threshold():
    from unittest.mock import MagicMock

    from cap.core.patch import ActivationPatcher

    patcher = MagicMock(spec=ActivationPatcher)
    patcher.layer_stats = {
        "layer.0": {
            "cohens_d": np.array([2.0, 0.1, 3.0]),
            "pooled_std": np.array([0.5, 0.5, 0.5]),
            "mean_diff": np.array([1.0, 0.05, 1.5]),
        },
        "layer.1": {
            "cohens_d": np.array([0.1, 0.2, 0.1]),
            "pooled_std": np.array([0.5, 0.5, 0.5]),
            "mean_diff": np.array([0.05, 0.1, 0.05]),
        },
    }
    patcher.identify_significant_neurons = ActivationPatcher.identify_significant_neurons.__get__(
        patcher
    )
    targets, _info = patcher.identify_significant_neurons(d_threshold=1.0, std_threshold=0.0)
    assert "layer.0" in targets
    assert "layer.1" not in targets
    assert set(targets["layer.0"]) == {0, 2}


def test_hook_scales_output():
    from unittest.mock import MagicMock

    from cap.core.patch import ActivationPatcher

    patcher = MagicMock(spec=ActivationPatcher)
    patcher.hooks = []
    patcher.patch_specs = {}
    patcher.model = nn.Sequential(nn.Linear(8, 8))
    patcher.setup_patches = ActivationPatcher.setup_patches.__get__(patcher)
    patcher.setup_patches_custom = ActivationPatcher.setup_patches_custom.__get__(patcher)
    patcher.clear_patches = ActivationPatcher.clear_patches.__get__(patcher)

    patcher.setup_patches(patch_targets={"0": [0, 1]}, scale_factor=0.0)
    x = torch.ones(1, 8)
    out = patcher.model(x)
    assert out[0, 0].item() == pytest.approx(0.0, abs=1e-5)
    assert out[0, 1].item() == pytest.approx(0.0, abs=1e-5)
    patcher.clear_patches()
    assert patcher.hooks == []
    assert patcher.patch_specs == {}


def test_load_statistics_success(tmp_path):
    patcher = _bound_patcher()
    fake_stats = {"layer.0": {"cohens_d": np.zeros(3)}}
    with mock_patch("cap.core.patch.H5Store.load_statistics", return_value=fake_stats):
        result = patcher.load_statistics(h5_path=tmp_path)
    assert result is fake_stats
    assert patcher.layer_stats is fake_stats


def test_load_statistics_missing_raises(tmp_path):
    patcher = _bound_patcher()
    with (
        mock_patch("cap.core.patch.H5Store.load_statistics", return_value=None),
        pytest.raises(ValueError, match="No statistics"),
    ):
        patcher.load_statistics(h5_path=tmp_path)


def test_identify_before_load_raises():
    patcher = _bound_patcher()
    patcher.layer_stats = None
    with pytest.raises(ValueError, match="load_statistics"):
        patcher.identify_significant_neurons(d_threshold=1.0)


class _FourDimModule(nn.Module):
    """Returns a 4-D activation (b, s, h, d) to exercise the ndim==4 reshape branch."""

    def forward(self, x):
        return torch.ones(1, 2, 2, 2)


def test_hook_reshapes_4d_output():
    patcher = _bound_patcher()
    patcher.model = nn.Sequential(_FourDimModule())  # child module is named "0"
    # flattened channel dim is h*d = 4; zero out channels 0 and 1
    patcher.setup_patches(patch_targets={"0": [0, 1]}, scale_factor=0.0)
    out = patcher.model(torch.zeros(1))
    assert out.shape == (1, 2, 2, 2)
    flat = out.reshape(1, 2, 4)
    assert torch.all(flat[:, :, 0:2] == 0.0)
    assert torch.all(flat[:, :, 2:4] == 1.0)


class _TupleModule(nn.Module):
    """Returns (tensor, extra) so the tuple-output branch is exercised."""

    def forward(self, x):
        return torch.ones(1, 4), "extra"


def test_hook_preserves_tuple_output():
    patcher = _bound_patcher()
    patcher.model = nn.Sequential(_TupleModule())  # child module is named "0"
    patcher.setup_patches(patch_targets={"0": [0]}, scale_factor=0.0)
    out = patcher.model(torch.zeros(1))
    assert isinstance(out, tuple)
    tensor, extra = out
    assert extra == "extra"
    assert tensor[0, 0].item() == pytest.approx(0.0, abs=1e-5)


class _UnderscoreNamed(nn.Module):
    """Submodule whose name uses underscores, matched via the '_'->'.' fallback."""

    def __init__(self):
        super().__init__()
        self.layer_0 = nn.Linear(4, 4)

    def forward(self, x):
        return self.layer_0(x)


def test_hook_matches_underscore_normalized_key():
    patcher = _bound_patcher()
    patcher.model = _UnderscoreNamed()
    # patch spec keyed with a dot; module is named "layer_0" -> matched after replace
    patcher.setup_patches_custom(patch_dict={"layer.0": ([0, 1], 0.0)})
    assert len(patcher.hooks) == 1
    out = patcher.model(torch.ones(1, 4))
    assert out[0, 0].item() == pytest.approx(0.0, abs=1e-5)
    assert out[0, 1].item() == pytest.approx(0.0, abs=1e-5)
