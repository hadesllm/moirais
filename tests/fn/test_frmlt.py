"""Test framelet_decompose (frmlt)."""
import numpy as np
import pytest
from morie.fn.frmlt import framelet_decompose, frmlt
from morie.fn._containers import DescriptiveResult


class TestFrmlt:
    def test_basic_haar(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(32)
        result = framelet_decompose(x, frame_type="haar")
        assert isinstance(result, DescriptiveResult)
        assert result.name == "framelet_decompose"
        assert result.value >= 2

    def test_linear_frame(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(32)
        r = framelet_decompose(x, frame_type="linear")
        assert r.extra["frame_type"] == "linear"
        assert r.value >= 3

    def test_low_band_exists(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(16)
        r = framelet_decompose(x)
        assert r.extra["low"] is not None
        assert len(r.extra["high_bands"]) >= 1

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            framelet_decompose(np.ones(16), frame_type="unknown")

    def test_alias(self):
        assert frmlt is framelet_decompose
