"""Test variational_mode (vmd)."""
import numpy as np
from morie.fn.vmd import variational_mode, vmd
from morie.fn._containers import DescriptiveResult


class TestVmd:
    def test_basic(self):
        t = np.linspace(0, 1, 256)
        x = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 20 * t)
        result = variational_mode(x, K=2, alpha=2000, max_iter=50)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "variational_mode"
        assert result.value == 2

    def test_modes_shape(self):
        x = np.random.default_rng(42).standard_normal(128)
        r = variational_mode(x, K=3, max_iter=30)
        assert r.extra["modes"].shape == (3, 128)
        assert len(r.extra["center_frequencies"]) == 3

    def test_alias(self):
        assert vmd is variational_mode
