"""Test variational_mode_decompose (vmdfn)."""
import numpy as np
from moirais.fn.vmdfn import variational_mode_decompose, vmdfn
from moirais.fn._containers import DescriptiveResult


class TestVmdfn:
    def test_basic(self):
        rng = np.random.default_rng(42)
        t = np.linspace(0, 1, 256)
        x = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 20 * t) + rng.standard_normal(256) * 0.1
        result = variational_mode_decompose(x, K=2)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "variational_mode_decompose"
        assert result.value == 2

    def test_modes_shape(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(128)
        r = variational_mode_decompose(x, K=3)
        assert r.extra["modes"].shape == (3, 128)
        assert len(r.extra["center_frequencies"]) == 3

    def test_single_mode(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(64)
        r = variational_mode_decompose(x, K=1)
        assert r.value == 1

    def test_alias(self):
        assert vmdfn is variational_mode_decompose
