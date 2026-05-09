"""Test ceemdan (ceemd)."""
import numpy as np
from moirais.fn.ceemd import ceemdan, ceemd
from moirais.fn._containers import DescriptiveResult


class TestCeemd:
    def test_basic(self):
        t = np.linspace(0, 1, 128)
        x = np.sin(2 * np.pi * 3 * t) + 0.5 * np.sin(2 * np.pi * 15 * t)
        result = ceemdan(x, num_sifts=10, seed=42, max_imfs=3)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "ceemdan"
        assert result.value >= 1

    def test_residual(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(64)
        r = ceemdan(x, num_sifts=5, seed=0, max_imfs=2)
        recon = np.sum(r.extra["imfs"], axis=0) + r.extra["residual"]
        np.testing.assert_allclose(recon, x, atol=0.5)

    def test_alias(self):
        assert ceemd is ceemdan
