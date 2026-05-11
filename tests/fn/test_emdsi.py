"""Test emd_sifting (emdsi)."""
import numpy as np
from morie.fn.emdsi import emd_sifting, emdsi
from morie.fn._containers import DescriptiveResult


class TestEmdsi:
    def test_basic(self):
        t = np.linspace(0, 1, 128)
        x = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 20 * t)
        result = emd_sifting(x, max_iter=50)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "emd_sifting"
        assert result.value >= 1

    def test_imf_plus_residual(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(64)
        r = emd_sifting(x)
        recon = r.extra["imf"] + r.extra["residual"]
        np.testing.assert_allclose(recon, x, atol=1e-10)

    def test_alias(self):
        assert emdsi is emd_sifting
