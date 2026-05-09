"""Test morphological_ca (mca)."""
import numpy as np
from moirais.fn.mca import morphological_ca, mca
from moirais.fn._containers import DescriptiveResult


class TestMca:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = np.sin(np.linspace(0, 4 * np.pi, 128)) + 0.3 * rng.standard_normal(128)
        result = morphological_ca(x, n_iter=20)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "morphological_ca"

    def test_decomposition_sums(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(64)
        r = morphological_ca(x, n_iter=50)
        recon = r.extra["smooth"] + r.extra["transient"]
        np.testing.assert_allclose(recon, x, atol=1e-10)

    def test_alias(self):
        assert mca is morphological_ca
