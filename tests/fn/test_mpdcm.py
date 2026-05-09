"""Test matching_pursuit_decompose (mpdcm)."""
import numpy as np
from moirais.fn.mpdcm import matching_pursuit_decompose, mpdcm
from moirais.fn._containers import DescriptiveResult


class TestMpdcm:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(32)
        result = matching_pursuit_decompose(x, sparsity=5)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "matching_pursuit_decompose"
        assert result.value >= 0

    def test_residual_decreases(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(32)
        r1 = matching_pursuit_decompose(x, sparsity=2)
        r2 = matching_pursuit_decompose(x, sparsity=10)
        assert r2.value <= r1.value + 1e-10

    def test_approximation(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(16)
        r = matching_pursuit_decompose(x, sparsity=5)
        recon = r.extra["approximation"] + r.extra["residual"]
        np.testing.assert_allclose(recon, x, atol=1e-10)

    def test_custom_dictionary(self):
        rng = np.random.default_rng(42)
        n = 10
        D = np.eye(n)
        x = rng.standard_normal(n)
        r = matching_pursuit_decompose(x, dictionary=D, sparsity=3)
        assert len(r.extra["atoms_used"]) == 3

    def test_alias(self):
        assert mpdcm is matching_pursuit_decompose
