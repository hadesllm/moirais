"""Test cosine_decompose (cosdc)."""
import numpy as np
from morie.fn.cosdc import cosine_decompose, cosdc
from morie.fn._containers import DescriptiveResult


class TestCosdc:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(64)
        result = cosine_decompose(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "cosine_decompose"
        assert result.value > 0

    def test_reconstruction(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(32)
        r = cosine_decompose(x)
        np.testing.assert_allclose(r.extra["reconstruction"], x, atol=1e-10)

    def test_energy_ratio(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        r = cosine_decompose(x)
        assert 0 <= r.extra["energy_ratio"] <= 1.0

    def test_parseval(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(64)
        r = cosine_decompose(x)
        np.testing.assert_allclose(r.value, np.sum(x ** 2), rtol=1e-6)

    def test_alias(self):
        assert cosdc is cosine_decompose
