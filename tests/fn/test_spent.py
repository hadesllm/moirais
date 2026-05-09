"""Test spectral_entropy (spent)."""
import numpy as np
from moirais.fn.spent import spectral_entropy, spent
from moirais.fn._containers import DescriptiveResult


class TestSpent:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = spectral_entropy(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "spectral_entropy"

    def test_positive(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = spectral_entropy(x)
        assert result.value > 0

    def test_alias(self):
        assert spent is spectral_entropy
