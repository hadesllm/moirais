"""Tests for rboot -- bootstrap reliability."""
import numpy as np
from morie.fn.rboot import bootstrap_reliability
from morie.fn._containers import ESRes


class TestBootstrapReliability:
    def test_basic(self):
        rng = np.random.default_rng(42)
        latent = rng.standard_normal(100)
        X = np.column_stack([latent + rng.standard_normal(100) * 0.3 for _ in range(5)])
        result = bootstrap_reliability(X, n_boot=100)
        assert isinstance(result, ESRes)
        assert result.ci_lower <= result.estimate <= result.ci_upper

    def test_custom_fn(self):
        rng = np.random.default_rng(42)
        X = rng.integers(1, 6, (50, 3)).astype(float)
        result = bootstrap_reliability(X, stat_fn=lambda d: np.mean(d), n_boot=50)
        assert result.se > 0
