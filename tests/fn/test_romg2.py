"""Tests for romg2 -- omega per subscale."""
import numpy as np
from moirais.fn.romg2 import omega_subscale
from moirais.fn._containers import ESRes


class TestOmegaSubscale:
    def test_basic(self):
        rng = np.random.default_rng(42)
        latent = rng.standard_normal(100)
        X = np.column_stack([latent + rng.standard_normal(100) * 0.3 for _ in range(5)])
        result = omega_subscale(X)
        assert isinstance(result, ESRes)
        assert result.estimate > 0.5

    def test_low_corr(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5))
        result = omega_subscale(X)
        assert result.estimate < 0.7
