"""Tests for vconv -- convergent validity."""
import numpy as np
from moirais.fn.vconv import convergent_validity
from moirais.fn._containers import ESRes


class TestConvergentValidity:
    def test_high_corr(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        y = x + rng.standard_normal(100) * 0.1
        result = convergent_validity(x, y)
        assert isinstance(result, ESRes)
        assert result.estimate > 0.9

    def test_ci_covers(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(50)
        y = x + rng.standard_normal(50) * 0.5
        result = convergent_validity(x, y)
        assert result.ci_lower <= result.estimate <= result.ci_upper
