"""Tests for rform -- parallel forms reliability."""
import numpy as np
from moirais.fn.rform import parallel_form_reliability
from moirais.fn._containers import ESRes


class TestParallelForms:
    def test_high_corr(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal(100)
        b = a + rng.standard_normal(100) * 0.1
        result = parallel_form_reliability(a, b)
        assert isinstance(result, ESRes)
        assert result.estimate > 0.9

    def test_ci_covers_estimate(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal(50)
        b = a + rng.standard_normal(50) * 0.5
        result = parallel_form_reliability(a, b)
        assert result.ci_lower <= result.estimate <= result.ci_upper
