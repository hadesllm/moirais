"""Tests for morie.fn.mipol — multiple imputation pooling (Rubin's rules)."""
import numpy as np
import pytest
from morie.fn.mipol import mi_pool


class TestMIPool:
    def test_pooled_is_mean(self):
        estimates = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        variances = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        res = mi_pool(estimates, variances)
        assert abs(res.extra["pooled_estimate"] - 3.0) < 1e-10

    def test_too_few_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            mi_pool(np.array([1.0]), np.array([0.1]))

    def test_se_positive(self):
        rng = np.random.default_rng(42)
        estimates = rng.standard_normal(10) + 5.0
        variances = rng.uniform(0.01, 0.5, size=10)
        res = mi_pool(estimates, variances)
        assert res.extra["se"] > 0
        assert res.extra["m"] == 10
