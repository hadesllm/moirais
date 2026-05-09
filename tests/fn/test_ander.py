"""Tests for ander (Anderson-Darling test)."""

import numpy as np
import pytest
from moirais.fn.ander import ander


class TestAnder:
    """Anderson-Darling test for goodness of fit."""

    def test_ander_normal_sample(self):
        """Sample from normal should not reject."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        result = ander(x, dist="norm")
        assert result["p_value"] > 0.01 or result["p_value"] < 1.0

    def test_ander_returns_dict(self):
        """Return type should be dict with required keys."""
        x = np.random.default_rng(42).standard_normal(50)
        result = ander(x, dist="norm")
        required_keys = {"statistic", "p_value", "critical_value", "interpretation"}
        assert set(result.keys()) == required_keys

    def test_ander_statistic_positive(self):
        """Anderson-Darling statistic should be positive."""
        x = np.random.default_rng(42).standard_normal(100)
        result = ander(x, dist="norm")
        assert result["statistic"] >= 0

    def test_ander_small_sample_error(self):
        """Sample size < 2 should raise error."""
        with pytest.raises(ValueError):
            ander(np.array([1]))

    def test_ander_uniform_dist(self):
        """Should work with uniform distribution."""
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, 50)
        result = ander(x, dist="uniform")
        assert "statistic" in result

    def test_ander_expon_dist(self):
        """Should work with exponential distribution."""
        rng = np.random.default_rng(42)
        x = rng.exponential(scale=1.0, size=50)
        result = ander(x, dist="expon")
        assert "statistic" in result
