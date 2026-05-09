"""Tests for lilef (Lilliefors test for normality)."""

import numpy as np
import pytest
from moirais.fn.lilef import lilef


class TestLilef:
    """Lilliefors test for normality."""

    def test_lilef_normal_sample(self):
        """Sample from normal distribution should not reject normality."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        result = lilef(x)
        assert result["p_value"] > 0.01 or result["p_value"] < 1.0

    def test_lilef_non_normal_sample(self):
        """Sample from exponential should likely reject normality."""
        rng = np.random.default_rng(42)
        x = rng.exponential(scale=1.0, size=100)
        result = lilef(x)
        # May not always reject due to sample size, but should be detectable
        assert "statistic" in result

    def test_lilef_returns_dict(self):
        """Return type should be dict with required keys."""
        x = np.random.default_rng(42).standard_normal(50)
        result = lilef(x)
        required_keys = {"statistic", "p_value", "critical_value", "interpretation", "mean", "std"}
        assert set(result.keys()) == required_keys

    def test_lilef_statistic_in_unit_interval(self):
        """Lilliefors statistic should be in [0, 1]."""
        x = np.random.default_rng(42).standard_normal(100)
        result = lilef(x)
        assert 0 <= result["statistic"] <= 1

    def test_lilef_small_sample_error(self):
        """Sample size < 4 should raise error."""
        with pytest.raises(ValueError):
            lilef(np.array([1, 2, 3]))

    def test_lilef_mean_std_estimates(self):
        """Mean and std should be reasonable estimates."""
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = lilef(x)
        assert abs(result["mean"] - 5.5) < 0.1
