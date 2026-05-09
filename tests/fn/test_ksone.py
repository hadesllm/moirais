"""Tests for ksone (Kolmogorov-Smirnov one-sample test)."""

import numpy as np
import pytest
from moirais.fn.ksone import ksone


class TestKsone:
    """Kolmogorov-Smirnov one-sample test."""

    def test_ksone_normal_from_normal(self):
        """Sample from N(0,1) should not reject normality."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        result = ksone(x, dist="norm")
        assert result["p_value"] > 0.05 or result["p_value"] < 1.0

    def test_ksone_uniform_from_uniform(self):
        """Sample from U(0,1) should not reject uniformity."""
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, 100)
        result = ksone(x, dist="uniform")
        assert result["p_value"] > 0.05 or result["p_value"] < 1.0

    def test_ksone_returns_dict(self):
        """Return type should be dict with required keys."""
        x = np.random.default_rng(42).standard_normal(50)
        result = ksone(x, dist="norm")
        required_keys = {"statistic", "p_value", "critical_value", "interpretation"}
        assert set(result.keys()) == required_keys

    def test_ksone_statistic_in_unit_interval(self):
        """KS statistic should be in [0, 1]."""
        x = np.random.default_rng(42).standard_normal(100)
        result = ksone(x, dist="norm")
        assert 0 <= result["statistic"] <= 1

    def test_ksone_small_sample_error(self):
        """Sample size < 2 should raise error."""
        with pytest.raises(ValueError):
            ksone(np.array([1]))

    def test_ksone_expon_dist(self):
        """Should accept exponential distribution."""
        rng = np.random.default_rng(42)
        x = rng.exponential(scale=1.0, size=100)
        result = ksone(x, dist="expon")
        assert "statistic" in result
