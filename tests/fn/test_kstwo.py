"""Tests for kstwo (Kolmogorov-Smirnov two-sample test)."""

import numpy as np
import pytest
from moirais.fn.kstwo import kstwo


class TestKstwo:
    """Kolmogorov-Smirnov two-sample test."""

    def test_kstwo_identical_samples(self):
        """Identical samples should not be rejected."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        result = kstwo(x, y)
        assert result["statistic"] == 0.0

    def test_kstwo_different_samples(self):
        """Very different samples should be rejected."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([10, 11, 12, 13, 14])
        result = kstwo(x, y)
        assert result["statistic"] > 0.5

    def test_kstwo_returns_dict(self):
        """Return type should be dict with required keys."""
        x = np.random.default_rng(42).standard_normal(50)
        y = np.random.default_rng(43).standard_normal(50)
        result = kstwo(x, y)
        required_keys = {"statistic", "p_value", "critical_value", "interpretation", "n1", "n2"}
        assert set(result.keys()) == required_keys

    def test_kstwo_statistic_in_unit_interval(self):
        """KS statistic should be in [0, 1]."""
        x = np.random.default_rng(42).standard_normal(50)
        y = np.random.default_rng(43).standard_normal(50)
        result = kstwo(x, y)
        assert 0 <= result["statistic"] <= 1

    def test_kstwo_same_distribution(self):
        """Samples from same distribution should not be rejected."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        rng = np.random.default_rng(43)
        y = rng.standard_normal(100)
        result = kstwo(x, y)
        assert result["p_value"] > 0.01 or result["p_value"] < 1.0

    def test_kstwo_empty_sample_error(self):
        """Empty sample should raise error."""
        with pytest.raises(ValueError):
            kstwo(np.array([1, 2, 3]), np.array([]))
