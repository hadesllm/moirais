"""Tests for ktaub (Kendall's tau-b)."""

import numpy as np
import pytest
from morie.fn.ktaub import ktaub


class TestKtaub:
    """Kendall's tau-b correlation."""

    def test_ktaub_perfect_positive(self):
        """Perfect positive correlation should have τ-b ≈ 1."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        result = ktaub(x, y)
        assert result["correlation"] > 0.9

    def test_ktaub_perfect_negative(self):
        """Perfect negative correlation should have τ-b ≈ -1."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 4, 3, 2, 1])
        result = ktaub(x, y)
        assert result["correlation"] < -0.9

    def test_ktaub_returns_dict(self):
        """Return type should be dict with required keys."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 3, 4, 5, 6])
        result = ktaub(x, y)
        required_keys = {"correlation", "p_value", "n", "concordant", "discordant"}
        assert set(result.keys()) == required_keys

    def test_ktaub_correlation_in_unit_interval(self):
        """Correlation should be in [-1, 1]."""
        x = np.random.default_rng(42).standard_normal(20)
        y = np.random.default_rng(43).standard_normal(20)
        result = ktaub(x, y)
        assert -1 <= result["correlation"] <= 1

    def test_ktaub_small_sample_error(self):
        """Sample size < 2 should raise error."""
        with pytest.raises(ValueError):
            ktaub(np.array([1]), np.array([2]))
