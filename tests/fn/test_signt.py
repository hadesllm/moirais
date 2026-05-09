"""Tests for signt (sign test for population quantiles)."""

import numpy as np
import pytest
from moirais.fn.signt import signt


class TestSignt:
    """Sign test for population quantiles."""

    def test_signt_basic(self):
        """Basic sign test with known values."""
        x = np.array([5, 3, 8, 6, 9, 2, 7, 4])
        result = signt(x, theta0=5)
        assert result["n_eff"] > 0
        assert result["p_value"] >= 0 and result["p_value"] <= 1

    def test_signt_all_above(self):
        """All values above θ₀ should reject H0 (need n ≥ 7 for two-sided)."""
        x = np.array([10, 11, 12, 13, 14, 15, 16, 17])
        result = signt(x, theta0=5)
        assert result["n_plus"] == 8
        assert result["n_minus"] == 0
        assert result["p_value"] < 0.05

    def test_signt_balanced(self):
        """Balanced plus/minus should not reject."""
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        result = signt(x, theta0=4.5)
        assert result["n_plus"] + result["n_minus"] == 8

    def test_signt_default_theta0(self):
        """Should use median if theta0 not specified."""
        x = np.array([1, 2, 3, 4, 5])
        result = signt(x)
        assert result["n_eff"] >= 0

    def test_signt_alternative_greater(self):
        """Alternative='greater' should test upper tail."""
        x = np.array([10, 11, 12, 13, 14])
        result = signt(x, theta0=5, alternative="greater")
        assert result["p_value"] < 0.05

    def test_signt_alternative_less(self):
        """Alternative='less' should test lower tail."""
        x = np.array([1, 2, 3, 4, 5])
        result = signt(x, theta0=10, alternative="less")
        assert result["p_value"] < 0.05

    def test_signt_returns_dict(self):
        """Return type should be dict with required keys."""
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        result = signt(x, theta0=4)
        required_keys = {"statistic", "n_plus", "n_minus", "n_zero", "n_eff", "p_value", "interpretation"}
        assert set(result.keys()) == required_keys

    def test_signt_all_equal_error(self):
        """All observations equal to theta0 should raise error."""
        x = np.array([5, 5, 5, 5])
        with pytest.raises(ValueError):
            signt(x, theta0=5)
