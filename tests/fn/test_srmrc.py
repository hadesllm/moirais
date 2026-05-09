"""Tests for srmrc (Spearman rank correlation)."""

import numpy as np
import pytest
from moirais.fn.srmrc import srmrc


class TestSrmrc:
    """Spearman rank correlation with CI."""

    def test_srmrc_perfect_positive(self):
        """Perfect monotone increasing should have ρ ≈ 1."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        result = srmrc(x, y)
        assert result["correlation"] > 0.99

    def test_srmrc_ci_ordered(self):
        """CI lower should be ≤ upper."""
        x = np.random.default_rng(42).standard_normal(20)
        y = np.random.default_rng(43).standard_normal(20)
        result = srmrc(x, y)
        assert result["ci_lower"] <= result["ci_upper"]

    def test_srmrc_returns_dict(self):
        """Return type should be dict with required keys."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 4, 3, 2, 1])
        result = srmrc(x, y)
        required_keys = {"correlation", "p_value", "ci_lower", "ci_upper", "n"}
        assert set(result.keys()) == required_keys

    def test_srmrc_small_sample_error(self):
        """Sample size < 3 should raise error."""
        with pytest.raises(ValueError):
            srmrc(np.array([1, 2]), np.array([1, 2]))
