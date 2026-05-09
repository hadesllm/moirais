"""Tests for wsrkt (Wilcoxon signed-rank test)."""

import numpy as np
import pytest
from moirais.fn.wsrkt import wsrkt


class TestWsrkt:
    """Wilcoxon signed-rank test."""

    def test_wsrkt_one_sample_basic(self):
        """Basic one-sample test."""
        x = np.array([5, 3, 8, 6, 9, 2, 7, 4])
        result = wsrkt(x, theta0=5)
        assert result["n_eff"] > 0
        assert 0 <= result["p_value"] <= 1

    def test_wsrkt_paired_identical(self):
        """Paired test with identical samples should not reject."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        result = wsrkt(x, y=y)
        assert result["n_pos"] == 0
        assert result["n_neg"] == 0

    def test_wsrkt_paired_shifted(self):
        """Paired test with shifted samples should reject."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 3, 4, 5, 6])
        result = wsrkt(x, y=y)
        assert result["p_value"] < 0.05

    def test_wsrkt_one_sample_all_above(self):
        """All above θ₀ should reject."""
        x = np.array([10, 11, 12, 13, 14])
        result = wsrkt(x, theta0=5)
        assert result["p_value"] < 0.05

    def test_wsrkt_returns_dict(self):
        """Return type should be dict with required keys."""
        x = np.array([1, 2, 3, 4, 5])
        result = wsrkt(x, theta0=3)
        required_keys = {
            "statistic",
            "n_pos",
            "n_neg",
            "n_zero",
            "n_eff",
            "p_value",
            "interpretation",
        }
        assert set(result.keys()) == required_keys

    def test_wsrkt_all_equal_error(self):
        """All zeros should raise error."""
        x = np.array([5, 5, 5, 5])
        with pytest.raises(ValueError):
            wsrkt(x, theta0=5)

    def test_wsrkt_alternative_greater(self):
        """Alternative='greater'."""
        x = np.array([10, 11, 12, 13, 14])
        result = wsrkt(x, theta0=5, alternative="greater")
        assert result["p_value"] < 0.05

    def test_wsrkt_alternative_less(self):
        """Alternative='less'."""
        x = np.array([1, 2, 3, 4, 5])
        result = wsrkt(x, theta0=10, alternative="less")
        assert result["p_value"] < 0.05
