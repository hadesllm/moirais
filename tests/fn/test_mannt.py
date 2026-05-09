"""Tests for mannt (Mann-Whitney U test)."""

import numpy as np
import pytest
from moirais.fn.mannt import mannt


class TestMannt:
    """Mann-Whitney U test for two independent samples."""

    def test_mannt_identical_samples(self):
        """Identical samples should not be rejected."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        result = mannt(x, y)
        assert result["p_value"] > 0.05

    def test_mannt_different_samples(self):
        """Very different samples should be rejected."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([10, 11, 12, 13, 14])
        result = mannt(x, y)
        assert result["p_value"] < 0.05

    def test_mannt_returns_dict(self):
        """Return type should be dict with required keys."""
        x = np.random.default_rng(42).standard_normal(50)
        y = np.random.default_rng(43).standard_normal(50)
        result = mannt(x, y)
        required_keys = {"statistic", "U_x", "U_y", "n_x", "n_y", "p_value", "interpretation"}
        assert set(result.keys()) == required_keys

    def test_mannt_u_values_valid(self):
        """U_x + U_y should equal n_x * n_y."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([6, 7, 8, 9, 10])
        result = mannt(x, y)
        assert abs(result["U_x"] + result["U_y"] - result["n_x"] * result["n_y"]) < 0.01

    def test_mannt_alternative_greater(self):
        """Alternative='greater' for right-shifted x."""
        x = np.array([10, 11, 12, 13, 14])
        y = np.array([1, 2, 3, 4, 5])
        result = mannt(x, y, alternative="greater")
        assert result["p_value"] < 0.05

    def test_mannt_alternative_less(self):
        """Alternative='less' for left-shifted x."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([10, 11, 12, 13, 14])
        result = mannt(x, y, alternative="less")
        assert result["p_value"] < 0.05

    def test_mannt_empty_sample_error(self):
        """Empty sample should raise error."""
        with pytest.raises(ValueError):
            mannt(np.array([1, 2, 3]), np.array([]))
