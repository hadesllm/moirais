"""Tests for kwone (Kruskal-Wallis one-way ANOVA)."""

import numpy as np
import pytest
from morie.fn.kwone import kwone


class TestKwone:
    """Kruskal-Wallis one-way ANOVA."""

    def test_kwone_identical_groups(self):
        """Identical groups should not reject."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        z = np.array([1, 2, 3, 4, 5])
        result = kwone(x, y, z)
        assert result["p_value"] > 0.05

    def test_kwone_different_groups(self):
        """Very different groups should reject."""
        x = np.array([1, 2, 3])
        y = np.array([10, 11, 12])
        z = np.array([20, 21, 22])
        result = kwone(x, y, z)
        assert result["p_value"] < 0.05

    def test_kwone_returns_dict(self):
        """Return type should be dict with required keys."""
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        result = kwone(x, y)
        required_keys = {"statistic", "p_value", "k", "n_total", "interpretation"}
        assert set(result.keys()) == required_keys

    def test_kwone_single_sample_error(self):
        """Fewer than 2 samples should raise error."""
        with pytest.raises(ValueError):
            kwone(np.array([1, 2, 3]))
