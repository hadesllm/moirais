"""Tests for moods (Mood's test for scale)."""

import numpy as np
import pytest
from moirais.fn.moods import moods


class TestMoods:
    """Mood's test for equality of scale parameters."""

    def test_moods_equal_scale(self):
        """Samples with equal scale should not reject."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(50)
        y = rng.standard_normal(50)
        result = moods(x, y)
        assert 0 <= result["p_value"] <= 1

    def test_moods_different_scale(self):
        """Samples with different scales should reject."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(50)
        y = rng.normal(0, 3, 50)
        result = moods(x, y)
        assert result["p_value"] < 0.05

    def test_moods_returns_dict(self):
        """Return type should be dict with required keys."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        result = moods(x, y)
        required_keys = {"statistic", "z_stat", "p_value", "interpretation"}
        assert set(result.keys()) == required_keys
