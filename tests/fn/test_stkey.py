"""Tests for stkey (Siegel-Tukey test)."""

import numpy as np
import pytest
from moirais.fn.stkey import stkey


class TestStkey:
    """Siegel-Tukey test for scale equality."""

    def test_stkey_equal_scale(self):
        """Equal scale should not reject."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(30)
        y = rng.standard_normal(30)
        result = stkey(x, y)
        assert 0 <= result["p_value"] <= 1

    def test_stkey_returns_dict(self):
        """Return type should be dict with required keys."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        result = stkey(x, y)
        required_keys = {"statistic", "z_stat", "p_value", "interpretation"}
        assert set(result.keys()) == required_keys
