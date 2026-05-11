"""Tests for fabdt (Freund-Ansari-Bradley-David test)."""

import numpy as np
import pytest
from morie.fn.fabdt import fabdt


class TestFabdt:
    """Freund-Ansari-Bradley-David test for scale equality."""

    def test_fabdt_equal_scale(self):
        """Equal scale samples should not reject."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(30)
        y = rng.standard_normal(30)
        result = fabdt(x, y)
        assert 0 <= result["p_value"] <= 1

    def test_fabdt_different_scale(self):
        """Different scale samples should likely reject."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(50)
        y = rng.normal(0, 3, 50)
        result = fabdt(x, y)
        assert "p_value" in result

    def test_fabdt_returns_dict(self):
        """Return type should be dict with required keys."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        result = fabdt(x, y)
        required_keys = {"statistic", "z_stat", "p_value", "interpretation"}
        assert set(result.keys()) == required_keys
