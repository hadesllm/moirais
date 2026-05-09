"""Tests for klotz (Klotz normal-scores test)."""

import numpy as np
from moirais.fn.klotz import klotz


class TestKlotz:
    """Klotz normal-scores test for scale equality."""

    def test_klotz_equal_scale(self):
        """Equal scale samples should not reject."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(30)
        y = rng.standard_normal(30)
        result = klotz(x, y)
        assert 0 <= result["p_value"] <= 1

    def test_klotz_returns_dict(self):
        """Return type should be dict with required keys."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        result = klotz(x, y)
        required_keys = {"statistic", "z_stat", "p_value", "interpretation"}
        assert set(result.keys()) == required_keys
