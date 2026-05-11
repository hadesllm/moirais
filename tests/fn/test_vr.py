"""Tests for morie.fn.vr -- variance ratio (F-test)."""

import numpy as np
import pytest
from morie.fn.vr import variance_ratio


class TestVarianceRatio:
    def test_equal_variances(self, rng):
        """Samples from same distribution should give F near 1."""
        x = rng.normal(0, 1, 100)
        y = rng.normal(0, 1, 100)
        result = variance_ratio(x, y)
        assert result.estimate == pytest.approx(1.0, rel=0.5)

    def test_different_variances(self, rng):
        """sd=1 vs sd=3 should give F near 1/9 or 9."""
        x = rng.normal(0, 1, 200)
        y = rng.normal(0, 3, 200)
        result = variance_ratio(x, y)
        # F = var(x)/var(y) ~ 1/9
        assert result.estimate < 0.3

    def test_has_p_value(self, rng):
        x = rng.normal(0, 1, 50)
        y = rng.normal(0, 2, 50)
        result = variance_ratio(x, y)
        assert "p_value" in result.extra
        assert 0 <= result.extra["p_value"] <= 1
