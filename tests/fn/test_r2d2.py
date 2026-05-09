"""Tests for moirais.fn.r2d2 — alias for R-squared."""
import numpy as np

from moirais.fn.r2d2 import r2d2, r_squared


def test_r2d2_is_callable():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    y = 2 * x + rng.standard_normal(50) * 0.5
    result = r2d2(x, y)
    assert hasattr(result, "estimate")
    assert result.estimate > 0


def test_r_squared_alias():
    assert r2d2 is r_squared
