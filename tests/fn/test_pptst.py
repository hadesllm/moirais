"""Tests for morie.fn.pptst — Phillips-Perron unit root test."""

import numpy as np
import pytest

from morie.fn.pptst import pp_test, pptst


def _ar1(phi: float, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.standard_normal()
    return x


def test_returns_test_result():
    """Return type has the standard TestResult interface."""
    x = _ar1(0.5, 100, seed=10)
    r = pp_test(x)
    assert hasattr(r, "statistic")
    assert hasattr(r, "p_value")
    assert 0.0 <= r.p_value <= 1.0


def test_random_walk_high_pvalue():
    """Unit root series — PP should NOT reject H0 (p >= 0.05)."""
    x = _ar1(1.0, 200, seed=1)
    r = pp_test(x)
    assert r.p_value >= 0.05, f"Expected p >= 0.05 for RW, got {r.p_value}"


def test_stationary_ar1_rejected():
    """Stationary AR(1) — PP should reject H0 (p <= 0.10)."""
    x = _ar1(0.2, 200, seed=2)
    r = pp_test(x)
    assert r.p_value <= 0.10, f"Expected p <= 0.10, got {r.p_value}"


def test_explicit_lags():
    """Explicit lag count is stored in extra."""
    x = _ar1(0.5, 100, seed=3)
    r = pp_test(x, lags=5)
    assert r.extra["lags"] == 5


def test_critical_values_present():
    x = _ar1(0.5, 80, seed=4)
    r = pp_test(x)
    cv = r.extra["critical_values"]
    assert "1%" in cv and "5%" in cv and "10%" in cv
    assert cv["1%"] < cv["5%"] < cv["10%"]


def test_lrvar_positive():
    """Long-run variance estimate should be positive."""
    x = _ar1(0.4, 100, seed=5)
    r = pp_test(x)
    assert r.extra["lrvar"] > 0.0


def test_too_short_raises():
    with pytest.raises(ValueError):
        pp_test([1.0, 2.0, 3.0])


def test_alias():
    assert pptst is pp_test
