"""Tests for morie.fn.adfts — Augmented Dickey-Fuller unit root test."""

import numpy as np
import pytest

from morie.fn.adfts import adf_test, adfts


def _ar1(phi: float, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.standard_normal()
    return x


def test_returns_test_result():
    """Return type has the standard TestResult interface."""
    x = _ar1(0.5, 100, seed=10)
    r = adf_test(x)
    assert hasattr(r, "statistic")
    assert hasattr(r, "p_value")
    assert 0.0 <= r.p_value <= 1.0
    assert isinstance(r.statistic, float)


def test_random_walk_high_pvalue():
    """Unit root (phi=1) — ADF should NOT reject H0 (p_value >= 0.05)."""
    x = _ar1(1.0, 150, seed=1)
    result = adf_test(x, regression="c")
    assert result.p_value >= 0.05, f"Expected p >= 0.05 for RW, got {result.p_value}"


def test_stationary_series_low_pvalue():
    """Stationary AR(1) (phi=0.2) — ADF should reject H0 (p_value <= 0.10)."""
    x = _ar1(0.2, 200, seed=2)
    result = adf_test(x, regression="c")
    assert result.p_value <= 0.10, f"Expected p <= 0.10, got {result.p_value}"


def test_statistic_negative_for_stationary():
    """ADF t-statistic should be strongly negative for a stationary series."""
    x = _ar1(0.1, 200, seed=3)
    result = adf_test(x)
    assert result.statistic < -2.0


def test_regression_types_all_run():
    x = _ar1(0.5, 100, seed=4)
    for reg in ("c", "ct", "nc"):
        r = adf_test(x, regression=reg)
        assert np.isfinite(r.statistic)


def test_critical_values_in_extra():
    x = _ar1(0.5, 80, seed=5)
    result = adf_test(x)
    cv = result.extra["critical_values"]
    assert "1%" in cv and "5%" in cv and "10%" in cv
    # 1% < 5% < 10% (more negative = stronger evidence against unit root)
    assert cv["1%"] < cv["5%"] < cv["10%"]


def test_invalid_regression_raises():
    with pytest.raises(ValueError):
        adf_test([1.0, 2.0, 3.0, 4.0, 5.0], regression="xy")


def test_too_short_raises():
    with pytest.raises(ValueError):
        adf_test([1.0, 2.0, 3.0])


def test_alias():
    assert adfts is adf_test
