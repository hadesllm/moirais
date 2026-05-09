"""Tests for moirais.fn.kpss — KPSS stationarity test."""
import numpy as np
import pytest

from moirais.fn.kpss import kpss_test, kpss


def test_stationary_series():
    """Stationary white noise: KPSS should not reject (p > 0.05)."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(200)
    result = kpss_test(x)
    assert result.p_value > 0.05


def test_trending_series():
    """Linear trend: KPSS should reject stationarity."""
    rng = np.random.default_rng(42)
    x = np.arange(200, dtype=float) + rng.standard_normal(200) * 0.5
    result = kpss_test(x)
    assert result.p_value <= 0.05


def test_kpss_alias():
    assert kpss is kpss_test


def test_too_few_obs():
    with pytest.raises(ValueError):
        kpss_test(np.ones(5))
