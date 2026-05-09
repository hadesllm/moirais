"""Tests for moirais.fn.grnge — Granger causality test."""

import numpy as np
import pytest

from moirais.fn.grnge import granger_test, grnge


def test_returns_test_result():
    """Return type has the standard TestResult interface."""
    rng = np.random.default_rng(0)
    y1 = rng.standard_normal(100)
    y2 = rng.standard_normal(100)
    r = granger_test(y1, y2, maxlag=2)
    assert hasattr(r, "statistic")
    assert hasattr(r, "p_value")
    assert 0.0 <= r.p_value <= 1.0
    assert r.df == 2.0


def test_independent_series_not_causal():
    """Unrelated series — Granger test should NOT reject H0 (p > 0.05)."""
    rng = np.random.default_rng(42)
    y1 = rng.standard_normal(200)
    y2 = rng.standard_normal(200)
    r = granger_test(y1, y2, maxlag=2)
    assert r.p_value > 0.05, f"Expected p > 0.05 for unrelated series, got {r.p_value}"


def test_causal_series_detected():
    """y1[t] = 0.8 * y2[t-1] + noise — y2 should Granger-cause y1 (p < 0.05)."""
    rng = np.random.default_rng(1)
    n = 200
    y2 = rng.standard_normal(n)
    eps = rng.standard_normal(n) * 0.1
    y1 = np.zeros(n)
    y1[1:] = 0.8 * y2[:-1] + eps[1:]
    r = granger_test(y1, y2, maxlag=2)
    assert r.p_value < 0.05, f"Expected p < 0.05 for Granger-causal series, got {r.p_value}"


def test_f_statistic_nonneg():
    """F-statistic must be non-negative."""
    rng = np.random.default_rng(3)
    y1 = rng.standard_normal(100)
    y2 = rng.standard_normal(100)
    r = granger_test(y1, y2, maxlag=3)
    assert r.statistic >= 0.0


def test_extra_fields():
    """Extra dict contains rss_restricted and rss_unrestricted."""
    rng = np.random.default_rng(4)
    y1 = rng.standard_normal(80)
    y2 = rng.standard_normal(80)
    r = granger_test(y1, y2, maxlag=2)
    assert "rss_restricted" in r.extra
    assert "rss_unrestricted" in r.extra
    # Restricted model must have RSS >= unrestricted model.
    assert r.extra["rss_restricted"] >= r.extra["rss_unrestricted"] - 1e-9


def test_mismatched_length_raises():
    y1 = np.ones(50)
    y2 = np.ones(49)
    with pytest.raises(ValueError):
        granger_test(y1, y2)


def test_too_short_raises():
    y1 = np.ones(5)
    y2 = np.ones(5)
    with pytest.raises(ValueError):
        granger_test(y1, y2, maxlag=4)


def test_alias():
    assert grnge is granger_test
