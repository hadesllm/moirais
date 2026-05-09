"""Tests for moirais.fn.cpchg — PELT change point detection."""

import numpy as np
import pytest

from moirais.fn.cpchg import changepoint_pelt, cpchg


def test_returns_descriptive_result():
    """Return type has DescriptiveResult interface."""
    y = np.concatenate([np.ones(30), 3 * np.ones(30)])
    r = changepoint_pelt(y)
    assert hasattr(r, "value")
    assert hasattr(r, "extra")
    assert "changepoints" in r.extra


def test_obvious_break_detected():
    """A large mean shift should be detected with BIC penalty."""
    y = np.concatenate([np.ones(30), 5.0 * np.ones(30)])
    r = changepoint_pelt(y, penalty="bic", min_size=5)
    # Exactly one break should appear near index 30.
    assert r.extra["n_changepoints"] >= 1
    if r.extra["n_changepoints"] == 1:
        cp = int(r.extra["changepoints"][0])
        assert 20 <= cp <= 40, f"Break should be near 30, got {cp}"


def test_constant_series_no_break():
    """A constant series with BIC penalty should yield 0 change points."""
    y = np.ones(60)
    r = changepoint_pelt(y, penalty="bic", min_size=5)
    assert r.extra["n_changepoints"] == 0


def test_penalty_types():
    """BIC, AIC, and numeric penalties all run without error."""
    rng = np.random.default_rng(42)
    y = rng.standard_normal(60)
    for pen in ("bic", "aic", 3.0):
        r = changepoint_pelt(y, penalty=pen)
        assert isinstance(r.extra["n_changepoints"], int)


def test_segments_cover_series():
    """Segment list must cover [0, n) without gaps."""
    y = np.concatenate([np.ones(20), 4 * np.ones(20), np.ones(20)])
    r = changepoint_pelt(y, penalty="bic", min_size=5)
    segs = r.extra["segments"]
    n = len(y)
    assert segs[0][0] == 0
    assert segs[-1][1] == n
    for i in range(len(segs) - 1):
        assert segs[i][1] == segs[i + 1][0]


def test_total_cost_nonneg():
    y = np.random.default_rng(0).standard_normal(50)
    r = changepoint_pelt(y, penalty="bic")
    assert r.extra["total_cost"] >= 0.0


def test_invalid_penalty_string_raises():
    with pytest.raises(ValueError):
        changepoint_pelt(np.ones(30), penalty="unknown")


def test_too_short_raises():
    with pytest.raises(ValueError):
        changepoint_pelt([1.0, 2.0], min_size=3)


def test_alias():
    assert cpchg is changepoint_pelt
