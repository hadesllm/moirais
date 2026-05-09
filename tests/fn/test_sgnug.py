"""Tests for nugget effect estimation."""
import numpy as np
from moirais.fn.sgnug import sgnug


def test_sgnug_smoke():
    lags = np.array([1, 2, 3, 4, 5], dtype=float)
    gamma = np.array([0.5, 0.8, 0.95, 1.0, 1.0], dtype=float)
    r = sgnug(gamma, lags)
    assert r.name == "nugget_effect_estimate"
    assert "nugget" in r.extra
    assert "sill" in r.extra
    assert "range" in r.extra


def test_sgnug_sill():
    lags = np.array([1, 2, 3, 4, 5], dtype=float)
    gamma = np.array([0.2, 0.6, 0.9, 1.0, 1.0], dtype=float)
    r = sgnug(gamma, lags)
    assert r.extra["sill"] == 1.0
