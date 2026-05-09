"""Tests for moirais.fn.grngr — Granger causality test."""
import numpy as np

from moirais.fn.grngr import granger_cause, grngr


def test_x_causes_y():
    """x(t) causes y(t+1): should reject."""
    rng = np.random.default_rng(42)
    n = 300
    x = rng.standard_normal(n)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.5 * x[i - 1] + rng.standard_normal() * 0.5
    result = granger_cause(x, y, max_lag=2)
    assert result.p_value < 0.05


def test_independent_not_rejected():
    """Two independent series: should not reject."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(300)
    y = rng.standard_normal(300)
    result = granger_cause(x, y, max_lag=2)
    assert result.p_value > 0.01


def test_grngr_alias():
    assert grngr is granger_cause
