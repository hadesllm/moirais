"""Tests for moirais.fn.rmst — Restricted mean survival time."""
import numpy as np

from moirais.fn.rmst import rmst_estimate, rmst


def test_rmst_positive():
    """RMST should be positive."""
    rng = np.random.default_rng(42)
    time = rng.exponential(5, size=100)
    event = np.ones(100, dtype=int)
    result = rmst_estimate(time, event)
    assert result.value > 0


def test_rmst_less_than_tau():
    """RMST should be less than or equal to tau."""
    rng = np.random.default_rng(42)
    time = rng.exponential(5, size=100)
    event = np.ones(100, dtype=int)
    tau = 10.0
    result = rmst_estimate(time, event, tau=tau)
    assert result.value <= tau


def test_rmst_alias():
    assert rmst is rmst_estimate
