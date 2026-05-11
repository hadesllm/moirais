"""Tests for morie.fn.cox — Cox proportional hazards."""
import numpy as np

from morie.fn.cox import cox_ph, cox


def test_cox_positive_hazard_ratio():
    """Positive covariate effect should give HR > 1."""
    rng = np.random.default_rng(42)
    n = 200
    x = rng.standard_normal(n)
    # Higher x -> shorter time (higher hazard)
    time = rng.exponential(np.exp(-0.5 * x))
    event = np.ones(n, dtype=int)
    result = cox_ph(time, event, x)
    assert result.extra["hazard_ratios"]["x0"] > 1.0
    assert result.n == n


def test_cox_converges():
    rng = np.random.default_rng(42)
    n = 100
    x = rng.standard_normal(n)
    time = rng.exponential(1, size=n)
    event = rng.integers(0, 2, size=n)
    result = cox_ph(time, event, x)
    assert result.extra["converged"]


def test_cox_alias():
    assert cox is cox_ph
