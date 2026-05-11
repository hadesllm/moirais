"""Tests for morie.fn.bhawz -- Bayesian hazard."""

import numpy as np
from morie.fn.bhawz import bayesian_hazard


def test_returns_dict():
    times = np.random.default_rng(42).exponential(5, 50)
    events = np.ones(50)
    result = bayesian_hazard(times, events, n_intervals=5)
    assert isinstance(result, dict)
    assert "hazard" in result


def test_survival_decreasing():
    times = np.random.default_rng(42).exponential(5, 100)
    events = np.ones(100)
    result = bayesian_hazard(times, events, n_intervals=5)
    surv = result["survival"]
    for i in range(1, len(surv)):
        assert surv[i] <= surv[i - 1] + 1e-10


def test_hazard_positive():
    times = np.random.default_rng(42).exponential(5, 50)
    events = np.ones(50)
    result = bayesian_hazard(times, events, n_intervals=5)
    assert all(h >= 0 for h in result["hazard"])
