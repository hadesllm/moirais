"""Tests for moirais.fn.durat — Semiparametric duration model."""

import numpy as np
import pytest
from moirais.fn.durat import durat


def test_returns_dict():
    rng = np.random.default_rng(42)
    t = rng.exponential(1, 100)
    result = durat(t)
    assert isinstance(result, dict)
    for key in ("t_grid", "hazard", "survival", "cumulative_hazard", "bandwidth", "n_obs", "n_events"):
        assert key in result


def test_survival_decreasing():
    rng = np.random.default_rng(42)
    t = rng.exponential(2, 200)
    result = durat(t)
    s = np.asarray(result["survival"])
    assert s[0] >= s[-1]


def test_hazard_nonnegative():
    rng = np.random.default_rng(42)
    t = rng.exponential(1, 100)
    result = durat(t)
    assert all(h >= -1e-10 for h in result["hazard"])


def test_negative_times_raises():
    with pytest.raises(ValueError, match="non-negative"):
        durat(np.array([-1, 2, 3, 4, 5]))


def test_censoring():
    rng = np.random.default_rng(42)
    t = rng.exponential(1, 50)
    event = np.ones(50)
    event[40:] = 0
    result = durat(t, event=event)
    assert result["n_events"] == 40
