"""Tests for morie.fn.aalen -- Aalen additive hazards model."""

import numpy as np
import pytest

from morie.fn.aalen import aalen


@pytest.fixture()
def surv_data():
    rng = np.random.default_rng(42)
    n = 80
    X = rng.standard_normal((n, 2))
    t = rng.exponential(5, n)
    c = rng.exponential(8, n)
    time = np.minimum(t, c)
    event = (t <= c).astype(float)
    return time, event, X


def test_returns_dict(surv_data):
    time, event, X = surv_data
    r = aalen(time, event, X)
    assert isinstance(r, dict)
    for k in ("times", "cumulative_coefficients", "se", "n_obs", "n_events"):
        assert k in r


def test_times_sorted(surv_data):
    time, event, X = surv_data
    r = aalen(time, event, X)
    assert np.all(np.diff(r["times"]) > 0)


def test_cum_coef_shape(surv_data):
    time, event, X = surv_data
    r = aalen(time, event, X)
    assert r["cumulative_coefficients"].shape[1] == 3


def test_cheatsheet():
    from morie.fn.aalen import cheatsheet
    assert "aalen" in cheatsheet().lower()
