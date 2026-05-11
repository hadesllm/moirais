"""Tests for morie.fn.phreg -- Piecewise constant hazard regression."""

import numpy as np
import pytest

from morie.fn.phreg import phreg


@pytest.fixture()
def surv_data():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 2))
    t = rng.exponential(5, n)
    c = rng.exponential(8, n)
    time = np.minimum(t, c)
    event = (t <= c).astype(float)
    return time, event, X


def test_returns_dict(surv_data):
    time, event, X = surv_data
    r = phreg(time, event, X)
    assert isinstance(r, dict)
    for k in ("coefficients", "baseline_hazards", "breaks"):
        assert k in r


def test_coefficients_shape(surv_data):
    time, event, X = surv_data
    r = phreg(time, event, X)
    assert r["coefficients"].shape == (2,)


def test_baseline_positive(surv_data):
    time, event, X = surv_data
    r = phreg(time, event, X)
    assert np.all(r["baseline_hazards"] >= 0)


def test_cheatsheet():
    from morie.fn.phreg import cheatsheet
    assert "piecewise" in cheatsheet().lower()
