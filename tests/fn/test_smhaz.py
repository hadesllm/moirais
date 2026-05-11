"""Tests for morie.fn.smhaz -- Smoothed hazard function."""

import numpy as np
import pytest

from morie.fn.smhaz import smhaz


@pytest.fixture()
def surv_data():
    rng = np.random.default_rng(42)
    n = 80
    t = rng.exponential(5, n)
    c = rng.exponential(8, n)
    time = np.minimum(t, c)
    event = (t <= c).astype(float)
    return time, event


def test_returns_dict(surv_data):
    time, event = surv_data
    r = smhaz(time, event)
    assert isinstance(r, dict)
    for k in ("grid", "hazard", "cumulative_hazard", "bandwidth"):
        assert k in r


def test_hazard_nonneg(surv_data):
    time, event = surv_data
    r = smhaz(time, event)
    assert np.all(r["hazard"] >= 0)


def test_grid_length(surv_data):
    time, event = surv_data
    r = smhaz(time, event, n_grid=50)
    assert len(r["grid"]) == 50


def test_cheatsheet():
    from morie.fn.smhaz import cheatsheet
    assert "smooth" in cheatsheet().lower()
