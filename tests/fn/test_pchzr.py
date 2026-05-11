"""Tests for morie.fn.pchzr -- Piecewise constant hazard rate."""

import numpy as np
import pytest

from morie.fn.pchzr import pchzr


@pytest.fixture()
def surv_data():
    rng = np.random.default_rng(42)
    n = 100
    t = rng.exponential(5, n)
    c = rng.exponential(8, n)
    time = np.minimum(t, c)
    event = (t <= c).astype(float)
    return time, event


def test_returns_dict(surv_data):
    time, event = surv_data
    r = pchzr(time, event)
    assert isinstance(r, dict)
    for k in ("breaks", "hazard_rates", "n_events_per_interval"):
        assert k in r


def test_hazard_nonneg(surv_data):
    time, event = surv_data
    r = pchzr(time, event)
    assert np.all(r["hazard_rates"] >= 0)


def test_events_sum(surv_data):
    time, event = surv_data
    r = pchzr(time, event)
    assert np.sum(r["n_events_per_interval"]) == int(np.sum(event))


def test_cheatsheet():
    from morie.fn.pchzr import cheatsheet
    assert "piecewise" in cheatsheet().lower()
