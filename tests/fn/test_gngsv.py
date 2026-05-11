"""Tests for morie.fn.gngsv -- Generalized gamma survival."""

import numpy as np
import pytest

from morie.fn.gngsv import gngsv


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
    r = gngsv(time, event)
    assert isinstance(r, dict)
    for k in ("mu", "sigma", "Q", "log_likelihood"):
        assert k in r


def test_sigma_positive(surv_data):
    time, event = surv_data
    r = gngsv(time, event)
    assert r["sigma"] > 0


def test_n_events_correct(surv_data):
    time, event = surv_data
    r = gngsv(time, event)
    assert r["n_events"] == int(np.sum(event))


def test_cheatsheet():
    from morie.fn.gngsv import cheatsheet
    assert "gamma" in cheatsheet().lower()
