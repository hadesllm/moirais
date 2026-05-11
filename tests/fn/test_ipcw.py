"""Tests for morie.fn.ipcw -- Inverse probability of censoring weights."""

import numpy as np
import pytest

from morie.fn.ipcw import ipcw


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
    r = ipcw(time, event)
    assert isinstance(r, dict)
    for k in ("weights", "censoring_survival", "n_obs", "n_censored"):
        assert k in r


def test_weights_positive(surv_data):
    time, event = surv_data
    r = ipcw(time, event)
    assert np.all(r["weights"] > 0)


def test_weights_geq_one(surv_data):
    time, event = surv_data
    r = ipcw(time, event)
    assert np.all(r["weights"] >= 1.0 - 1e-10)


def test_censoring_bounded(surv_data):
    time, event = surv_data
    r = ipcw(time, event)
    assert np.all(r["censoring_survival"] >= 0)
    assert np.all(r["censoring_survival"] <= 1)


def test_cheatsheet():
    from morie.fn.ipcw import cheatsheet
    assert "censoring" in cheatsheet().lower()
