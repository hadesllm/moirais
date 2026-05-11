"""Tests for morie.fn.gompr -- Gompertz survival model."""

import numpy as np
import pytest

from morie.fn.gompr import gompr


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
    r = gompr(time, event)
    assert isinstance(r, dict)
    for k in ("lambda_", "gamma", "log_likelihood"):
        assert k in r


def test_lambda_positive(surv_data):
    time, event = surv_data
    r = gompr(time, event)
    assert r["lambda_"] > 0


def test_n_events(surv_data):
    time, event = surv_data
    r = gompr(time, event)
    assert r["n_events"] == int(np.sum(event))


def test_cheatsheet():
    from morie.fn.gompr import cheatsheet
    assert "gompertz" in cheatsheet().lower()
