"""Tests for moirais.fn.expsv -- Exponential survival model."""

import numpy as np
import pytest

from moirais.fn.expsv import expsv


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
    r = expsv(time, event)
    assert isinstance(r, dict)
    assert "rate" in r and "log_likelihood" in r


def test_rate_positive(surv_data):
    time, event = surv_data
    r = expsv(time, event)
    assert r["rate"] > 0


def test_with_covariates(surv_data):
    time, event = surv_data
    X = np.random.default_rng(99).standard_normal((len(time), 2))
    r = expsv(time, event, X)
    assert r["coefficients"].shape == (2,)


def test_cheatsheet():
    from moirais.fn.expsv import cheatsheet
    assert "exponential" in cheatsheet().lower()
