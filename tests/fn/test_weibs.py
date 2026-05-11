"""Tests for morie.fn.weibs -- Weibull survival model."""

import numpy as np
import pytest

from morie.fn.weibs import weibs


@pytest.fixture()
def surv_data():
    rng = np.random.default_rng(42)
    n = 100
    t = rng.weibull(1.5, n) * 5
    c = rng.exponential(8, n)
    time = np.minimum(t, c)
    event = (t <= c).astype(float)
    return time, event


def test_returns_dict(surv_data):
    time, event = surv_data
    r = weibs(time, event)
    assert isinstance(r, dict)
    for k in ("shape", "scale", "log_likelihood", "n_obs", "n_events"):
        assert k in r


def test_shape_positive(surv_data):
    time, event = surv_data
    r = weibs(time, event)
    assert r["shape"] > 0


def test_scale_positive(surv_data):
    time, event = surv_data
    r = weibs(time, event)
    assert r["scale"] > 0


def test_with_covariates(surv_data):
    time, event = surv_data
    X = np.random.default_rng(99).standard_normal((len(time), 2))
    r = weibs(time, event, X)
    assert r["coefficients"].shape == (2,)


def test_cheatsheet():
    from morie.fn.weibs import cheatsheet
    assert "weibull" in cheatsheet().lower()
