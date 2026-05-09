"""Tests for moirais.fn.logns -- Log-normal survival model."""

import numpy as np
import pytest

from moirais.fn.logns import logns


@pytest.fixture()
def surv_data():
    rng = np.random.default_rng(42)
    n = 100
    t = rng.lognormal(1, 0.5, n)
    c = rng.exponential(8, n)
    time = np.minimum(t, c)
    event = (t <= c).astype(float)
    return time, event


def test_returns_dict(surv_data):
    time, event = surv_data
    r = logns(time, event)
    assert isinstance(r, dict)
    for k in ("mu", "sigma", "log_likelihood"):
        assert k in r


def test_sigma_positive(surv_data):
    time, event = surv_data
    r = logns(time, event)
    assert r["sigma"] > 0


def test_with_covariates(surv_data):
    time, event = surv_data
    X = np.random.default_rng(99).standard_normal((len(time), 1))
    r = logns(time, event, X)
    assert r["coefficients"].shape == (1,)


def test_cheatsheet():
    from moirais.fn.logns import cheatsheet
    assert "log-normal" in cheatsheet().lower()
