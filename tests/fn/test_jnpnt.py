"""Tests for morie.fn.jnpnt -- Joinpoint survival regression."""

import numpy as np
import pytest

from morie.fn.jnpnt import jnpnt


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
    r = jnpnt(time, event, max_joinpoints=1)
    assert isinstance(r, dict)
    for k in ("n_joinpoints", "joinpoints", "slopes", "bic"):
        assert k in r


def test_n_joinpoints_bounded(surv_data):
    time, event = surv_data
    r = jnpnt(time, event, max_joinpoints=2)
    assert 0 <= r["n_joinpoints"] <= 2


def test_bic_exists(surv_data):
    time, event = surv_data
    r = jnpnt(time, event, max_joinpoints=1)
    assert "bic" in r


def test_cheatsheet():
    from morie.fn.jnpnt import cheatsheet
    assert "joinpoint" in cheatsheet().lower()
