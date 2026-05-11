"""Tests for morie.fn.mxcrk -- Mixture cure model."""

import numpy as np
import pytest

from morie.fn.mxcrk import mxcrk


@pytest.fixture()
def cure_data():
    rng = np.random.default_rng(42)
    n = 100
    cured = rng.binomial(1, 0.3, n)
    t_uncured = rng.exponential(3, n)
    t = np.where(cured, 100, t_uncured)
    c = rng.exponential(10, n)
    time = np.minimum(t, c)
    event = (t <= c).astype(float)
    return time, event


def test_returns_dict(cure_data):
    time, event = cure_data
    r = mxcrk(time, event)
    assert isinstance(r, dict)
    for k in ("cure_fraction", "survival_shape", "survival_scale", "log_likelihood"):
        assert k in r


def test_cure_fraction_bounded(cure_data):
    time, event = cure_data
    r = mxcrk(time, event)
    assert 0 <= r["cure_fraction"] <= 1


def test_shape_positive(cure_data):
    time, event = cure_data
    r = mxcrk(time, event)
    assert r["survival_shape"] > 0


def test_cheatsheet():
    from morie.fn.mxcrk import cheatsheet
    assert "cure" in cheatsheet().lower()
