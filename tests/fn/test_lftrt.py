"""Tests for morie.fn.lftrt -- Left truncation adjustment."""

import numpy as np
import pytest

from morie.fn.lftrt import lftrt


@pytest.fixture()
def trunc_data():
    rng = np.random.default_rng(42)
    n = 80
    entry = rng.uniform(0, 2, n)
    t = entry + rng.exponential(5, n)
    c = entry + rng.exponential(8, n)
    time = np.minimum(t, c)
    event = (t <= c).astype(float)
    return entry, time, event


def test_returns_dict(trunc_data):
    entry, time, event = trunc_data
    r = lftrt(entry, time, event)
    assert isinstance(r, dict)
    for k in ("times", "survival", "se", "n_obs", "n_events"):
        assert k in r


def test_survival_monotone(trunc_data):
    entry, time, event = trunc_data
    r = lftrt(entry, time, event)
    assert np.all(np.diff(r["survival"]) <= 1e-12)


def test_survival_bounded(trunc_data):
    entry, time, event = trunc_data
    r = lftrt(entry, time, event)
    assert np.all(r["survival"] >= 0)
    assert np.all(r["survival"] <= 1)


def test_cheatsheet():
    from morie.fn.lftrt import cheatsheet
    assert "truncat" in cheatsheet().lower()
