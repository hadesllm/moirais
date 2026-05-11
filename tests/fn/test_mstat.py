"""Tests for morie.fn.mstat -- Multi-state model."""

import numpy as np
import pytest

from morie.fn.mstat import mstat


@pytest.fixture()
def ms_data():
    rng = np.random.default_rng(42)
    n = 50
    time = rng.exponential(3, n)
    states = [0, 1, 2]
    state_from = rng.choice([0, 1], n)
    state_to = rng.choice([1, 2], n)
    return time, state_from, state_to


def test_returns_dict(ms_data):
    time, sf, st = ms_data
    r = mstat(time, sf, st)
    assert isinstance(r, dict)
    for k in ("states", "transition_counts", "transition_rates", "n_transitions"):
        assert k in r


def test_transition_matrix_shape(ms_data):
    time, sf, st = ms_data
    r = mstat(time, sf, st)
    k = len(r["states"])
    assert r["transition_counts"].shape == (k, k)


def test_rates_bounded(ms_data):
    time, sf, st = ms_data
    r = mstat(time, sf, st)
    assert np.all(r["transition_rates"] >= 0)
    assert np.all(r["transition_rates"] <= 1)


def test_cheatsheet():
    from morie.fn.mstat import cheatsheet
    assert "multi-state" in cheatsheet().lower()
