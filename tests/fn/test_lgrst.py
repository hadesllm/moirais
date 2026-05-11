"""Tests for morie.fn.lgrst -- Log-rank test."""

import numpy as np
import pytest

from morie.fn.lgrst import lgrst


@pytest.fixture()
def two_group_data():
    rng = np.random.default_rng(42)
    n = 100
    group = np.array([0] * 50 + [1] * 50)
    t0 = rng.exponential(5, 50)
    t1 = rng.exponential(3, 50)
    t = np.concatenate([t0, t1])
    c = rng.exponential(8, n)
    time = np.minimum(t, c)
    event = (t <= c).astype(float)
    return time, event, group


def test_returns_dict(two_group_data):
    time, event, group = two_group_data
    r = lgrst(time, event, group)
    assert isinstance(r, dict)
    for k in ("statistic", "p_value", "n_obs", "n_events"):
        assert k in r


def test_statistic_nonneg(two_group_data):
    time, event, group = two_group_data
    r = lgrst(time, event, group)
    assert r["statistic"] >= 0


def test_p_value_bounded(two_group_data):
    time, event, group = two_group_data
    r = lgrst(time, event, group)
    assert 0 <= r["p_value"] <= 1


def test_identical_groups():
    time = np.array([1, 2, 3, 4, 5, 6])
    event = np.ones(6)
    group = np.array([0, 0, 0, 1, 1, 1])
    r = lgrst(time, event, group)
    assert r["p_value"] > 0.01


def test_wrong_groups():
    with pytest.raises(ValueError):
        lgrst(np.array([1, 2, 3]), np.ones(3), np.array([0, 1, 2]))


def test_cheatsheet():
    from morie.fn.lgrst import cheatsheet
    assert "log-rank" in cheatsheet().lower()
