"""Tests for morie.fn.grest -- Gehan-Breslow test."""

import numpy as np
import pytest

from morie.fn.grest import grest


@pytest.fixture()
def two_group_data():
    rng = np.random.default_rng(42)
    group = np.array([0] * 50 + [1] * 50)
    t0 = rng.exponential(5, 50)
    t1 = rng.exponential(3, 50)
    t = np.concatenate([t0, t1])
    c = rng.exponential(8, 100)
    time = np.minimum(t, c)
    event = (t <= c).astype(float)
    return time, event, group


def test_returns_dict(two_group_data):
    time, event, group = two_group_data
    r = grest(time, event, group)
    assert isinstance(r, dict)
    assert "statistic" in r and "p_value" in r


def test_statistic_nonneg(two_group_data):
    time, event, group = two_group_data
    r = grest(time, event, group)
    assert r["statistic"] >= 0


def test_p_value_bounded(two_group_data):
    time, event, group = two_group_data
    r = grest(time, event, group)
    assert 0 <= r["p_value"] <= 1


def test_wrong_groups():
    with pytest.raises(ValueError):
        grest(np.array([1, 2, 3]), np.ones(3), np.array([0, 1, 2]))


def test_cheatsheet():
    from morie.fn.grest import cheatsheet
    assert "gehan" in cheatsheet().lower()
