"""Tests for morie.fn.trent -- Trend test for survival."""

import numpy as np
import pytest

from morie.fn.trent import trent


@pytest.fixture()
def ordered_data():
    rng = np.random.default_rng(42)
    group = np.array([0] * 30 + [1] * 30 + [2] * 30)
    rates = np.array([5.0] * 30 + [3.0] * 30 + [1.0] * 30)
    t = rng.exponential(rates)
    c = rng.exponential(8, 90)
    time = np.minimum(t, c)
    event = (t <= c).astype(float)
    return time, event, group


def test_returns_dict(ordered_data):
    time, event, group = ordered_data
    r = trent(time, event, group)
    assert isinstance(r, dict)
    assert "statistic" in r and "p_value" in r


def test_statistic_nonneg(ordered_data):
    time, event, group = ordered_data
    r = trent(time, event, group)
    assert r["statistic"] >= 0


def test_p_value_bounded(ordered_data):
    time, event, group = ordered_data
    r = trent(time, event, group)
    assert 0 <= r["p_value"] <= 1


def test_single_group_raises():
    with pytest.raises(ValueError):
        trent(np.array([1, 2]), np.ones(2), np.array([0, 0]))


def test_cheatsheet():
    from morie.fn.trent import cheatsheet
    assert "trend" in cheatsheet().lower()
