"""Tests for moirais.fn.rmstd -- Restricted mean survival time difference."""

import numpy as np
import pytest

from moirais.fn.rmstd import rmstd


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
    r = rmstd(time, event, group)
    assert isinstance(r, dict)
    for k in ("rmst_0", "rmst_1", "difference", "se", "ci_lower", "ci_upper", "p_value"):
        assert k in r


def test_rmst_positive(two_group_data):
    time, event, group = two_group_data
    r = rmstd(time, event, group)
    assert r["rmst_0"] > 0
    assert r["rmst_1"] > 0


def test_ci_contains_diff(two_group_data):
    time, event, group = two_group_data
    r = rmstd(time, event, group)
    assert r["ci_lower"] <= r["difference"] <= r["ci_upper"]


def test_wrong_groups():
    with pytest.raises(ValueError):
        rmstd(np.array([1, 2, 3]), np.ones(3), np.array([0, 1, 2]))


def test_cheatsheet():
    from moirais.fn.rmstd import cheatsheet
    assert "restricted" in cheatsheet().lower()
