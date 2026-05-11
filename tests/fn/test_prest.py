"""Tests for morie.fn.prest -- PRE statistic."""
import numpy as np
from morie.fn.prest import pre_statistic, prest


def test_alias():
    assert prest is pre_statistic


def test_smoke():
    pred = np.array([1, 0, 1, 0, 1])
    obs = np.array([1, 0, 1, 0, 1])
    r = pre_statistic(pred, obs)
    assert r.name == "pre_statistic"
    assert r.value == 1.0


def test_no_improvement():
    pred = np.array([1, 1, 1, 1])
    obs = np.array([1, 0, 1, 0])
    r = pre_statistic(pred, obs)
    assert r.value <= 1.0
    assert "errors_null" in r.extra
