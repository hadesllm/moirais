"""Tests for morie.fn.gmpst -- geometric mean probability."""
import numpy as np
from morie.fn.gmpst import gmp_statistic, gmpst


def test_alias():
    assert gmpst is gmp_statistic


def test_smoke():
    probs = np.array([0.9, 0.8, 0.7, 0.6])
    obs = np.array([1, 1, 1, 1])
    r = gmp_statistic(probs, obs)
    assert r.name == "gmp_statistic"
    assert 0 < r.value <= 1.0


def test_perfect_prediction():
    probs = np.array([0.99, 0.99])
    obs = np.array([1, 1])
    r = gmp_statistic(probs, obs)
    assert r.value > 0.9


def test_log_gmp_in_extra():
    r = gmp_statistic([0.5, 0.5], [1, 0])
    assert "log_gmp" in r.extra
