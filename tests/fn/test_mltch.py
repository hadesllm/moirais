"""Tests for morie.fn.mltch -- combine multiple chains."""
import numpy as np
from morie.fn.mltch import multiple_chains_combine, mltch


def test_alias():
    assert mltch is multiple_chains_combine


def test_smoke():
    c1 = np.random.default_rng(42).standard_normal(100)
    c2 = np.random.default_rng(43).standard_normal(100)
    r = multiple_chains_combine([c1, c2])
    assert r.name == "multiple_chains_combine"
    assert r.extra["n_chains"] == 2
    assert r.extra["total_samples"] == 200


def test_2d():
    c1 = np.ones((50, 3))
    c2 = np.zeros((50, 3))
    r = multiple_chains_combine([c1, c2])
    assert r.extra["total_samples"] == 100
