"""Tests for morie.fn.rcsub -- subset roll calls."""
import numpy as np
from morie.fn.rcsub import subset_roll_calls, rcsub


def test_alias():
    assert rcsub is subset_roll_calls


def test_smoke():
    rng = np.random.default_rng(42)
    votes = rng.choice([0.0, 1.0], size=(20, 10))
    r = subset_roll_calls(votes, min_margin=0.025)
    assert r.name == "subset_roll_calls"
    assert "n_kept" in r.extra
    assert "n_dropped" in r.extra


def test_lopsided_dropped():
    votes = np.ones((10, 3))
    votes[:, 2] = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    r = subset_roll_calls(votes, min_margin=0.1)
    assert r.extra["n_kept"] <= 3
