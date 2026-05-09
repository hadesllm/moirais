"""Tests for dynamic IRT."""
import numpy as np
from moirais.fn.dirt import dirt


def test_dirt_smoke():
    rng = np.random.default_rng(42)
    votes = (rng.random((8, 12)) > 0.4).astype(float)
    time_periods = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
    r = dirt(votes, time_periods, n_samples=20, burn_in=10)
    assert r.name == "dynamic_irt_model"
    assert "ideal_trajectories" in r.extra
    assert r.extra["ideal_trajectories"].shape[0] == 8


def test_cheatsheet():
    from moirais.fn.dirt import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
