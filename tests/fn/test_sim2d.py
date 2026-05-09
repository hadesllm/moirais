"""Tests for moirais.fn.sim2d -- simulate 2D unfolding."""

import numpy as np
from moirais.fn.sim2d import simulate_2d_unfolding, sim2d


def test_sim2d_smoke():
    r = sim2d(n_resp=10, n_stim=3)
    assert r.name == "simulate_2d_unfolding"
    assert r.value.shape == (10, 3)
    assert np.all(r.value >= 0)
    assert "X_resp" in r.extra
    assert r.extra["X_resp"].shape == (10, 2)


def test_sim2d_alias():
    assert sim2d is simulate_2d_unfolding
