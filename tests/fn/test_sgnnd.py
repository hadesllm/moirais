"""Tests for nearest-neighbor distances."""
import numpy as np
from morie.fn.sgnnd import sgnnd


def test_sgnnd_smoke():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 10, (30, 2))
    r = sgnnd(pts)
    assert r.name == "nearest_neighbor_distances"
    assert "nn_distances" in r.extra
    assert r.extra["mean_nn"] > 0
    assert len(r.extra["nn_distances"]) == 30


def test_cheatsheet():
    from morie.fn.sgnnd import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
