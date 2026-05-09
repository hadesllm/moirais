"""Tests for G function."""
import numpy as np
from moirais.fn.sggfn import sggfn


def test_sggfn_smoke():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 10, (40, 2))
    r = sggfn(pts)
    assert r.name == "g_function_nearest_neighbor"
    assert "G_values" in r.extra
    assert "nn_distances" in r.extra
    assert r.extra["mean_nn_distance"] > 0


def test_cheatsheet():
    from moirais.fn.sggfn import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
