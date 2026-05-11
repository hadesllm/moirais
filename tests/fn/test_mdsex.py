"""Tests for morie.fn.mdsex -- convex hull MDS."""

import numpy as np
from morie.fn.mdsex import convex_hull_mds, mdsex


def test_mdsex_smoke():
    X = np.array([[0, 0], [1, 0], [0.5, 1], [2, 0], [2.5, 1], [2, 1.5]], dtype=float)
    groups = np.array(["A", "A", "A", "B", "B", "B"])
    r = mdsex(X, groups)
    assert r.name == "convex_hull_mds"
    assert "A" in r.value
    assert r.extra["n_groups"] == 2


def test_mdsex_alias():
    assert mdsex is convex_hull_mds
