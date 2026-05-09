"""Tests for moirais.fn.ctprx -- cutting plane proximity."""
import numpy as np
from moirais.fn.ctprx import cutting_plane_proximity, ctprx


def test_alias():
    assert ctprx is cutting_plane_proximity


def test_smoke():
    X = np.array([[0.5, 0.5], [-0.5, -0.5], [0.0, 0.0]])
    r = cutting_plane_proximity(X, normal=[1.0, 0.0], cutpoint=0.0)
    assert r.name == "cutting_plane_proximity"
    assert "distances" in r.extra
    assert r.extra["n_legislators"] == 3
