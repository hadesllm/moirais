"""Tests for morie.fn.scidl — scatter ideal points."""
import numpy as np
from morie.fn.scidl import scidl


def test_scidl_smoke():
    X = np.array([[1, 2], [3, 4], [5, 6.0]])
    r = scidl(X, groups=["A", "B", "A"])
    assert r.name == "scatter_ideal_points"
    assert r.value == 3
    assert r.extra["groups"] == ["A", "B", "A"]


def test_cheatsheet():
    from morie.fn.scidl import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
