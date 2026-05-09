"""Tests for cutting lines."""
import numpy as np
from moirais.fn.cutln import cutln


def test_cutln_smoke():
    normals = np.array([[1.0, 0.0], [0.0, 1.0], [0.707, 0.707]])
    cutpoints = np.array([0.0, 0.5, 0.3])
    r = cutln(normals, cutpoints)
    assert r.name == "cutting_line_mesh"
    assert r.value == 3
    assert len(r.extra["endpoints"]) == 3
    assert len(r.extra["angles"]) == 3


def test_cheatsheet():
    from moirais.fn.cutln import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
