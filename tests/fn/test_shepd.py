"""Tests for morie.fn.shepd -- Shepard diagram."""

import numpy as np
from morie.fn.shepd import shepard_diagram, shepd


def test_shepd_smoke():
    D_obs = np.array([[0, 1, 3], [1, 0, 2], [3, 2, 0]], dtype=float)
    D_mod = np.array([[0, 1.1, 2.8], [1.1, 0, 2.2], [2.8, 2.2, 0]], dtype=float)
    r = shepd(D_obs, D_mod)
    assert r.name == "shepard_diagram"
    assert r.value > 0.9
    assert "observed" in r.extra


def test_shepd_alias():
    assert shepd is shepard_diagram
