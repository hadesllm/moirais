"""Tests for morie.fn.strpp -- stress per point."""

import numpy as np
from morie.fn.strpp import stress_per_point, strpp


def test_strpp_smoke():
    D_obs = np.array([[0, 1, 3], [1, 0, 2], [3, 2, 0]], dtype=float)
    D_mod = np.array([[0, 1.5, 2.5], [1.5, 0, 2.5], [2.5, 2.5, 0]], dtype=float)
    r = strpp(D_obs, D_mod)
    assert r.name == "stress_per_point"
    assert len(r.value) == 3
    assert np.isclose(np.sum(r.value), 1.0)


def test_strpp_alias():
    assert strpp is stress_per_point
