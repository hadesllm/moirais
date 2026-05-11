"""Tests for morie.fn.ufcmp -- compare unfolding methods."""

import numpy as np
from morie.fn.ufcmp import compare_unfolding_methods, ufcmp


def test_ufcmp_smoke():
    r1 = {"stress": 0.05, "coords": np.array([1, 2, 3])}
    r2 = {"stress": 0.08, "coords": np.array([1.1, 2.1, 2.9])}
    r = ufcmp(r1, r2)
    assert r.name == "compare_unfolding_methods"
    assert r.value["better"] == 1
    assert r.value["coord_correlation"] > 0.9


def test_ufcmp_alias():
    assert ufcmp is compare_unfolding_methods
