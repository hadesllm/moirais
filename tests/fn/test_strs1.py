"""Tests for moirais.fn.strs1 -- Kruskal stress-1."""

import numpy as np
from moirais.fn.strs1 import stress1_measure, strs1


def test_strs1_perfect():
    D = np.array([[0, 1, 2], [1, 0, 1.5], [2, 1.5, 0]], dtype=float)
    r = strs1(D, D)
    assert r.name == "stress1_measure"
    assert r.value < 1e-10


def test_strs1_nonzero():
    D_obs = np.array([[0, 1, 3], [1, 0, 2], [3, 2, 0]], dtype=float)
    D_mod = np.array([[0, 1.5, 2.5], [1.5, 0, 2.5], [2.5, 2.5, 0]], dtype=float)
    r = strs1(D_obs, D_mod)
    assert r.value > 0


def test_strs1_alias():
    assert strs1 is stress1_measure
