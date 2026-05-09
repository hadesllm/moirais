"""Tests for moirais.fn.strs2 -- stress-2."""

import numpy as np
from moirais.fn.strs2 import stress2_measure, strs2


def test_strs2_perfect():
    D = np.array([[0, 1, 2], [1, 0, 1.5], [2, 1.5, 0]], dtype=float)
    r = strs2(D, D)
    assert r.name == "stress2_measure"
    assert r.value < 1e-10


def test_strs2_alias():
    assert strs2 is stress2_measure
