"""Tests for morie.fn.major -- SMACOF majorize step."""

import numpy as np
from morie.fn.major import majorize_step, major


def test_major_smoke():
    D = np.array([[0, 1, 2], [1, 0, 1.5], [2, 1.5, 0]], dtype=float)
    X = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=float)
    r = major(X, D)
    assert r.name == "majorize_step"
    assert r.value.shape == (3, 2)


def test_major_alias():
    assert major is majorize_step
