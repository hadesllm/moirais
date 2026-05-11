"""Tests for morie.fn.dstcm -- distance comparison."""

import numpy as np
from morie.fn.dstcm import distance_comparison, dstcm


def test_dstcm_perfect():
    D = np.array([[0, 1, 2], [1, 0, 1.5], [2, 1.5, 0]], dtype=float)
    r = dstcm(D, D)
    assert r.name == "distance_comparison"
    assert np.isclose(r.value, 1.0)
    assert r.extra["rmse"] < 1e-10


def test_dstcm_alias():
    assert dstcm is distance_comparison
