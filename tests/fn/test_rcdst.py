"""Tests for morie.fn.rcdst -- distance from agreement."""

import numpy as np
from morie.fn.rcdst import distance_from_agreement, rcdst


def test_rcdst_smoke():
    A = np.array([[1.0, 0.8], [0.8, 1.0]])
    r = rcdst(A)
    assert r.name == "distance_from_agreement"
    assert np.isclose(r.value[0, 0], 0.0)
    assert np.isclose(r.value[0, 1], 0.2)


def test_rcdst_alias():
    assert rcdst is distance_from_agreement
