"""Tests for morie.fn.rcagr -- agreement scores."""

import numpy as np
from morie.fn.rcagr import agreement_scores, rcagr


def test_rcagr_perfect():
    votes = np.array([[1, 0, 1], [1, 0, 1]], dtype=float)
    r = rcagr(votes)
    assert r.name == "agreement_scores"
    assert np.isclose(r.value[0, 1], 1.0)


def test_rcagr_opposite():
    votes = np.array([[1, 0, 1], [0, 1, 0]], dtype=float)
    r = rcagr(votes)
    assert np.isclose(r.value[0, 1], 0.0)


def test_rcagr_alias():
    assert rcagr is agreement_scores
