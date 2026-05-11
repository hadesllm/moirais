"""Tests for morie.fn.rcmat -- rectangular matrix."""

import numpy as np
from morie.fn.rcmat import rectangular_matrix, rcmat


def test_rcmat_smoke():
    R = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    r = rcmat(R, ["A", "B", "C"])
    assert r.name == "rectangular_matrix"
    assert r.extra["n_resp"] == 2
    assert r.extra["n_stim"] == 3


def test_rcmat_default_labels():
    R = np.ones((3, 4))
    r = rcmat(R, None)
    assert r.extra["stimuli"] == ["S1", "S2", "S3", "S4"]


def test_rcmat_alias():
    assert rcmat is rectangular_matrix
