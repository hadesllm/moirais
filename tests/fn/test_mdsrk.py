"""Tests for morie.fn.mdsrk -- reconstruct distances."""

import numpy as np
from morie.fn.mdsrk import mds_reconstruct_distances, mdsrk


def test_mdsrk_smoke():
    X = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    r = mdsrk(X)
    assert r.name == "mds_reconstruct_distances"
    D = r.value
    assert D.shape == (3, 3)
    assert np.isclose(D[0, 1], 1.0)
    assert np.isclose(D[0, 0], 0.0)


def test_mdsrk_alias():
    assert mdsrk is mds_reconstruct_distances
