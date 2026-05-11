"""Tests for morie.fn.mdsmm -- MDS with missing data."""

import numpy as np
from morie.fn.mdsmm import mds_missing_data, mdsmm


def test_mdsmm_smoke():
    D = np.array([[0, 1, 2], [1, 0, 1.5], [2, 1.5, 0]], dtype=float)
    W = np.ones((3, 3))
    r = mdsmm(D, W, n_dims=2, max_iter=20)
    assert r.name == "mds_missing_data"
    assert r.value.shape == (3, 2)
    assert "weighted_stress" in r.extra


def test_mdsmm_alias():
    assert mdsmm is mds_missing_data
