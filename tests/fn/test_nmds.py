"""Tests for moirais.fn.nmds -- Non-metric MDS."""

import numpy as np
from moirais.fn.nmds import nmds
from moirais.fn._containers import MdsRes


class TestNmds:
    def test_returns_mds_res(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 4))
        res = nmds(X, n_dims=2)
        assert isinstance(res, MdsRes)

    def test_shape(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 4))
        res = nmds(X, n_dims=2)
        assert res.coordinates.shape == (20, 2)

    def test_stress_nonneg(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((15, 3))
        res = nmds(X, n_dims=2)
        assert res.stress >= 0

    def test_from_distance_matrix(self):
        from scipy.spatial.distance import pdist, squareform
        rng = np.random.default_rng(42)
        X = rng.standard_normal((15, 3))
        D = squareform(pdist(X))
        res = nmds(D, n_dims=2)
        assert res.coordinates.shape == (15, 2)
