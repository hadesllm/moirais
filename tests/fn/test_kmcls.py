"""Tests for morie.fn.kmcls -- K-means clustering."""

import numpy as np
from morie.fn.kmcls import kmeans, kmcls
from morie.fn._containers import KmeansRes


class TestKmeans:
    def test_alias(self):
        assert kmcls is kmeans

    def test_returns_kmeans_res(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 3))
        res = kmeans(X, k=3)
        assert isinstance(res, KmeansRes)

    def test_correct_k(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 3))
        res = kmeans(X, k=4)
        assert res.centers.shape[0] == 4
        assert len(np.unique(res.labels)) <= 4

    def test_inertia_positive(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((40, 2))
        res = kmeans(X, k=3)
        assert res.inertia > 0

    def test_well_separated_clusters(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(0, 0.3, (30, 2)), rng.normal(5, 0.3, (30, 2))])
        res = kmeans(X, k=2)
        assert len(np.unique(res.labels)) == 2
