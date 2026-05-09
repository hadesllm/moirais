"""Tests for moirais.fn.hclus -- Hierarchical clustering."""

import numpy as np
from moirais.fn.hclus import hierarchical_cluster, hclus
from moirais.fn._containers import HclstRes


class TestHierarchicalCluster:
    def test_alias(self):
        assert hclus is hierarchical_cluster

    def test_returns_hclst_res(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 3))
        res = hierarchical_cluster(X, n_clusters=3)
        assert isinstance(res, HclstRes)

    def test_correct_n_clusters(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 3))
        res = hierarchical_cluster(X, n_clusters=4)
        assert len(np.unique(res.labels)) <= 4

    def test_ward_method(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 2))
        res = hierarchical_cluster(X, n_clusters=2, method="ward")
        assert len(res.labels) == 20

    def test_linkage_matrix(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((15, 3))
        res = hierarchical_cluster(X, n_clusters=3, method="average")
        assert res.linkage_matrix.shape[0] == 14
        assert res.linkage_matrix.shape[1] == 4
