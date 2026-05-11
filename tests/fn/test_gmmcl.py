"""Tests for morie.fn.gmmcl -- Gaussian mixture model clustering."""

import numpy as np
from morie.fn.gmmcl import gmm_cluster, gmmcl
from morie.fn._containers import DescriptiveResult


class TestGmmCluster:
    def test_alias(self):
        assert gmmcl is gmm_cluster

    def test_returns_result(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 3))
        res = gmm_cluster(X, n_components=2)
        assert isinstance(res, DescriptiveResult)

    def test_labels_valid(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 3))
        res = gmm_cluster(X, n_components=3)
        labels = res.value
        assert len(labels) == 50
        assert set(np.unique(labels)).issubset({0, 1, 2})

    def test_has_bic_aic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 2))
        res = gmm_cluster(X, n_components=2)
        assert "bic" in res.extra
        assert "aic" in res.extra

    def test_well_separated(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(0, 0.3, (30, 2)), rng.normal(5, 0.3, (30, 2))])
        res = gmm_cluster(X, n_components=2)
        assert len(np.unique(res.value)) == 2
