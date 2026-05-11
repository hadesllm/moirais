"""Tests for morie.fn.dbind -- Davies-Bouldin index."""

import numpy as np
from morie.fn.dbind import davies_bouldin, dbind
from morie.fn._containers import DescriptiveResult


class TestDaviesBouldin:
    def test_alias(self):
        assert dbind is davies_bouldin

    def test_returns_result(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 2))
        labels = np.array([0]*15 + [1]*15)
        res = davies_bouldin(X, labels)
        assert isinstance(res, DescriptiveResult)

    def test_nonnegative(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 2))
        labels = np.array([0]*15 + [1]*15)
        res = davies_bouldin(X, labels)
        assert res.value >= 0

    def test_low_for_good_clustering(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(0, 0.2, (20, 2)), rng.normal(10, 0.2, (20, 2))])
        labels = np.array([0]*20 + [1]*20)
        res = davies_bouldin(X, labels)
        assert res.value < 1.0
