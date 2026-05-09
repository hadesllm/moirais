"""Tests for moirais.fn.sihlh -- Silhouette score."""

import numpy as np
from moirais.fn.sihlh import silhouette, sihlh
from moirais.fn._containers import DescriptiveResult


class TestSilhouette:
    def test_alias(self):
        assert sihlh is silhouette

    def test_returns_result(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 2))
        labels = np.array([0]*10 + [1]*10 + [2]*10)
        res = silhouette(X, labels)
        assert isinstance(res, DescriptiveResult)

    def test_bounded(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 2))
        labels = np.array([0]*15 + [1]*15)
        res = silhouette(X, labels)
        assert -1.0 <= res.value <= 1.0

    def test_good_clustering_positive(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(0, 0.3, (20, 2)), rng.normal(5, 0.3, (20, 2))])
        labels = np.array([0]*20 + [1]*20)
        res = silhouette(X, labels)
        assert res.value > 0.5

    def test_per_sample(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 2))
        labels = np.array([0]*10 + [1]*10)
        res = silhouette(X, labels)
        assert len(res.extra["per_sample"]) == 20
