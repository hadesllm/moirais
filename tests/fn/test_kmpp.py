"""Tests for morie.fn.kmpp — k-means++."""
import numpy as np
from morie.fn.kmpp import kmeans_pp


class TestKMeansPP:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(0, 0.5, (30, 2)), rng.normal(5, 0.5, (30, 2))])
        res = kmeans_pp(X, k=2)
        assert len(np.unique(res.extra["labels"])) == 2

    def test_inertia_positive(self):
        X = np.random.default_rng(42).standard_normal((50, 3))
        res = kmeans_pp(X, k=3)
        assert res.extra["inertia"] > 0
