"""Tests for moirais.fn.chand -- Calinski-Harabasz index."""

import numpy as np
from moirais.fn.chand import calinski_harabasz, chand
from moirais.fn._containers import DescriptiveResult


class TestCalinskiHarabasz:
    def test_alias(self):
        assert chand is calinski_harabasz

    def test_returns_result(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 2))
        labels = np.array([0]*15 + [1]*15)
        res = calinski_harabasz(X, labels)
        assert isinstance(res, DescriptiveResult)

    def test_positive_for_good_clustering(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(0, 0.3, (20, 2)), rng.normal(5, 0.3, (20, 2))])
        labels = np.array([0]*20 + [1]*20)
        res = calinski_harabasz(X, labels)
        assert res.value > 0

    def test_single_cluster(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 2))
        labels = np.zeros(20, dtype=int)
        res = calinski_harabasz(X, labels)
        assert res.value == 0.0
