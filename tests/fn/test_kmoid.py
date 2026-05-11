"""Tests for morie.fn.kmoid -- K-medoids (PAM)."""

import numpy as np
from morie.fn.kmoid import kmedoids, kmoid
from morie.fn._containers import DescriptiveResult


class TestKmedoids:
    def test_alias(self):
        assert kmoid is kmedoids

    def test_returns_result(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 3))
        res = kmedoids(X, k=3)
        assert isinstance(res, DescriptiveResult)

    def test_labels_valid(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 3))
        res = kmedoids(X, k=3)
        labels = res.value
        assert len(labels) == 30
        assert set(np.unique(labels)).issubset({0, 1, 2})

    def test_medoids_from_data(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 2))
        res = kmedoids(X, k=2)
        for idx in res.extra["medoid_indices"]:
            np.testing.assert_array_equal(res.extra["medoids"][np.where(res.extra["medoid_indices"] == idx)[0][0]], X[idx])

    def test_cost_positive(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 2))
        res = kmedoids(X, k=3)
        assert res.extra["cost"] > 0
