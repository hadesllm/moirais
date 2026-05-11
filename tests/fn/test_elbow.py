"""Tests for morie.fn.elbow -- Elbow method for optimal k."""

import numpy as np
from morie.fn.elbow import elbow_method


class TestElbowMethod:
    def test_two_clusters(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.standard_normal((30, 2)) + [5, 5],
                        rng.standard_normal((30, 2)) - [5, 5]])
        result = elbow_method(X, max_k=6)
        assert "k_values" in result
        assert "inertias" in result
        assert "optimal_k" in result
        assert result["optimal_k"] >= 2

    def test_inertia_decreases(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 2))
        result = elbow_method(X, max_k=5)
        # Inertia should generally decrease with k
        assert result["inertias"][0] >= result["inertias"][-1]

    def test_max_k_capped(self):
        X = np.ones((3, 2))
        result = elbow_method(X, max_k=10)
        assert len(result["k_values"]) == 3  # capped at n=3
