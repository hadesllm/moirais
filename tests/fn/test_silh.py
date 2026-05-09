"""Tests for moirais.fn.silh -- Silhouette score."""

import numpy as np
import pytest
from moirais.fn.silh import silhouette_score


class TestSilhouetteScore:
    def test_well_separated(self):
        X = np.array([[0, 0], [0.1, 0], [0, 0.1],
                       [10, 10], [10.1, 10], [10, 10.1]], dtype=float)
        labels = np.array([0, 0, 0, 1, 1, 1])
        result = silhouette_score(X, labels)
        assert result["mean_score"] > 0.8
        assert result["per_sample_scores"].shape == (6,)

    def test_single_cluster_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            silhouette_score(np.ones((5, 2)), np.zeros(5))

    def test_range(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 2))
        labels = np.array([0] * 10 + [1] * 10)
        result = silhouette_score(X, labels)
        assert -1 <= result["mean_score"] <= 1
