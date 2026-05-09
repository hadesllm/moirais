"""Tests for moirais.fn.kmean — K-means clustering."""

import numpy as np
import pandas as pd
import pytest
from moirais.fn.kmean import kmean
from moirais.fn._containers import KmeansRes


class TestKmean:
    """Tests for K-means clustering."""

    def test_returns_kmeans_res(self, rng):
        X = rng.standard_normal((60, 3))
        result = kmean(X, k=3)
        assert isinstance(result, KmeansRes)

    def test_labels_count(self, rng):
        """Should produce exactly k unique labels."""
        X = np.vstack([
            rng.standard_normal((30, 2)) + [5, 5],
            rng.standard_normal((30, 2)) + [-5, -5],
            rng.standard_normal((30, 2)) + [5, -5],
        ])
        result = kmean(X, k=3)
        assert len(np.unique(result.labels)) == 3
        assert len(result.labels) == 90

    def test_well_separated_clusters(self, rng):
        """K-means should perfectly assign well-separated clusters."""
        c1 = rng.standard_normal((40, 2)) + [20, 20]
        c2 = rng.standard_normal((40, 2)) + [-20, -20]
        X = np.vstack([c1, c2])
        result = kmean(X, k=2)
        # All of group 1 should share one label, all of group 2 another
        assert len(np.unique(result.labels[:40])) == 1
        assert len(np.unique(result.labels[40:])) == 1
        assert result.labels[0] != result.labels[40]

    def test_inertia_decreases_with_k(self, rng):
        X = rng.standard_normal((80, 3))
        r2 = kmean(X, k=2)
        r4 = kmean(X, k=4)
        assert r4.inertia < r2.inertia

    def test_dataframe_input(self, rng):
        df = pd.DataFrame(rng.standard_normal((50, 2)), columns=["x", "y"])
        result = kmean(df, k=2)
        assert len(result.labels) == 50
