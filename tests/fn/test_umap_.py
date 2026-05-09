"""Tests for moirais.fn.umap_ — UMAP dimensionality reduction."""

import numpy as np
import pytest
from moirais.fn.umap_ import umap_
from moirais.fn._containers import UmapRes


class TestUmap:
    """Tests for UMAP."""

    def test_returns_umap_res(self, rng):
        X = rng.standard_normal((50, 5))
        result = umap_(X, n_epochs=10, n_neighbors=10)
        assert isinstance(result, UmapRes)

    def test_output_shape(self, rng):
        X = rng.standard_normal((40, 6))
        result = umap_(X, n_dims=3, n_epochs=10, n_neighbors=10)
        assert result.embedding.shape == (40, 3)

    def test_clusters_separated(self, rng):
        """Well-separated clusters should be distinguishable."""
        c1 = rng.standard_normal((30, 4)) + 8
        c2 = rng.standard_normal((30, 4)) - 8
        X = np.vstack([c1, c2])
        result = umap_(X, n_epochs=50, n_neighbors=10)
        emb = result.embedding
        centroid1 = emb[:30].mean(axis=0)
        centroid2 = emb[30:].mean(axis=0)
        between = np.linalg.norm(centroid1 - centroid2)
        assert between > 0.01  # they should not collapse to a point
