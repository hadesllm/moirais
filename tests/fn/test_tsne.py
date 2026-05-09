"""Tests for moirais.fn.tsne — t-SNE."""

import numpy as np
import pytest
from moirais.fn.tsne import tsne
from moirais.fn._containers import TsneRes


class TestTsne:
    """Tests for t-SNE."""

    def test_returns_tsne_res(self, rng):
        X = rng.standard_normal((50, 5))
        result = tsne(X, n_iter=50, perplexity=10)
        assert isinstance(result, TsneRes)

    def test_output_shape(self, rng):
        X = rng.standard_normal((40, 8))
        result = tsne(X, n_dims=2, n_iter=50, perplexity=10)
        assert result.embedding.shape == (40, 2)

    def test_clusters_separated(self, rng):
        """Two well-separated clusters should remain separated after t-SNE."""
        c1 = rng.standard_normal((25, 5)) + 10
        c2 = rng.standard_normal((25, 5)) - 10
        X = np.vstack([c1, c2])
        result = tsne(X, n_iter=200, perplexity=10)
        emb = result.embedding
        # Centroids of the two groups should be far apart relative to within-group spread
        centroid1 = emb[:25].mean(axis=0)
        centroid2 = emb[25:].mean(axis=0)
        between = np.linalg.norm(centroid1 - centroid2)
        within1 = np.mean(np.linalg.norm(emb[:25] - centroid1, axis=1))
        within2 = np.mean(np.linalg.norm(emb[25:] - centroid2, axis=1))
        assert between > (within1 + within2) / 2
