"""Tests for moirais.fn.embd — embedding similarity."""
import numpy as np
import pytest
from moirais.fn.embd import embedding_similarity


class TestEmbeddingSimilarity:
    def test_similarity_matrix_shape(self):
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((5, 3))
        res = embedding_similarity(embeddings)
        sim = res.extra["similarity_matrix"]
        assert sim.shape == (5, 5)

    def test_diagonal_zeroed(self):
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((4, 6))
        res = embedding_similarity(embeddings, metric="cosine")
        sim = res.extra["similarity_matrix"]
        np.testing.assert_allclose(np.diag(sim), 0.0, atol=1e-10)

    def test_euclidean_metric(self):
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((3, 4))
        res = embedding_similarity(embeddings, metric="euclidean")
        sim = res.extra["similarity_matrix"]
        assert sim.shape == (3, 3)
        assert res.extra["metric"] == "euclidean"
