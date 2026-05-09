"""Tests for moirais.fn.morph -- t-SNE dimensionality reduction."""

import numpy as np
from moirais.fn.morph import tsne_reduce, morph
from moirais.fn._containers import TsneRes


class TestMorph:
    def test_alias(self):
        assert morph is tsne_reduce

    def test_basic_reduction(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (20, 5))
        result = tsne_reduce(X, n_iter=50, seed=42)
        assert isinstance(result, TsneRes)
        assert result.embedding.shape == (20, 2)

    def test_3d(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (15, 4))
        result = tsne_reduce(X, n_dims=3, n_iter=30, seed=42)
        assert result.embedding.shape == (15, 3)
