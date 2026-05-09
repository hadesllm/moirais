"""Tests for moirais.fn.spclr -- Spectral clustering."""

import numpy as np
from moirais.fn.spclr import spectral_clustering, spclr
from moirais.fn._containers import DescriptiveResult


class TestSpectralClustering:
    def test_alias(self):
        assert spclr is spectral_clustering

    def test_returns_result(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 2))
        res = spectral_clustering(X, n_clusters=2)
        assert isinstance(res, DescriptiveResult)

    def test_labels_shape(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 2))
        res = spectral_clustering(X, n_clusters=3)
        assert len(res.value) == 30

    def test_has_eigenvalues(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 2))
        res = spectral_clustering(X, n_clusters=2)
        assert "eigenvalues" in res.extra
        assert "embedding" in res.extra
