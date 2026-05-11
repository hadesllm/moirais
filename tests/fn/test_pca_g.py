"""Tests for morie.fn.pca_g — genotype PCA."""
import numpy as np
import pytest
from morie.fn.pca_g import pca_genotype


class TestPcaGenotype:
    def test_basic(self):
        rng = np.random.default_rng(42)
        G = rng.choice([0, 1, 2], size=(50, 100))
        res = pca_genotype(G, n_components=2)
        assert res.extra["scores"].shape == (50, 2)
        assert res.value > 0

    def test_wrong_dim_raises(self):
        with pytest.raises(ValueError):
            pca_genotype(np.array([1, 2, 3]))
