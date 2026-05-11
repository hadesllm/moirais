"""Tests for morie.fn.grm — genomic relationship matrix."""
import numpy as np
import pytest
from morie.fn.grm import genetic_relatedness


class TestGRM:
    def test_symmetric(self):
        G = np.random.default_rng(42).choice([0, 1, 2], size=(20, 50))
        res = genetic_relatedness(G)
        grm = res.extra["grm"]
        np.testing.assert_allclose(grm, grm.T, atol=1e-10)

    def test_mean_diagonal(self):
        G = np.random.default_rng(42).choice([0, 1, 2], size=(20, 50))
        res = genetic_relatedness(G)
        assert res.value > 0
