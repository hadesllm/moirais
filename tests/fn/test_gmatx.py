"""Tests for moirais.fn.gmatx -- Genomic relationship matrix."""

import numpy as np
import pytest
from moirais.fn.gmatx import gmatx


class TestGmatx:
    def test_symmetric(self):
        rng = np.random.default_rng(42)
        Z = rng.choice([0, 1, 2], size=(10, 20)).astype(float)
        res = gmatx(Z)
        G = np.array(res.extra["G"])
        np.testing.assert_allclose(G, G.T, atol=1e-10)

    def test_diagonal_near_one(self):
        rng = np.random.default_rng(42)
        Z = rng.choice([0, 1, 2], size=(20, 100)).astype(float)
        res = gmatx(Z)
        G = np.array(res.extra["G"])
        diag = np.diag(G)
        assert np.all(diag > 0.5) and np.all(diag < 2.0)

    def test_method_2(self):
        rng = np.random.default_rng(42)
        Z = rng.choice([0, 1, 2], size=(10, 20)).astype(float)
        res = gmatx(Z, method=2)
        assert res.extra["method"] == 2
        G = np.array(res.extra["G"])
        assert G.shape == (10, 10)

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            gmatx(np.ones((5, 10)), method=3)

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            gmatx(np.ones(10))
