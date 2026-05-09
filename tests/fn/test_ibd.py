"""Tests for moirais.fn.ibd -- Identity by descent."""

import numpy as np
import pytest
from moirais.fn.ibd import ibd


class TestIbd:
    def test_identical_individuals(self):
        Z = np.array([[0, 1, 2, 0, 1], [0, 1, 2, 0, 1]], dtype=float)
        res = ibd(Z)
        assert res.extra["pi_hat"][0][1] > 0.5

    def test_diagonal_is_one(self):
        rng = np.random.default_rng(42)
        Z = rng.choice([0, 1, 2], size=(5, 10)).astype(float)
        res = ibd(Z)
        pi = np.array(res.extra["pi_hat"])
        np.testing.assert_allclose(np.diag(pi), 1.0)

    def test_symmetric(self):
        rng = np.random.default_rng(42)
        Z = rng.choice([0, 1, 2], size=(5, 10)).astype(float)
        res = ibd(Z)
        pi = np.array(res.extra["pi_hat"])
        np.testing.assert_allclose(pi, pi.T, atol=1e-10)

    def test_too_few_individuals(self):
        with pytest.raises(ValueError):
            ibd(np.array([[0, 1, 2]]))
