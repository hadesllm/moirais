"""Tests for moirais.fn.admix — admixture proportions."""
import numpy as np
import pytest
from moirais.fn.admix import admixture_proportions


class TestAdmixture:
    def test_q_sums_to_one(self):
        G = np.random.default_rng(42).choice([0, 1, 2], size=(30, 50))
        res = admixture_proportions(G, K=2)
        Q = res.extra["Q"]
        np.testing.assert_allclose(Q.sum(axis=1), 1.0, atol=1e-6)

    def test_k_too_small_raises(self):
        with pytest.raises(ValueError):
            admixture_proportions(np.zeros((5, 5)), K=1)
