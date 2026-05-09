"""Tests for moirais.fn.admxp -- Admixture proportions."""

import numpy as np
import pytest
from moirais.fn.admxp import admxp


class TestAdmxp:
    def test_q_sums_to_one(self):
        rng = np.random.default_rng(42)
        Z = rng.choice([0, 1, 2], size=(20, 30)).astype(float)
        res = admxp(Z, K=2, n_iter=50)
        Q = np.array(res.extra["Q"])
        np.testing.assert_allclose(np.sum(Q, axis=1), 1.0, atol=1e-6)

    def test_p_range(self):
        rng = np.random.default_rng(42)
        Z = rng.choice([0, 1, 2], size=(20, 30)).astype(float)
        res = admxp(Z, K=2, n_iter=50)
        P = np.array(res.extra["P"])
        assert np.all(P >= 0) and np.all(P <= 1)

    def test_k_equals_one(self):
        rng = np.random.default_rng(42)
        Z = rng.choice([0, 1, 2], size=(10, 20)).astype(float)
        res = admxp(Z, K=1, n_iter=50)
        Q = np.array(res.extra["Q"])
        np.testing.assert_allclose(Q, 1.0, atol=1e-6)

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            admxp(np.ones((5, 10)), K=10)
