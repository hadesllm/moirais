"""Tests for mantel_test."""
import numpy as np, pytest
from moirais.fn.mantel import mantel_test

class TestMantel:
    def test_identical(self):
        rng = np.random.default_rng(42)
        pts = rng.normal(size=(20, 2))
        D = np.sqrt(((pts[:, None] - pts[None, :]) ** 2).sum(axis=2))
        r = mantel_test(D, D, n_perm=999, seed=0)
        assert r.statistic == pytest.approx(1.0)
        assert r.p_value < 0.05

    def test_random(self):
        rng = np.random.default_rng(0)
        D1 = np.abs(rng.normal(0, 1, (5, 5)))
        D1 = (D1 + D1.T) / 2
        np.fill_diagonal(D1, 0)
        D2 = np.abs(rng.normal(0, 1, (5, 5)))
        D2 = (D2 + D2.T) / 2
        np.fill_diagonal(D2, 0)
        r = mantel_test(D1, D2, n_perm=99, seed=0)
        assert 0 <= r.p_value <= 1
