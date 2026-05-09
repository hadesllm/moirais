"""Tests for moirais.fn.gwreg — GWR."""
import numpy as np
from moirais.fn.gwreg import gwr


class TestGWR:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n = 20
        coords = rng.uniform(0, 10, (n, 2))
        X = rng.standard_normal((n, 2))
        y = X @ [1, 0.5] + rng.standard_normal(n) * 0.5
        res = gwr(y, X, coords)
        assert res.extra["local_betas"].shape == (n, 2)

    def test_r2_bounded(self):
        rng = np.random.default_rng(42)
        n = 15
        coords = rng.uniform(0, 5, (n, 2))
        X = rng.standard_normal((n, 1))
        y = rng.standard_normal(n)
        res = gwr(y, X, coords)
        assert -0.5 <= res.value <= 1.1
