"""Tests for breusch_pagan."""
import numpy as np, pytest
from morie.fn.brpgn import breusch_pagan

class TestBP:
    def test_homoscedastic(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (500, 2))
        e = rng.normal(0, 1, 500)
        r = breusch_pagan(e, X)
        assert r.p_value > 0.05

    def test_hetero(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (200, 1))
        e = rng.normal(0, 1, 200) * (1 + 2 * np.abs(X[:, 0]))
        r = breusch_pagan(e, X)
        assert r.test_name == "Breusch-Pagan"
